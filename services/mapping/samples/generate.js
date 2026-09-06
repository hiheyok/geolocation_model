'use strict';

/**
 * Regenerate the sample set against a running server.
 *
 *   node server.js &
 *   node samples/generate.js [baseUrl]
 *
 * Each sample is chosen to show a different mix of classes, so the folder
 * covers dense urban, water, parkland, countryside, a low zoom, and an
 * overzoomed tile above the source maxzoom.
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const sharp = require('sharp');
const { CLASSES, AMBIGUOUS } = require('../lib/styles');

const BASE = process.argv[2] || 'http://localhost:3000';
const OUT = __dirname;

const SAMPLES = [
  { name: '00-world-z0',      lat: 0,       lng:  0,      z: 0,
    about: 'The entire world in one tile. Web Mercator covers latitudes -85.05 to 85.05, so the poles are cut off. Coastlines and country-level landuse only; at this zoom the source data is heavily generalised and buildings and minor roads do not exist.' },
  { name: '01-urban-dense',   lat: 51.5045, lng: -0.0950, z: 14,
    about: 'Southwark, London. Dense buildings, full road hierarchy, a corner of the Thames.' },
  { name: '02-water-coast',   lat: 51.4600, lng:  0.3200, z: 13,
    about: 'Thames estuary. Water dominates; useful for checking the water class and coastline edges.' },
  { name: '03-park-green',    lat: 51.5073, lng: -0.1657, z: 14,
    about: 'Hyde Park, London. Large grass polygons against surrounding built-up landuse.' },
  { name: '04-rural',         lat: 52.2000, lng: -1.5000, z: 13,
    about: 'Warwickshire countryside. Mostly farmland and minor roads; sparse buildings.' },
  { name: '05-low-zoom',      lat: 51.5074, lng: -0.1278, z: 10,
    about: 'Greater London at z10. Whole-city context; note how little building detail survives.' },
  { name: '06-overzoom-z16',  lat: 51.5045, lng: -0.0950, z: 16,
    about: 'Same place as 01 but above the source maxzoom of 14, so the renderer overzooms a z14 parent. Label coordinates are scaled and offset to match.' },
];

const lngToX = (lng, z) => Math.floor(((lng + 180) / 360) * 2 ** z);
const latToY = (lat, z) => {
  const r = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.asinh(Math.tan(r)) / Math.PI) / 2) * 2 ** z);
};

function fetch(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => (res.statusCode === 200
        ? resolve(Buffer.concat(chunks))
        : reject(new Error(`${res.statusCode} ${url}`))));
    }).on('error', reject);
  });
}

const NAME = new Map(Object.entries(CLASSES).map(([k, v]) => [v, k]));
const nameOf = (v) => (v === AMBIGUOUS ? 'ambiguous' : NAME.get(v) ?? `UNKNOWN(${v})`);

/**
 * Single-channel pixel data for a mask PNG.
 *
 * toColourspace is not optional: sharp promotes a lone band to sRGB on decode,
 * which would return three interleaved copies of every pixel.
 */
const plane = (png) => sharp(png).toColourspace('b-w').raw().toBuffer();

/** Percentage of the mask occupied by each class, largest first. */
function classHistogram(px) {
  const counts = new Map();
  for (const v of px) counts.set(v, (counts.get(v) || 0) + 1);
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([v, n]) => ({
      class: nameOf(v),
      value: v,
      percent: +((100 * n) / px.length).toFixed(2),
    }));
}

/**
 * How much of each class `mode=ambiguous` declines to label.
 *
 * A flagged pixel is attributed to the class it carries in `mode=class`, so
 * this reads as "share of this class sitting on a contested boundary". Thin
 * linear classes score highest, since their antialiased edge is most of them.
 */
function flaggedByClass(classPx, ambPx) {
  const total = new Map(), flagged = new Map();
  for (let i = 0; i < classPx.length; i++) {
    const c = classPx[i];
    total.set(c, (total.get(c) || 0) + 1);
    if (ambPx[i] === AMBIGUOUS) flagged.set(c, (flagged.get(c) || 0) + 1);
  }
  return [...total.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([v, n]) => ({
      class: nameOf(v),
      value: v,
      flaggedPercent: +((100 * (flagged.get(v) || 0)) / n).toFixed(1),
    }));
}

(async () => {
  const index = [];

  for (const s of SAMPLES) {
    const x = lngToX(s.lng, s.z);
    const y = latToY(s.lat, s.z);
    const dir = path.join(OUT, s.name);
    fs.mkdirSync(dir, { recursive: true });

    const urls = {
      image:  `${BASE}/tile/${s.z}/${x}/${y}.png`,
      mask:   `${BASE}/tile/${s.z}/${x}/${y}/mask.png`,
      maskAmbiguous: `${BASE}/tile/${s.z}/${x}/${y}/mask.png?mode=ambiguous`,
      labels: `${BASE}/tile/${s.z}/${x}/${y}/labels.json`,
    };

    const [image, mask, maskAmb, labelsRaw] = await Promise.all(
      [urls.image, urls.mask, urls.maskAmbiguous, urls.labels].map(fetch)
    );
    const labels = JSON.parse(labelsRaw.toString());

    fs.writeFileSync(path.join(dir, 'image.png'), image);
    fs.writeFileSync(path.join(dir, 'mask.png'), mask);
    fs.writeFileSync(path.join(dir, 'mask_ambiguous.png'), maskAmb);
    fs.writeFileSync(path.join(dir, 'labels.json'), JSON.stringify(labels, null, 2) + '\n');

    const [classPx, ambPx] = await Promise.all([plane(mask), plane(maskAmb)]);
    let flaggedCount = 0;
    for (const v of ambPx) if (v === AMBIGUOUS) flaggedCount++;

    const meta = {
      about: s.about,
      tile: { z: s.z, x, y },
      center: { lat: s.lat, lng: s.lng },
      requests: {
        image: urls.image.replace(BASE, ''),
        mask: urls.mask.replace(BASE, ''),
        maskAmbiguous: urls.maskAmbiguous.replace(BASE, ''),
        labels: urls.labels.replace(BASE, ''),
      },
      files: {
        'image.png': { bytes: image.length, channels: 1, size: `${labels.size}x${labels.size}`,
          description: 'Photometric greyscale. Pixel values are cartographic brightness, not class ids.' },
        'mask.png': { bytes: mask.length, channels: 1, size: `${labels.size}x${labels.size}`,
          description: 'Class-index mask. Every pixel is one of the values in /classes.' },
        'mask_ambiguous.png': { bytes: maskAmb.length, channels: 1, size: `${labels.size}x${labels.size}`,
          description: 'As mask.png, but boundary pixels whose vote was split carry 255 '
            + '(ignore_index) instead of a guessed class. Unflagged pixels are identical.' },
        'labels.json': { bytes: labelsRaw.length, count: labels.count,
          description: 'Named features with x/y in pixels from the tile top-left.' },
      },
      maskClassBreakdown: classHistogram(classPx),
      ambiguous: {
        value: AMBIGUOUS,
        flaggedPercent: +((100 * flaggedCount) / classPx.length).toFixed(2),
        byClass: flaggedByClass(classPx, ambPx),
      },
      labelsByLayer: labels.labels.reduce((a, l) => (a[l.layer] = (a[l.layer] || 0) + 1, a), {}),
    };
    fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify(meta, null, 2) + '\n');

    index.push({ name: s.name, tile: meta.tile, labels: labels.count,
      ambiguousPercent: meta.ambiguous.flaggedPercent, about: s.about });
    console.log(`${s.name.padEnd(18)} z${s.z}/${x}/${y}  image ${(image.length / 1024).toFixed(0)}KB  ` +
      `mask ${(mask.length / 1024).toFixed(0)}KB  labels ${String(labels.count).padStart(4)}  ` +
      `flagged ${meta.ambiguous.flaggedPercent.toFixed(2)}%`);
  }

  fs.writeFileSync(path.join(OUT, 'index.json'),
    JSON.stringify({ generatedFrom: BASE, classes: CLASSES,
      ambiguous: AMBIGUOUS, samples: index }, null, 2) + '\n');
  console.log(`\nwrote ${SAMPLES.length} samples + index.json`);
})();
