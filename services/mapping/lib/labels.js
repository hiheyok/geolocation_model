'use strict';

const { VectorTile } = require('@mapbox/vector-tile');

// pbf v4 ships an ES module default export; v5 renamed the reader to
// PbfReader. Accept either so a minor bump does not break decoding.
const pbf = require('pbf');
const Pbf = pbf.PbfReader || pbf.default || pbf;

/** Source layers that carry names worth emitting as annotations. */
const LABEL_LAYERS = ['place', 'water_name', 'transportation_name', 'poi',
  'mountain_peak', 'park', 'aerodrome_label'];

/**
 * Which .mbtiles tile backs a rendered tile, and where the rendered tile sits
 * inside it. At or below the source maxzoom this is the identity; above it the
 * renderer overzooms a parent, so annotations must be scaled and offset to
 * match.
 */
function sourceTileFor(z, x, y, maxzoom) {
  if (z <= maxzoom) return { z, x, y, scale: 1, offsetX: 0, offsetY: 0 };
  const shift = z - maxzoom;
  const s = 2 ** shift;
  return {
    z: maxzoom, x: x >> shift, y: y >> shift,
    scale: s, offsetX: x - ((x >> shift) * s), offsetY: y - ((y >> shift) * s),
  };
}

/**
 * Decode named features from a vector tile into labels carrying pixel
 * coordinates relative to the rendered tile's top-left corner.
 *
 * Vector tile geometry is already in tile-local units (`extent`, normally
 * 4096), so the conversion is a scale — the rendered image *is* this tile.
 */
function extractLabels(buffer, {
  size = 512, src, layers = LABEL_LAYERS, lang = 'latin', limit = 0,
} = {}) {
  const tile = new VectorTile(new Pbf(buffer));
  const out = [];

  for (const name of layers) {
    const layer = tile.layers[name];
    if (!layer) continue;
    // Position within the source tile, then within the rendered sub-tile.
    const k = (size * src.scale) / layer.extent;

    for (let i = 0; i < layer.length; i++) {
      const f = layer.feature(i);
      const text = f.properties[`name:${lang}`] || f.properties.name;
      if (!text) continue;

      const points = f.loadGeometry().flat();
      if (!points.length) continue;
      // Points carry their own anchor; lines and polygons get their midpoint,
      // which approximates where a renderer would centre a label.
      const anchor = f.type === 1 ? points[0] : points[Math.floor(points.length / 2)];

      const px = anchor.x * k - src.offsetX * size;
      const py = anchor.y * k - src.offsetY * size;
      // Tiles carry geometry beyond their edges as a buffer; drop anchors that
      // fall outside the rendered frame.
      if (px < 0 || py < 0 || px > size || py > size) continue;

      out.push({
        layer: name,
        text,
        class: f.properties.class ?? f.properties.subclass ?? null,
        rank: f.properties.rank ?? null,
        geometry: ['', 'point', 'line', 'polygon'][f.type],
        x: Math.round(px * 10) / 10,
        y: Math.round(py * 10) / 10,
      });
    }
  }

  // Rank ascends with decreasing importance in the OpenMapTiles schema; keep
  // the most prominent features when a caller caps the count.
  out.sort((a, b) => (a.rank ?? 1e6) - (b.rank ?? 1e6));
  return limit > 0 ? out.slice(0, limit) : out;
}

module.exports = { extractLabels, sourceTileFor, LABEL_LAYERS };
