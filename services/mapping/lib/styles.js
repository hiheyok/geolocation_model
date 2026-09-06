'use strict';

/**
 * Two purpose-built styles, both label-free and both painted in pure greys
 * (r == g == b) so that channel 0 of the render *is* the value — no luma
 * conversion, no rounding, and a 1-channel extract costs nothing.
 *
 * Neither style references glyphs or sprites, so rendering issues zero network
 * requests: everything comes from the local .mbtiles.
 */

const SOURCE = 'omt';

/**
 * Semantic classes for the mask. Values are spaced >= 20 apart so that
 * antialiased edge pixels can be snapped back to a legal class (see
 * buildSnapLut) without collapsing two classes into each other.
 */
const CLASSES = {
  background:  0,
  water:      23,
  ice:        46,
  wood:       69,
  grass:      92,
  farmland:  115,
  landuse:   138,
  building:  161,
  railway:   184,
  road_minor: 207,
  road_major: 230,
  aeroway:   253,
};

/**
 * Greyscale paint values for the photometric image.
 *
 * Evenly spaced across the full 0..255 range (~25 apart) rather than clustered
 * at the light end. A narrow spread is what makes a conventional cartographic
 * palette collapse when reduced to one channel: neighbouring classes land
 * within a few luma of each other and become indistinguishable.
 */
const MONO = {
  background: 255,
  ice:        232,
  farmland:   209,
  landuse:    186,
  grass:      163,
  wood:       140,
  road_minor: 117,
  building:    94,
  road_major:  71,
  railway:     48,
  waterway:    25,
  water:        0,
};

/**
 * Reserved value for a pixel the renderer cannot attribute to one class.
 *
 * 255 is the conventional `ignore_index` in segmentation training (Cityscapes
 * and PASCAL VOC both use it), and sits clear of the class ids, which stop at
 * 253. It is only ever emitted in the `ambiguous` mask mode.
 */
const AMBIGUOUS = 255;

const grey = (v) => `rgb(${v},${v},${v})`;

const source = (maxzoom) => ({
  [SOURCE]: { type: 'vector', tiles: ['mbtiles://{z}/{x}/{y}'], minzoom: 0, maxzoom },
});

const MINOR_ROADS = ['minor', 'service', 'track', 'path'];
const MAJOR_ROADS = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary'];

/** Road widths grow with zoom so lines stay legible at every level. */
const roadWidth = (base) => ({
  base: 1.4,
  stops: [[6, base * 0.4], [12, base], [16, base * 2.5], [20, base * 6]],
});

/**
 * Layer skeleton shared by both styles. `p` maps a semantic name to the grey
 * value that style paints it with; layers whose class is absent are dropped.
 */
function layers(p) {
  const fill = (id, sourceLayer, key, filter) => (p[key] === undefined ? null : {
    id, type: 'fill', source: SOURCE, 'source-layer': sourceLayer,
    ...(filter ? { filter } : {}),
    // Fills are flat regions; antialiasing them only blurs class boundaries.
    paint: { 'fill-color': grey(p[key]), 'fill-antialias': false },
  });

  const line = (id, sourceLayer, key, width, filter) => (p[key] === undefined ? null : {
    id, type: 'line', source: SOURCE, 'source-layer': sourceLayer,
    ...(filter ? { filter } : {}),
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': grey(p[key]), 'line-width': roadWidth(width) },
  });

  return [
    { id: 'background', type: 'background', paint: { 'background-color': grey(p.background) } },
    fill('farmland', 'landcover', 'farmland', ['==', 'class', 'farmland']),
    fill('landuse', 'landuse', 'landuse', ['in', 'class', 'residential', 'commercial', 'industrial', 'retail']),
    fill('grass', 'park', 'grass'),
    fill('wood', 'landcover', 'wood', ['==', 'class', 'wood']),
    // Ice and glacier are the dominant land cover at low zoom, where landuse
    // and vegetation are not present at all — without this, Antarctica and
    // Greenland render as undifferentiated background.
    fill('ice', 'landcover', 'ice', ['==', 'class', 'ice']),
    fill('water', 'water', 'water', ['!=', 'brunnel', 'tunnel']),
    line('waterway', 'waterway', 'waterway', 1),
    fill('aeroway_area', 'aeroway', 'aeroway', ['==', '$type', 'Polygon']),
    fill('building', 'building', 'building'),
    line('railway', 'transportation', 'railway', 1.2, ['==', 'class', 'rail']),
    line('road_minor', 'transportation', 'road_minor', 1.5, ['in', 'class', ...MINOR_ROADS]),
    line('road_major', 'transportation', 'road_major', 3, ['in', 'class', ...MAJOR_ROADS]),
  ].filter(Boolean);
}

/** Photometric greyscale map image — visually map-like, wide luma spread. */
function buildMonoStyle(maxzoom = 14) {
  return { version: 8, name: 'mono', sources: source(maxzoom), layers: layers(MONO) };
}

/** Class-index mask — each pixel value is a semantic class id. */
function buildMaskStyle(maxzoom = 14) {
  return { version: 8, name: 'mask', sources: source(maxzoom), layers: layers(CLASSES) };
}

/**
 * 256-entry lookup table mapping any byte to the nearest legal class value.
 *
 * MapLibre always antialiases line geometry (`fill-antialias: false` only
 * covers fills), so roughly one pixel in six along road edges lands between two
 * class values. Snapping restores a valid label everywhere. The cost is that
 * an edge pixel is assigned by numeric proximity rather than semantics, which
 * leaves about half a pixel of boundary error along class edges.
 */
function buildSnapLut(classes = CLASSES) {
  const values = [...new Set(Object.values(classes))].sort((a, b) => a - b);
  return Buffer.from(Array.from({ length: 256 }, (_, v) =>
    values.reduce((best, c) => (Math.abs(c - v) < Math.abs(best - v) ? c : best), values[0])
  ));
}

module.exports = { CLASSES, MONO, AMBIGUOUS, buildMonoStyle, buildMaskStyle, buildSnapLut };
