'use strict';

const mbgl = require('@maplibre/maplibre-gl-native');
const sharp = require('sharp');
const { CLASSES, AMBIGUOUS, buildMonoStyle, buildMaskStyle, buildSnapLut } = require('./styles');

/** [lng, lat] of the centre of an XYZ tile. */
function tileCenter(z, x, y) {
  const n = 2 ** z;
  const lng = ((x + 0.5) / n) * 360 - 180;
  const lat = Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 0.5)) / n))) * 180 / Math.PI;
  return [lng, lat];
}

/**
 * One MapLibre map plus the serialisation around it.
 *
 * maplibre-gl-native is not thread-safe and a Map owns a run loop, so renders
 * are queued one at a time per instance. Parallelism comes from running several
 * worker processes, each with its own Renderer.
 */
class Renderer {
  constructor(style, { tiles, size = 512, ratio = 1 }) {
    this.size = size;
    this.ratio = ratio;
    this.tiles = tiles;
    this.queue = Promise.resolve();

    this.map = new mbgl.Map({
      request: (req, callback) => {
        const url = req.url;
        // Both styles are self-contained: no glyphs, no sprites, no HTTP. A
        // request for anything else is a bug in the style, not a fetch to make.
        if (!url.startsWith('mbtiles://')) {
          return callback(new Error(`unexpected non-mbtiles resource: ${url}`));
        }
        const [z, x, y] = url.slice('mbtiles://'.length).split('/').map(Number);
        const data = this.tiles.get(z, x, y);
        // An absent tile is normal (ocean, out of coverage) — render empty.
        return callback(null, data ? { data } : {});
      },
      // Supersampling is a pixel-ratio change, not a viewport change: at ratio N
      // the returned buffer is N times larger on each edge but covers exactly
      // the same ground, and line widths scale with it. Enlarging the viewport
      // instead would widen the extent to an N x N block of tiles.
      ratio,
    });
    this.map.load(style);
  }

  /** Raw RGBA pixels for a tile. Serialised against this instance. */
  renderRaw(z, x, y) {
    const result = this.queue.then(() => new Promise((resolve, reject) => {
      this.map.render(
        { zoom: z, center: tileCenter(z, x, y), width: this.size, height: this.size },
        (err, buffer) => (err ? reject(err) : resolve(buffer))
      );
    }));
    // The queue is only a sequencing token: absorb failures here so one bad
    // render cannot leave a rejected promise for every later render to chain on.
    this.queue = result.catch(() => {});
    return result;
  }

  release() {
    this.map.release();
  }
}

/**
 * Owns both renderers for a worker and turns tiles into encoded PNG bytes.
 * The mask renderer is created on first use, since most callers only ever ask
 * for imagery.
 */
class TileRenderer {
  constructor({ tiles, size = 512, compression = 1, maskSupersample = 2,
                ambiguousThreshold = 0.5 }) {
    this.opts = { tiles, size };
    this.compression = compression;
    this.ss = Math.max(1, maskSupersample);
    // A pixel is called ambiguous when its winning class holds no more than
    // this share of the legal samples. At ss = 2 that is a 2-2 tie.
    this.ambiguousThreshold = ambiguousThreshold;
    this.mono = new Renderer(buildMonoStyle(tiles.maxzoom), this.opts);
    this.mask = null;
    this.lut = buildSnapLut();
    this.legal = new Set(Object.values(CLASSES));
  }

  /**
   * Reduce a supersampled mask to output resolution by majority vote.
   *
   * Snapping alone cannot tell a genuine class from a blend of two classes
   * either side of it: mixing values 23 and 69 lands on 46, which is a
   * different class entirely. Rendering larger means each output pixel is
   * backed by several samples, most of which are interior — so taking the most
   * common legal value recovers the true class and confines error to pixels
   * that really are on a boundary.
   */
  downsampleMode(plane, size, ss, ambiguous = false) {
    const out = Buffer.allocUnsafe(size * size);
    const wide = size * ss;
    const values = new Uint8Array(ss * ss);
    const counts = new Uint8Array(ss * ss);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let seen = 0, sum = 0;
        for (let dy = 0; dy < ss; dy++) {
          for (let dx = 0; dx < ss; dx++) {
            const v = plane[(y * ss + dy) * wide + (x * ss + dx)];
            sum += v;
            if (!this.legal.has(v)) continue;
            let k = 0;
            while (k < seen && values[k] !== v) k++;
            if (k === seen) { values[seen] = v; counts[seen] = 1; seen++; }
            else counts[k]++;
          }
        }
        if (seen === 0) {
          // Every sample was a blend. There is nothing to vote on, so the
          // honest answer is "unknown"; otherwise fall back to snapping the mean.
          out[y * size + x] = ambiguous ? AMBIGUOUS : this.lut[Math.round(sum / (ss * ss))];
        } else {
          let best = 0;
          for (let k = 1; k < seen; k++) if (counts[k] > counts[best]) best = k;
          // A split vote means the samples genuinely disagree — distinct from a
          // pixel that is merely blended but still dominated by one class.
          out[y * size + x] = ambiguous && counts[best] / seen <= this.ambiguousThreshold
            ? AMBIGUOUS
            : values[best];
        }
      }
    }
    return out;
  }

  /**
   * A single-channel PNG. `kind` is 'image' (photometric greyscale) or 'mask'
   * (class indices).
   *
   * Mask `mode` selects how antialiased boundary pixels are resolved:
   *   class      every pixel gets a class, decided by majority vote
   *   ambiguous  as above, but pixels whose vote is split become AMBIGUOUS
   *   raw        no supersampling and no snapping — blends as rendered
   */
  async render(z, x, y, { kind = 'image', mode = 'class' } = {}) {
    const { size } = this.opts;
    // Masks are supersampled so the downsample has interior samples to vote
    // with; imagery, and raw masks, render at output size.
    const ss = kind === 'mask' && mode !== 'raw' ? this.ss : 1;

    if (kind === 'mask') {
      // Two mask renderers only exist when supersampling is on and a caller
      // also asks for raw values; each is built on first use.
      if (ss > 1 && !this.mask) {
        this.mask = new Renderer(buildMaskStyle(this.opts.tiles.maxzoom), {
          ...this.opts, ratio: ss,
        });
      } else if (ss === 1 && !this.maskRaw) {
        this.maskRaw = new Renderer(buildMaskStyle(this.opts.tiles.maxzoom), this.opts);
      }
    }
    const renderer = kind === 'mask' ? (ss === 1 ? this.maskRaw : this.mask) : this.mono;
    const rendered = size * ss;
    const rgba = await renderer.renderRaw(z, x, y);

    // Every paint colour is a pure grey, so channel 0 is the value exactly —
    // cheaper and lossless compared with a luma conversion.
    let plane = await sharp(rgba, { raw: { width: rendered, height: rendered, channels: 4 } })
      .extractChannel(0).raw().toBuffer();

    if (kind === 'mask' && mode !== 'raw') {
      // At ss = 1 this degenerates to a one-sample vote, which is exactly
      // snapping (or flagging) each blended pixel on its own.
      plane = this.downsampleMode(plane, size, ss, mode === 'ambiguous');
    }

    // toColourspace('b-w') is required: sharp promotes a single band to sRGB on
    // encode by default, which writes PNG colour type 2 (RGB) with r==g==b.
    // That still decodes to the right values but triples the channels, so
    // consumers get (H, W, 3) instead of (H, W).
    return sharp(plane, { raw: { width: size, height: size, channels: 1 } })
      .toColourspace('b-w')
      .png({ compressionLevel: this.compression })
      .toBuffer();
  }

  release() {
    this.mono.release();
    if (this.mask) this.mask.release();
    if (this.maskRaw) this.maskRaw.release();
  }
}

module.exports = { TileRenderer, Renderer, tileCenter };
