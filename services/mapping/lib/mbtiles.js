'use strict';

const Database = require('better-sqlite3');
const zlib = require('zlib');

/**
 * Read-only accessor for an OpenMapTiles-schema .mbtiles file.
 *
 * The `tiles` relation is a view joining `map` -> `images`, and the MapTiler
 * planet stores rows in TMS order, so the XYZ y is flipped on the way in.
 */
class MBTiles {
  constructor(file, { cacheMb = 64 } = {}) {
    this.db = new Database(file, { readonly: true, fileMustExist: true });
    this.db.pragma(`cache_size = -${cacheMb * 1024}`);
    this.db.pragma('temp_store = MEMORY');
    this.stmt = this.db.prepare(
      'SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?'
    );

    const meta = {};
    for (const { name, value } of this.db.prepare('SELECT name, value FROM metadata').all()) {
      meta[name] = value;
    }
    this.minzoom = parseInt(meta.minzoom ?? '0');
    this.maxzoom = parseInt(meta.maxzoom ?? '14');
    this.format = meta.format || 'pbf';
  }

  /** Raw (still gzipped) tile bytes, or null when the tile is absent. */
  getRaw(z, x, y) {
    const row = this.stmt.get(z, x, (2 ** z) - 1 - y);
    return row ? Buffer.from(row.tile_data) : null;
  }

  /** Decompressed vector tile bytes, or null. */
  get(z, x, y) {
    const data = this.getRaw(z, x, y);
    if (!data) return null;
    return (data[0] === 0x1f && data[1] === 0x8b) ? zlib.gunzipSync(data) : data;
  }

  close() {
    this.db.close();
  }
}

module.exports = { MBTiles };
