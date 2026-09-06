'use strict';

const fs = require('fs');
const path = require('path');

/**
 * Disk cache in slippy-map layout: {root}/{kind}/{z}/{x}/{y}.{ext}
 *
 * Writes go to a temporary file and are renamed into place, so a reader never
 * observes a partially written tile even when two workers race on the same
 * cold entry.
 */
class TileCache {
  /**
   * `onHit` and `onWrite` report activity to whatever is managing the size cap.
   * They are left unset when no cap is configured, so the default path keeps
   * exactly the cost it had before.
   */
  constructor(root, { ttlMs = 0, onHit = null, onWrite = null } = {}) {
    // Resolved so that pathFor always yields an absolute path: the reaper keys
    // its index off these, and a relative root would make them ambiguous.
    this.root = path.resolve(root);
    this.ttlMs = ttlMs;
    this.onHit = onHit;
    this.onWrite = onWrite;
    this.seq = 0;
    fs.mkdirSync(root, { recursive: true });
  }

  pathFor(kind, z, x, y, ext) {
    return path.join(this.root, kind, String(z), String(x), `${y}.${ext}`);
  }

  /** Cached bytes when present and fresh, otherwise null. */
  read(kind, z, x, y, ext) {
    const file = this.pathFor(kind, z, x, y, ext);
    try {
      if (this.ttlMs > 0 && Date.now() - fs.statSync(file).mtimeMs > this.ttlMs) return null;
      const data = fs.readFileSync(file);
      // Reported after the read succeeds, so a miss never counts as a use.
      if (this.onHit) this.onHit(file);
      return data;
    } catch {
      return null;
    }
  }

  write(kind, z, x, y, ext, data) {
    const file = this.pathFor(kind, z, x, y, ext);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const tmp = `${file}.${process.pid}.${this.seq++}.tmp`;
    fs.writeFileSync(tmp, data);
    fs.renameSync(tmp, file);
    if (this.onWrite) this.onWrite(file, data.length);
    return file;
  }
}

module.exports = { TileCache };
