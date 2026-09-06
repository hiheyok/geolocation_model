'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');

/**
 * Size cap and LRU eviction for the tile cache.
 *
 * Eviction is centralised in one process — the cluster primary — because the
 * workers all write into the same directory. If each evicted independently they
 * would double-count the total and delete entries the others had just written.
 * Workers therefore only *report* what they touched; the reaper decides.
 *
 * Recency comes from those reports rather than from the filesystem. atime looks
 * like the obvious source, but `relatime` (the Linux default) only advances it
 * once a day, and `noatime` never does, which would silently degrade LRU into
 * "least recently written" and evict tiles that are being read constantly.
 * atime is used only to seed entries already on disk at startup, where it is
 * the sole evidence available.
 */

const UNITS = { b: 1, kb: 1024, mb: 1024 ** 2, gb: 1024 ** 3, tb: 1024 ** 4 };

/** Parse "20GB", "1.5tb", "500mb", or a plain byte count. 0 means unlimited. */
function parseSize(value) {
  const m = /^(\d+(?:\.\d+)?)\s*(b|kb|mb|gb|tb)?$/i.exec(String(value ?? '').trim());
  if (!m) return 0;
  return Math.floor(parseFloat(m[1]) * (m[2] ? UNITS[m[2].toLowerCase()] : 1));
}

const humanSize = (n) => {
  if (n < 1024) return `${n} B`;
  const u = ['KB', 'MB', 'GB', 'TB'];
  let v = n / 1024, i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(1)} ${u[i]}`;
};

class CacheReaper {
  /**
   * @param root            cache directory
   * @param maxBytes        hard cap; 0 disables eviction entirely
   * @param lowWaterRatio   sweep down to this fraction of the cap, so that a
   *                        cache sitting at the limit does not evict on every
   *                        single write
   * @param maxTmpAgeMs     age at which an orphaned .tmp file (a writer that
   *                        died between write and rename) is cleaned up
   */
  constructor(root, { maxBytes = 0, lowWaterRatio = 0.9, maxTmpAgeMs = 5 * 60_000 } = {}) {
    this.root = path.resolve(root);
    this.maxBytes = maxBytes;
    this.lowWater = Math.floor(maxBytes * lowWaterRatio);
    this.maxTmpAgeMs = maxTmpAgeMs;
    // relative path -> { bytes, used }. Relative keeps the index materially
    // smaller: at ~62 KB per tile a 100 GB cache holds ~1.6M of these.
    this.entries = new Map();
    this.total = 0;
    this.scanned = false;
    this.evictedTotal = 0;
  }

  /**
   * Index key for a path. Resolved against the cwd first, so a caller reporting
   * a relative path cannot land outside the root and silently fail to evict.
   * Returns null for anything not under the cache root.
   */
  key(file) {
    const rel = path.relative(this.root, path.resolve(file));
    return rel && !rel.startsWith('..') ? rel : null;
  }

  /** Seed the index from whatever is already on disk. */
  async scan() {
    const walk = async (dir) => {
      let items;
      try {
        items = await fsp.readdir(dir, { withFileTypes: true });
      } catch {
        return;                                   // raced with an eviction
      }
      for (const item of items) {
        const full = path.join(dir, item.name);
        if (item.isDirectory()) { await walk(full); continue; }
        if (!item.isFile()) continue;

        let st;
        try { st = await fsp.stat(full); } catch { continue; }

        if (item.name.endsWith('.tmp')) {
          // A writer died between writeFileSync and renameSync. Only reap ones
          // old enough that no live writer could still be holding them.
          if (Date.now() - st.mtimeMs > this.maxTmpAgeMs) {
            await fsp.unlink(full).catch(() => {});
          }
          continue;
        }
        this.entries.set(path.relative(this.root, full), {
          bytes: st.size,
          // Best available evidence for a file this process has never served.
          used: Math.max(st.atimeMs, st.mtimeMs),
        });
        this.total += st.size;
      }
    };

    await walk(this.root);
    this.scanned = true;
    return { files: this.entries.size, bytes: this.total };
  }

  /** A worker wrote a tile. */
  note(file, bytes) {
    const k = this.key(file);
    if (k === null) return;
    const prev = this.entries.get(k);
    if (prev) this.total -= prev.bytes;
    this.entries.set(k, { bytes, used: Date.now() });
    this.total += bytes;
  }

  /** A worker served a tile from cache — this is what makes it LRU. */
  touch(file) {
    const k = this.key(file);
    if (k === null) return;
    const e = this.entries.get(k);
    if (e) e.used = Date.now();
  }

  /**
   * Delete least-recently-used entries until the cache is under the low-water
   * mark. Sorting the whole index is only paid when actually over the cap.
   */
  sweep() {
    if (!this.maxBytes || !this.scanned || this.total <= this.maxBytes) {
      return { evicted: 0, freed: 0, total: this.total };
    }

    const byAge = [...this.entries].sort((a, b) => a[1].used - b[1].used);
    let evicted = 0, freed = 0, missing = 0;

    for (const [rel, entry] of byAge) {
      if (this.total <= this.lowWater) break;
      const full = path.join(this.root, rel);
      let removed = true;
      try {
        fs.unlinkSync(full);
      } catch (err) {
        // The entry leaves the index either way, but a file that was never
        // there is counted separately. Reporting it as freed would let a path
        // mismatch look like a working cap while the cache grew unbounded.
        if (err.code !== 'ENOENT') continue;
        removed = false;
        missing++;
      }
      this.entries.delete(rel);
      this.total -= entry.bytes;
      if (removed) { freed += entry.bytes; evicted++; this.pruneDirs(path.dirname(full)); }
    }

    this.evictedTotal += evicted;
    return { evicted, freed, missing, total: this.total };
  }

  /** Remove directories left empty by eviction, up to the cache root. */
  pruneDirs(dir) {
    while (dir.startsWith(this.root) && dir !== this.root) {
      try { fs.rmdirSync(dir); } catch { return; }   // ENOTEMPTY ends the walk
      dir = path.dirname(dir);
    }
  }

  stats() {
    return {
      files: this.entries.size,
      bytes: this.total,
      size: humanSize(this.total),
      maxBytes: this.maxBytes,
      maxSize: this.maxBytes ? humanSize(this.maxBytes) : 'unlimited',
      usedPercent: this.maxBytes ? +((100 * this.total) / this.maxBytes).toFixed(1) : null,
      evictedTotal: this.evictedTotal,
      scanned: this.scanned,
    };
  }
}

module.exports = { CacheReaper, parseSize, humanSize };
