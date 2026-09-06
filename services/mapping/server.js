'use strict';

const cluster = require('cluster');
const express = require('express');
const os = require('os');
const path = require('path');

const { MBTiles } = require('./lib/mbtiles');
const { TileCache } = require('./lib/cache');
const { TileRenderer } = require('./lib/renderer');
const { extractLabels, sourceTileFor, LABEL_LAYERS } = require('./lib/labels');
const { CLASSES, AMBIGUOUS } = require('./lib/styles');
const { CacheReaper, parseSize } = require('./lib/reaper');
const { buildVariants, staleNamespaces } = require('./lib/variants');
const { rendererIdentity } = require('./lib/identity');

// ── Config ────────────────────────────────────────────────────────────────────

const CONFIG = {
  mbtiles:     process.env.MBTILES_PATH || './planet.mbtiles',
  outputDir:   process.env.OUTPUT_DIR   || './tiles',
  port:        parseInt(process.env.PORT || '3000'),
  // Renders are CPU-bound in native code; leave a couple of cores for the OS,
  // sharp's thread pool, and the SQLite reads.
  workers:     parseInt(process.env.WORKERS || String(Math.max(1, os.cpus().length - 2))),
  tileSize:    parseInt(process.env.TILE_SIZE || '512'),
  // Level 1 costs ~1.3ms per tile against ~2.7ms at level 6, for ~50% more
  // bytes. Worth it when the consumer is on a fast link.
  compression: parseInt(process.env.PNG_LEVEL || '1'),
  cacheTtlMs:  parseInt(process.env.CACHE_TTL_MS || '0'),
  // Hard cap on the disk cache, e.g. "20GB". Unset means unbounded growth,
  // which is the historical behaviour.
  cacheMaxBytes: parseSize(process.env.CACHE_MAX_SIZE || '0'),
  cacheSweepMs:  parseInt(process.env.CACHE_SWEEP_MS || '30000'),
  // Masks render at this multiple of the output size, then downsample by
  // majority vote. 1 disables it and falls back to plain nearest-value snapping.
  maskSupersample: parseInt(process.env.MASK_SUPERSAMPLE || '2'),
  // In mask mode=ambiguous, a pixel is flagged when its winning class holds no
  // more than this share of the samples. At supersample 2 that is a 2-2 tie.
  ambiguousThreshold: parseFloat(process.env.MASK_AMBIGUOUS_THRESHOLD || '0.5'),
  maxZoom:     parseInt(process.env.MAX_ZOOM || '20'),
};

// ── Primary: fork workers, keep them alive ────────────────────────────────────

if (cluster.isPrimary && CONFIG.workers > 1) {
  console.log(`Tile server on http://localhost:${CONFIG.port}  (${CONFIG.workers} workers)`);
  console.log(`MBTiles : ${CONFIG.mbtiles}`);
  console.log(`Output  : ${path.resolve(CONFIG.outputDir)}`);

  // Opened only to read metadata for the namespace hash, then closed.
  const probe = new MBTiles(CONFIG.mbtiles);
  const variants = variantsFor(probe.maxzoom);
  probe.close();
  console.log(`Cache ns: ${Object.values(variants).join('  ')}`);

  const stale = staleNamespaces(CONFIG.outputDir, variants);
  if (stale.length) {
    console.warn(`cache: ${stale.length} namespace(s) from a previous configuration `
      + `will never be read again: ${stale.join(', ')}`);
    console.warn('cache: delete them, or leave them to age out if CACHE_MAX_SIZE is set');
  }

  const reaper = startReaper();
  if (reaper) {
    // Workers cannot evict safely on their own, so they report what they
    // touched and the primary is the only process that deletes.
    cluster.on('message', (_worker, msg) => {
      if (!msg || msg.t !== 'cache') return;
      for (const [file, bytes] of msg.writes) reaper.note(file, bytes);
      for (const file of msg.hits) reaper.touch(file);
    });
  }

  for (let i = 0; i < CONFIG.workers; i++) cluster.fork();

  cluster.on('exit', (worker, code, signal) => {
    if (!worker.exitedAfterDisconnect) {
      console.error(`worker ${worker.process.pid} died (${signal || code}) — restarting`);
      cluster.fork();
    }
  });

  for (const sig of ['SIGINT', 'SIGTERM']) {
    process.on(sig, () => {
      for (const w of Object.values(cluster.workers)) w.kill(sig);
      process.exit(0);
    });
  }
} else {
  startWorker();
}

/** Cache namespaces for the running configuration. */
function variantsFor(maxzoom) {
  return buildVariants({
    mbtiles: CONFIG.mbtiles,
    maxzoom,
    tileSize: CONFIG.tileSize,
    compression: CONFIG.compression,
    maskSupersample: CONFIG.maskSupersample,
    ambiguousThreshold: CONFIG.ambiguousThreshold,
  });
}

// ── Cache size cap ────────────────────────────────────────────────────────────

/**
 * Build the reaper and run it on a timer. Returns null when no cap is set, in
 * which case no reporting hooks are installed anywhere and the cache behaves
 * exactly as it did before.
 */
function startReaper() {
  if (!CONFIG.cacheMaxBytes) return null;

  const reaper = new CacheReaper(CONFIG.outputDir, { maxBytes: CONFIG.cacheMaxBytes });

  // Scanning an existing cache can take a while on a large tree; sweeps are
  // suppressed until it finishes so a partial total cannot over-evict.
  reaper.scan().then(({ files, bytes }) => {
    console.log(`Cache   : ${reaper.stats().maxSize} cap, ${files} files (${reaper.stats().size}) indexed`);
    const r = reaper.sweep();
    if (r.evicted) console.log(`cache: evicted ${r.evicted} entries on startup`);
  }).catch((err) => console.error('cache scan failed:', err.message));

  const timer = setInterval(() => {
    const r = reaper.sweep();
    if (r.evicted || r.missing) {
      console.log(`cache: evicted ${r.evicted} entries, freed ${(r.freed / 1024 ** 2).toFixed(0)} MB, `
        + `now ${reaper.stats().size} (${reaper.stats().usedPercent}% of cap)`
        // Indexed but absent from disk. A handful is normal (external deletion);
        // a large share means the index and the cache have diverged.
        + (r.missing ? `  [${r.missing} indexed files were already gone]` : ''));
    }
    // Only the primary holds the index, so push its view out for /health.
    const stats = reaper.stats();
    for (const w of Object.values(cluster.workers || {})) {
      if (w.isConnected()) w.send({ t: 'cache-stats', stats });
    }
  }, CONFIG.cacheSweepMs);
  timer.unref();

  return reaper;
}

/**
 * Batched reporting from a worker to the primary.
 *
 * Cache hits run at well over 10,000/s, so one IPC message per hit would cost
 * more than the read it is describing. Hits are deduplicated into a set and
 * flushed once a second, which is ample resolution for eviction ordering.
 */
let lastCacheStats = null;

function cacheReporter(reaper) {
  if (!CONFIG.cacheMaxBytes) return {};

  // Single-process mode: no IPC, drive the reaper directly.
  if (reaper) return { onWrite: (f, b) => reaper.note(f, b), onHit: (f) => reaper.touch(f) };

  const writes = [];
  const hits = new Set();
  process.on('message', (msg) => {
    if (msg && msg.t === 'cache-stats') lastCacheStats = msg.stats;
  });
  const flush = () => {
    if (!writes.length && !hits.size) return;
    process.send({ t: 'cache', writes: writes.splice(0), hits: [...hits] });
    hits.clear();
  };
  const timer = setInterval(flush, 1000);
  timer.unref();

  return { onWrite: (f, b) => writes.push([f, b]), onHit: (f) => hits.add(f) };
}

/**
 * Cache state for /health. In cluster mode this is the primary's last broadcast,
 * so it lags by up to one sweep interval and is absent until the first sweep.
 */
function cacheStats(localReaper) {
  if (!CONFIG.cacheMaxBytes) return { maxSize: 'unlimited' };
  if (localReaper) return localReaper.stats();
  return lastCacheStats || { maxSize: 'pending', note: 'awaiting first sweep from primary' };
}

// ── Worker ────────────────────────────────────────────────────────────────────

function startWorker() {
  const tiles = new MBTiles(CONFIG.mbtiles);
  const variants = variantsFor(tiles.maxzoom);

  // Single process: no primary ran, so report namespaces here instead.
  if (!cluster.isWorker) {
    console.log(`Cache ns: ${Object.values(variants).join('  ')}`);
    const stale = staleNamespaces(CONFIG.outputDir, variants);
    if (stale.length) {
      console.warn(`cache: ${stale.length} namespace(s) from a previous configuration `
        + `will never be read again: ${stale.join(', ')}`);
    }
  }

  // With one process there is no primary to report to, so it owns the reaper.
  const localReaper = cluster.isWorker ? null : startReaper();
  const cache = new TileCache(CONFIG.outputDir, {
    ttlMs: CONFIG.cacheTtlMs,
    ...cacheReporter(localReaper),
  });
  const renderer = new TileRenderer({
    tiles, size: CONFIG.tileSize, compression: CONFIG.compression,
    maskSupersample: CONFIG.maskSupersample,
    ambiguousThreshold: CONFIG.ambiguousThreshold,
  });

  // Coalesce concurrent requests for the same cold entry: without this, N
  // simultaneous callers each pay a full render for one tile.
  const inFlight = new Map();
  function once(key, fn) {
    const running = inFlight.get(key);
    if (running) return running;
    const p = fn().finally(() => inFlight.delete(key));
    inFlight.set(key, p);
    return p;
  }

  /** Reject anything that is not a valid tile address before it reaches the fs. */
  function parseTile(req, res) {
    const z = Number(req.params.z), x = Number(req.params.x), y = Number(req.params.y);
    const ok = [z, x, y].every(Number.isInteger)
      && z >= 0 && z <= CONFIG.maxZoom
      && x >= 0 && x < 2 ** z && y >= 0 && y < 2 ** z;
    if (!ok) {
      res.status(400).json({ error: 'invalid tile address', z: req.params.z, x: req.params.x, y: req.params.y });
      return null;
    }
    return { z, x, y };
  }

  function sendPng(res, body, cached, mode) {
    res.set({
      'Content-Type': 'image/png',
      'Cache-Control': 'public, max-age=31536000, immutable',
      'X-Tile-Cached': cached ? '1' : '0',
      'X-Tile-Channels': '1',
      // Echo the scheme back so a caller can tell from the response alone
      // whether 255 may appear in the pixels.
      ...(mode ? { 'X-Mask-Mode': mode } : {}),
    });
    res.send(body);
  }

  const app = express();
  app.disable('x-powered-by');

  const MASK_MODES = ['class', 'ambiguous', 'raw'];

  /**
   * Which resolution scheme a mask request asked for.
   *
   *   class      (default) every pixel carries a class id
   *   ambiguous  boundary pixels whose vote is split carry 255 instead, for
   *              use as ignore_index in training
   *   raw        blends exactly as rendered, no supersampling or snapping
   *
   * ?snap=0 is the original spelling of mode=raw and still works.
   */
  function maskMode(req) {
    if (req.query.mode !== undefined) {
      const m = String(req.query.mode);
      return MASK_MODES.includes(m) ? m : null;
    }
    return req.query.snap === '0' ? 'raw' : 'class';
  }

  /**
   * GET /tile/:z/:x/:y.png        single-channel photometric greyscale
   * GET /tile/:z/:x/:y/mask.png   single-channel class-index mask
   */
  for (const [route, kind] of [['/tile/:z/:x/:y.png', 'image'], ['/tile/:z/:x/:y/mask.png', 'mask']]) {
    app.get(route, async (req, res) => {
      const t = parseTile(req, res);
      if (!t) return;

      let mode = 'class';
      if (kind === 'mask') {
        mode = maskMode(req);
        if (!mode) {
          return res.status(400).json({ error: 'unknown mask mode', mode: req.query.mode, valid: MASK_MODES });
        }
      }
      // Each mode is a distinct artefact, and each namespace additionally
      // carries a hash of the settings that produced it.
      const artefact = kind === 'mask' && mode !== 'class' ? `mask_${mode}` : kind;
      const variant = variants[artefact];

      try {
        const hit = cache.read(variant, t.z, t.x, t.y, 'png');
        if (hit) return sendPng(res, hit, true, kind === 'mask' ? mode : null);

        const body = await once(`${variant}/${t.z}/${t.x}/${t.y}`, async () => {
          const png = await renderer.render(t.z, t.x, t.y, { kind, mode });
          cache.write(variant, t.z, t.x, t.y, 'png', png);
          return png;
        });
        sendPng(res, body, false, kind === 'mask' ? mode : null);
      } catch (err) {
        console.error(`render ${kind}/${mode} z=${t.z} x=${t.x} y=${t.y}:`, err.message);
        res.status(500).json({ error: err.message });
      }
    });
  }

  /**
   * GET /tile/:z/:x/:y/labels.json
   *
   * Named features from the same vector tile the image was rendered from, with
   * coordinates in pixels relative to the tile's top-left corner.
   *
   *   ?layers=place,transportation_name   restrict source layers
   *   ?lang=latin                          name:<lang>, falling back to name
   *   ?limit=50                            keep the N most prominent
   */
  app.get('/tile/:z/:x/:y/labels.json', (req, res) => {
    const t = parseTile(req, res);
    if (!t) return;

    const layers = req.query.layers
      ? String(req.query.layers).split(',').filter((l) => LABEL_LAYERS.includes(l))
      : LABEL_LAYERS;
    const lang = /^[a-z_-]{2,12}$/i.test(String(req.query.lang || 'latin'))
      ? String(req.query.lang || 'latin') : 'latin';
    const limit = Math.max(0, parseInt(req.query.limit || '0') || 0);

    try {
      const src = sourceTileFor(t.z, t.x, t.y, tiles.maxzoom);
      const buffer = tiles.get(src.z, src.x, src.y);
      const labels = buffer
        ? extractLabels(buffer, { size: CONFIG.tileSize, src, layers, lang, limit })
        : [];
      res.set('Cache-Control', 'public, max-age=31536000, immutable');
      res.json({ z: t.z, x: t.x, y: t.y, size: CONFIG.tileSize, count: labels.length, labels });
    } catch (err) {
      console.error(`labels z=${t.z} x=${t.x} y=${t.y}:`, err.message);
      res.status(500).json({ error: err.message });
    }
  });

  /**
   * GET /classes — the mask's pixel-value to class-name mapping.
   *
   * ?mode=ambiguous includes the reserved ignore_index value, so a pipeline can
   * fetch exactly the vocabulary of the masks it is about to request.
   */
  app.get('/classes', (req, res) => {
    if (req.query.mode === 'ambiguous') return res.json({ ...CLASSES, ambiguous: AMBIGUOUS });
    res.json(CLASSES);
  });

  app.get('/health', (_req, res) => {
    res.json({
      ok: true,
      pid: process.pid,
      mbtiles: CONFIG.mbtiles,
      sourceZoom: { min: tiles.minzoom, max: tiles.maxzoom },
      outputDir: path.resolve(CONFIG.outputDir),
      tileSize: CONFIG.tileSize,
      channels: 1,
      workers: CONFIG.workers,
      cache: cacheStats(localReaper),
      cacheNamespaces: variants,
      // Fingerprints the *configuration* above; `renderer` fingerprints the
      // rendering behaviour. Both are needed and neither substitutes for the
      // other — see lib/identity.js for the two servers that agreed on every
      // field above and still disagreed on 0.03%-0.28% of mask pixels.
      renderer: rendererIdentity(),
    });
  });

  app.listen(CONFIG.port, () => {
    if (CONFIG.workers <= 1) {
      console.log(`Tile server on http://localhost:${CONFIG.port} (single process)`);
    }
  });

  for (const sig of ['SIGINT', 'SIGTERM']) {
    process.on(sig, () => { renderer.release(); tiles.close(); process.exit(0); });
  }
}
