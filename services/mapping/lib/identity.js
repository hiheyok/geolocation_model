'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

/**
 * Renderer identity: a fingerprint of everything that decides what a mask
 * pixel actually becomes.
 *
 * ── Why cacheNamespaces could not do this job ────────────────────────────────
 *
 * buildVariants() in variants.js hashes the *configuration* — the built style
 * objects, the .mbtiles basename and byte count, tile size, PNG level,
 * supersample factor, ambiguity threshold. That is the right key for a cache,
 * because it answers "were these tiles produced under the settings I am
 * running now".
 *
 * It is the wrong answer to "were these tiles produced by the same renderer",
 * and we have the measurement to prove it. Two deployments — this one and
 * 10.0.0.84:/mnt/storage/mapping_service — report byte-identical /health
 * metadata and byte-identical cacheNamespaces (image-a54b3a89, mask-87addb24,
 * mask_ambiguous-61a2e49b, mask_raw-fca59587), off the same
 * maptiler-osm-2020-02-10-v3.11-planet.mbtiles. Their masks nonetheless
 * disagree on 0.03%–0.28% of pixels at every zoom from 0 to 16. The class
 * tables are identical, the value sets are identical, and every differing
 * pixel sits on a feature boundary (23<->138, 0<->230). That is rasterisation
 * drift — a different maplibre-gl-native prebuild laying down an edge half a
 * pixel differently — not a data or configuration difference.
 *
 * The divergence is benign today: at the 12-d per-patch class fractions the
 * model consumes it is max 7.8e-3, mean 3.3e-4. The problem is not its size,
 * it is that nothing in the cache records which of the two renderers produced
 * a given tile, so a cache built across a renderer change is undetectable.
 *
 * Configuration is hashed from values this process chose. Rendering behaviour
 * lives in code and in native libraries this process merely loaded, so it has
 * to be hashed from what is actually on disk. That is what this module does.
 *
 * ── What goes into the id ────────────────────────────────────────────────────
 *
 * `id` is deliberately a function of only (sources digest, libs) — the things
 * that determine pixels. `commit` and `dirty` are reported alongside it for
 * provenance but are NOT folded in: a commit that touches the README or
 * lib/reaper.js does not change a single rendered pixel, and an id that moved
 * on such a commit would invalidate caches for no reason and train everyone to
 * ignore it. Conversely an edit to a rendering file changes the sources digest
 * whether or not it was ever committed, because the digest is taken over the
 * working tree.
 *
 * `id` is the single value a downstream cache should record, the way
 * tile_cache records rows_digest.
 */

const ROOT = path.join(__dirname, '..');

/**
 * Source files that can change a rendered pixel, hashed as a set.
 *
 * Deliberately narrow. Every file here is on the mask path:
 *   server.js       render parameters, mask mode selection, overzoom entry
 *   lib/renderer.js the render call, the supersample majority vote, the encode
 *   lib/styles.js   the class table, the layer set, the snap LUT
 *   lib/mbtiles.js  which source tile's bytes get fed to the renderer
 *
 * lib/cache.js and lib/reaper.js are excluded on purpose: they decide what is
 * stored and evicted, never what is drawn. lib/labels.js serves a different
 * endpoint. lib/variants.js only names directories. Widening this list would
 * make the id churn on changes that cannot affect a mask, which is the failure
 * mode that makes a fingerprint worthless.
 */
const RENDER_SOURCES = [
  'lib/mbtiles.js',
  'lib/renderer.js',
  'lib/styles.js',
  'server.js',
];

const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

/** Hash of the rendering sources, order-independent and length-delimited. */
function sourcesDigest() {
  const h = crypto.createHash('sha256');
  for (const rel of [...RENDER_SOURCES].sort()) {
    const bytes = fs.readFileSync(path.join(ROOT, rel));
    // Delimit with the path and the length so that moving a byte from the end
    // of one file to the start of the next cannot collide.
    h.update(`${rel}\0${bytes.length}\0`);
    h.update(bytes);
  }
  return h.digest('hex');
}

/** A package's version, without assuming its package.json is exported. */
function pkgDir(name) {
  try {
    let d = path.dirname(require.resolve(`${name}/package.json`));
    return d;
  } catch { /* package.json not in "exports" — fall through */ }
  try {
    let d = path.dirname(require.resolve(name));
    for (let i = 0; i < 8; i++) {
      const p = path.join(d, 'package.json');
      if (fs.existsSync(p) && JSON.parse(fs.readFileSync(p, 'utf8')).name === name) return d;
      const up = path.dirname(d);
      if (up === d) break;
      d = up;
    }
  } catch { /* not resolvable */ }
  return null;
}

function pkgVersion(name) {
  const d = pkgDir(name);
  if (!d) return null;
  try { return JSON.parse(fs.readFileSync(path.join(d, 'package.json'), 'utf8')).version; }
  catch { return null; }
}

/**
 * Digest of a package's compiled native addons.
 *
 * This is the part a version string cannot give us. Both deployments run
 * @maplibre/maplibre-gl-native 6.4.1, but they load *different prebuilt
 * binaries* — different platform, different toolchain, different vendored
 * GL stack — and that is where the boundary-pixel divergence comes from.
 * Hashing the .node file identifies the rasteriser that is actually loaded.
 */
function nativeDigest(name) {
  const d = pkgDir(name);
  if (!d) return null;
  const found = [];
  const walk = (dir, depth) => {
    if (depth > 4 || found.length > 8) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p, depth + 1);
      else if (e.name.endsWith('.node')) found.push(p);
    }
  };
  walk(d, 0);
  if (!found.length) return null;
  const h = crypto.createHash('sha256');
  for (const p of found.sort()) {
    h.update(`${path.relative(d, p)}\0`);
    h.update(fs.readFileSync(p));
  }
  return h.digest('hex').slice(0, 12);
}

/** Short commit of the repo this service is checked out from, or null. */
function gitInfo() {
  const run = (args) => execFileSync('git', args, {
    cwd: ROOT, encoding: 'utf8', timeout: 3000,
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim();
  try {
    const commit = run(['rev-parse', '--short', 'HEAD']);
    // Scoped to this directory: a modification elsewhere in the enclosing
    // repository says nothing about the renderer.
    const dirty = run(['status', '--porcelain', '--', '.']).length > 0;
    return { commit, dirty };
  } catch {
    // Not a repository, git missing, or a detached/empty state. Reported as
    // unknown rather than guessed — see README, both current deployments run
    // from plain directories.
    return { commit: null, dirty: null };
  }
}

/**
 * Versions of every library that participates in turning vector tiles into
 * mask bytes.
 *
 * The curated fields are the ones on the mask path: maplibre rasterises,
 * libvips (via sharp) extracts the channel, and spng/libpng + zlib-ng encode.
 * sharp ships two dozen other libraries — AVIF, HEIF, TIFF, text shaping —
 * that a mask never touches, so listing them individually would produce false
 * alarms. They are still covered: sharp_stack_digest hashes the whole
 * sharp.versions map, so anything moving in that native stack is visible
 * without a wall of irrelevant version strings.
 */
function libs() {
  let sharpVersions = {};
  try { sharpVersions = require('sharp').versions || {}; } catch { /* not loadable */ }

  return {
    node: process.versions.node,
    v8: process.versions.v8,
    platform: process.platform,
    arch: process.arch,

    maplibre_gl_native: pkgVersion('@maplibre/maplibre-gl-native'),
    maplibre_native_digest: nativeDigest('@maplibre/maplibre-gl-native'),

    sharp: sharpVersions.sharp || pkgVersion('sharp'),
    vips: sharpVersions.vips || null,
    png: sharpVersions.png || null,
    spng: sharpVersions.spng || null,
    zlib_ng: sharpVersions['zlib-ng'] || null,
    sharp_stack_digest: sha256(JSON.stringify(
      Object.keys(sharpVersions).sort().map((k) => [k, sharpVersions[k]])
    )).slice(0, 12),
  };
}

let cached = null;

/**
 * Computed once per process: it reads several files and shells out to git, and
 * nothing it measures can change while the process is alive.
 */
function rendererIdentity() {
  if (cached) return cached;

  const { commit, dirty } = gitInfo();
  const sources = sourcesDigest();
  const l = libs();

  cached = {
    // The one value to record downstream. See the note above on why commit and
    // dirty are excluded from it.
    id: `r-${sha256(JSON.stringify({ sources, libs: l })).slice(0, 12)}`,
    commit,
    dirty,
    sources: { digest: sources.slice(0, 16), files: [...RENDER_SOURCES].sort() },
    libs: l,
  };
  return cached;
}

module.exports = { rendererIdentity, RENDER_SOURCES };
