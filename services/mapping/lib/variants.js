'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { buildMonoStyle, buildMaskStyle } = require('./styles');

/**
 * Cache namespaces, keyed by everything that changes the bytes of an artefact.
 *
 * The cache is otherwise addressed only by z/x/y, which means a change to the
 * style or to a render setting leaves the old output in place and serves it
 * forever — the tiles are still perfectly valid PNGs, just produced by a
 * different configuration. Folding that configuration into the directory name
 * means a change lands in a fresh namespace instead of quietly lying.
 *
 * Inputs are per artefact rather than global, so tuning MASK_SUPERSAMPLE does
 * not throw away the image cache, which it cannot affect.
 */

const shortHash = (obj) =>
  crypto.createHash('sha256').update(JSON.stringify(obj)).digest('hex').slice(0, 8);

/**
 * Identity of the source file. Basename rather than full path, so relocating
 * the project does not invalidate anything, plus the byte count to catch a
 * different extract dropped in at the same filename.
 */
function sourceIdentity(file) {
  try {
    return { name: path.basename(file), bytes: fs.statSync(file).size };
  } catch {
    return { name: path.basename(file), bytes: null };
  }
}

/**
 * Namespace per logical artefact, e.g. { image: 'image-3f9a1c22', ... }.
 *
 * The styles are hashed as built objects, so editing a colour or adding a layer
 * in styles.js changes the namespace with no bookkeeping here.
 */
function buildVariants({ mbtiles, maxzoom, tileSize, compression,
                         maskSupersample, ambiguousThreshold }) {
  const source = sourceIdentity(mbtiles);
  const base = { source, size: tileSize, png: compression };
  const mono = buildMonoStyle(maxzoom);
  const mask = buildMaskStyle(maxzoom);

  return {
    image: `image-${shortHash({ ...base, style: mono })}`,
    // Supersampling changes the vote, so it changes both resolved mask modes.
    mask: `mask-${shortHash({ ...base, style: mask, ss: maskSupersample })}`,
    mask_ambiguous: `mask_ambiguous-${shortHash({
      ...base, style: mask, ss: maskSupersample, threshold: ambiguousThreshold,
    })}`,
    // Raw output is whatever the renderer produced, before any of the vote.
    mask_raw: `mask_raw-${shortHash({ ...base, style: mask })}`,
  };
}

/**
 * Namespaces present on disk that no longer correspond to the running config.
 *
 * Reported rather than deleted: a stale namespace is still valid output, and
 * flipping a setting back should find its cache intact. With a size cap set
 * they age out on their own, since nothing reads them and eviction is LRU.
 */
function staleNamespaces(root, variants) {
  const current = new Set(Object.values(variants));
  try {
    return fs.readdirSync(root, { withFileTypes: true })
      .filter((e) => e.isDirectory() && !current.has(e.name))
      .map((e) => e.name);
  } catch {
    return [];
  }
}

module.exports = { buildVariants, staleNamespaces, shortHash };
