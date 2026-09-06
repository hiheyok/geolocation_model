"""Embed the image tiles once and keep them.

`tile_probe.py` spent 1.7 h computing 120,000 x 6 tile vectors, scored one
comparison with them, and dropped them on exit.  Every experiment in
`docs/tile-fusion.md` -- mean pooling, per-block normalisation, attention
pooling, a trained fusion head -- wants exactly those vectors, and none of them
wants to pay that 1.7 h again.  So this script is the same forward pass with the
result written to disk.

Layout is the one the fusion head needs, which is *not* the one the probe used
internally:

    tile6.f16.npy       (n, n_tiles, 1536) float16
                        axis 1 is the tile, row-major over the grid
                        axis 2 is [DINOv2 768 | SigLIP 768] for that tile

The probe concatenated as [dino t0..t5 | siglip t0..t5], one flat 9,216-d
vector, which is what made per-tile anything impossible to recover.  The layout
above is a strict superset: the probe's vector is

    np.concatenate([T[:, :, :768].reshape(n, -1),
                    T[:, :, 768:].reshape(n, -1)], axis=1)

so the losing arm can still be reproduced exactly as a control, and the six
tokens are also addressable individually.

Two other things the probe did not do:

  * **Resumable.**  A `_done.u8.npy` marks finished rows, so a crash or a
    deliberate stop costs whatever is in flight rather than the whole run.
  * **Batched across images.**  The probe forwarded one image's six tiles at a
    time -- its `--batch` argument was accepted and then never used -- which
    leaves an RTX 3070 mostly idle between kernel launches.  Grouping images
    into one forward is the same arithmetic at a fraction of the wall time.
  * **Preloaded, which is the whole ballgame.**  The shards live on a spinning
    disk, and 12,000 scattered `ZipFile.read` calls into a 2.5 GB archive run at
    the drive's seek-bound rate -- ~3 MB/s, which is ~38 ms per image with the
    GPU at 1% utilisation.  `embed_street.py` already solved this and the probe
    never inherited it: one sequential read of the archive turns every later
    member read into a memory hit.  A shard is ~20 s to slurp against ~8 min to
    seek through, so this is not a tuning knob, it is the difference between
    a disk-bound job and a compute-bound one.

Sampling is `rng.permutation(N)[:n]`, deliberately, so a later run with a larger
--n is a *superset* of this one and only embeds the difference.  That is worth
more than reproducing tile_probe's exact draw, because the control for the draw
is available separately and for free: crop3 for these same rows already exists
in the shipping cache, so if crop3 scores what it scored there, the draw is
equivalent and any tile6 difference belongs to the tiles.
"""

import argparse
import io
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import maskio
import provenance as prov
import shards

DINO = "vit_base_patch14_dinov2.lvd142m"
SIGLIP = "vit_base_patch16_siglip_224.v2_webli"
S = 224
D_ENC = 768


def tile_uint8(blob, gc, gr):
    """JPEG bytes -> (gc*gr, 224, 224, 3) uint8, row-major over the grid.

    Identical geometry to tile_probe.tiles(): resize the whole frame to the
    grid's exact pixel size, then cut it up.  No draft() here -- draft decodes
    at a reduced DCT scale, which is right when the target is 224 on the short
    side and self-defeating when the whole point is to keep resolution.
    """
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    img = img.resize((gc * S, gr * S), Image.BILINEAR)
    a = np.asarray(img, dtype=np.uint8)
    out = np.empty((gc * gr, S, S, 3), dtype=np.uint8)
    for r in range(gr):
        for c in range(gc):
            out[r * gc + c] = a[r * S:(r + 1) * S, c * S:(c + 1) * S]
    return out


def check_resume(meta, want_grid, want_src, stem, explicit_src):
    """Refuse to extend a cache that was built to different terms.

    A resume writes new rows into an existing memmap and keeps the old ones, so
    both halves have to mean the same thing.  Two ways they silently do not:

    * **grid** -- 3x2 and 2x3 both hold six tiles, so a cache built as one and
      resumed as the other passes every count-based check downstream while the
      tokens are transposed.  The geometry has to be compared, not the product.
    * **row source** -- `rows` holds *indices into an image list*.  Resume a
      release cache against `bank_ext.parquet` and index 41,000 addresses a
      different photograph in each half.  Shapes agree, the mask agrees, and
      every row is a real embedding of a real image.

    `explicit_src` is what keeps the second check from failing open.  Metadata
    written before `rows_parquet` existed can only describe a release cache --
    there was no other way to build one -- so a default-source run may resume
    it.  A `--parquet` run may not: that is the exact case the field was added
    to catch, and reading silence as agreement would let it straight through.
    """
    had = tuple(int(v) for v in meta["grid"]) if "grid" in meta.files else None
    if had is not None and had != tuple(want_grid):
        raise SystemExit(
            "cache {} was built at grid {}x{} and this run wants {}x{}; "
            "both hold {} tiles, so nothing downstream would notice. "
            "Delete it or use --out.".format(
                stem, had[0], had[1], want_grid[0], want_grid[1],
                had[0] * had[1]))
    src = (str(meta["rows_parquet"]) if "rows_parquet" in meta.files
           else (None if explicit_src else config.DATASET_PARQUET.name))
    if src != want_src:
        raise SystemExit(
            "cache {} was built over {} and this run wants {}; `rows` is "
            "indices into that list, so resuming would mix two row spaces. "
            "Delete it or use --out.".format(
                stem, src or "an unrecorded image list (it pre-dates "
                "--parquet, so it is a release cache)", want_src))


def check_rows_identity(meta, rows_old, ids_now, stem):
    """Do the rows this cache already holds still name the same photographs?

    `rows` is a set of *positions* in an image list. Positions are not identity:
    rebuild or reorder the list at the same length and the seeded selection is
    numerically identical, every old tile embedding is retained, and each one
    now sits under a row that names a different photograph. The completion mask
    proves some tensor was written, never whose.

    Nothing downstream can catch it. `pool_pyramid` and `fuse_head` read current
    coordinates, split labels and crop embeddings at `sel` and pair them with
    the tile embeddings at the matching cache positions -- so every row is a
    real image's crops beside another image's tiles, at the right shape, with a
    complete mask. REVIEW4 #19.

    So the digest is taken over the ids at the rows the cache covers, in cache
    order, and re-derived from the current list on every resume. A cache with no
    recorded digest predates the field: it is reported and allowed, because
    refusing would strand `tile6` and its 500,000 finished rows, but it is
    never treated as agreement.
    """
    if "rows_digest" not in meta.files:
        print("warning: {} records no rows digest (it predates the field), so "
              "its {:,} existing rows cannot be shown to still name the images "
              "they were embedded from".format(stem, len(rows_old)), flush=True)
        return
    want = str(meta["rows_digest"])
    got = prov.rows_digest(ids_now[rows_old])
    if got != want:
        raise SystemExit(
            "{} covers {:,} rows whose image ids no longer match what it was "
            "built from ({} now, {} recorded). The row numbers are the same, "
            "so every count and every mask still agrees -- the images behind "
            "them changed. Delete the cache and rebuild it."
            .format(stem, len(rows_old), got, want))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120000,
                    help="images to embed. A larger value later is a superset "
                         "of this one and re-embeds nothing.")
    ap.add_argument("--grid", default="3x2", help="cols x rows of 224 tiles")
    ap.add_argument("--batch", type=int, default=32,
                    help="images per forward pass; tiles per forward is this "
                         "times the grid size. Measured flat from 8 to 64 on a "
                         "3070 -- 18.1-18.5 ms an image, the forward being 96%% "
                         "of the pipeline -- and 2x worse at 96. Spare VRAM is "
                         "idle capacity here, not headroom: the card is compute "
                         "saturated at batch 8 and a bigger batch buys nothing.")
    ap.add_argument("--workers", type=int, default=4, help="decode threads")
    ap.add_argument("--no-preload", dest="preload", action="store_false",
                    help="read shard members straight off the disk. Only worth "
                         "it if RAM is tighter than a 2.5 GB shard.")
    ap.set_defaults(preload=True)
    ap.add_argument("--out", default=None, help="stem, default tile{n_tiles}")
    ap.add_argument("--parquet", default=None,
                    help="image list to tile instead of the release's "
                         "dataset.parquet; needs zip_name. This is how the "
                         "3M-row bank extension gets tiled: build_bank_ext.py "
                         "writes bank_ext{,2,3,4}.parquet with zip_name "
                         "already materialised, in the same row order as "
                         "bank_ext70_meta.npz.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # Same guard embed_street.py carries, for the same reason. `tile6` is the
    # release cache every pyramid number on record was measured against, and a
    # run over a different image list is not that cache: without --out it would
    # leave a file of the right name and the wrong rows, and `sel` -- which is
    # indices into whatever list was read -- would silently address the wrong
    # images. The row space is not recoverable from the file afterwards.
    if a.parquet and not a.out:
        raise SystemExit(
            "--parquet tiles a different image list than the release, but "
            "there is no --out, so it would overwrite the release's own tile "
            "cache in place. Name it: --out <stem>.")

    import pyarrow.parquet as pq
    import timm
    from timm.data import resolve_model_data_config

    gc, gr = (int(v) for v in a.grid.lower().split("x"))
    # 3x2 and 2x3 both hold six tiles, so a cache built as one can be reused
    # as the other and every count-based check passes. The geometry has to be
    # compared, not the product.
    _want_grid = (gc, gr)
    _want_src = a.parquet or config.DATASET_PARQUET.name
    n_tile = gc * gr
    stem = a.out or "tile{}".format(n_tile)
    emb_p = config.STREET_CACHE / (stem + ".f16.npy")
    rows_p = config.STREET_CACHE / (stem + "_rows.i64.npy")
    done_p = config.STREET_CACHE / (stem + "_done.u8.npy")
    meta_p = config.STREET_CACHE / (stem + "_meta.npz")
    if meta_p.exists():
        check_resume(np.load(meta_p, allow_pickle=True), _want_grid,
                     _want_src, stem, explicit_src=bool(a.parquet))

    rows_src = config.PROCESSED / a.parquet if a.parquet else config.DATASET_PARQUET
    _src_tbl = pq.read_table(rows_src, columns=["zip_name", "image_id"])
    zn = np.asarray(_src_tbl["zip_name"]).astype("U40")
    ids_now = np.asarray(_src_tbl["image_id"])
    n = min(a.n, len(zn))
    sel = np.sort(np.random.default_rng(a.seed).permutation(len(zn))[:n])

    # Resume, or start.  An existing cache is extended in place when the new
    # selection contains the old one -- which the permutation guarantees for a
    # larger --n at the same seed, and which is checked rather than assumed.
    if rows_p.exists():
        old = np.load(rows_p)
        if meta_p.exists():
            check_rows_identity(np.load(meta_p, allow_pickle=True), old,
                                ids_now, stem)
        if not np.isin(old, sel, assume_unique=True).all():
            sys.exit("existing {} is not a subset of the new selection; "
                     "delete it or use a different --out".format(rows_p.name))
        idx = np.searchsorted(sel, old)
        cur = np.load(emb_p, mmap_mode="r")
        grew = cur.shape[0] < n
        del cur                       # Windows will not unlink a mapped file
        if grew:
            tmp = emb_p.with_suffix(".tmp.npy")
            grow = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.float16, shape=(n, n_tile, 2 * D_ENC))
            src = np.load(emb_p, mmap_mode="r")
            grow[idx] = src[:len(old)]
            del grow, src
            emb_p.unlink()
            tmp.rename(emb_p)
        E = np.lib.format.open_memmap(emb_p, mode="r+")
        done = np.zeros(n, np.uint8)
        # Validated, not trusted. This was the one resume path with no shape
        # or value check on its mask, so a 2 -- from a torn write, or a writer
        # that once used another convention -- counted as complete and the row
        # under it stayed zero-filled. fuse_head then casts the mask to bool
        # and agrees with it.
        done[idx] = maskio.load_mask(done_p, len(old), stem)[:len(old)]
    else:
        E = np.lib.format.open_memmap(emb_p, mode="w+", dtype=np.float16,
                                      shape=(n, n_tile, 2 * D_ENC))
        done = np.zeros(n, np.uint8)
    np.save(rows_p, sel)
    np.save(done_p, done)

    todo = np.flatnonzero(done != 1)
    print("release {}  grid {}x{}  {:,} images  {:,} to embed  {:.2f} GB".format(
        config.RELEASE, gc, gr, n, len(todo), E.nbytes / 1e9), flush=True)
    if len(todo) == 0:
        # Still verify and still republish the metadata. Returning here meant
        # a resumed-but-already-complete cache skipped both, so a build whose
        # metadata was never written (or was written by an older layout) could
        # report success forever without either being produced.
        if not maskio.is_complete(done, n):
            raise SystemExit(
                "{:,} of {:,} rows are unfilled but there is nothing queued to "
                "fill them -- the mask and the selection disagree. Delete {} "
                "and rebuild.".format(n - maskio.complete(done), n, done_p.name))
        np.savez(meta_p, release=config.RELEASE, grid=np.array([gc, gr]),
                 tile_px=S, encoders=np.array([DINO, SIGLIP]), seed=a.seed,
                 n=n, layout="(n, n_tiles, dino768|siglip768)",
                 rows_parquet=_want_src,
                 rows_digest=prov.rows_digest(ids_now[sel]))
        print("nothing to do; {:,} rows verified complete, metadata rewritten"
              .format(n), flush=True)
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    encs = []
    for name in (DINO, SIGLIP):
        m = timm.create_model(name, pretrained=True, num_classes=0,
                              img_size=S).eval().to(dev)
        cfg = resolve_model_data_config(m)
        stat = [torch.tensor(cfg[k], device=dev).view(1, 3, 1, 1)
                for k in ("mean", "std")]
        encs.append((m, stat[0], stat[1]))
        print("encoder    {}  mean {}".format(
            name, [round(float(v), 3) for v in cfg["mean"]]), flush=True)

    by_zip = defaultdict(list)
    for i in todo:
        by_zip[zn[sel[i]].split("/")[0]].append((int(i), zn[sel[i]]))

    t0 = time.time()
    state = {"done": 0, "bad": 0}
    pool = ThreadPoolExecutor(a.workers)

    def decode(item):
        i, blob = item
        try:
            return i, tile_uint8(blob, gc, gr)
        except Exception:
            return i, None

    def flush(chunk):
        """chunk: list of (row, (n_tile,224,224,3) uint8) -> written to E."""
        if not chunk:
            return
        rows = [c[0] for c in chunk]
        x = torch.from_numpy(np.concatenate([c[1] for c in chunk])).to(dev)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        out = []
        with torch.no_grad():
            for m, mean, std in encs:
                with torch.autocast(dev, dtype=torch.bfloat16,
                                    enabled=(dev == "cuda")):
                    f = m((x - mean) / std)
                out.append(f.float())
        v = torch.cat(out, dim=1).reshape(len(rows), n_tile, 2 * D_ENC)
        v = v.to(torch.float16).cpu().numpy()
        for k, r in enumerate(rows):
            E[r] = v[k]
            done[r] = 1
        before, state["done"] = state["done"], state["done"] + len(rows)
        if state["done"] // 2000 != before // 2000:
            el = time.time() - t0
            print("  {:>7,}/{:,}  {:.0f}s  eta {:.0f}s".format(
                state["done"], len(todo), el,
                el * (len(todo) - state["done"]) / max(state["done"], 1)),
                flush=True)
            np.save(done_p, done)

    def decoded(items, zf, window):
        """Decode `items` on the pool, at most `window` in flight.

        Not ThreadPoolExecutor.map: that consumes its input iterable eagerly,
        which here would read a whole 2.5 GB shard into RAM and queue every
        decoded tile stack behind the GPU.  A sliding window keeps both bounded.
        """
        q, it = deque(), iter(items)
        def top_up():
            while len(q) < window:
                nxt = next(it, None)
                if nxt is None:
                    return
                q.append(pool.submit(decode, (nxt[0], zf.read(nxt[1]))))
        top_up()
        while q:
            yield q.popleft().result()
            top_up()

    chunk = []
    for shard, items in sorted(by_zip.items()):
        with shards.archive(shard, preload=a.preload) as zf:
            for i, parts in decoded(items, zf, max(4 * a.workers, 2 * a.batch)):
                if parts is None:
                    state["bad"] += 1
                    continue
                chunk.append((i, parts))
                if len(chunk) == a.batch:
                    flush(chunk)
                    chunk = []
        del src                       # the next shard wants the 2.5 GB back
    flush(chunk)
    pool.shutdown()

    E.flush()
    np.save(done_p, done)
    el = time.time() - t0
    np.savez(meta_p, release=config.RELEASE, grid=np.array([gc, gr]),
             tile_px=S, encoders=np.array([DINO, SIGLIP]), seed=a.seed, n=n,
             layout="(n, n_tiles, dino768|siglip768)", rows_parquet=_want_src,
             rows_digest=prov.rows_digest(ids_now[sel]))
    print("\nembedded {:,} in {:.0f}s ({:.1f} ms/img), {} unreadable".format(
        state["done"], el, 1000 * el / max(state["done"], 1), state["bad"]),
        flush=True)
    print("cache      {}  {:,} of {:,} rows filled".format(
        emb_p.name, int(done.sum()), n), flush=True)

    # Exit non-zero on an incomplete build. fetch_tiles and pyramid_cache were
    # fixed; this one still returned success after unreadable images, so a
    # marker-gated runner wrote its done-marker over a cache with zero-filled
    # rows and every later stage read them as embeddings.
    # `done.sum() < n` is not a completion test: a mask holding one 2 and one
    # 0 sums to n while a row is still blank. Count the ones.
    if state["bad"] or not maskio.is_complete(done, n):
        raise SystemExit(
            "{:,} unreadable and {:,} of {:,} rows unfilled -- this cache is "
            "incomplete. Re-run to fill the gaps; do not mark it done."
            .format(state["bad"], n - maskio.complete(done), n))

    # Windows does not OOM when VRAM runs out -- WDDM pages GPU allocations
    # into system RAM and the job simply gets slower, which is how a 2x
    # slowdown at 96 images per forward looked like nothing but a bad number.
    # torch cannot see that paging (cudaMalloc succeeds), so report the two
    # things it can see and let the operator compare against nvidia-smi.
    if dev == "cuda":
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_reserved()
        retries = torch.cuda.memory_stats().get("num_alloc_retries", 0)
        print("vram       peak reserved {:.2f} GB of {:.2f} GB, {} alloc "
              "retries".format(peak / 1e9, total / 1e9, retries), flush=True)
        if peak > 0.70 * total or retries:
            print("           ^ close to the limit. On Windows this does not "
                  "raise, it pages to system RAM and halves throughput -- "
                  "check Shared Usage in Task Manager and lower --batch.",
                  flush=True)


if __name__ == "__main__":
    main()
