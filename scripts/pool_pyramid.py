"""Write the pyramid's 1536-d street vector: crops alone, or crops plus tiles.

`runs/RESMATCH.md` decided the rebuild is tiles at 224, on a retrieval probe.
The agent consumes neighbours through a *trained* retrieval prior, so only a
retrain converts that into an agent number -- and this project has twice had
inference-level reasoning predict the wrong sign. This script exists to make
that retrain a clean paired experiment: **one code path, one flag between the
arms.**

    py scripts/pool_pyramid.py --out pyr_l0.f16.npy                  # crops
    py scripts/pool_pyramid.py --tiles tile6 --out pyr_mix.f16.npy   # crops+tiles

The arithmetic is exactly what the probes measured, which is *not* what
`pool_street.py` does. `pool_street` takes a plain mean of the crop tokens; the
probes normalise per token, per encoder half, and again after the join:

    C   = l2(crop tokens)                      (n, crops, 2, 768)
    L0  = l2([ l2(mean_c C_dino) | l2(mean_c C_siglip) ])
    L1  = l2([ l2(mean_t T_dino) | l2(mean_t T_siglip) ])
    out = L0                     or     l2((1-w)*L0 + w*L1)

Those two happen to score the same (`pool_bal` 226.0 km / 27.8% against `L0`
223.0 km / 27.7%, identical any-of-32), which is why `L0` can stand in as the
untiled arm -- but "scores the same" is not "is the same", and a paired
experiment wants the untiled arm produced by this code rather than by the other
one. Otherwise the contrast carries a normalisation difference as well as the
tiles.

**Per-token normalisation subsumes `--scale-b`.** `nrm` divides each 768-d
token by its own norm, so any positive per-encoder rescaling cancels: measured,
`L0(dual_c3)` and `L0(dual_bal)` agree at **cosine 0.99999976** over 200 rows
(max abs difference 9.3e-05, which is fp16 storage noise) while their raw
SigLIP norms are 20.53 and 82.73. So it does not matter whether this is fed the
balanced or unbalanced dual cache -- including on the extension side, where
both `bank_ext_dual` and `bank_ext_bal` exist and picking wrong would otherwise
be a silent seam between the two halves of a bank.

`w` defaults to 0.5, the equal blend every pyramid number on record was
measured at. It is deliberately *not* the 0.04 of `runs/PYR_BLEND.md`: that
weight blends a learned fusion head against a level mean, which is a different
pair of things.

**A partly-tiled output is refused, not filled.** A bank whose rows are a mix
of `l2((L0+L1)/2)` and `L0` is not a smaller tiled bank, it is a broken one --
one cosine ranks both, so an untiled row has the query's `L1` half scored
against its `L0`, cross-space and systematically lower, and it is silently
demoted. Nothing downstream can see that: every row is a unit vector of the
right width. So every requested row must be present *and* marked done in the
tile cache, or this exits.
"""

import argparse
import sys
import time
from pathlib import Path

# BEFORE torch and before any large mmap, deliberately. `pq.read_table`
# imports `pyarrow.dataset` lazily on first use, and that DLL load takes an
# access violation once CUDA has been initialised OR a multi-gigabyte mmap is
# open -- either alone is enough, measured. The tiles arm hit both, so it died
# at 0xC0000005 with an empty log and three identical retries. Importing here
# costs nothing and the crash does not reproduce.
import pyarrow.dataset          # noqa: F401  (imported for its side effect)
import pyarrow.parquet as pq
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config                                        # noqa: E402
import maskio                                        # noqa: E402
import provenance as prov                            # noqa: E402

D_ENC = 768


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def levels(C, T=None):
    """(b, crops, 2, 768) and (b, tiles, 2, 768) -> L0, L1 as the probes pool.

    Both inputs are token-normalised first, which is what makes the DINOv2 and
    SigLIP halves contribute equally without a `--scale-b` constant: their raw
    norms are 82.96 and 20.58, so an unnormalised join gives DINOv2 81% of the
    cosine.
    """
    C = nrm(C)
    L0 = nrm(torch.cat([nrm(C[:, :, 0].mean(1)), nrm(C[:, :, 1].mean(1))], 1))
    if T is None:
        return L0, None
    T = nrm(T)
    L1 = nrm(torch.cat([nrm(T[:, :, 0].mean(1)), nrm(T[:, :, 1].mean(1))], 1))
    return L0, L1


def check_tile_identity(stem, rows):
    """Are this cache's rows still the photographs it was embedded from?

    `rows` holds *positions* in an image list, and positions are not identity:
    rebuild that list at the same length and every tile embedding stays put
    under a row that now names a different photograph (REVIEW4 #19). This is
    the consumption half of the check `tile_cache` writes -- a recorded digest
    nobody reads is not a guard.

    It matters most precisely here. This function pairs each row's crops with
    the tile embeddings at the matching cache position, so a stale cache
    produces one image's crops beside another image's tiles: right shape, unit
    norm, complete mask, wrong photograph.

    Missing metadata is reported and allowed -- `tile6` predates both fields
    and holds 500,000 finished rows -- but never treated as agreement.
    """
    import provenance as prov

    meta_p = config.STREET_CACHE / (stem + "_meta.npz")
    if not meta_p.exists():
        return
    meta = np.load(meta_p, allow_pickle=True)
    if "rows_digest" not in meta.files or "rows_parquet" not in meta.files:
        print("warning: {} records no row identity, so its {:,} rows cannot be "
              "shown to still name the images they were embedded from"
              .format(stem, len(rows)), flush=True)
        return
    src = str(meta["rows_parquet"])
    pq_path = (config.DATASET_PARQUET if src == config.DATASET_PARQUET.name
               else config.PROCESSED / src)
    ids = np.asarray(pq.read_table(pq_path, columns=["image_id"])["image_id"])
    got = prov.rows_digest(ids[rows])
    if got != str(meta["rows_digest"]):
        sys.exit(
            "{} was built over {} whose rows now digest {} against the "
            "recorded {}. The row numbers still line up, so every shape and "
            "every mask agrees -- the images behind them changed, and pooling "
            "would pair each image's crops with another image's tiles."
            .format(stem, src, got, str(meta["rows_digest"])))


def tile_positions(stem, want):
    """Position of each wanted row in the tile cache, or exit.

    The cache stores a *selection*: `rows` holds the release rows it covers, in
    order, and `done` marks which of those were actually written. A row that is
    absent, or present but unwritten, is a zero fill -- and a zero row
    L2-normalises to a unit-length nothing that ranks like a real vector. This
    is the check whose absence would have built a bank 76% zeros.
    """
    rows_p = config.STREET_CACHE / (stem + "_rows.i64.npy")
    done_p = config.STREET_CACHE / (stem + "_done.u8.npy")
    if not rows_p.exists():
        sys.exit("{} has no _rows.i64.npy; cannot tell which rows it covers"
                 .format(stem))
    rows = np.load(rows_p)
    done = maskio.load_mask(done_p, len(rows), stem)
    if done.ndim > 1:
        done = done.min(axis=1)
    check_tile_identity(stem, rows)
    have = rows[done == 1]
    # Sized by both, not just `want`: the cache may legitimately cover rows the
    # caller did not ask for (a 750k cache queried for 400k of it), and sizing
    # this by the request alone makes writing the map itself go out of bounds.
    pos = np.full(int(max(rows.max(initial=-1), want.max(initial=-1))) + 1,
                  -1, np.int64)
    pos[rows] = np.arange(len(rows))
    missing = np.setdiff1d(want, have, assume_unique=False)
    if len(missing):
        sys.exit(
            "{} covers {:,} finished rows but {:,} of the {:,} requested are "
            "missing or unwritten (first: {}). A partly-tiled cache cannot be "
            "pooled: the untiled rows would be L0 while the rest are the "
            "blend, one cosine would rank both, and the untiled ones would be "
            "silently demoted. Finish the pass or narrow the selection."
            .format(stem, len(have), len(missing), len(want), missing[:5]))
    return pos[want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dual_c3.f16.npy",
                    help="dual crop cache, (n, crops*2*768)")
    ap.add_argument("--tiles", default=None,
                    help="tile cache stem, e.g. tile6; omitted = crops only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=0.5,
                    help="weight on the tile level in l2((1-w)*L0 + w*L1)")
    ap.add_argument("--block", type=int, default=20000)
    a = ap.parse_args()

    if not 0.0 <= a.w <= 1.0:
        sys.exit("--w is a blend weight in [0, 1], got {}".format(a.w))
    src = config.STREET_CACHE / a.src
    out = config.STREET_CACHE / a.out
    if out == src:
        sys.exit("refusing to write over the source")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(src, mmap_mode="r")
    n, d = X.shape
    if d % (2 * D_ENC):
        sys.exit("{} is {}-d, not a whole number of 768-d blocks per encoder"
                 .format(a.src, d))
    crops, half = d // (2 * D_ENC), d // 2

    T_all, tpos = None, None
    if a.tiles:
        tpos = tile_positions(a.tiles, np.arange(n))
        T_all = np.load(config.STREET_CACHE / (a.tiles + ".f16.npy"),
                        mmap_mode="r")
        if T_all.shape[2] != 2 * D_ENC:
            sys.exit("{} is {}-d per tile, expected {}".format(
                a.tiles, T_all.shape[2], 2 * D_ENC))

    print("{}  {:,} x {} = {} crops x [dino {} | siglip {}]".format(
        a.src, n, d, crops, D_ENC, D_ENC), flush=True)
    print("{}  {:,} x {}   {}".format(
        a.out, n, 2 * D_ENC,
        "crops only" if not a.tiles else
        "l2({:.2f}*L0 + {:.2f}*L1) over {} tiles from {}".format(
            1 - a.w, a.w, T_all.shape[1], a.tiles)), flush=True)

    Y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(n, 2 * D_ENC))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, a.block):
            e = min(s + a.block, n)
            B = torch.from_numpy(np.asarray(X[s:e], np.float32)).to(dev)
            C = torch.stack([B[:, :half].reshape(-1, crops, D_ENC),
                             B[:, half:].reshape(-1, crops, D_ENC)], dim=2)
            T = None
            if a.tiles:
                Tb = torch.from_numpy(
                    np.asarray(T_all[tpos[s:e]], np.float32)).to(dev)
                T = torch.stack([Tb[:, :, :D_ENC], Tb[:, :, D_ENC:]], dim=2)
            L0, L1 = levels(C, T)
            V = L0 if L1 is None else nrm((1 - a.w) * L0 + a.w * L1)
            Y[s:e] = V.cpu().numpy().astype(np.float16)
            el = time.time() - t0
            print("  {:>9,}/{:,}  {:.0f}s  eta {:.0f}s".format(
                e, n, el, el * (n - e) / max(e, 1)), flush=True)
    Y.flush()

    # Cheap recompute rather than trust, on rows the loop did not special-case.
    rng = np.random.default_rng(0)
    for i in rng.choice(n, 5, replace=False):
        v = torch.from_numpy(np.asarray(X[i:i + 1], np.float32)).to(dev)
        C = torch.stack([v[:, :half].reshape(-1, crops, D_ENC),
                         v[:, half:].reshape(-1, crops, D_ENC)], dim=2)
        T = None
        if a.tiles:
            Tb = torch.from_numpy(
                np.asarray(T_all[tpos[i:i + 1]], np.float32)).to(dev)
            T = torch.stack([Tb[:, :, :D_ENC], Tb[:, :, D_ENC:]], dim=2)
        L0, L1 = levels(C, T)
        want = (L0 if L1 is None else nrm((1 - a.w) * L0 + a.w * L1))[0]
        got = torch.from_numpy(np.asarray(Y[i], np.float32)).to(dev)
        assert float((want - got).abs().max()) < 2e-3, "row {} disagrees".format(i)
    # Every unit vector is a plausible one, so a row-count check is all the
    # protection there is against pooling a cache that describes other images.
    prov.carry(src, out, n, pooled_from=a.src, crops=crops,
               tiles=a.tiles or "none", w=a.w if a.tiles else 0.0)
    print("\nwrote {} in {:.0f}s; 5 rows verified against a recompute".format(
        out.name, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
