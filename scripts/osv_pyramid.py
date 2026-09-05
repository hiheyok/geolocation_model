"""The pyramid question, asked on the shipping benchmark's own data.

Everything measured about the pyramid so far ran on `pyr47` -- 47,646 KartaView
images, their own bank, and a *retrieval probe* rather than the agent. Three
things separated those numbers from anything shippable, and this closes two of
them: the corpus and the split are now OSV-5M's, exactly the ones the shipping
benchmark uses.

    queries and bank   OSV-5M release rows, `sequence` split -- the benchmark's
    representation      3 crops (L0) + 6 tiles (L1), per-token L2
    still not the agent one matmul to a neighbour, no beam, no map, no click

**Only two levels exist here, and that is a fact about the data.** OSV-5M
frames are 682x512. At 224 px that supports 3 full-height crops and a 3x2 tile
grid; the 24-tile L2 of `pyr47` would need 1344x896, which is inventing pixels.
So the level-weighting result from the high-resolution harvest -- deepest level
wants the most weight -- **cannot be tested here at all**. What can be tested is
the question underneath it: on the shipping corpus, does adding a tile level to
the crops help, and at what weight?

That matters because **L0 alone is essentially what ships**: the production
street vector is the mean of 3 crops (`pool_street.py`), later projected to
768. So `L0 alone` is the incumbent and every other row is a candidate to beat
it, on its own data.

Weights are explicit for the same reason as everywhere else in this family: a
concatenation of unit blocks is an exactly weighted sum of their cosines, and
`stack(per).mean(0)` picks 1:1 without saying so.

    OSV_RELEASE=s10 py scripts/osv_pyramid.py --queries 3000
"""

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import maskio                                    # noqa: E402
import splits as sp                              # noqa: E402
from tile_pool import l2, paired                 # noqa: E402
from tile_match import dense_sim, topk_stats     # noqa: E402
from fuse_head import load_tokens                # noqa: E402  (reference)

THRESH = (1, 25, 200, 750, 2500)
D_ENC = 768


def complete_rows(stem="tile6"):
    """(release row ids, their positions in the cache) for FINISHED rows only.

    `tile_cache.py` grows in place: `--n 500000` immediately extends the array
    and the rows file to 500,000 and then fills the new entries, so a cache
    caught mid-pass is 500,000 rows of which most are the zero fill. That is a
    legal resume state and it is also a trap -- a zero row L2-normalises to a
    unit-length nothing that no metric can distinguish from a real vector.

    `fuse_head.load_tokens` guards this with `require_written`. The blocked
    loader here did not, and the first partially-grown cache it met would have
    built a bank that was 76% zeros. Reading the mask and keeping only the
    finished rows makes every consumer correct during a pass rather than only
    between passes.
    """
    rows = np.load(config.STREET_CACHE / (stem + "_rows.i64.npy"))
    done_p = config.STREET_CACHE / (stem + "_done.u8.npy")
    if not done_p.exists():
        return rows, np.arange(len(rows))
    d = maskio.load_mask(done_p, len(rows), stem)
    pos = np.flatnonzero((d == 1).all(-1) if d.ndim == 2 else (d == 1))
    if len(pos) < len(rows):
        print("{}: {:,} of {:,} rows written; using the finished ones"
              .format(stem, len(pos), len(rows)), flush=True)
    return rows[pos], pos


def load_tokens_blocked(sel, pos=None, block=8192):
    """(n, 9, 2, 768) fp16: 3 crops then 6 tiles, per-token L2 -- in blocks.

    Same result as `fuse_head.load_tokens`, which builds it in one shot and
    peaks near 13 GB: 2.2 GB for the crops as float32, 4.4 GB for the tiles,
    and 6.6 GB for the concatenation before it is cast down. That does not fit
    beside a loaded demo server. Blocked, the peak is the 3.3 GB output plus a
    working block.
    """
    C_all = np.load(config.STREET_CACHE / "dual_c3.f16.npy", mmap_mode="r")
    T_all = np.load(config.STREET_CACHE / "tile6.f16.npy", mmap_mode="r")
    h = C_all.shape[1] // 2
    nc = h // D_ENC
    if pos is None:
        pos = np.arange(len(sel))
    n, nt = len(sel), T_all.shape[1]
    out = np.empty((n, nc + nt, 2, D_ENC), np.float16)
    for s in range(0, n, block):
        e = min(s + block, n)
        C = np.asarray(C_all[sel[s:e]], np.float32)
        C = np.stack([C[:, :h].reshape(-1, nc, D_ENC),
                      C[:, h:].reshape(-1, nc, D_ENC)], axis=2)
        T = np.asarray(T_all[pos[s:e]], np.float32)
        T = np.stack([T[:, :, :D_ENC], T[:, :, D_ENC:]], axis=2)
        X = np.concatenate([C, T], axis=1)
        X /= np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-6)
        out[s:e] = X.astype(np.float16)
    return out


def level_vectors(tok, level_of, dev, block=2048):
    """One unit vector per level: mean the level's tokens, L2 per encoder.

    Mean rather than anything cleverer: a GeM sweep over the deep level of the
    high-resolution pyramid found every p inside noise against p=1, and the one
    separated result was a loss. Within a level the mean is already right.
    """
    lo = torch.as_tensor(np.asarray(level_of))
    n_lvl = int(lo.max()) + 1
    idx = [torch.nonzero(lo == lv, as_tuple=True)[0].to(dev)
           for lv in range(n_lvl)]
    out = [np.empty((len(tok), 2 * D_ENC), np.float32) for _ in range(n_lvl)]
    nrm = lambda t: t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        for s in range(0, len(tok), block):
            e = min(s + block, len(tok))
            B = torch.from_numpy(np.asarray(tok[s:e], np.float32)).to(dev)
            for lv in range(n_lvl):
                m = B.index_select(1, idx[lv])
                v = torch.cat([nrm(m[:, :, 0].mean(1)),
                               nrm(m[:, :, 1].mean(1))], dim=1)
                out[lv][s:e] = nrm(v).cpu().numpy()
            del B
    return out


def row(name, e, rep):
    return "%-26s %9.1f %s" % (name, np.median(e[rep]), " ".join(
        "%7.1f%%" % (100 * (e[rep] < t).mean()) for t in THRESH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--split-mode", default="sequence")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    sel_rows, tile_pos = complete_rows("tile6")
    ds = pq.read_table(config.DATASET_PARQUET)
    lat = np.asarray(ds["lat"], np.float64)[sel_rows]
    lon = np.asarray(ds["lon"], np.float64)[sel_rows]
    seq = np.asarray(ds["sequence"]).astype("U40")[sel_rows]
    spl = sp.read(ds, a.split_mode)[0][sel_rows]
    tr, te = np.flatnonzero(spl == "train"), np.flatnonzero(spl == "test")

    nq = min(a.queries, len(te))
    qi, bi = te[:nq], tr
    same = seq[qi][:, None] == seq[bi][None, :]
    # Half of the queries choose the weighting, the other half report it. A
    # weighting picked as the best of a sweep and scored on the same queries
    # counts the noise that favoured it as gain -- measured, that inflated a
    # 25 km figure by 0.5 pp and turned a per-step result from positive to
    # negative.
    # crc32 of the sequence, not hash(): Python salts string hashing per
    # process, so the halves would reshuffle every run and no two arms would
    # be comparable. Splitting on the sequence rather than the row keeps
    # adjacent frames of one drive on the same side.
    hq = np.array([zlib.crc32(s.encode()) % 2 for s in seq[qi].tolist()])
    sel, rep = np.flatnonzero(hq == 0), np.flatnonzero(hq == 1)

    print("OSV-5M rows with tiles cached : {:,}".format(len(sel_rows)))
    print("split {!r}: {:,} train (bank), {:,} test".format(
        a.split_mode, len(tr), len(te)))
    print("{:,} queries used   {:,} choose / {:,} report".format(
        nq, len(sel), len(rep)))
    print("{:,} same-sequence bank cells masked\n".format(int(same.sum())),
          flush=True)

    X = load_tokens_blocked(sel_rows, tile_pos)
    level_of = [0] * 3 + [1] * (X.shape[1] - 3)
    print("tokens {}  levels {}  in {:.0f}s".format(
        tuple(X.shape), np.bincount(level_of), time.time() - t0), flush=True)

    lv = level_vectors(X, level_of, dev)
    S = [dense_sim(V[qi], V[bi], dev) for V in lv]
    print("similarities in {:.0f}s\n".format(time.time() - t0), flush=True)

    hdr = "%-26s %9s %s" % ("arm", "median km",
                            " ".join("%8s" % ("<%gkm" % t) for t in THRESH))
    print(hdr); print("-" * len(hdr))
    err = {}
    grid = [(1, 0), (0, 1), (3, 1), (2, 1), (3, 2), (1, 1), (2, 3), (1, 2),
            (1, 3)]
    for g in grid:
        w = np.asarray(g, np.float64) / sum(g)
        e, _ = topk_stats(w[0] * S[0] + w[1] * S[1], lat, lon, qi, bi, same)
        err[g] = e
        note = {(1, 0): "   <- L0 only, what ships",
                (1, 1): "   <- equal, the baseline"}.get(g, "")
        print(row("L0:L1 = %d:%d%s" % (g + (note,)), e, rep))

    best = max(grid, key=lambda g: (err[g][sel] < 25).mean())
    b_rep = max(grid, key=lambda g: (err[g][rep] < 25).mean())
    print("\nbest on <25 km: selection half {}, reporting half {}   {}".format(
        best, b_rep, "agree" if best == b_rep else "DISAGREE, the peak is flat"))

    rng = np.random.default_rng(0)
    print("\n--- against L0 only (the shipping representation), {:,} held-out "
          "queries ---".format(len(rep)))
    base = err[(1, 0)]
    for g in grid:
        if g == (1, 0):
            continue
        cells = []
        for t in THRESH:
            lo_, hi_ = paired(base[rep] < t, err[g][rep] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[g][rep] < t).mean() - (base[rep] < t).mean()),
                lo_, hi_, " " if lo_ * hi_ > 0 else "~"))
        print("%-14s %s" % ("L0:L1 = %d:%d" % g, " ".join(cells)))
    print("\n~ marks an interval spanning zero. {:.0f}s total"
          .format(time.time() - t0))


if __name__ == "__main__":
    main()
