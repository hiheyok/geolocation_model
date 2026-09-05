"""Does the blend weight want to change with the zoom step, and can it?

`pyr_blend.py` found one weighting for all of retrieval: levels 1:1:2 plus a
head weight near 0.10. But the agent does not use the prior once. It uses it at
every step of a descent whose cell goes 2504 km -> 156 km -> 9.8 km -> 611 m,
and the components measurably disagree about scale -- L0-heavy mixes win at
1 km while L2-heavy mixes win from 25 km out.

So the weight might want to be `w_t`. Two questions, in order, because the
second is only worth asking if the first says yes:

**1. How much headroom is there?** Sweep the whole 4-component simplex and, for
each distance threshold, take the best weighting *for that threshold*. The gap
between that and the single global weighting is the most per-step tuning could
ever buy. If it is small, stop -- the plumbing is not free.

**2. Can it be reached without four banks?** The prior is served from one k-NN
cache, so a genuinely per-step vector would mean one bank per step: four times
the memory, against a neighbour table already at 5.38 GB. The implementable
version is to **retrieve once and re-rank per step**: take the cached top-k
under a recall-oriented weighting, then re-score those k with the step's own
weighting. Re-scoring 32 candidates is free; the question is whether the right
neighbour survives the shared retrieval to be re-ranked at all.

Deliberately swept, not learned. A learned per-step gate over the DINOv2 /
SigLIP scale split learned nothing, and that split is the same shape as this.

    OSV_RELEASE=s10 py scripts/pyr_perstep.py --pyr-stem pyr47 --queries 3000
"""

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import maskio                                    # noqa: E402
from tile_pool import paired                     # noqa: E402
from tile_match import dense_sim, topk_stats     # noqa: E402
from pyr_blend import per_level, level_mean, head_vectors, THRESH  # noqa: E402

# Which cell width each step descends into, so a threshold can be read as the
# step it informs. g=16 over 4 steps: 2504 km, 156 km, 9.8 km, 611 m.
STEP_KM = (2504, 156, 9.8, 0.611)


def simplex(n, step):
    """Every non-negative weight vector of length n summing to 1, on a grid.

    Enumerated as compositions of an integer, which is the version that cannot
    go wrong: the stars-and-bars spelling produced negative weights and rows
    summing to 0.8, and a weight vector that does not sum to 1 silently
    rescales every similarity it combines.
    """
    q = int(round(1.0 / step))

    def comps(k, total):
        if k == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in comps(k - 1, total - i):
                yield (i,) + rest

    for c in comps(n, q):
        yield tuple(x / float(q) for x in c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyr-stem", default="pyr47")
    ap.add_argument("--head", default="pyr47_fuse_p05_head.pt")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--step", type=float, default=0.2,
                    help="simplex granularity for the level weights")
    ap.add_argument("--retr-k", type=int, default=32,
                    help="neighbours the shared retrieval hands to the "
                         "per-step re-ranker; matches the agent's --retr-k")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = np.load(config.STREET_CACHE / (a.pyr_stem + "_meta.npz"),
                allow_pickle=True)
    lat, lon = m["lat"], m["lon"]
    seq = m["sequence"].astype("U40")
    level_of = m["level_of"].tolist()

    pyr = np.load(config.STREET_CACHE / (a.pyr_stem + ".f16.npy"), mmap_mode="r")
    done_p = config.STREET_CACHE / (a.pyr_stem + "_done.u8.npy")
    if done_p.exists():
        d = maskio.load_mask(done_p, len(pyr), a.pyr_stem)
        if not maskio.is_complete(d, len(pyr)):
            raise SystemExit("{} is incomplete".format(done_p.name))

    h = np.array([zlib.crc32(x.encode()) % 10 for x in seq.tolist()])
    tr, te = np.flatnonzero(h < 8), np.flatnonzero(h >= 8)
    nq = min(a.queries, len(te))
    qi, bi = te[:nq], tr
    same = seq[qi][:, None] == seq[bi][None, :]
    # Choosing the best of ~126 weightings and reporting it on the same
    # queries is selection on the test set: the winner carries whatever noise
    # favoured it. Half the queries pick the weighting, the other half report
    # it, and every number below is the reporting half. Split by sequence for
    # the same reason the train/test split is -- adjacent frames of one drive
    # would otherwise sit on both sides.
    hq = h[qi]
    pick_m = (hq % 2) == 0
    sel, rep = np.flatnonzero(pick_m), np.flatnonzero(~pick_m)
    print("{}  {:,} rows, levels {}\n{:,} queries  {:,} bank"
          .format(a.pyr_stem, len(pyr), np.bincount(level_of), nq, len(bi)))
    print("{:,} queries choose the weighting, {:,} report it\n"
          .format(len(sel), len(rep)), flush=True)

    levels = per_level(pyr, level_of)
    Z = head_vectors(pyr, level_of, config.STREET_CACHE / a.head, dev)
    S = [dense_sim(V[qi], V[bi], dev) for V in levels]
    S.append(dense_sim(Z[qi], Z[bi], dev))
    base_err, _ = topk_stats(dense_sim(level_mean(levels)[qi],
                                       level_mean(levels)[bi], dev),
                             lat, lon, qi, bi, same)
    print("components + similarities in {:.0f}s".format(time.time() - t0),
          flush=True)

    # ---- 1. the whole simplex, scored at every threshold ------------------
    # (1-h) on the level mix, h on the head, so the four weights sum to 1 and
    # `h` means exactly what it means in pyr_blend.py. Ranking is invariant to
    # a global scale, so leaving the sum at 1+h would not corrupt a result --
    # but it would stop the two scripts' numbers being the same quantity.
    grid = [((1 - hw) * l0, (1 - hw) * l1, (1 - hw) * l2, hw)
            for (l0, l1, l2) in simplex(3, a.step)
            for hw in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)]
    buf = np.empty_like(S[0])
    hits, rhits, errs = {}, {}, {}
    for w in grid:
        buf[:] = 0.0
        for c in range(4):
            if w[c]:
                buf += w[c] * S[c]
        e, _ = topk_stats(buf, lat, lon, qi, bi, same)
        errs[w] = e
        hits[w] = tuple((e[sel] < t).mean() for t in THRESH)    # chooses
        rhits[w] = tuple((e[rep] < t).mean() for t in THRESH)   # reports
    print("swept {} weightings in {:.0f}s\n".format(len(grid), time.time() - t0),
          flush=True)

    # normalise the level part so the printed ratio is readable
    def label(w):
        lv = np.array(w[:3])
        g = lv / lv[lv > 0].min() if (lv > 0).any() else lv
        return "L %s  head %.2f" % (":".join("%.3g" % x for x in g), w[3])

    glob = max(grid, key=lambda w: hits[w][THRESH.index(25)])
    print("single global weighting, chosen on <25 km over the selection half:")
    print("  {}".format(label(glob)))
    print("\nEvery figure below is the REPORTING half. Each row's weighting was"
          "\nchosen on the selection half, so the headroom is what per-step"
          "\ntuning would actually transfer -- not what it fits.")
    print("\n%-10s %-30s %9s %9s %9s" % (
        "threshold", "best weighting for it", "best", "global", "headroom"))
    print("-" * 72)
    head_room = {}
    for i, t in enumerate(THRESH):
        b = max(grid, key=lambda w: hits[w][i])          # chosen on sel
        gap = 100 * (rhits[b][i] - rhits[glob][i])       # measured on rep
        head_room[t] = (b, gap)
        step = min(range(len(STEP_KM)), key=lambda s: abs(STEP_KM[s] - t))
        print("%-4gkm s%d %-30s %8.2f%% %8.2f%% %+8.2f pp" % (
            t, step, label(b), 100 * rhits[b][i], 100 * rhits[glob][i], gap))

    # ---- 2. one retrieval, per-step re-ranking ---------------------------
    # The implementable version: the agent gets ONE cached neighbour list, so a
    # per-step weighting can only reorder what that list already contains.
    print("\n--- one retrieval (global weighting, top-{}), re-ranked per "
          "threshold ---".format(a.retr_k))
    buf[:] = 0.0
    for c in range(4):
        if glob[c]:
            buf += glob[c] * S[c]
    masked = np.where(same, -2.0, buf)
    k = min(a.retr_k, masked.shape[1])
    keep = np.argpartition(-masked, k - 1, axis=1)[:, :k]

    print("%-10s %9s %9s %9s %9s" % ("threshold", "global", "re-ranked",
                                     "gain", "of headroom"))
    print("-" * 52)
    for i, t in enumerate(THRESH):
        b, gap = head_room[t]
        buf[:] = 0.0
        for c in range(4):
            if b[c]:
                buf += b[c] * S[c]
        M = np.full_like(buf, -2.0)
        np.put_along_axis(M, keep, np.take_along_axis(buf, keep, 1), 1)
        e, _ = topk_stats(M, lat, lon, qi, bi, same)
        got = 100 * ((e[rep] < t).mean() - rhits[glob][i])
        frac = "n/a" if abs(gap) < 1e-9 else "%.0f%%" % (100 * got / gap)
        print("%-4gkm %11.2f%% %9.2f%% %+8.2f pp %9s" % (
            t, 100 * rhits[glob][i], 100 * (e[rep] < t).mean(), got, frac))

    # ---- paired intervals for the global arm against the published pool ---
    rng = np.random.default_rng(0)
    print("\n--- global weighting against the published level mean ---")
    cells = []
    for i, t in enumerate(THRESH):
        lo_, hi_ = paired(base_err[rep] < t, errs[glob][rep] < t, rng)
        cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
            100 * (rhits[glob][i] - (base_err[rep] < t).mean()),
            lo_, hi_, " " if lo_ * hi_ > 0 else "~"))
    print("  " + " ".join(cells))
    print("\n~ marks an interval spanning zero. {:.0f}s total"
          .format(time.time() - t0))


if __name__ == "__main__":
    main()
