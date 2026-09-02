"""What does pure co-location score, with no images and no model?

At 50k images the empirical table `p(child | parent tile)` scored 6.58% at step
2 where the trained dual-encoder model scored 5.9%, and that single fact is why
tile-ID memory was rejected: a mechanism whose ceiling is a count table is
measuring how the split was drawn, not what the photograph shows.

Both halves of that comparison are now stale.  The model is an order of
magnitude better and the data is an order of magnitude larger, so the ceiling
has to be re-measured before a per-tile embedding table can be interpreted.

This is also the exact control for the cheapest rung of that table.  Under
teacher forcing a per-tile *scalar* bias receives positive gradient when its
tile is the target and negative gradient when it is a sibling of the target, so
integrated over an epoch it converges to precisely this count table.  Whatever
this scores is what a scalar arm is worth; anything a 128-d key earns above it
is the part that is genuinely query-dependent rather than co-location.

Teacher-forced per-step accuracy, so it is directly comparable to the `s0..s3`
row the training log prints.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import splits as sp
import tile_math as tm


def z16_xy(lat, lon):
    lat = np.clip(np.asarray(lat, np.float64), -tm.MAX_LAT, tm.MAX_LAT)
    n = 1 << 16
    s = np.sin(np.radians(lat))
    x = (np.asarray(lon, np.float64) + 180.0) / 360.0 * n
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n
    return (np.clip(x, 0, n - 1).astype(np.int64),
            np.clip(y, 0, n - 1).astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-mode", default="sequence")
    ap.add_argument("--eval-split", default="test")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    ds = pq.read_table(config.DATASET_PARQUET)
    x16, y16 = z16_xy(np.asarray(ds["lat"]), np.asarray(ds["lon"]))
    spl, h = sp.read(ds, a.split_mode)
    tr, te = spl == "train", spl == a.eval_split
    g, steps = tm.G, tm.STEPS
    print("release {}  split {} {}  train {:,}  {} {:,}".format(
        config.RELEASE, a.split_mode, h, int(tr.sum()), a.eval_split,
        int(te.sum())), flush=True)
    print()
    print("%-6s %11s %14s %12s %14s" % (
        "step", "cell width", "parent cells", "accuracy", "unseen parent"))
    print("-" * 62)

    width = [2504.0, 156.0, 9.8, 0.611]
    for t in range(steps):
        shift_p = 16 - 4 * t             # parent level z = 4t
        shift_c = 16 - 4 * (t + 1)       # child level
        # parent id; at t=0 every image shares the single z0 root
        px, py = x16 >> shift_p, y16 >> shift_p
        parent = py * (1 << (4 * t)) + px if t else np.zeros_like(px)
        act = (((y16 >> shift_c) % g) * g + ((x16 >> shift_c) % g))

        # counts[parent][action] from train only, held as a dict of arrays so
        # the deep levels stay sparse rather than allocating the full grid
        order = np.argsort(parent[tr], kind="stable")
        ptr, atr = parent[tr][order], act[tr][order]
        bounds = np.flatnonzero(np.diff(ptr)) + 1
        best = {}
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(ptr)]):
            c = np.bincount(atr[lo:hi], minlength=g * g)
            best[int(ptr[lo])] = int(c.argmax())

        pt, at = parent[te], act[te]
        pred = np.array([best.get(int(p), -1) for p in pt])
        unseen = pred < 0
        acc = (pred == at).mean()
        print("s%-5d %8.3f km %14s %11.1f%% %13.1f%%" % (
            t, width[t], "{:,}".format(len(best)), 100 * acc,
            100 * unseen.mean()), flush=True)

    print()
    print("compare against the model's teacher-forced per-step accuracy, the "
          "'s0 .. s3' row in a training log.")


if __name__ == "__main__":
    main()
