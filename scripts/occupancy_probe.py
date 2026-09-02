"""How much of the policy's per-step accuracy is just "where is there data?"

On the `cell8` holdout the policy scores 0.0% at s1 against a 0.39% chance
level -- below chance, which means it is not merely ignorant of held-out cells
but actively steering away from them.  The explanation that fits is an
occupancy prior: it has learned which cells contain images rather than how a map
view matches a street view, and on `cell8` the correct cell is guaranteed to be
one with no data.

That raises a sharper question about the split that actually matters. On
`sequence` the policy scores s1 76.2%, which looks like real map reading. But
`sequence` does not hold cells out, so an occupancy prior would score *well*
there too. This measures exactly that: a baseline that ignores the image
entirely and picks, among the 256 children of the true parent tile, whichever
has the most corpus images.

Two corpora, because the model sees both:

    train   the 400k training images alone
    corpus  training images plus every bank extension, which is what the
            retrieval prior actually votes with (2.25M images at bank55)

`scripts/count_table.py` already measured the training-corpus version of this
and agrees digit for digit (20.5 / 9.7 / 8.0 / 1.5%); what is new here is the
bank-inclusive corpus and the split-mode switch, and the answer barely moves.

Read it as a floor, not as a ceiling. The gap between this baseline and the
model is the part of the policy that needed the street image; whatever the
baseline already gets is available without looking at the photograph at all.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config
import tile_math as tm

G = 16
STEPS = tm.STEPS


def tile_for_vec(lat, lon, z):
    """Vectorised tile_math.tile_for -- the scalar one clamps with min/max.

    Same formula, so the two must agree exactly; checked against the scalar
    version on a sample before use.
    """
    lat = np.clip(lat, -tm.MAX_LAT, tm.MAX_LAT)
    x = (lon + 180.0) / 360.0
    sn = np.sin(np.radians(lat))
    y = 0.5 - np.log((1.0 + sn) / (1.0 - sn)) / (4.0 * np.pi)
    n = 1 << z
    return (np.clip((x * n).astype(np.int64), 0, n - 1),
            np.clip((y * n).astype(np.int64), 0, n - 1))


def digits(x16, y16):
    """(action, parent) per step, from z16 addresses -- see tile_math."""
    out = []
    for t in range(STEPS):
        shift = G ** (STEPS - 1 - t)
        col = (x16 // shift) % G
        row = (y16 // shift) % G
        z = 4 * t
        px, py = x16 >> (4 * (STEPS - t)), y16 >> (4 * (STEPS - t))
        out.append((row * G + col, py * (1 << z) + px))
    return out


def best_child(parent_tr, action_tr, parent_q):
    """For each query parent, the child action with the most corpus images."""
    key = parent_tr.astype(np.int64) * (G * G) + action_tr
    uniq, cnt = np.unique(key, return_counts=True)
    up, ua = uniq // (G * G), uniq % (G * G)
    # uniq is sorted, so each parent's entries are contiguous; order by
    # (parent, -count) and the first row of each group is its argmax
    order = np.lexsort((-cnt, up))
    up_s, ua_s = up[order], ua[order]
    first = np.ones(len(up_s), dtype=bool)
    first[1:] = up_s[1:] != up_s[:-1]
    par_top, act_top = up_s[first], ua_s[first]
    pos = np.searchsorted(par_top, parent_q)
    pos = np.clip(pos, 0, len(par_top) - 1)
    hit = par_top[pos] == parent_q
    return np.where(hit, act_top[pos], -1), hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--split-mode", default="sequence",
                    choices=("sequence", "cell8"))
    ap.add_argument("--exts", default="bank_ext,bank_ext2",
                    help="bank extensions to include in the 'corpus' arm")
    a = ap.parse_args()

    col = "split_" + a.split_mode          # see src/splits.py MODES
    ds = pq.read_table(config.DATASET_PARQUET, columns=["lat", "lon", col])
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    split = np.asarray(ds[col]).astype(str)
    x16, y16 = tile_for_vec(lat, lon, 4 * STEPS)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(lat), 200, replace=False):
        _, sx, sy = tm.tile_for(float(lat[i]), float(lon[i]), 4 * STEPS)
        assert (sx, sy) == (int(x16[i]), int(y16[i])), "vectorised tile_for "            "disagrees with tile_math at row {}".format(i)

    tr = split == "train"
    qi = split == a.split
    print("{:,} images on {}: {:,} train, {:,} {}".format(
        len(lat), col, int(tr.sum()), int(qi.sum()), a.split))

    ex, ey = [x16[tr]], [y16[tr]]
    for stem in [s.strip() for s in a.exts.split(",") if s.strip()]:
        m = np.load(config.bank_meta(stem), allow_pickle=True)
        ex.append(m["x16"].astype(np.int64))
        ey.append(m["y16"].astype(np.int64))
        print("  + {:<12} {:,} images".format(stem, len(m["x16"])))
    cx, cy = np.concatenate(ex), np.concatenate(ey)
    print("corpus {:,} images\n".format(len(cx)))

    dq = digits(x16[qi], y16[qi])
    banks = [("train", x16[tr], y16[tr]), ("corpus", cx, cy)]

    print("%-8s %s" % ("", "  ".join("%14s" % ("s%d" % t)
                                     for t in range(STEPS))))
    print("-" * 70)
    rows = {}
    for name, bx, by in banks:
        db = digits(bx, by)
        cells = []
        for t in range(STEPS):
            pred, seen = best_child(db[t][1], db[t][0], dq[t][1])
            acc = float((pred == dq[t][0]).mean())
            cells.append("%6.1f%% (%3.0f%%)" % (100 * acc, 100 * seen.mean()))
            rows[(name, t)] = acc
        print("%-8s %s" % (name, "  ".join(cells)))
    print("\nchance is 1/256 = 0.39%.  the bracket is the share of queries "
          "whose\nparent tile appears in that corpus at all; the rest are "
          "counted wrong.")

    print("\nThe model, for comparison (val, teacher-forced, from the training "
          "logs):")
    print("  d1536-b190-e4 on sequence   s0 90.6%  s1 76.2%  s2 50.2%  s3 17.0%")
    print("  d1536-b115-e4-cell8 on cell8  s0 71.4%  s1  0.0%  s2  0.5%  s3  4.3%")
    print("\nThe gap above the 'corpus' row is what the street image bought.")


if __name__ == "__main__":
    main()
