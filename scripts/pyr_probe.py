"""Mean-pooled pyramid arms on the high-resolution cache: the control.

The earlier finding -- three levels lose to two, and level-weighting beats
token-weighting by 4 pp at three levels -- came from 5,426 US images. This
repeats it on 18,812 global ones from the same source, and it is the baseline
the attention arm has to beat, so it needs to be right.

The two weightings are the point. Pooling the union of levels weights *tokens*,
and a 3+6+24 pyramid then gives the deepest level 73% of the vector; pooling the
per-level means weights *scales*, one third each. If L2 only hurts under token
weighting, "L2 hurts" is a statement about the pooling and not about the
information, and attention -- which reweights by content -- is not bound by it.

Every arm is a 1536-d [DINOv2 | SigLIP] vector, so no arm wins on width. Whole
sequences are held out, not rows: consecutive frames of one drive are near
duplicates, and a row-wise split would put a near-copy of every query in the
bank.
"""

import argparse
import os
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config
from tile_pool import score, paired, l2

THRESH = (1, 25, 200, 750, 2500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="pyr33")
    ap.add_argument("--queries", type=int, default=3000)
    a = ap.parse_args()

    m = np.load(config.STREET_CACHE / (a.cache + "_meta.npz"), allow_pickle=True)
    lat, lon = m["lat"], m["lon"]
    seq = m["sequence"].astype("U40")
    lv = m["level_of"]
    X = np.asarray(np.load(config.STREET_CACHE / (a.cache + ".f16.npy"),
                           mmap_mode="r"), np.float32)
    X /= np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-6)

    h = np.array([zlib.crc32(s.encode()) % 10 for s in seq.tolist()])
    te = np.flatnonzero(h >= 8)
    tr = np.flatnonzero(h < 8)
    rng = np.random.default_rng(0)
    qi = te[rng.choice(len(te), min(a.queries, len(te)), replace=False)]
    qi.sort()
    bi = tr
    same = seq[qi][:, None] == seq[bi][None, :]
    print("{:,} images  {:,} queries  {:,} bank  {:,} same-sequence masked"
          .format(len(lat), len(qi), len(bi), int(same.sum())), flush=True)
    print("levels: {} tokens at L0, {} at L1, {} at L2\n".format(
        *[int((lv == i).sum()) for i in range(3)]), flush=True)

    def by_token(levels):
        """mean over every token of every level -- deep levels dominate"""
        mask = np.isin(lv, levels)
        return np.concatenate([l2(X[:, mask, 0].mean(1)),
                               l2(X[:, mask, 1].mean(1))], axis=1)

    def by_level(levels):
        """mean of the per-level means -- every scale weighted the same"""
        per = [np.stack([l2(X[:, lv == i, 0].mean(1)),
                         l2(X[:, lv == i, 1].mean(1))], 0) for i in levels]
        v = np.stack(per).mean(0)
        return np.concatenate([l2(v[0]), l2(v[1])], axis=1)

    arms = {"L0": by_token([0])}
    for tag, ls in (("L0+L1", [0, 1]), ("L0+L1+L2", [0, 1, 2]),
                    ("L1+L2", [1, 2])):
        arms["{} [token]".format(tag)] = by_token(ls)
        arms["{} [level]".format(tag)] = by_level(ls)
    arms["L2 alone"] = by_token([2])

    err = {}
    print("%-18s %6s %12s %s" % ("arm", "d", "top-1 median",
                                 " ".join("%9s" % ("<%gkm" % t)
                                          for t in THRESH)))
    print("-" * 92)
    for k, V in arms.items():
        d1, _ = score(V[qi], V[bi], lat, lon, qi, bi, same)
        err[k] = d1
        print("%-18s %6d %12.1f %s" % (k, V.shape[1], np.median(d1), " ".join(
            "%8.1f%%" % (100 * (d1 < t).mean()) for t in THRESH)), flush=True)

    base = "L0"
    rng2 = np.random.default_rng(0)
    print("\npaired against {}, percentage points, ~ spans zero".format(base))
    print("%-18s %s" % ("arm", " ".join("%18s" % ("<%gkm" % t)
                                        for t in THRESH)))
    print("-" * 112)
    for k in arms:
        if k == base:
            continue
        cells = []
        for t in THRESH:
            lo, hi = paired(err[base] < t, err[k] < t, rng2)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[k] < t).mean() - (err[base] < t).mean()), lo, hi,
                " " if lo * hi > 0 else "~"))
        print("%-18s %s" % (k, " ".join(cells)))

    print("\nRead the two weightings against each other, not against L0: if the "
          "gap\nbetween [token] and [level] at three levels is large, the "
          "deepest level is\nbeing drowned by count rather than being "
          "uninformative -- and that is a\nstatement about mean pooling, not "
          "about the pyramid.")


if __name__ == "__main__":
    main()
