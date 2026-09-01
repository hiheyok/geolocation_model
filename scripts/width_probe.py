"""Can the street vector be halved again? 1536 -> 768 and below.

Pooling three crop tokens into one 1536-d vector instead of concatenating them
into 4608 measured as free twice over -- inside noise on `sequence` and again on
`cell8` -- and that is what paid for the bank going to 1.90M images and the
median halving to 3.6 km. Corpus is still the best axis in the project, so the
obvious question is whether the same trick works once more: at 768-d a 2.65M
bank is 4.07 GB instead of 8.14, and the same host RAM would hold about 5.3M
images.

Two families of arm, and the comparison between them is the point:

    PCA d       a learned projection of the full [DINOv2 | SigLIP] vector
    half        just one encoder's 768 dimensions, no projection at all

If PCA-768 is inside noise of 1536, the width can be halved. If it is also no
better than taking a single encoder's half, then whatever the second encoder
contributes does not survive compression, and the dual-encoder result -- worth
62 km when it landed -- is really a claim about width rather than about having
two views. Those are very different things to believe, and only the second arm
separates them.

Retrieval-level only, and deliberately so: this costs minutes and decides
whether a training arm is worth an hour. It runs on CPU, so it does not compete
with whatever is on the GPU.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config
import splits as sp
from tile_pool import pca_fit, score, paired, l2

D_ENC = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street-file", default="pool_bal.f16.npy")
    ap.add_argument("--split-mode", default="sequence")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--bank", type=int, default=100000)
    ap.add_argument("--widths", default="768,384,192")
    a = ap.parse_args()

    X = np.load(config.STREET_CACHE / a.street_file, mmap_mode="r")
    ds = pq.read_table(config.DATASET_PARQUET, columns=["lat", "lon", "sequence"])
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    seq = np.asarray(ds["sequence"]).astype("U40")
    labels = sp.read(pq.read_table(config.DATASET_PARQUET), a.split_mode)[0]

    rng = np.random.default_rng(0)
    te = np.flatnonzero(labels == "test")
    tr = np.flatnonzero(labels == "train")
    qi = te[rng.choice(len(te), min(a.queries, len(te)), replace=False)]
    bi = tr[rng.choice(len(tr), min(a.bank, len(tr)), replace=False)]
    qi.sort()
    bi.sort()
    same = seq[qi][:, None] == seq[bi][None, :]
    print("{}  {:,} x {}".format(a.street_file, X.shape[0], X.shape[1]))
    print("{:,} queries  {:,} bank  {:,} same-sequence masked\n"
          .format(len(qi), len(bi), int(same.sum())), flush=True)

    rows = np.union1d(qi, bi)
    V = np.asarray(X[rows], dtype=np.float32)
    pos = {r: i for i, r in enumerate(rows.tolist())}
    q = np.array([pos[r] for r in qi.tolist()])
    b = np.array([pos[r] for r in bi.tolist()])

    arms = [("1536 as shipped", V)]
    for d in [int(w) for w in a.widths.split(",") if w.strip()]:
        mu, P = pca_fit(V, d, np.random.default_rng(0))
        arms.append(("PCA {}".format(d), (V - mu) @ P))
    # No projection at all -- just one encoder's block. The pooled layout is
    # [DINOv2 768 | SigLIP 768], see pool_street.py.
    arms.append(("DINOv2 half 768", V[:, :D_ENC].copy()))
    arms.append(("SigLIP half 768", V[:, D_ENC:].copy()))

    res = {}
    print("%-18s %6s %12s %12s" % ("arm", "d", "top-1 median", "any-of-32<25"))
    print("-" * 52)
    for name, E in arms:
        d1, hit = score(E[q], E[b], lat, lon, qi, bi, same)
        res[name] = (d1, hit)
        print("%-18s %6d %12.1f %11.1f%%"
              % (name, E.shape[1], np.median(d1), 100 * hit.mean()), flush=True)

    base = "1536 as shipped"
    rng2 = np.random.default_rng(0)
    print("\npaired against {}, percentage points on any-of-32 <25 km"
          .format(base))
    print("-" * 52)
    for name in res:
        if name == base:
            continue
        lo, hi = paired(res[base][1], res[name][1], rng2)
        delta = 100 * (res[name][1].mean() - res[base][1].mean())
        flag = "separated" if lo * hi > 0 else "inside noise"
        print("%-18s %+6.2f  [%+.2f, %+.2f]  %s" % (name, delta, lo, hi, flag))

    print("\nBank bytes at 2.65M images: "
          + "  ".join("%d-d %.2f GB" % (d, 2.65e6 * d * 2 / 1e9)
                      for d in (1536, 768, 384, 192)))


if __name__ == "__main__":
    main()
