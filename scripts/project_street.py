"""Project a street cache down to a narrower width with a fitted PCA.

`pool_street.py` halved 4608 -> 1536 by averaging crop tokens, which is free
because the three crops live in one space. Halving again cannot work that way:
[DINOv2 | SigLIP] are unrelated bases and averaging them is arbitrary. A fitted
projection is the honest version, and at 768 it measured inside noise on
any-of-32 `<25 km` (-0.17 pp [-1.00, +0.63], `scripts/width_probe.py`) while
384 did not, so 768 is where the knee is.

Two things this has to get right.

**One basis for every row.** Queries and bank must land in the same space, so
the basis is fitted once on a subsample of the *release* rows and then applied
to everything, extensions included. Fitting per-file would silently put the two
corpora in different spaces -- the failure would look like a weaker bank rather
than an error.

**The basis is saved.** Without it no future extension can be projected to
match, and the cache becomes a dead end.

Writes `<out>.f16.npy` plus `<out>_pca.npz` holding the mean and the basis.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config
from tile_pool import pca_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--fit-rows", type=int, default=200000,
                    help="rows sampled to fit the basis")
    ap.add_argument("--fit-from", type=int, default=500000,
                    help="fit only within the first N rows, i.e. the release; "
                         "0 to sample the whole file")
    ap.add_argument("--basis", default="",
                    help="reuse a saved <stem>_pca.npz instead of fitting")
    ap.add_argument("--block", type=int, default=100000)
    a = ap.parse_args()

    src = config.STREET_CACHE / a.src
    out = config.STREET_CACHE / a.out
    if src == out:
        sys.exit("refusing to write over the source")
    X = np.load(src, mmap_mode="r")
    n, d = X.shape
    print("{}  {:,} x {}  ->  {:,} x {}   {:.2f} -> {:.2f} GB".format(
        a.src, n, d, n, a.dim, X.nbytes / 1e9, n * a.dim * 2 / 1e9), flush=True)

    if a.basis:
        z = np.load(config.STREET_CACHE / a.basis)
        mu, P = z["mu"], z["P"]
        if P.shape != (d, a.dim):
            sys.exit("basis is {} but this needs {}".format(P.shape, (d, a.dim)))
        print("reusing basis {}".format(a.basis), flush=True)
    else:
        hi = min(a.fit_from or n, n)
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(hi, min(a.fit_rows, hi), replace=False))
        print("fitting on {:,} rows drawn from the first {:,}".format(
            len(pick), hi), flush=True)
        F = np.asarray(X[pick], dtype=np.float32)
        mu, P = pca_fit(F, a.dim, np.random.default_rng(0), fit_rows=len(pick))
        del F
        np.savez(str(out).replace(".f16.npy", "") + "_pca.npz", mu=mu, P=P,
                 src=a.src, dim=a.dim)
        print("saved basis alongside the output", flush=True)

    Y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(n, a.dim))
    t0 = time.time()
    for s in range(0, n, a.block):
        e = min(s + a.block, n)
        Y[s:e] = ((np.asarray(X[s:e], dtype=np.float32) - mu) @ P
                  ).astype(np.float16)
        el = time.time() - t0
        print("  {:>9,}/{:,}  {:.0f}s  eta {:.0f}s".format(
            e, n, el, el * (n - e) / max(e, 1)), flush=True)
    Y.flush()

    # the projection must be reproducible from the saved basis, not just from
    # this process; recompute a handful of rows the long way round
    rng = np.random.default_rng(1)
    for i in rng.choice(n, 5, replace=False):
        want = (np.asarray(X[i], dtype=np.float32) - mu[0]) @ P
        got = np.asarray(Y[i], dtype=np.float32)
        assert np.allclose(want, got, atol=3e-2), "row {} disagrees".format(i)
    zeros = 0
    for s in range(0, n, a.block):
        blk = np.asarray(Y[s:min(s + a.block, n)], dtype=np.float32)
        zeros += int((np.abs(blk).sum(1) == 0).sum())
    if zeros:
        raise SystemExit("{:,} all-zero rows".format(zeros))
    print("\nwrote {} in {:.0f}s; 5 rows verified, no zero rows".format(
        out.name, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
