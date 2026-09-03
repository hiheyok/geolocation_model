"""Is the fusion head bad, or is its 768-d random projection bad?

`fuse-attn-pyr47` reported the learned head at 20.9% within 25 km against the
1536-d level mean at 33.8%, and that -11.30 pp was recorded as closing the
direction. A review pointed out the comparison is not apples to apples, and it
is right: `FuseHead` is a residual on the mean pooled through a **fixed random**
1536->768 matrix (`base_w`), not on the mean itself. The comment there claims
"start at the baseline is exact"; it is exact only up to a random projection
that halves the width.

So the reported gap confounds three things: the width change, the random
projection, and the learned residual. This separates them by scoring the head
**at initialisation**, where the output layer is zeroed and the forward pass is
exactly `normalize(baseline @ base_w)`. Against that:

  L0+L1+L2 level (mean), 1536-d    what the head was compared against
  head at init, 768-d              the projection alone, nothing learned
  mean -> PCA 768 (train-fitted)   what a fair 768-d arm can do
  fusion head (trained), 768-d     the reported number, recomputed

If the trained head lands near its own initialisation, the learned part is a
null and the gap is the projection. If it lands well below, training actively
hurt. Either way the -11.30 pp against a 1536-d arm does not mean what it was
recorded to mean.

The PCA control is fitted on TRAIN ROWS ONLY, unlike the `mean + head, PCA to
1536` arm in fuse_head.py, which fits over every row including the test queries.

    OSV_RELEASE=s10 python scripts/fuse_init_probe.py --pyr-stem pyr47
"""

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from fuse_head import D_ENC, THRESH, FuseHead  # noqa: E402
from tile_match import dense_sim, topk_stats  # noqa: E402
from tile_pool import l2  # noqa: E402


def pca_fit_apply(fit_rows, all_rows, d, seed=0):
    """Randomised PCA fitted on `fit_rows` only, applied to everything."""
    rng = np.random.default_rng(seed)
    F = fit_rows[rng.choice(len(fit_rows), min(40000, len(fit_rows)),
                            replace=False)]
    mu = F.mean(0, keepdims=True)
    Fc = F - mu
    Om = rng.standard_normal((Fc.shape[1], d + 64)).astype(np.float32)
    Y = Fc @ Om
    for _ in range(2):
        Y = Fc @ (Fc.T @ Y)
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ Fc, full_matrices=False)
    return (all_rows - mu) @ Vt[:d].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyr-stem", default="pyr47")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--d", type=int, default=256)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    m = np.load(config.STREET_CACHE / (a.pyr_stem + "_meta.npz"),
                allow_pickle=True)
    lat, lon = m["lat"], m["lon"]
    seq = m["sequence"].astype("U40")
    level_of = m["level_of"].tolist()
    h = np.array([zlib.crc32(x.encode()) % 10 for x in seq.tolist()])
    tr, te = np.flatnonzero(h < 8), np.flatnonzero(h >= 8)

    X = np.asarray(np.load(config.STREET_CACHE / (a.pyr_stem + ".f16.npy"),
                           mmap_mode="r"), np.float32)
    X /= np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-6)
    print("{:,} rows: {:,} train, {:,} test   levels {}"
          .format(len(X), len(tr), len(te), np.bincount(level_of)), flush=True)

    # the 1536-d level-weighted mean, exactly as fuse_head prints it
    lo = np.asarray(level_of)
    n_lvl = int(lo.max()) + 1
    lvl = lambda i: np.stack([l2(X[:, lo == l, i].mean(1))
                              for l in range(n_lvl)]).mean(0)
    base_un = np.concatenate([l2(lvl(0)), l2(lvl(1))], 1)

    # the head at initialisation: out[1] is zeroed, so forward() is exactly
    # normalize(baseline @ base_w) -- the random projection and nothing else
    torch.manual_seed(0)
    head = FuseHead(d=a.d, out=768, n_reg=X.shape[1],
                    level_of=level_of).to(dev).eval()
    Z0 = np.zeros((len(X), 768), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            xb = torch.from_numpy(X[s:s + 1024]).to(dev)
            Z0[s:s + 1024] = head(xb).float().cpu().numpy()

    pca768 = pca_fit_apply(base_un[tr], base_un, 768)

    nq = min(a.queries, len(te))
    qi, bi = te[:nq], tr
    same = seq[qi][:, None] == seq[bi][None, :]
    print("{:,} test queries  {:,} train bank  {:,} same-sequence masked\n"
          .format(nq, len(bi), int(same.sum())), flush=True)

    print("%-34s %7s %10s %s" % ("arm", "dim", "median km",
                                 " ".join("%9s" % ("<%gkm" % t) for t in THRESH)))
    print("-" * 96)
    errs = {}
    for name, V in (("L0+L1+L2 level (mean)", base_un),
                    ("head at INIT (random proj only)", Z0),
                    ("mean -> PCA 768 (train-fitted)", pca768)):
        e, _ = topk_stats(dense_sim(V[qi], V[bi], dev), lat, lon, qi, bi, same)
        errs[name] = e
        print("%-34s %6dd %10.1f %s" % (
            name, V.shape[1], np.median(e),
            " ".join("%8.1f%%" % (100 * (e < t).mean()) for t in THRESH)),
            flush=True)

    b = errs["L0+L1+L2 level (mean)"]
    print("\nagainst the 1536-d mean, on <25 km:")
    for name in ("head at INIT (random proj only)", "mean -> PCA 768 (train-fitted)"):
        e = errs[name]
        print("  {:<34} {:+.2f} pp".format(
            name, 100 * ((e < 25).mean() - (b < 25).mean())))
    print("\n  fusion head (TRAINED), reported by the pyr47 run: -11.30 pp")
    print("\nRead the init row first. Whatever it gives is the part of that")
    print("-11.30 pp that the learned residual never had a chance to earn.")


if __name__ == "__main__":
    main()
