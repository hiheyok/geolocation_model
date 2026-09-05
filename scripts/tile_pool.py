"""Steps 2-3 of the tile-fusion ladder: is it the tiles, or was it the joining?

`tile_probe.py` measured six tiles *concatenated* against three crops
concatenated, and the tiles lost.  That comparison confounds two things.
Concatenated cosine compares tile *i* of the query only against tile *i* of the
bank image, so two photographs of the same street from ten metres apart put the
same building in different slots and every per-tile term misses at once.  Three
full-height crops are coarser and far more tolerant of that shift.

So the question this script asks is not "tiles or crops" but:

    with the joining held fixed, do six tiles carry more geolocation signal
    than three crops?

Both sources are sequences of 1536-d tokens -- crop3 is (3, 1536), tile6 is
(6, 1536), each token being [DINOv2 768 | SigLIP 768] -- so every pooling scheme
below applies unchanged to both, and the pooled widths are *identical*.  That
makes the headline comparison equal-bytes with no PCA in the way of it, which
is what the earlier probe could not do.

Pooling schemes, cheapest first, each isolating one suspected defect:

  concat  the probe's arm.  Present as a control: it must reproduce the losing
          numbers, or the cache is wrong.
  mean    order-free.  If this closes the gap, rigid slot correspondence was
          the defect and attention is the upgrade path.  If it does not, the
          tiles genuinely carry less than full-height crops and no amount of
          attention will rescue that.
  bnorm   L2-normalise the DINOv2 and SigLIP halves of each token separately,
          then mean.  The 81/19 imbalance measured across the whole vector also
          operates *within* each token, so a token's two encoders enter the sum
          at whatever raw activation norm they happen to have.
  tnorm   L2-normalise each whole token, then mean.  Separates "one encoder
          shouts" from "one tile shouts": a high-contrast road tile has a larger
          activation norm than a flat sky tile and silently outweighs it.
  gem     generalised mean, p=3, on the bnorm tokens.  A non-learned stand-in
          for the selectivity attention pooling is supposed to provide: it
          weights peaked dimensions above flat ones without any parameters.

Doing both normalisations is not a fifth arm: after per-block normalisation
every token has norm sqrt(2) exactly, so a subsequent per-token normalisation
divides every token by the same constant and the pooled direction -- hence the
cosine -- is unchanged.  bnorm already subsumes tnorm; the two are alternatives,
not a ladder.

`crop3` is read from the shipping cache rather than recomputed, which makes it
the control for the *draw* as well as for the pooling: if it scores what it
scored in the earlier probe, this sample is equivalent to that one and any
difference in the tile arms belongs to the tiles.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
from tile_math import great_circle_km as great_circle

K = 32
R_EARTH = 6371.0088


def l2(X, axis=-1):
    return X / np.linalg.norm(X, axis=axis, keepdims=True).clip(1e-6)


def pool(T, scheme, d_enc=768):
    """(n, tokens, 1536) float32 -> (n, width) float32."""
    if scheme == "concat":
        n = T.shape[0]
        return np.concatenate([T[:, :, :d_enc].reshape(n, -1),
                               T[:, :, d_enc:].reshape(n, -1)], axis=1)
    X = T
    if scheme in ("bnorm", "gem"):
        X = np.concatenate([l2(X[:, :, :d_enc]), l2(X[:, :, d_enc:])], axis=2)
    elif scheme == "tnorm":
        X = l2(X)
    if scheme == "gem":
        # The p-th root is what makes this a generalised *mean*. Without it
        # this returned a signed third moment, which is not scale-equivalent
        # to the inputs and is not what the arm claims to measure.
        p = 3.0
        m = (np.sign(X) * np.abs(X) ** p).mean(1)
        return np.sign(m) * np.abs(m) ** (1.0 / p)
    return X.mean(1)


def pca_fit(X, d, rng, fit_rows=40000):
    """Randomised PCA, fitted on a subsample.

    The concat arm is 9,216-d over 117k rows -- 4.3 GB, and centring makes a
    second copy of it.  A 40k-row fit estimates a 512-d subspace far more
    precisely than this measurement can resolve, at a third of the peak.
    """
    if X.shape[0] > fit_rows:
        X = X[rng.choice(X.shape[0], fit_rows, replace=False)]
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    Om = rng.standard_normal((Xc.shape[1], d + 64)).astype(np.float32)
    Y = Xc @ Om
    for _ in range(2):
        Y = Xc @ (Xc.T @ Y)
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return mu, Vt[:d].T


def score(Eq, Eb, lat, lon, qi, bi, same):
    """Per-query top-1 error and any-of-K hit, so arms can be paired later."""
    # min(K, n), not n-1: the -1 assumed the query sits inside the bank,
    # which is only true when the two sets are the same. These are
    # disjoint, so it dropped one legal neighbour and gave k=0 on a
    # one-row bank. argpartition takes an INDEX though, so its kth is
    # k-1 -- passing k is out of bounds once K reaches the bank size,
    # which is a crash the first version of this fix introduced.
    k = max(1, min(K, Eb.shape[0]))
    Sm = l2(Eq) @ l2(Eb).T
    Sm[same] = -2.0
    j = np.argpartition(-Sm, k - 1, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(Sm, j, 1), axis=1)
    got = np.take_along_axis(j, o, 1)
    d1 = great_circle(lat[qi], lon[qi], lat[bi[got[:, 0]]], lon[bi[got[:, 0]]])
    dk = great_circle(np.repeat(lat[qi], k), np.repeat(lon[qi], k),
                      lat[bi[got.ravel()]], lon[bi[got.ravel()]]).reshape(-1, k)
    return d1, (dk < 25).any(1)


def paired(a, b, rng, n=2000):
    """95% interval on (b - a) in percentage points, resampling queries."""
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = 100 * (b[idx].mean(1) - a[idx].mean(1))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="tile6", help="cache stem")
    ap.add_argument("--crop3", default="dual_c3.f16.npy")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--schemes", default="concat,mean,bnorm,tnorm,gem")
    ap.add_argument("--dims", default="512",
                    help="PCA widths; the pooled arms are already equal-width, "
                         "so this only matters for concat")
    ap.add_argument("--n", type=int, default=0, help="cap on cache rows used")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    sel = np.load(config.STREET_CACHE / (a.tiles + "_rows.i64.npy"))
    done = np.load(config.STREET_CACHE / (a.tiles + "_done.u8.npy"))
    T = np.load(config.STREET_CACHE / (a.tiles + ".f16.npy"), mmap_mode="r")
    keep = np.flatnonzero(done == 1)
    if a.n:
        keep = keep[:a.n]
    sel = sel[keep]

    ds = pq.read_table(config.DATASET_PARQUET,
                       columns=["lat", "lon", "sequence"])
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    seq = np.asarray(ds["sequence"]).astype("U40")

    nq = a.queries
    qi, bi = sel[:nq], sel[nq:]
    same = seq[qi][:, None] == seq[bi][None, :]
    print("release {}  {:,} images  {:,} queries  {:,} bank".format(
        config.RELEASE, len(sel), nq, len(bi)), flush=True)
    print("same-sequence pairs masked: {:,}\n".format(int(same.sum())), flush=True)

    # Both sources as (n, tokens, 1536).  dual_c3 is stored as
    # [dino c0c1c2 | siglip c0c1c2], the same block-major order the tile cache
    # reverses, so the crop arm has to be un-interleaved to match.
    C = np.asarray(np.load(config.STREET_CACHE / a.crop3,
                           mmap_mode="r")[sel], dtype=np.float32)
    h = C.shape[1] // 2
    crops = h // 768
    C = np.concatenate([C[:, :h].reshape(-1, crops, 768),
                        C[:, h:].reshape(-1, crops, 768)], axis=2)
    sources = [("crop3", C), (a.tiles, np.asarray(T[keep], dtype=np.float32))]

    dims = [int(x) for x in a.dims.split(",") if x]
    rng = np.random.default_rng(0)
    rows, hits = [], {}

    def emit(tag, width, d1, hk):
        hit = (d1 < 25)
        hits[tag] = (width, hit, hk)
        rows.append((tag, width, np.median(d1), hit.mean(), hk.mean()))
        print("%-22s %7d %11.1f %12.1f%% %13.1f%%" % (
            tag, width, rows[-1][2], 100 * rows[-1][3], 100 * rows[-1][4]),
            flush=True)

    print("%-22s %7s %11s %12s %13s" %
          ("arm", "dims", "top1 km", "top1 <25km", "any32 <25km"))
    print("-" * 70)
    for scheme in a.schemes.split(","):
        for name, X in sources:
            P = pool(X, scheme)
            emit("{} {}".format(name, scheme), P.shape[1],
                 *score(P[:nq], P[nq:], lat, lon, qi, bi, same))
            for d in dims:
                if d >= P.shape[1]:
                    continue
                mu, W = pca_fit(P[nq:], d, rng)
                emit("{} {} pca{}".format(name, scheme, d), d,
                     *score((P[:nq] - mu) @ W, (P[nq:] - mu) @ W,
                            lat, lon, qi, bi, same))
            del P
        print()

    # Every arm against the representation that actually ships, on both
    # metrics and with the storage it would cost.  The shipping bank is the
    # only baseline that makes a width comparison mean anything: an arm is
    # interesting when it is no worse at fewer bytes, not when it wins at more.
    base = "crop3 concat"
    w0 = hits[base][0]
    print()
    print("paired against the shipping representation ({}, {}-d)".format(
        base, w0))
    print("%-22s %7s %20s %20s" % ("arm", "bytes", "top1 <25km", "any32 <25km"))
    print("-" * 72)
    for tag, (w, h1, hk) in hits.items():
        if tag == base:
            continue
        cell = []
        for a_, b_ in ((hits[base][1], h1), (hits[base][2], hk)):
            lo, hi = paired(a_, b_, rng)
            cell.append("%+5.2f [%+.2f,%+.2f]%s" % (
                100 * (b_.mean() - a_.mean()), lo, hi,
                " " if lo * hi > 0 else "~"))
        print("%-22s %6.2fx %20s %20s" % (tag, w / w0, cell[0], cell[1]))
    print("~ marks an interval spanning zero")

    # And the ladder's own question: tiles against crops, joining held fixed.
    print()
    print("{} - crop3, at identical width and identical pooling".format(
        a.tiles))
    print("%-22s %7s %20s %20s" % ("scheme", "dims", "top1 <25km",
                                   "any32 <25km"))
    print("-" * 72)
    for scheme in a.schemes.split(","):
        for suffix in ("", " pca{}".format(dims[0]) if dims else ""):
            lhs = "crop3 " + scheme + suffix
            rhs = a.tiles + " " + scheme + suffix
            if lhs not in hits or rhs not in hits:
                continue
            cell = []
            for i in (1, 2):
                lo, hi = paired(hits[lhs][i], hits[rhs][i], rng)
                cell.append("%+5.2f [%+.2f,%+.2f]%s" % (
                    100 * (hits[rhs][i].mean() - hits[lhs][i].mean()), lo, hi,
                    " " if lo * hi > 0 else "~"))
            print("%-22s %7d %20s %20s" % (
                scheme + suffix, hits[lhs][0], cell[0], cell[1]))


if __name__ == "__main__":
    main()
