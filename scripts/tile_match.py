"""Upper-bound what any pooling could do, instead of asserting it.

`tile_pool.py` compared tiles against crops under mean pooling and found no
gain, and I read that as "attention cannot add signal that is not there".  That
claim was not earned.  Mean pooling is *an* order-free match, but a poor one: a
pooled cosine is

    <mean_i q_i, mean_j b_j>  ~  sum_ij <q_i, b_j>

which adds every cross pair at equal weight, including a query sky tile against
a bank road tile.  Attention would assign selectively.  Showing that mean
pooling does not help says nothing about whether selective matching would.

So bound it.  Two matchers that drop the index constraint entirely:

  chamfer   mean_i max_j <q_i, b_j>   -- every query tile finds its own best
            partner anywhere in the bank image, with no index constraint and no
            interference from mismatched pairs.  This is the soft assignment
            attention pooling approximates, computed exactly.
  maxmax    max_ij <q_i, b_j>         -- one good tile pair carries the image.

And one that peeks at the label, to bound tile *selection*:

  oracle    each query tile retrieves independently; the query counts as a hit
            if ANY of its tiles lands within 25 km.  No learned weighting of
            query tiles can beat this, because it is the best case over all of
            them.

None of the three is shippable -- they need per-tile storage and 36x the
compare cost, where the bank is the memory-bound part of the system.  That is
the point: they are references, not products.

Be careful what they bound.  Chamfer says how much the *index constraint* was
costing, given these tokens and a fixed cosine.  It is NOT a ceiling on a
learned metric over the same tokens, which could weight the dimensions that
carry geography and beat a fixed heuristic, and it is not a ceiling on
contextualisation, which changes the tokens before matching.  What it does
settle is whether rearranging matches is enough on its own.

Tokens are per-block L2-normalised throughout, which also disposes of the
baseline question: normalising the DINOv2 and SigLIP halves of a token
separately makes `dual_c3` and `dual_bal` bit-identical inputs, so no arm here
can be flattered by the 81/19 imbalance.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from tile_pool import great_circle, l2, paired

K = 32
D_ENC = 768


def oracle_at_k(errs, rng, n_sub=256):
    """Per-query chance that a size-k subset of tiles contains a 25 km hit.

    The plain oracle is not comparable across sources: it is 6 draws for tiles
    against 3 for crops, and more independent retrievals win by arithmetic even
    when each is no better.  Averaging over random subsets of a fixed size k
    removes that, and the curve over k separates "each tile is good" from
    "the tiles are diverse".
    """
    nq, n = errs.shape
    hit = errs < 25
    out = {}
    for k in range(1, n + 1):
        if k == n:
            out[k] = hit.any(1).astype(np.float64)
        else:
            acc = np.zeros(nq)
            for _ in range(n_sub):
                acc += hit[:, rng.choice(n, k, replace=False)].any(1)
            out[k] = acc / n_sub
    return out


def bnorm(T):
    """(n, tokens, 1536) -> per-block unit tokens, float32."""
    X = np.asarray(T, dtype=np.float32)
    return np.concatenate([l2(X[:, :, :D_ENC]), l2(X[:, :, D_ENC:])], axis=2)


def topk_stats(S, lat, lon, qi, bi, same):
    """(nq, nbank) similarity -> per-query top-1 error and any-of-K hit."""
    S = S.copy()
    S[same] = -2.0
    k = min(K, S.shape[1] - 1)
    j = np.argpartition(-S, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, j, 1), axis=1)
    got = np.take_along_axis(j, o, 1)
    d1 = great_circle(lat[qi], lon[qi], lat[bi[got[:, 0]]], lon[bi[got[:, 0]]])
    dk = great_circle(np.repeat(lat[qi], k), np.repeat(lon[qi], k),
                      lat[bi[got.ravel()]], lon[bi[got.ravel()]]).reshape(-1, k)
    return d1, (dk < 25).any(1)


def dense_sim(Q, B, dev, block=8192):
    """Cosine of two pooled matrices, blocked over the bank."""
    q = torch.from_numpy(l2(Q)).to(dev, torch.float16)
    out = np.empty((Q.shape[0], B.shape[0]), dtype=np.float32)
    for s in range(0, B.shape[0], block):
        b = torch.from_numpy(l2(B[s:s + block])).to(dev, torch.float16)
        out[:, s:s + block] = (q @ b.T).float().cpu().numpy()
    return out


def set_sim(Qt, Bt, dev, block=3072):
    """Chamfer, max-max, and per-query-tile best bank row.

    Qt (nq, tq, d) and Bt (nb, tb, d) hold unit-ish tokens.  For a bank block
    the full (nq*tq, nb*tb) table is formed once and reduced three ways, which
    keeps the arithmetic on the GPU and the peak at a few hundred MB.
    """
    nq, tq, d = Qt.shape
    nb, tb, _ = Bt.shape
    q = torch.from_numpy(l2(Qt).reshape(nq * tq, d)).to(dev, torch.float16)
    cham = np.empty((nq, nb), dtype=np.float32)
    mmax = np.empty((nq, nb), dtype=np.float32)
    best_v = torch.full((nq, tq), -2.0, device=dev)
    best_i = torch.zeros((nq, tq), dtype=torch.long, device=dev)
    for s in range(0, nb, block):
        e = min(s + block, nb)
        b = torch.from_numpy(l2(Bt[s:e]).reshape((e - s) * tb, d))
        S = (q @ b.to(dev, torch.float16).T).float().view(nq, tq, e - s, tb)
        per_tile = S.amax(3)                       # (nq, tq, blk)
        cham[:, s:e] = per_tile.mean(1).cpu().numpy()
        mmax[:, s:e] = per_tile.amax(1).cpu().numpy()
        v, i = per_tile.max(2)                     # best bank row per query tile
        upd = v > best_v
        best_v = torch.where(upd, v, best_v)
        best_i = torch.where(upd, i + s, best_i)
        del S, per_tile
    return cham, mmax, best_i.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="tile6")
    ap.add_argument("--crop3", default="dual_c3.f16.npy")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--block", type=int, default=3072)
    a = ap.parse_args()

    import pyarrow.parquet as pq

    sel = np.load(config.STREET_CACHE / (a.tiles + "_rows.i64.npy"))
    done = np.load(config.STREET_CACHE / (a.tiles + "_done.u8.npy"))
    T = np.load(config.STREET_CACHE / (a.tiles + ".f16.npy"), mmap_mode="r")
    keep = np.flatnonzero(done == 1)
    sel = sel[keep]

    ds = pq.read_table(config.DATASET_PARQUET,
                       columns=["lat", "lon", "sequence"])
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    seq = np.asarray(ds["sequence"]).astype("U40")

    nq = a.queries
    qi, bi = sel[:nq], sel[nq:]
    same = seq[qi][:, None] == seq[bi][None, :]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("release {}  {:,} queries  {:,} bank  device {}".format(
        config.RELEASE, nq, len(bi), dev), flush=True)

    C = np.asarray(np.load(config.STREET_CACHE / a.crop3,
                           mmap_mode="r")[sel], dtype=np.float32)
    h = C.shape[1] // 2
    nc = h // D_ENC
    C = np.concatenate([C[:, :h].reshape(-1, nc, D_ENC),
                        C[:, h:].reshape(-1, nc, D_ENC)], axis=2)
    sources = [("crop3", bnorm(C)), (a.tiles, bnorm(T[keep]))]
    del C

    hits, order, curves, per_tile_err = {}, [], {}, {}

    def emit(tag, d1, hk):
        hits[tag] = ((d1 < 25), hk)
        order.append(tag)
        print("%-24s %11.1f %12.1f%% %13.1f%%" % (
            tag, np.median(d1), 100 * hits[tag][0].mean(), 100 * hk.mean()),
            flush=True)

    print("\n%-24s %11s %12s %13s" %
          ("arm", "top1 km", "top1 <25km", "any32 <25km"))
    print("-" * 64)
    for name, X in sources:
        n_tok = X.shape[1]
        flat = X.reshape(X.shape[0], -1)
        emit("{} concat".format(name),
             *topk_stats(dense_sim(flat[:nq], flat[nq:], dev),
                         lat, lon, qi, bi, same))
        pooled = X.mean(1)
        emit("{} mean".format(name),
             *topk_stats(dense_sim(pooled[:nq], pooled[nq:], dev),
                         lat, lon, qi, bi, same))
        cham, mmax, best = set_sim(X[:nq], X[nq:], dev, a.block)
        emit("{} chamfer".format(name), *topk_stats(cham, lat, lon, qi, bi, same))
        emit("{} maxmax".format(name), *topk_stats(mmax, lat, lon, qi, bi, same))

        # Oracle over query tiles: each tile retrieves on its own, and the
        # query counts as a hit if any of them lands. Uses the label, so it is
        # a ceiling on tile weighting, not a method.
        errs = np.stack([great_circle(lat[qi], lon[qi],
                                      lat[bi[best[:, t]]], lon[bi[best[:, t]]])
                         for t in range(n_tok)], axis=1)
        hit = (errs < 25).any(1)
        hits["{} oracle".format(name)] = (hit, hit)
        order.append("{} oracle".format(name))
        print("%-24s %11.1f %12.1f%% %13s" % (
            "{} oracle({})".format(name, n_tok), np.median(errs.min(1)),
            100 * hit.mean(), "--"), flush=True)
        ok = oracle_at_k(errs, np.random.default_rng(1))
        curves[name] = ok
        per_tile_err[name] = errs
        print("%-24s %s" % (
            "  oracle@k, k=1..{}".format(n_tok),
            "  ".join("%.1f%%" % (100 * ok[k].mean()) for k in sorted(ok))),
            flush=True)
        print("%-24s %s" % (
            "  each tile alone",
            "  ".join("%.1f%%" % (100 * (errs[:, t] < 25).mean())
                      for t in range(n_tok))), flush=True)
        del flat, pooled, cham, mmax
        print()

    rng = np.random.default_rng(0)
    print("{} - crop3, paired, percentage points".format(a.tiles))
    print("%-24s %22s %22s" % ("matcher", "top1 <25km", "any32 <25km"))
    print("-" * 70)
    for m in ("concat", "mean", "chamfer", "maxmax", "oracle"):
        lhs, rhs = "crop3 " + m, a.tiles + " " + m
        cell = []
        for i in (0, 1):
            lo, hi = paired(hits[lhs][i], hits[rhs][i], rng)
            cell.append("%+5.2f [%+.2f,%+.2f]%s" % (
                100 * (hits[rhs][i].mean() - hits[lhs][i].mean()), lo, hi,
                " " if lo * hi > 0 else "~"))
        print("%-24s %22s %22s" % (m, cell[0], cell[1]))
    print("~ marks an interval spanning zero")

    # The oracle at matched draw count: 6 tiles against 3 crops is not a fair
    # oracle, so compare both at k=3.
    kk = min(min(len(c) for c in curves.values()), 3)
    la, lb = curves["crop3"][kk], curves[a.tiles][kk]
    lo, hi = paired(la, lb, rng)
    print()
    print("oracle at matched k={}: crop3 {:.1f}%  {} {:.1f}%   {:+.2f} pp "
          "[{:+.2f}, {:+.2f}]{}".format(
              kk, 100 * la.mean(), a.tiles, 100 * lb.mean(),
              100 * (lb.mean() - la.mean()), lo, hi,
              "" if lo * hi > 0 else "  (spans zero)"))

    # Do tiles carry information the crops do not?  "More vectors is more
    # information" only matters if the extra information is *different*.  If
    # tiles rescue queries where every crop missed, fusing both is worth
    # something even though tiles lose on their own.
    ec, et = per_tile_err["crop3"], per_tile_err[a.tiles]
    hc, ht = (ec < 25).any(1), (et < 25).any(1)
    print()
    print("complementarity, over {:,} queries".format(len(hc)))
    print("  a crop lands                            {:.1f}%".format(100 * hc.mean()))
    print("  a tile lands                            {:.1f}%".format(100 * ht.mean()))
    print("  either lands                            {:.1f}%".format(100 * (hc | ht).mean()))
    print("  a tile lands where every crop missed    {:.1f}%  ({:,} of {:,})".format(
        100 * ht[~hc].mean(), int(ht[~hc].sum()), int((~hc).sum())))
    print("  a crop lands where every tile missed    {:.1f}%  ({:,} of {:,})".format(
        100 * hc[~ht].mean(), int(hc[~ht].sum()), int((~ht).sum())))
    lo, hi = paired(hc, hc | ht, rng)
    print("  union - crops alone                     {:+.2f} pp [{:+.2f}, {:+.2f}]{}".format(
        100 * ((hc | ht).mean() - hc.mean()), lo, hi,
        "" if lo * hi > 0 else "  (spans zero)"))
    print()

    print("\nheadroom within a source: matcher - its own concat")
    print("%-24s %22s %22s" % ("arm", "top1 <25km", "any32 <25km"))
    print("-" * 70)
    for name, _ in sources:
        for m in ("mean", "chamfer", "maxmax"):
            base, tag = name + " concat", "{} {}".format(name, m)
            cell = []
            for i in (0, 1):
                lo, hi = paired(hits[base][i], hits[tag][i], rng)
                cell.append("%+5.2f [%+.2f,%+.2f]%s" % (
                    100 * (hits[tag][i].mean() - hits[base][i].mean()), lo, hi,
                    " " if lo * hi > 0 else "~"))
            print("%-24s %22s %22s" % (tag, cell[0], cell[1]))


if __name__ == "__main__":
    main()
