"""Compare whole encoders on the metric the bank is actually used for.

`probe_encoders.py` compares two caches by kNN median and a linear probe.  This
asks a narrower question that today's results made the important one: **at what
spatial scale does each encoder win, and does a stronger encoder absorb the
other's advantage or leave it standing?**

The reason it matters here is not curiosity about encoders.  DINOv2 and SigLIP
currently split by scale -- SigLIP alone beats DINOv2 alone by 5 pp at a 2500 km
threshold and loses by 2.2 pp at 25 km -- and that split is the entire
justification for carrying two encoders at twice the bank cost.  If a newer
encoder is simply better at fine scale, the pair still pays.  If it also picks
up the coarse advantage, the second encoder stops earning its keep and the right
move is one better encoder rather than two.  A single hit rate at 25 km cannot
tell those apart, which is the mistake this project already made once.

Every arm is reduced the same way: per-crop-token L2 normalisation, then mean
pooling to 768-d -- the representation validated end to end today.  That removes
any advantage from raw activation scale, which is what silently gave DINOv2 81%
of the cosine when the blocks were merely concatenated.

    python scripts/enc_ladder.py --arms dinov2=embeddings_c3.f16.npy,\\
        dinov3=dinov3_c3.f16.npy,siglip=siglip_c3.f16.npy --pairs dinov2+siglip,dinov3+siglip
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
from tile_match import dense_sim, topk_stats

D_ENC = 768
THRESH = (25, 200, 750, 2500)


def pooled(path, rows):
    """(n, crops*768) cache -> (n, 768), tokens unit-normalised then averaged."""
    X = np.asarray(np.load(config.STREET_CACHE / path, mmap_mode="r")[rows],
                   dtype=np.float32)
    return l2(X.reshape(len(rows), -1, D_ENC)).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="name=file,name=file — single-encoder caches")
    ap.add_argument("--pairs", default="",
                    help="a+b,c+d — joined arms, each side unit-normalised")
    ap.add_argument("--rows", default="tile6_rows.i64.npy",
                    help="row selection, shared with the other probes so the "
                         "numbers are comparable")
    ap.add_argument("--queries", type=int, default=3000)
    a = ap.parse_args()

    import pyarrow.parquet as pq

    sel = np.load(config.STREET_CACHE / a.rows)
    ds = pq.read_table(config.DATASET_PARQUET,
                       columns=["lat", "lon", "sequence"])
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    seq = np.asarray(ds["sequence"]).astype("U40")

    nq = a.queries
    qi, bi = sel[:nq], sel[nq:]
    same = seq[qi][:, None] == seq[bi][None, :]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("release {}  {:,} queries  {:,} bank".format(
        config.RELEASE, nq, len(bi)), flush=True)

    vecs, err = {}, {}
    for spec in a.arms.split(","):
        name, path = spec.split("=", 1)
        vecs[name] = pooled(path, sel)
        print("  {:<10} {:>5}-d  from {}".format(
            name, vecs[name].shape[1], path), flush=True)
    for spec in [p for p in a.pairs.split(",") if p]:
        x, y = spec.split("+")
        vecs[spec] = np.concatenate([l2(vecs[x]), l2(vecs[y])], axis=1)

    print("\ntop-1 error under increasing thresholds")
    print("%-16s %10s %s" % ("arm", "median km",
                             " ".join("%9s" % ("<%dkm" % t) for t in THRESH)))
    print("-" * 74)
    for name, V in vecs.items():
        d1, _ = topk_stats(dense_sim(V[:nq], V[nq:], dev),
                           lat, lon, qi, bi, same)
        err[name] = d1
        print("%-16s %10.1f %s" % (
            name, np.median(d1),
            " ".join("%8.1f%%" % (100 * (d1 < t).mean()) for t in THRESH)),
            flush=True)

    rng = np.random.default_rng(0)

    def contrast(lhs, rhs):
        if lhs not in err or rhs not in err:
            return
        cells = []
        for t in THRESH:
            lo, hi = paired(err[lhs] < t, err[rhs] < t, rng)
            d = 100 * ((err[rhs] < t).mean() - (err[lhs] < t).mean())
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                d, lo, hi, " " if lo * hi > 0 else "~"))
        print("%-30s %s" % ("{} - {}".format(rhs, lhs), " ".join(cells)))

    print("\npaired differences in percentage points, ~ spans zero")
    print("%-30s %s" % ("contrast", " ".join("%18s" % ("<%dkm" % t)
                                             for t in THRESH)))
    print("-" * 106)
    names = [s.split("=")[0] for s in a.arms.split(",")]
    base = names[0]
    for n in names[1:]:
        contrast(base, n)
    for spec in [p for p in a.pairs.split(",") if p]:
        contrast(spec.split("+")[0], spec)
    pairs = [p for p in a.pairs.split(",") if p]
    if len(pairs) == 2:
        contrast(pairs[0], pairs[1])

    # The decisive question: does the second encoder still rescue failures, and
    # does that change when the first encoder gets stronger?
    print("\nrescue rate at each threshold -- how often the partner lands where "
          "the lead encoder missed entirely")
    for spec in pairs:
        x, y = spec.split("+")
        cells = []
        for t in THRESH:
            hx, hy = err[x] < t, err[y] < t
            cells.append("%5.1f%%" % (100 * hy[~hx].mean()))
        print("  %-14s %s" % ("{} rescues {}".format(y, x), " ".join(cells)))


if __name__ == "__main__":
    main()
