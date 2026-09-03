"""Is there headroom for cross-encoder fusion, before anyone builds it?

The proposal is to fuse DINOv2 and SigLIP with cross-attention -- DINOv2 tokens
as queries over SigLIP tokens -- instead of the fixed position-wise
concatenation that ships today, where tile i's DINOv2 vector is simply glued to
tile i's SigLIP vector and the two never interact.

That is worth building only if the encoders are *complementary*.  A learned
content-dependent blend can only beat a fixed global blend to the extent the
two encoders succeed on different images.  If SigLIP retrieves essentially the
same queries DINOv2 does, cross-attention has nothing to route and the ceiling
is whatever a scalar blend already gets -- which this project has measured, at
+3.0 pp for equal-norm against the accidental 81/19 mix.

So measure the ceiling first:

  solo        each encoder alone, per-token L2-normalised then mean-pooled
  either      oracle over the two, per query -- an upper bound on *any* rule
              that picks between them, learned or not, including cross-attention
  rescue      how often one lands where the other missed entirely

The oracle is over 2 draws against each solo arm's 1, so it is inflated the same
way the tile oracle was; what it bounds is the *shape* of the headroom, and the
rescue rates say whether the two encoders fail on the same images or different
ones.  Redundant encoders make cross-attention pointless however it is wired.

Run for both token sources, because the answer could differ: tiles are weaker
individually and may lean on the two encoders differently than crops do.
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
from tile_pool import l2, paired
from tile_match import D_ENC, dense_sim, topk_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="tile6")
    ap.add_argument("--crop3", default="dual_c3.f16.npy")
    ap.add_argument("--queries", type=int, default=3000)
    a = ap.parse_args()

    import pyarrow.parquet as pq

    sel = np.load(config.STREET_CACHE / (a.tiles + "_rows.i64.npy"))
    done = np.load(config.STREET_CACHE / (a.tiles + "_done.u8.npy"))
    T = np.load(config.STREET_CACHE / (a.tiles + ".f16.npy"), mmap_mode="r")
    sel = sel[np.flatnonzero(done == 1)]

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

    C = np.asarray(np.load(config.STREET_CACHE / a.crop3,
                           mmap_mode="r")[sel], dtype=np.float32)
    h = C.shape[1] // 2
    nc = h // D_ENC
    C = np.concatenate([C[:, :h].reshape(-1, nc, D_ENC),
                        C[:, h:].reshape(-1, nc, D_ENC)], axis=2)
    keep = np.flatnonzero(done == 1)
    sources = [("crop3", C),
               (a.tiles, np.asarray(T[keep], dtype=np.float32))]
    del C

    rng = np.random.default_rng(0)
    print("\n%-22s %11s %12s %13s" %
          ("arm", "top1 km", "top1 <25km", "any32 <25km"))
    print("-" * 62)
    for name, X in sources:
        parts = {"dino": l2(X[:, :, :D_ENC]),
                 "siglip": l2(X[:, :, D_ENC:])}
        parts["both"] = np.concatenate([parts["dino"], parts["siglip"]], axis=2)
        res = {}
        for tag, P in parts.items():
            pooled = P.mean(1)
            d1, hk = topk_stats(dense_sim(pooled[:nq], pooled[nq:], dev),
                                lat, lon, qi, bi, same)
            res[tag] = (d1 < 25, hk, d1)
            print("%-22s %11.1f %12.1f%% %13.1f%%" % (
                "{} {}".format(name, tag), np.median(d1),
                100 * res[tag][0].mean(), 100 * hk.mean()), flush=True)

        # Where does the second encoder actually pay?  A 25 km hit rate is
        # blind to an encoder that fixes continent-scale errors, and the median
        # column says that is most of what SigLIP does. Walk the thresholds.
        print("  %-18s %s" % ("threshold ladder",
                              "  ".join("%8s" % ("<%dkm" % t)
                                        for t in (1, 25, 200, 750, 2500))))
        for tag in ("dino", "siglip", "both"):
            print("  %-18s %s" % (tag, "  ".join(
                "%7.1f%%" % (100 * (res[tag][2] < t).mean())
                for t in (1, 25, 200, 750, 2500))))
        for t in (200, 750, 2500):
            lo, hi = paired(res["dino"][2] < t, res["both"][2] < t, rng)
            print("  both - dino at <%-5d km          %+6.2f pp [%+.2f, %+.2f]%s"
                  % (t, 100 * ((res["both"][2] < t).mean()
                               - (res["dino"][2] < t).mean()), lo, hi,
                     "" if lo * hi > 0 else "  (spans zero)"))

        hd, hs = res["dino"][0], res["siglip"][0]
        hb = res["both"][0]
        print("  %-20s %11s %12.1f%% %13s" % (
            "either (oracle)", "--", 100 * (hd | hs).mean(), "--"))
        print("  siglip lands where dino missed   {:.1f}%  ({:,} of {:,})".format(
            100 * hs[~hd].mean(), int(hs[~hd].sum()), int((~hd).sum())))
        print("  dino lands where siglip missed   {:.1f}%  ({:,} of {:,})".format(
            100 * hd[~hs].mean(), int(hd[~hs].sum()), int((~hs).sum())))
        lo, hi = paired(hb, hd | hs, rng)
        print("  oracle - fixed concat            {:+.2f} pp [{:+.2f}, {:+.2f}]{}"
              .format(100 * ((hd | hs).mean() - hb.mean()), lo, hi,
                      "" if lo * hi > 0 else "  (spans zero)"))
        lo, hi = paired(hd, hb, rng)
        print("  fixed concat - dino alone        {:+.2f} pp [{:+.2f}, {:+.2f}]{}"
              .format(100 * (hb.mean() - hd.mean()), lo, hi,
                      "" if lo * hi > 0 else "  (spans zero)"))
        print()


if __name__ == "__main__":
    main()
