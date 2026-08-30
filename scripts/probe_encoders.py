"""Do two encoders with different training objectives carry complementary
geolocation signal, or the same signal twice?

DINOv2 is self-supervised on visual structure and correspondence.  SigLIP is
language-supervised on web images, so it encodes nameable scene semantics and
has seen captions full of place names.  The hypothesis is that their biases
differ enough that the union beats either -- but "concatenation wins" is also
what you would see from simply doubling the feature budget, so the honest test
is a capacity-matched arm.

Four arms:

  dinov2      2304-d, the encoder the agent uses today
  siglip      2304-d, on its own
  concat      4608-d, both -- twice the probe parameters, so not yet evidence
  concat-pca  4608 -> 2304, matched to the single-encoder arms

Two readouts: kNN median km, which is the metric the project reports, and a
linear probe on s0/s1, which isolates whether the signal is linearly available
rather than merely present.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from baselines import baseline_knn, print_table, report
from dataset import GeoStepDataset
from probe_street import fit_probe, pca_to


def feats(ds, name):
    arr = np.load(config.STREET_CACHE / name, mmap_mode="r")
    return np.asarray(arr[ds.rows], dtype=np.float32)


def unit(x):
    """L2-normalise per encoder before concatenating.

    DINOv2 embeddings have mean L2 norm ~110 and SigLIP ~21.  Concatenating raw
    would let DINOv2 dominate every distance and every initial gradient, so the
    comparison would measure scale, not complementarity.
    """
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="embeddings_c3.f16.npy")
    ap.add_argument("--b", default="siglip_c3.f16.npy")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--knn-only", action="store_true")
    o = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = GeoStepDataset("train", street_file=o.a)
    va = GeoStepDataset("val", street_file=o.a)
    print("train {:,}   val {:,}".format(len(tr), len(va)))

    atr, ava = feats(tr, o.a), feats(va, o.a)
    btr, bva = feats(tr, o.b), feats(va, o.b)
    print("dinov2 {}  mean L2 {:.1f}".format(atr.shape, np.linalg.norm(atr, axis=1).mean()))
    print("siglip {}  mean L2 {:.1f}\n".format(btr.shape, np.linalg.norm(btr, axis=1).mean()))

    au, bu = unit(atr), unit(btr)
    avu, bvu = unit(ava), unit(bva)
    ctr = np.concatenate([au, bu], 1)
    cva = np.concatenate([avu, bvu], 1)
    ptr, pva = pca_to(ctr, cva, atr.shape[1])

    arms = {"dinov2": (au, avu), "siglip": (bu, bvu),
            "concat": (ctr, cva), "concat-pca": (ptr, pva)}

    rows = []
    for name, (xt, xv) in arms.items():
        r = baseline_knn(tr, va, np.ascontiguousarray(xt), np.ascontiguousarray(xv),
                         ks=(1,), dev=dev)
        for k in r:
            k["name"] = "kNN " + name
        rows.extend(r)
    print()
    print_table(rows)

    if o.knn_only:
        return

    tasks = {
        "s0": (tr.action[:, 0].astype(np.int64), va.action[:, 0].astype(np.int64), None),
        "s1": (tr.action[:, 1].astype(np.int64), va.action[:, 1].astype(np.int64),
               (tr.action[:, 0].astype(np.int64), va.action[:, 0].astype(np.int64))),
    }
    print("\n{:<6} {:<13} {:>8} {:>8} {:>9}".format("task", "features", "dims", "train", "val"))
    print("-" * 50)
    res = {}
    for task, (ytr, yva, side) in tasks.items():
        for name, (xt, xv) in arms.items():
            if side is not None:
                xt = np.concatenate([xt, np.eye(256, dtype=np.float32)[side[0]]], 1)
                xv = np.concatenate([xv, np.eye(256, dtype=np.float32)[side[1]]], 1)
            xt = np.ascontiguousarray(xt, dtype=np.float32)
            xv = np.ascontiguousarray(xv, dtype=np.float32)
            m, s = xt.mean(0, keepdims=True), xt.std(0, keepdims=True) + 1e-6
            trn, val, _ = fit_probe((xt - m) / s, ytr, (xv - m) / s, yva, 256, dev, o.epochs)
            res[(task, name)] = val
            print("{:<6} {:<13} {:>8,} {:>7.1%} {:>8.1%}".format(
                task, name, xt.shape[1], trn, val))

    print("\nverdict")
    for task in tasks:
        d = res[(task, "dinov2")]
        for name in ("siglip", "concat-pca"):
            print("  {} {:<11} {:+.2f} pp vs dinov2 ({:.1%} vs {:.1%})".format(
                task, name, 100 * (res[(task, name)] - d), res[(task, name)], d))


if __name__ == "__main__":
    main()
