"""Gate 2: does a spatial street representation carry transferable evidence
that the pooled CLS vector destroyed?

The question is not whether patch tokens hold more information -- of course they
do -- but whether the extra information *generalises*.  A linear probe answers
that for a fraction of the cost of building the cross-attention that would read
them, and it is the same discipline that caught the centre-crop bug: test the
input before building the architecture on top of it.

Three arms, all fit on train and scored on val:

  pooled      the 2304-d CLS concatenation the agent uses today
  spatial     a 2x2 pooled grid per crop, 12 tokens x 768 = 9216-d
  spatial-pca spatial reduced to 2304-d, so the probe has exactly the
              parameter count of the pooled arm

The third arm is the one that matters.  Without it, "spatial wins" is
indistinguishable from "four times as many probe parameters wins".

Two tasks:

  s0   which of 256 z4 world cells -- a pure street->region problem
  s1   which of 256 z8 sub-cells, given the true z4 cell as a one-hot side
       input, since without it the target is ambiguous.  Both arms carry the
       same side input, so the comparison stays fair.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from dataset import GeoStepDataset


def load(split, street_file):
    ds = GeoStepDataset(split, street_file=street_file)
    return ds


def features(ds, path, spatial, chunk=4096):
    """Pool in chunks: materialising all 48 tokens as float32 is ~5.9 GB."""
    arr = np.load(config.STREET_CACHE / path, mmap_mode="r")
    rows = ds.rows
    if not spatial:
        return np.asarray(arr[rows], dtype=np.float32)
    t, d = arr.shape[1], arr.shape[2]
    per = t // 3
    g = int(round(per ** 0.5))
    h = g // 2
    out = np.empty((len(rows), 3 * 4 * d), dtype=np.float32)
    for lo in range(0, len(rows), chunk):
        sel = rows[lo:lo + chunk]
        c = np.asarray(arr[sel], dtype=np.float32).reshape(-1, 3, g, g, d)
        # 4x4 per crop -> 2x2 per crop
        c = c.reshape(-1, 3, 2, h, 2, h, d).mean(axis=(3, 5))
        out[lo:lo + len(sel)] = c.reshape(len(sel), -1)
    return out


def fit_probe(xtr, ytr, xva, yva, n_cls, dev, epochs=60, lr=3e-3, wd=1e-4, bs=1024):
    """Multinomial logistic regression, full-batch AdamW on the GPU."""
    d = xtr.shape[1]
    w = torch.nn.Linear(d, n_cls).to(dev)
    opt = torch.optim.AdamW(w.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # features stay on the host and stream: the spatial arm is 1.5 GB and the
    # 3070 has 8 GB with a training job liable to want it too
    xtr_t = torch.from_numpy(xtr)
    ytr_t = torch.from_numpy(ytr)
    xva_t = torch.from_numpy(xva)
    yva_t = torch.from_numpy(yva)
    ce = torch.nn.CrossEntropyLoss()

    def acc(x, y):
        hit = 0
        with torch.no_grad():
            for lo in range(0, len(x), 8192):
                xb = x[lo:lo + 8192].to(dev, non_blocking=True)
                hit += (w(xb).argmax(-1).cpu() == y[lo:lo + 8192]).sum().item()
        return hit / len(x)

    best = 0.0
    for ep in range(epochs):
        w.train()
        perm = torch.randperm(len(xtr_t))
        for lo in range(0, len(perm), bs):
            sel = perm[lo:lo + bs]
            xb = xtr_t[sel].to(dev, non_blocking=True)
            yb = ytr_t[sel].to(dev, non_blocking=True)
            loss = ce(w(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        w.eval()
        best = max(best, acc(xva_t, yva_t))
    return acc(xtr_t, ytr_t), acc(xva_t, yva_t), best


def pca_to(xtr, xva, k, seed=0):
    """Fit on train only, so val never informs the projection."""
    mu = xtr.mean(0, keepdims=True)
    a = xtr - mu
    # randomised range finder: 9216 -> 2304 exactly is a big SVD otherwise
    rng = np.random.default_rng(seed)
    om = rng.standard_normal((a.shape[1], k + 32)).astype(np.float32)
    q, _ = np.linalg.qr(a @ om)
    b = q.T @ a
    _, _, vt = np.linalg.svd(b, full_matrices=False)
    p = vt[:k].T
    return (xtr - mu) @ p, (xva - mu) @ p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled", default="embeddings_c3.f16.npy")
    ap.add_argument("--spatial", default="patch4_c3.f16.npy")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = load("train", a.pooled)
    va = load("val", a.pooled)
    print("train {:,}   val {:,}\n".format(len(tr), len(va)))

    arms = {}
    arms["pooled"] = (features(tr, a.pooled, False), features(va, a.pooled, False))
    sp_tr = features(tr, a.spatial, True)
    sp_va = features(va, a.spatial, True)
    arms["spatial"] = (sp_tr, sp_va)
    k = arms["pooled"][0].shape[1]
    t0 = time.time()
    arms["spatial-pca"] = pca_to(sp_tr, sp_va, k)
    print("pca {} -> {} in {:.0f}s".format(sp_tr.shape[1], k, time.time() - t0))
    for name, (xt, xv) in arms.items():
        print("  {:<12} train {}  val {}".format(name, xt.shape, xv.shape))

    # targets
    tasks = {}
    tasks["s0"] = (tr.action[:, 0].astype(np.int64), va.action[:, 0].astype(np.int64), None)
    tasks["s1"] = (tr.action[:, 1].astype(np.int64), va.action[:, 1].astype(np.int64),
                   (tr.action[:, 0].astype(np.int64), va.action[:, 0].astype(np.int64)))

    print("\n{:<6} {:<13} {:>8} {:>8} {:>9}".format(
        "task", "features", "dims", "train", "val"))
    print("-" * 50)
    results = {}
    for task, (ytr, yva, side) in tasks.items():
        for name, (xt, xv) in arms.items():
            if side is not None:
                oh_t = np.eye(256, dtype=np.float32)[side[0]]
                oh_v = np.eye(256, dtype=np.float32)[side[1]]
                xt = np.concatenate([xt, oh_t], 1)
                xv = np.concatenate([xv, oh_v], 1)
            xt = np.ascontiguousarray(xt, dtype=np.float32)
            xv = np.ascontiguousarray(xv, dtype=np.float32)
            m = xt.mean(0, keepdims=True)
            s = xt.std(0, keepdims=True) + 1e-6
            xt, xv = (xt - m) / s, (xv - m) / s
            trn, val, best = fit_probe(xt, ytr, xv, yva, 256, dev, a.epochs)
            results[(task, name)] = (trn, val, best)
            print("{:<6} {:<13} {:>8,} {:>7.1%} {:>8.1%}".format(
                task, name, xt.shape[1], trn, val))

    print("\nverdict")
    for task in tasks:
        p = results[(task, "pooled")][1]
        q = results[(task, "spatial-pca")][1]
        print("  {}: spatial-pca {:+.2f} pp over pooled at matched capacity "
              "({:.1%} vs {:.1%})".format(task, 100 * (q - p), q, p))


if __name__ == "__main__":
    main()
