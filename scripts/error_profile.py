"""Why the mean is 30x the median: decompose the error by first wrong step.

A wrong action at step t puts the answer in the wrong cell at zoom 4t, and the
cell widths are 2504 km, 156 km, 9.8 km, 611 m.  So the step at which a path
first leaves the truth sets the *scale* of the resulting error, and the mean is
dominated by whichever step contributes catastrophes rather than by the typical
case.  The median cannot see that at all, which is why both belong in the table.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import splits as sp
import tile_math as tm
from baselines import great_circle_km
from beam import TokenSource, search
from dataset import GeoStepDataset, street_table
from evaluate import load_model, street_file_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s10_n400k_e2")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--score-steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, d_street = load_model(a.tag, dev)
    sf = street_file_for(d_street)
    tbl = (street_table(config.STREET_CACHE / sf, dev)
           if ck.get("retr_mode") in ("pos", "dual") else None)
    ds = GeoStepDataset("test", street_file=sf,
                        split_mode=ck.get("split_mode", sp.PRIMARY),
                        knn_file=ck.get("knn_file"),
                        knn_k=ck.get("retr_k", 0) if ck.get("retr") else 0)
    src = TokenSource(tm.G)

    idx = np.arange(min(a.n, len(ds)))
    errs, firstwrong = [], []
    for lo in range(0, len(idx), a.batch):
        sel = idx[lo:lo + a.batch]
        street = torch.from_numpy(
            np.asarray(ds.street[ds.rows[sel]], dtype=np.float32)).to(dev)
        nbrs = None
        if getattr(ds, "knn_k", 0):
            r = ds.rows[sel]
            nbrs = [torch.from_numpy(ds.all_x16[ds.knn_idx[r]].astype(np.int64)).to(dev),
                    torch.from_numpy(ds.all_y16[ds.knn_idx[r]].astype(np.int64)).to(dev),
                    torch.from_numpy(ds.knn_sim[r]).to(dev)]
            if tbl is not None:
                i2 = torch.from_numpy(ds.knn_idx[r].astype(np.int64))
                nbrs.append(tbl[i2 if i2.device == tbl.device
                                else i2.to(tbl.device)].to(dev).float())
        res = search(model, street, src, dev, a.beam, max(4, a.beam),
                     greedy=(a.beam == 1), score_steps=a.score_steps, nbrs=nbrs)
        for i, r in zip(sel, res):
            b = r["best"]
            errs.append(great_circle_km(np.array(b["lat"]), np.array(b["lon"]),
                                        np.array(ds.lat[i]), np.array(ds.lon[i])))
            p = np.array(b["path"])
            bad = np.flatnonzero(p != ds.action[i])
            firstwrong.append(int(bad[0]) if len(bad) else 4)

    e = np.array(errs)
    fw = np.array(firstwrong)
    n = len(e)
    print("\n{}  {:,} test images, k={}, ranked s0-s{}".format(
        a.tag, n, a.beam, a.score_steps - 1))
    print("median {:.1f} km    mean {:.1f} km    ratio {:.0f}x".format(
        np.median(e), e.mean(), e.mean() / max(np.median(e), 1e-9)))

    print("\npercentiles")
    for p in (10, 25, 50, 75, 90, 95, 99, 100):
        print("  p{:<4} {:>9.1f} km".format(p, np.percentile(e, p)))

    print("\ncontribution to the mean, by first wrong step")
    print("  {:<22} {:>6} {:>7}  {:>10} {:>12} {:>8}".format(
        "first wrong step", "n", "share", "mean km", "of total mean", "cum"))
    cum = 0.0
    names = {0: "s0  (z4, 2504 km)", 1: "s1  (z8, 156 km)",
             2: "s2  (z12, 9.8 km)", 3: "s3  (z16, 611 m)",
             4: "none - path exact"}
    for k in (0, 1, 2, 3, 4):
        m = fw == k
        if not m.any():
            continue
        contrib = e[m].sum() / n
        cum += contrib
        print("  {:<22} {:>6,} {:>6.1f}% {:>10.1f} {:>11.1f} km {:>7.1f}%".format(
            names[k], int(m.sum()), 100 * m.mean(), e[m].mean(), contrib,
            100 * cum / e.mean()))

    print("\ntrimmed means")
    for q in (100, 99, 95, 90, 75):
        cut = np.percentile(e, q)
        print("  drop worst {:>2}%   mean {:>8.1f} km".format(
            100 - q, e[e <= cut].mean()))

    print("\ntail mass")
    for t in (1000, 2504, 5000, 10000):
        m = e > t
        print("  > {:>6,} km : {:>5.1f}% of images, {:>5.1f}% of the total error"
              .format(t, 100 * m.mean(), 100 * e[m].sum() / e.sum()))


if __name__ == "__main__":
    main()
