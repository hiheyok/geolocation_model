"""Paired bootstrap over arms, so an ordering is a claim rather than a ranking.

The median of a heavy-tailed great-circle error carries a ~16 km 95% interval at
n ~ 5,000.  Most differences that have separated arms in this project are 5-10
km, which is inside that.  The hit rate `<25 km` carries ~0.6 pp at the same n
and usually can separate them.

Arms see the same images, so the difference is paired and the interval is much
tighter than two independent intervals would suggest -- which is the whole point
of doing it this way rather than eyeballing two medians.

Per-image errors are cached, so re-running a comparison costs nothing.

    python scripts/bootstrap.py --tags a,b,c --split test
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import names
import splits as sp
import tile_math as tm
from beam import TokenSource
from dataset import GeoStepDataset, street_table
from evaluate import check_split, evaluate, load_model, street_file_for

CACHE = ROOT / "runs" / "errs"


def errors_for(tag, split, n, beam_k, score_steps, dev, source):
    """Per-image great-circle error, cached by (tag, split, n, k, depth)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # "r" marks the seeded random sample; the old caches were the first n rows
    key = "{}_{}_{}r_k{}_d{}.npy".format(tag, split, n, beam_k, score_steps)
    p = CACHE / key
    if p.exists():
        return np.load(p)

    model, ck, d_street = load_model(tag, dev)
    mode = ck.get("split_mode", sp.PRIMARY)
    check_split(ck, mode, split)
    sf = street_file_for(ck, d_street)
    tbl = None
    if ck.get("retr_mode") in ("pos", "dual"):
        tbl = street_table(config.STREET_CACHE / sf, dev)
    ds = GeoStepDataset(split, street_file=sf, split_mode=mode,
                        knn_file=ck.get("knn_file"),
                        knn_k=ck.get("retr_k", 0) if ck.get("retr") else 0)
    m = evaluate(model, ds, source, dev, n, beam_k=beam_k,
                 top_m=max(4, beam_k), greedy=(beam_k == 1),
                 score_steps=score_steps, street_gpu=tbl)
    np.save(p, m["err"])
    del model, tbl
    torch.cuda.empty_cache()
    return m["err"]


def paired(a, b, reps, rng, thresh=25.0):
    """95% CI on (a - b) for the median and for the <thresh km hit rate."""
    n = len(a)
    dm = np.empty(reps)
    dh = np.empty(reps)
    for i in range(reps):
        s = rng.integers(0, n, n)
        dm[i] = np.median(a[s]) - np.median(b[s])
        dh[i] = (a[s] < thresh).mean() - (b[s] < thresh).mean()
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(dm), q(dh)


def verdict(lo, hi):
    return "separated" if (lo > 0) == (hi > 0) else "inside noise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma separated checkpoint tags")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--score-steps", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    source = TokenSource(tm.G)
    rng = np.random.default_rng(config.SPLIT_SEED)

    errs = {}
    for t in tags:
        errs[t] = errors_for(t, a.split, a.n, a.beam, a.score_steps, dev, source)
        print("{:<24} n={:,}  median {:7.1f} km  mean {:8.1f}  <25km {:5.1%}"
              .format(t, len(errs[t]), float(np.median(errs[t])),
                      float(errs[t].mean()), float((errs[t] < 25).mean())),
              flush=True)

    lens = {len(v) for v in errs.values()}
    if len(lens) != 1:
        raise SystemExit("arms cover different image counts {} -- a paired test "
                         "needs the same images in the same order".format(lens))

    L = ["", "paired bootstrap, {} split, {:,} images, {:,} resamples, k={}, "
         "ranked on s0-s{}".format(a.split, len(errs[tags[0]]), a.reps,
                                   a.beam, a.score_steps - 1), ""]
    # A legend, because the tags alone do not say what differs between arms:
    # d1536-b265-e6 and d768-b265-e4 differ in width AND bank AND epochs, and
    # nothing in either name says so. names.describe reads from an explicit
    # table, so an arm it cannot name honestly is simply left out.
    known = [(t, names.describe(t)) for t in tags if names.describe(t)]
    if known:
        L.append("| arm | what it is |")
        L.append("|---|---|")
        for t, d in known:
            L.append("| `{}` | {} |".format(t, d))
        L.append("")
    L.append("| contrast | median diff, 95% CI | | <25km diff, 95% CI | |")
    L.append("|---|---|---|---|---|")
    for x, y in itertools.combinations(tags, 2):
        (ml, mh), (hl, hh) = paired(errs[x], errs[y], a.reps, rng)
        L.append("| {} vs {} | [{:+.1f}, {:+.1f}] km | {} | [{:+.2f}, {:+.2f}] pp | {} |"
                 .format(x, y, ml, mh, verdict(ml, mh),
                         100 * hl, 100 * hh, verdict(hl, hh)))
    L.append("")
    L.append("A positive median difference means the first arm is worse "
             "(more km); a positive <25km difference means it is better.")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
