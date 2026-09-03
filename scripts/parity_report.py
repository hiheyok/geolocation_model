"""What the shipping-parity re-runs did to the external numbers.

Every KartaView figure reported before 2026-09-03 was measured with
eval_highres searching the whole embedding file rather than the rows the
checkpoint's kNN was built over, and with a hardcoded K=32 where these arms
train at retr_k=16. Both were fixed; the hrfix-* stages re-measure the same
5,000 images at parity.

The point of this script is the second half. Directions should survive a
uniform change, because every arm was measured the same wrong way -- but
neighbour count is exactly what --retr-drop manipulates, so the p=0.7 optimum
has to be reconfirmed at K=16 before it can be called tuned. That is the one
conclusion the old measurement could not support.

    python scripts/parity_report.py

Arms are listed explicitly. The old and new exports do not share a naming
convention (hr_d1536_drop30_n5k vs hrfix_d1536drop30), and deriving one name
from the other is the exact move that has produced most of this project's
silent failures.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# label, old export, parity export
ARMS = [
    ("d768-b265-e6",         "hr_b265_n5k.npz",        "hrfix_b265.npz"),
    ("d768-b350-e6",         "hr_b350_n5k.npz",        "hrfix_b350.npz"),
    ("  p=0.1",              "hr_drop10_n5k.npz",      "hrfix_drop10.npz"),
    ("  p=0.3",              "hr_drop30_n5k.npz",      "hrfix_drop30.npz"),
    ("  p=0.5",              "hr_drop50_n5k.npz",      "hrfix_drop50.npz"),
    ("  p=0.7",              "hr_drop70_n5k.npz",      "hrfix_drop70.npz"),
    ("  p=0.9",              "hr_drop90_n5k.npz",      "hrfix_drop90.npz"),
    ("d768-b350-e6-sub2",    "hr_sub2_n5k.npz",        "hrfix_sub2.npz"),
    ("d1536-b350-e6",        "hr_d1536_n5k.npz",       "hrfix_d1536.npz"),
    ("d1536-drop30",         "hr_d1536_drop30_n5k.npz", "hrfix_d1536drop30.npz"),
]

# The p-curve, in order, as it will be read.  p=0 is the plain b350 arm.
PCURVE = [("0.0", "hrfix_b350.npz"), ("0.1", "hrfix_drop10.npz"),
          ("0.3", "hrfix_drop30.npz"), ("0.5", "hrfix_drop50.npz"),
          ("0.7", "hrfix_drop70.npz"), ("0.9", "hrfix_drop90.npz")]


def load(name):
    p = RUNS / name
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["err"], float), np.asarray(z["image_id"])


def align(a, b):
    """Paired tests need the same images in the same order, and say so if not."""
    (ea, ia), (eb, ib) = a, b
    if len(ea) != len(eb):
        return None
    if not (ia == ib).all():
        oa, ob = np.argsort(ia), np.argsort(ib)
        if not (ia[oa] == ib[ob]).all():
            return None
        ea, eb = ea[oa], eb[ob]
    return ea, eb


def paired(ea, eb, thresh=25.0, reps=5000, seed=0):
    """(hit-rate difference, its CI, median difference, its CI); b minus a."""
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ea), (reps, len(ea)))
    d = (eb < thresh).astype(float) - (ea < thresh)
    bs = d[ix].mean(1) * 100
    md = np.median(eb[ix], 1) - np.median(ea[ix], 1)
    return (100 * d.mean(), np.percentile(bs, [2.5, 97.5]),
            np.median(eb) - np.median(ea), np.percentile(md, [2.5, 97.5]))


def verdict(lo, hi):
    # A zero-touching interval is not separated.  bootstrap.verdict called
    # those "separated" once; this keeps the two files saying the same thing.
    return "separated" if lo * hi > 0 else "noise"


def main():
    print("\nShipping parity: whole embedding file / K=32  ->  checkpoint's own "
          "bank / its retr_k\n")
    print("  {:<22} {:>17}  {:>19}  {}".format(
        "arm", "<25 km", "median km", "paired shift, <25 km"))
    have = {}
    for label, old_f, new_f in ARMS:
        o, n = load(old_f), load(new_f)
        if n is None:
            print("  {:<22} {:>17}".format(label, "not yet run"))
            continue
        have[new_f] = n
        if o is None:
            print("  {:<22} {:>16.1%}  {:>18.1f}   (no prior export)".format(
                label, (n[0] < 25).mean(), np.median(n[0])))
            continue
        pair = align(o, n)
        if pair is None:
            print("  {:<22} {:>17}".format(label, "image sets differ"))
            continue
        ea, eb = pair
        dh, (lo, hi), dm, _ = paired(ea, eb)
        print("  {:<22} {:>6.1%} -> {:>6.1%}  {:>8.1f} -> {:>7.1f}   "
              "{:+.2f} pp [{:+.2f}, {:+.2f}] {}".format(
                  label, (ea < 25).mean(), (eb < 25).mean(),
                  np.median(ea), np.median(eb), dh, lo, hi, verdict(lo, hi)))

    ready = [(p, f) for p, f in PCURVE if f in have]
    if len(ready) < 2:
        print("\nThe p-curve needs at least two parity runs; {} so far."
              .format(len(ready)))
        return
    print("\n\nThe --retr-drop curve at parity, each p against p=0\n")
    print("  {:>4} {:>8} {:>8} {:>10}   {}".format(
        "p", "<1 km", "<25 km", "median", "vs p=0 on <25 km"))
    base = have.get("hrfix_b350.npz")
    for p, f in ready:
        e = have[f][0]
        row = "  {:>4} {:>7.1%} {:>8.1%} {:>9.1f}".format(
            p, (e < 1).mean(), (e < 25).mean(), np.median(e))
        if base is not None and f != "hrfix_b350.npz":
            pair = align(base, have[f])
            if pair is not None:
                dh, (lo, hi), _, _ = paired(*pair)
                row += "   {:+.2f} pp [{:+.2f}, {:+.2f}] {}".format(
                    dh, lo, hi, verdict(lo, hi))
        print(row)

    # Whether the confound was correlated with the treatment. Every arm in the
    # curve shares one bank and one restriction, so any difference in how much
    # parity costs them is about the arm, not the corpus -- and if the dropout
    # arms lose more, the old measurement was flattering the very thing under
    # test. K=32 gives twice the neighbours these arms train with, and a model
    # trained on a randomly thinned neighbour set has more to gain from extras
    # than one tuned to exactly 16.
    shifts = []
    for label, old_f, new_f in ARMS:
        if new_f not in have:
            continue
        o = load(old_f)
        if o is None:
            continue
        pair = align(o, have[new_f])
        if pair is not None:
            shifts.append((label.strip(), paired(*pair)[0]))
    if len(shifts) > 2:
        print("\n\nWhat parity cost each arm, on <25 km\n")
        for label, d in shifts:
            print("  {:<22} {:+.2f} pp".format(label, d))
        print("\n  Every row but d768-b265-e6 shares one bank and "
              "one restriction, so a spread among" + " those is about the "
              "arm, not the corpus.")
        print("  If the dropout arms lose more than p=0, the old "
              "measurement flattered the treatment: K=32 is twice the "
              "neighbours they train with, and a model trained on a "
              "thinned set has more to gain from extras than one tuned "
              "to exactly 16.")

    best = max(ready, key=lambda pf: (have[pf[1]][0] < 25).mean())
    print("\n  best <25 km at parity: p = {}".format(best[0]))
    print("  The claim on file is p = 0.7. If this disagrees, the optimum was "
          "an artefact of\n  measuring at K=32 -- which is twice the neighbour "
          "count these arms train with,\n  and neighbour count is the thing "
          "--retr-drop manipulates.")
    if len(ready) < len(PCURVE):
        print("\n  ({} of {} p-curve arms have run; treat the argmax as "
              "provisional.)".format(len(ready), len(PCURVE)))


if __name__ == "__main__":
    main()
