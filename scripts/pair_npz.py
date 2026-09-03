"""Paired bootstrap over two exported per-image error arrays.

Two arms evaluated on the same images are not compared by putting their point
estimates side by side -- that is how a 1.4 pp difference on 1,000 images gets
read as a finding when the interval spans zero. This resamples *images*, which
is the paired unit, and reports the difference with a 95% interval on both the
hit rate and the median.

    python scripts/pair_npz.py runs/hr_b265.npz runs/hr_b350.npz

A positive hit-rate difference means the SECOND file is better; a positive
median difference means the second file is worse (more km), matching the
convention in bootstrap.py.
"""

import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    a, b = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
    ea, eb = np.asarray(a["err"], float), np.asarray(b["err"], float)
    if len(ea) != len(eb):
        sys.exit("different lengths: {} vs {}".format(len(ea), len(eb)))
    ia, ib = a["image_id"], b["image_id"]
    if not (ia == ib).all():
        # Order matters: a paired test on differently-ordered rows silently
        # compares unrelated images and usually reports "inside noise".
        oa, ob = np.argsort(ia), np.argsort(ib)
        if not (ia[oa] == ib[ob]).all():
            sys.exit("the two runs scored different image sets")
        ea, eb = ea[oa], eb[ob]
        print("reordered to a common image order")

    print("{}  vs  {}".format(sys.argv[1], sys.argv[2]))
    for name, t in (("<1km", 1.0), ("<25km", 25.0), ("<200km", 200.0)):
        print("  {:<7} {:>6.1%} -> {:>6.1%}".format(
            name, (ea < t).mean(), (eb < t).mean()), end="")
        d = (eb < t).astype(float) - (ea < t)
        rng = np.random.default_rng(0)
        bs = d[rng.integers(0, len(d), (5000, len(d)))].mean(1) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print("   {:+.2f} pp [{:+.2f}, {:+.2f}] {}".format(
            100 * d.mean(), lo, hi,
            "separated" if lo * hi > 0 else "inside noise"))

    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(ea), (5000, len(ea)))
    md = np.median(eb[ix], 1) - np.median(ea[ix], 1)
    lo, hi = np.percentile(md, [2.5, 97.5])
    print("  median  {:>6.1f} -> {:>6.1f} km   {:+.1f} km [{:+.1f}, {:+.1f}] {}"
          .format(np.median(ea), np.median(eb),
                  np.median(eb) - np.median(ea), lo, hi,
                  "separated" if lo * hi > 0 else "inside noise"))
    print("  n = {:,} images, paired".format(len(ea)))


if __name__ == "__main__":
    main()
