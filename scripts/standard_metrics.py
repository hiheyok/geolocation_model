"""Report this agent in the metric format the OSV-5M literature uses.

GeoScore is the Geoguessr-style score used by the OSV-5M benchmark:
5000 * exp(-d / 1492.7) per image, averaged.  Recalls are the fraction within
the street / city / region / country / continent radii.

The numbers this prints are NOT comparable to the published OSV-5M leaderboard:
that benchmark enforces a 1 km spatial separation between train and test and
keeps one image per capture sequence, while the split here only guarantees that
a sequence never spans the train/test boundary.  See the note printed at the
end.  This exists so the two can be lined up on the same axes once the
geographic-holdout arm is trained.
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "runs" / "errs"

RADII = [(1, "street"), (25, "city"), (200, "region"),
         (750, "country"), (2500, "continent")]


def geoscore(e):
    return float((5000 * np.exp(-e / 1492.7)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default=None,
                    help="comma separated .npy of per-image errors; "
                         "default = every cached test file")
    a = ap.parse_args()

    files = ([Path(f) for f in a.files.split(",")] if a.files
             else sorted(CACHE.glob("*_test_*.npy")))
    if not files:
        raise SystemExit("no cached error files in {}".format(CACHE))

    hdr = ("{:<26} {:>8} {:>9} {:>9}  " + " ".join("{:>8}" for _ in RADII)).format(
        "model", "GeoScore", "mean km", "median km",
        *["<{}km".format(r) for r, _ in RADII])
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        e = np.load(f)
        tag = f.stem.split("_test_")[0]
        rec = [100 * float((e < r).mean()) for r, _ in RADII]
        print(("{:<26} {:>8.0f} {:>9.1f} {:>9.1f}  " +
               " ".join("{:>7.1f}%" for _ in RADII)).format(
                   tag, geoscore(e), e.mean(), np.median(e), *rec))

    print("\nreference points, published, on their own test sets")
    print("  {:<26} {:>8} {:>9} {:>9}".format(
        "OSV-5M baseline (CVPR24)", 3361, 1814, "n/r"))
    print("\nNOT COMPARABLE AS-IS: the OSV-5M benchmark holds out test points by")
    print("1 km of physical distance and keeps one image per capture sequence.")
    print("This split only guarantees a sequence never spans train/test, so a")
    print("test image can sit on a street the training set and the retrieval")
    print("bank both cover. Train a --split-mode cell8 arm for the comparable")
    print("number.")


if __name__ == "__main__":
    main()
