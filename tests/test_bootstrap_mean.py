"""The mean was a number in every report and a claim in none of them.

`runs/TILESHIP.md` said it plainly -- "bootstrap.py does not resample the
mean, so this has no interval and is not a claim" -- and then the mean was
read as an ordering anyway, there and again on the rotation arms, because a
column of numbers invites one. The fix is not a warning; it is an interval.

It cannot borrow the median's. The mean of a great-circle error is a TAIL
statistic: at 5,000 test images the worst 1% sit near 8,600 km, wrong
hemisphere, and carry about a third of a ~350 km mean. Which of those fifty
land in a resample moves it by tens of kilometres, so the mean's interval is
roughly thirty times the median's on the same data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bootstrap as B  # noqa: E402


def heavy(n=5000, seed=0, shift=0.0):
    """Errors shaped like the real ones: a tight body and an antipodal tail."""
    rng = np.random.default_rng(seed)
    e = rng.lognormal(mean=2.8, sigma=1.9, size=n) + shift
    e[rng.integers(0, n, n // 100)] += 9000.0
    return e


def test_paired_returns_three_intervals():
    a, b = heavy(seed=1), heavy(seed=2)
    out = B.paired(a, b, 200, np.random.default_rng(0))
    assert len(out) == 3
    for lo, hi in out:
        assert lo <= hi


def test_the_mean_interval_is_far_wider_than_the_median_s():
    """Why it needs its own. Borrowing the median's would understate it by
    more than an order of magnitude, which is worse than having none."""
    a, b = heavy(seed=1), heavy(seed=2)
    (ml, mh), (ul, uh), _ = B.paired(a, b, 400, np.random.default_rng(0))
    assert (uh - ul) > 10 * (mh - ml), ((ml, mh), (ul, uh))


def test_all_three_share_one_resample():
    """Three views of one comparison, not three comparisons. Drawing separate
    resamples would let the median and the mean disagree about which images
    they were computed on."""
    a, b = heavy(seed=1), heavy(seed=2)
    one = B.paired(a, b, 200, np.random.default_rng(7))
    two = B.paired(a, b, 200, np.random.default_rng(7))
    assert one == two
    assert B.paired(a, b, 200, np.random.default_rng(8)) != one


def test_a_real_mean_shift_is_detected():
    """So the width above is not the interval being useless."""
    a = heavy(seed=3)
    b = a + 400.0
    _, (ul, uh), _ = B.paired(a, b, 400, np.random.default_rng(0))
    assert uh < 0, (ul, uh)
    assert B.verdict(ul, uh) == "separated"


def test_a_tail_only_difference_moves_the_mean_and_not_the_median():
    """The case the mean exists to catch, and the reason it is worth its
    width: an arm that fixes wrong-hemisphere misses and nothing else."""
    a = heavy(seed=4)
    b = a.copy()
    worst = np.argsort(a)[-50:]
    b[worst] = 20.0
    (ml, mh), (ul, uh), _ = B.paired(a, b, 400, np.random.default_rng(0))
    assert B.verdict(ul, uh) == "separated"
    assert B.verdict(ml, mh) == "inside noise"


def test_the_table_columns_line_up():
    """A markdown table whose header, rule and rows disagree on column count
    renders as garbage, and that is invisible in the source."""
    row = B.TABLE_ROW.format("a", "b", -1.0, 1.0, "v", -1, 1, "v", -1.0, 1.0,
                             "v")
    assert B.TABLE_HEAD.count("|") == B.TABLE_RULE.count("|") == row.count("|")


def test_the_table_reports_the_mean():
    assert "mean diff" in B.TABLE_HEAD
    assert B.TABLE_HEAD.count("95% CI") == 3
