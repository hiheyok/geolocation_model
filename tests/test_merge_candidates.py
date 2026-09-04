"""Equal slots are not equal weight (review item 23).

`rr` hands each photograph in a group the same number of candidate slots, which
is what makes it "equal-photo".  It then carried each candidate's *raw* cosine
into a single softmax downstream.  Similarity scale varies per photograph --
that is the very fact `rr` exists to work around -- so a photograph whose best
match is 0.72 has every candidate suppressed against one whose best is 0.95,
and the equal share it was just given back is spent.

`calib="top1"` subtracts each photograph's own top-1, so every photograph's
best candidate enters at 0.  `calib="none"` reproduces the runs on record,
including the +1.2 pp second-angle result, so that number stays reproducible.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from multiquery import merge_candidates  # noqa: E402

# two photographs, four candidates each, disjoint bank rows.
# photo 0 is the "confident" one; photo 1 sits half a point lower throughout.
IDX = np.array([[10, 11, 12, 13],
                [20, 21, 22, 23]], dtype=np.int64)
SIM = np.array([[0.95, 0.90, 0.85, 0.80],
                [0.45, 0.40, 0.35, 0.30]], dtype=np.float32)


def test_rr_gives_each_photograph_the_same_number_of_slots():
    ci, _ = merge_candidates([0, 1], IDX, SIM, 4, "rr")
    assert sum(b < 20 for b in ci) == 2
    assert sum(b >= 20 for b in ci) == 2


def test_uncalibrated_slots_are_not_uncalibrated_weight():
    """The defect: equal slots, then a 0.50 gap decides the softmax anyway."""
    _, cs = merge_candidates([0, 1], IDX, SIM, 4, "rr", calib="none")
    lo = [s for s in cs if s < 0.5]
    hi = [s for s in cs if s >= 0.5]
    assert min(hi) - max(lo) > 0.3


def test_top1_calibration_puts_each_photographs_best_at_zero():
    _, cs = merge_candidates([0, 1], IDX, SIM, 4, "rr", calib="top1")
    assert max(cs) == 0.0
    # each photograph contributes one 0.0 (its own best) and one -0.05
    assert sorted(np.round(cs, 4).tolist()) == [-0.05, -0.05, 0.0, 0.0]


def test_calibration_preserves_the_order_within_one_photograph():
    _, cs = merge_candidates([0], IDX, SIM, 4, "rr", calib="top1")
    assert list(cs) == sorted(cs, reverse=True)


def test_one_photograph_is_unchanged_by_calibration_up_to_a_constant():
    """N=1 is the baseline every multi-photo number is measured against, so a
    shift there would move the baseline rather than the treatment."""
    _, a = merge_candidates([0], IDX, SIM, 4, "rr", calib="none")
    _, b = merge_candidates([0], IDX, SIM, 4, "rr", calib="top1")
    assert np.allclose(np.diff(a), np.diff(b))


def test_global_merge_is_calibrated_too():
    """Without this, `global` keeps the best K by raw score and one photograph
    takes every slot -- the failure `rr` was written to avoid."""
    ci, _ = merge_candidates([0, 1], IDX, SIM, 4, "global", calib="none")
    assert all(b < 20 for b in ci), "uncalibrated global: one photo takes all"
    ci, _ = merge_candidates([0, 1], IDX, SIM, 4, "global", calib="top1")
    assert sum(b >= 20 for b in ci) == 2


def test_padding_does_not_let_one_row_vote_twice():
    """Fewer than K distinct rows exist. The pad repeated the last candidate at
    its own score, reintroducing the double-count rr deduplicates to avoid."""
    idx = np.array([[7, 7, 7, 7]], dtype=np.int64)
    sim = np.array([[0.9, 0.9, 0.9, 0.9]], dtype=np.float32)
    ci, cs = merge_candidates([0], idx, sim, 4, "rr", calib="none")
    assert len(ci) == 4 and (ci == 7).all()
    assert (np.array(cs)[1:] < -1e3).all(), "pads must carry no weight"


def test_duplicates_across_photographs_are_not_counted_twice():
    idx = np.array([[5, 6], [5, 7]], dtype=np.int64)
    sim = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
    ci, _ = merge_candidates([0, 1], idx, sim, 3, "rr")
    assert len(set(int(b) for b in ci)) == 3
