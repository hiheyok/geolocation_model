"""`--limit` and `--overfit` together must not crash model construction.

`--limit` wraps the training set in a `Subset`; a valid `--overfit` wraps that
result again. Model construction unwrapped one level, so `_ds` was still a
`Subset` and the next attribute access raised `AttributeError` -- after the
dataset had been built and before the first epoch. The size guard accepts the
combination and overfit correctly switches selection to loss, so the run gets
all the way to construction and then dies (REVIEW8 #5).

The one-level form was written out at four sites in `train.py`, which is why
one flag combination broke all four at once. That is the recurring shape here:
a rule expressed once per caller instead of once.
"""

import sys
from pathlib import Path

from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import base_dataset  # noqa: E402


class Base:
    """Only the attributes construction actually reads off the dataset."""

    def __init__(self):
        self.dim_street = 768
        self.dim_cond = 0

    def __len__(self):
        return 8

    def __getitem__(self, i):
        return i


def test_an_unwrapped_dataset_is_returned_as_is():
    b = Base()
    assert base_dataset(b) is b


def test_one_wrapper_is_removed():
    b = Base()
    assert base_dataset(Subset(b, [0, 1, 2])) is b


def test_two_wrappers_are_removed():
    """`--limit 100 --overfit 10`. One level of unwrapping returns a Subset,
    and `Subset` has no `dim_street`."""
    b = Base()
    both = Subset(Subset(b, [0, 1, 2, 3]), [0, 1])
    assert base_dataset(both) is b
    assert base_dataset(both).dim_street == 768


def test_the_one_level_form_is_what_raised():
    """Recorded so the fix is not mistaken for defensive tidying."""
    both = Subset(Subset(Base(), [0, 1, 2, 3]), [0, 1])
    one_level = both.dataset if hasattr(both, "dataset") else both
    assert isinstance(one_level, Subset)
    assert not hasattr(one_level, "dim_street")


def test_arbitrary_nesting_terminates():
    b = Base()
    ds = b
    for _ in range(5):
        ds = Subset(ds, [0, 1])
    assert base_dataset(ds) is b


def test_train_has_no_hand_rolled_unwrap_left():
    """Four sites each unwrapped once, so one flag combination broke all four.
    A fifth copy would reintroduce it silently."""
    src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
    assert 'hasattr(tr, "dataset")' not in src, (
        "train.py still unwraps a dataset by hand; use base_dataset so the "
        "depth is decided in one place")
