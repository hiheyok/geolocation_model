"""Distinct restricted banks must get distinct cache names (REVIEW8 #11).

The name encoded `bank_limit // 1000`, so every limit inside a thousand-row
interval shared one path -- 1000 and 1999 both wrote `bank1k`. The builder
samples the exact requested count, so building both in sequence left the
second bank's neighbours under the first bank's filename, with the embeddings,
the split and the split hash all still agreeing. A checkpoint naming that path
then trained against a bank it did not ask for, and nothing downstream
compared the stored row count with the restriction the older run intended.

The compatibility half matters as much as the injectivity half: every
restricted cache on disk was built at a multiple of 1000, so those keep the
short form and no existing artifact is renamed by this change.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OSV_RELEASE", "s10")

import config  # noqa: E402


def name(limit):
    return config.knn_name("dual_c3.f16.npy", "sequence", 32, limit)


@pytest.mark.parametrize(("a", "b"), [(1, 999), (1000, 1999), (25000, 25500),
                                      (1, 1000), (999, 1000)])
def test_distinct_limits_get_distinct_names(a, b):
    assert name(a) != name(b), "{} and {} share {}".format(a, b, name(a))


def test_no_two_limits_collide_across_a_wide_sweep():
    """The pairs above are the review's; this is the property they sample."""
    seen = {}
    for n in list(range(1, 40)) + [999, 1000, 1001, 1999, 2000, 25000, 25500,
                                   999_999, 1_000_000, 1_000_001]:
        s = name(n)
        assert s not in seen, "{} and {} share {}".format(seen[s], n, s)
        seen[s] = n


def test_an_unrestricted_bank_stays_separate():
    assert name(0) == "knn_dual_c3_sequence_k32.npz"
    assert "bank" not in name(0)


def test_multiples_of_a_thousand_keep_the_name_they_have():
    """`knn_dual_c3_sequence_k32_bank25k.npz` is on disk; renaming it would
    orphan it from the checkpoints that record it."""
    assert name(25000).endswith("_bank25k.npz")
    assert name(1000).endswith("_bank1k.npz")
    assert name(1_000_000).endswith("_bank1000k.npz")


def test_the_two_forms_cannot_be_confused():
    """One ends in `k` and the other does not, which is what keeps the short
    form for multiples from colliding with a spelled-out count."""
    assert config.bank_suffix(2000) == "2k"
    assert config.bank_suffix(2001) == "2001"
    assert config.bank_suffix(2) == "2"
    assert config.bank_suffix(2000) != config.bank_suffix(2)
