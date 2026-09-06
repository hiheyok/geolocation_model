"""In range is not in the bank (REVIEW6 #4).

`GeoStepDataset` checked that every cached neighbour id resolves to some row
of the address tables. That is a much weaker statement than it looks, because
the wrong rows are ordinary rows: a held-out test image, the query itself, and
the frame captured ten metres earlier along the same road all have valid
non-negative ids inside the range, and all three are leaks.

The third one is not hypothetical here. Same-sequence exclusion has already
gone missing once in this project -- it stopped applying across the corpus
boundary when the extension's sequence ids were offset -- and it was worth 18
points of the headline. Nothing downstream could see it, which is the whole
argument for checking the artifact rather than trusting the builder.

Run against the caches actually on disk, this check fails six of nineteen.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import _check_neighbours  # noqa: E402


class Cache:
    def __init__(self, **kw):
        self._d = kw
        self.files = list(kw)

    def __getitem__(self, k):
        return self._d[k]


# four release rows, two extension rows; rows 0-1 are the bank
REL = np.array(["s0", "s7", "s1", "s2"], "U40")
EXT = np.array(["s9", "s1"], "U40")
BANK = np.array([0, 1, 4, 5], np.int64)


def bank(**kw):
    kw.setdefault("bank_rows", BANK)
    return Cache(**kw)


def test_a_clean_cache_passes():
    """Non-vacuity: the fixture must be acceptable before the rest means
    anything. Every query here draws from a bank row on another drive."""
    idx = np.array([[1, 4], [0, 4], [0, 4], [1, 5]], np.int64)
    _check_neighbours(idx, bank(), "t", 6, REL, EXT)


def test_a_neighbour_outside_the_bank_is_refused():
    """Row 2 is a perfectly valid release row; it is also held out."""
    idx = np.array([[1, 4], [0, 4], [0, 2], [1, 5]], np.int64)
    with pytest.raises(SystemExit, match="not in the 4 rows"):
        _check_neighbours(idx, bank(), "t", 6, REL, EXT)


def test_a_query_retrieving_itself_is_refused():
    idx = np.array([[1, 4], [0, 4], [0, 4], [1, 3]], np.int64)
    with pytest.raises(SystemExit, match="retrieve their own row"):
        _check_neighbours(idx, Cache(bank_rows=np.array([0, 1, 3, 4, 5])),
                          "t", 6, REL, EXT)


def test_a_same_sequence_neighbour_inside_the_release_is_refused():
    """Rows 0 and 1 are both sequence s0 -- consecutive frames on one road."""
    idx = np.array([[1, 4], [0, 4], [0, 4], [1, 5]], np.int64)
    with pytest.raises(SystemExit, match="share their query's sequence"):
        _check_neighbours(idx, bank(), "t", 6,
                          np.array(["s0", "s0", "s1", "s2"], "U40"), EXT)


def test_a_same_sequence_neighbour_across_the_corpus_boundary_is_refused():
    """The failure that actually happened: one real drive whose frames landed
    on both sides of the release/extension boundary. Release row 2 and
    extension row 5 are both sequence s1, and offsetting the extension's ids
    made them look like different drives."""
    idx = np.array([[1, 4], [0, 4], [0, 5], [1, 5]], np.int64)
    with pytest.raises(SystemExit, match="share their query's sequence"):
        _check_neighbours(idx, bank(), "t", 6, REL, EXT)


def test_a_cache_without_bank_rows_warns_and_is_still_checked(capsys):
    """Membership cannot be checked; the other two still can."""
    idx = np.array([[1, 4], [0, 4], [0, 4], [1, 5]], np.int64)
    _check_neighbours(idx, Cache(), "t", 6, REL, EXT)
    assert "records no bank_rows" in capsys.readouterr().out
    idx[3, 1] = 3
    with pytest.raises(SystemExit, match="retrieve their own row"):
        _check_neighbours(idx, Cache(), "t", 6, REL, EXT)


def test_a_sequence_table_that_does_not_span_the_addresses_is_refused():
    """Silently short would make `sid[idx]` read the wrong drive, or raise
    deep inside the check rather than saying what disagreed."""
    idx = np.array([[1, 4], [0, 4], [0, 4], [1, 5]], np.int64)
    with pytest.raises(SystemExit, match="disagree about how many"):
        _check_neighbours(idx, bank(), "t", 6, REL, None)


def test_the_release_only_case_needs_no_extension():
    idx = np.array([[1], [0], [1], [0]], np.int64)
    _check_neighbours(idx, Cache(bank_rows=np.array([0, 1])), "t", 4,
                      np.array(["a", "b", "c", "d"], "U40"), None)
