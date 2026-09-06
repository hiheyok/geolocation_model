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


# --- REVIEW8 #1: membership does not establish that the bank is ALLOWED ------

def test_the_reviewers_repro_is_now_refused():
    """Verbatim from REVIEW8 #1, which returned successfully before this.

    Live labels are [train, train, test]. Row 2 is held out, appears in both
    `bank_rows` and `idx`, every query retrieves a different row, and every
    sequence differs -- so membership, range, self and sequence all pass.
    """
    idx = np.array([[2], [0], [1]], np.int64)
    z = Cache(bank_rows=np.array([0, 1, 2]))
    seq = np.array(["a", "b", "c"], "U40")
    labels = np.array(["train", "train", "test"], dtype=object)
    _check_neighbours(idx, z, "probe", 3, seq, None)      # the old contract
    with pytest.raises(SystemExit, match="this split holds out"):
        _check_neighbours(idx, z, "probe", 3, seq, None, labels=labels,
                          mode="sequence", x16=np.zeros(3), y16=np.zeros(3))


def test_a_restricted_training_bank_is_still_allowed():
    """Banking fewer rows than the split permits is a choice, not a leak, so
    the rule has to be one-directional."""
    idx = np.array([[1], [0], [0]], np.int64)
    z = Cache(bank_rows=np.array([0, 1]))
    seq = np.array(["a", "b", "c"], "U40")
    labels = np.array(["train", "train", "test"], dtype=object)
    _check_neighbours(idx, z, "probe", 3, seq, None, labels=labels,
                      mode="sequence", x16=np.zeros(3), y16=np.zeros(3))


def test_an_extension_row_in_a_held_out_cell_is_refused():
    """The extension carries no split label, so a cell split can only exclude
    it geographically -- and that decision lived in the builder alone."""
    labels = np.array(["train", "train", "test", "train"], dtype=object)
    seq = np.array(["a", "b", "c", "d"], "U40")
    ext = np.array(["e", "f"], "U40")
    # row 2 is held out and sits in z8 cell (1000>>8, 2000>>8); extension row 4
    # lands in the same cell, extension row 5 does not.
    x16 = np.array([10, 20, 1000, 30, 1001, 50000], np.int64)
    y16 = np.array([10, 20, 2000, 30, 2001, 50000], np.int64)
    idx = np.array([[1], [0], [0], [1]], np.int64)
    z = Cache(bank_rows=np.array([0, 1, 3, 4]))
    with pytest.raises(SystemExit, match="held-out"):
        _check_neighbours(idx, z, "probe", 6, seq, ext, labels=labels,
                          mode="cell8", x16=x16, y16=y16)
    z2 = Cache(bank_rows=np.array([0, 1, 3, 5]))
    _check_neighbours(idx, z2, "probe", 6, seq, ext, labels=labels,
                      mode="cell8", x16=x16, y16=y16)


def test_sequence_mode_holds_out_no_cells():
    """`sequence` excludes by drive, not by region, so an extension row is
    never geographically ineligible -- the same-sequence check covers it."""
    labels = np.array(["train", "train", "test", "train"], dtype=object)
    seq = np.array(["a", "b", "c", "d"], "U40")
    ext = np.array(["e", "f"], "U40")
    x16 = np.array([10, 20, 1000, 30, 1001, 50000], np.int64)
    y16 = np.array([10, 20, 2000, 30, 2001, 50000], np.int64)
    idx = np.array([[1], [0], [0], [1]], np.int64)
    _check_neighbours(idx, Cache(bank_rows=np.array([0, 1, 3, 4])), "probe", 6,
                      seq, ext, labels=labels, mode="sequence",
                      x16=x16, y16=y16)


def test_eligibility_is_skipped_when_the_caller_has_no_labels():
    """Consumers that genuinely cannot supply the live split still get every
    other check, rather than the none they had before."""
    idx = np.array([[2], [0], [1]], np.int64)
    _check_neighbours(idx, Cache(bank_rows=np.array([0, 1, 2])), "probe", 3,
                      np.array(["a", "b", "c"], "U40"), None)
