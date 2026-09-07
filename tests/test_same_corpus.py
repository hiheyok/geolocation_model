"""Corpora are compared by ordered identity, never by count.

All four bank extensions on disk hold exactly 750,000 rows. Two artifacts
built over different ones therefore agree on every length, every index stays
in range, and each row names a different photograph -- so a count is not
evidence of anything here, and three separate checks were written as if it
were.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import provenance as prov  # noqa: E402


def test_the_same_ids_in_the_same_order_pass():
    prov.require_same_corpus([10, 11, 12], [10, 11, 12], "a", "b")


def test_equal_length_different_ids_are_refused():
    """The shape every one of the three defects had."""
    with pytest.raises(SystemExit, match="different corpora"):
        prov.require_same_corpus([10, 11, 12], [10, 11, 99], "a", "b")


def test_a_reordering_is_refused():
    """Row k is whatever id sits at position k."""
    with pytest.raises(SystemExit, match="different corpora"):
        prov.require_same_corpus([10, 11, 12], [10, 12, 11], "a", "b")


def test_the_message_says_why_the_count_did_not_help():
    """The next person to reach for `len()` should be told why not, in the
    error rather than in a docstring they will not open."""
    with pytest.raises(SystemExit) as e:
        prov.require_same_corpus([1, 2], [1, 3], "a", "b")
    msg = str(e.value)
    assert "750,000" in msg and "lengths agree" in msg


def test_the_caller_can_say_what_breaks():
    with pytest.raises(SystemExit, match="scored against"):
        prov.require_same_corpus([1], [2], "a", "b",
                                 because="Both arms are scored against a.")


def test_string_ids_work_too():
    """The external corpora key on strings, not ints."""
    prov.require_same_corpus(np.array(["a", "b"]), np.array(["a", "b"]), "x", "y")
    with pytest.raises(SystemExit, match="different corpora"):
        prov.require_same_corpus(np.array(["a", "b"]), np.array(["b", "a"]),
                                 "x", "y")
