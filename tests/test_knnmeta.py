"""One validator for a k-NN cache, because three consumers had none.

`GeoStepDataset` checked a cache's split mode, hash, street filename, query
count and neighbour range. `serve.py`, `eval_highres.py` and `multiquery.py`
read `bank_rows` straight out of the `.npz` and turned it into a search mask,
so a stale or replaced cache of the right apparent name selected a different
split or bank while serving and evaluation carried on normally (REVIEW4 #14).

Two failures covered here that no consumer could catch at all.

**A negative row wraps.** `mask[rows] = True` with a negative row marks a row
from the *end* of the bank. NumPy is doing what it was asked; the mask covers
images nobody selected and every shape still agrees.

**The extension is never cross-checked against the embeddings** (REVIEW4 #1).
All four extensions on disk hold exactly 750,000 rows, so a `release ++
bank_ext` street file beside a cache stamped `bank_ext2` passes the filename
check, the query count and the index range -- and then gives every visually
matched embedding a different photograph's z16 address.

The extension comparison is by **content, not name**, and the test that matters
most is the one asserting it accepts the *shipping* pair: a bank whose sidecar
names its extensions and a cache naming the combined one are the same corpus,
and a name comparison would have rejected the path in production.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import knnmeta  # noqa: E402


class Fake:
    """An .npz stand-in: `files` plus item access, which is all knnmeta uses."""

    def __init__(self, **kw):
        self._d = kw
        self.files = list(kw)

    def __getitem__(self, k):
        return self._d[k]


def cache(**kw):
    kw.setdefault("idx", np.zeros((10, 32), np.int64))
    return Fake(**kw)


# --- bank_rows ---------------------------------------------------------------

def test_valid_rows_pass_through():
    z = cache(bank_rows=np.array([0, 3, 7]))
    assert np.array_equal(knnmeta.bank_rows(z, 10, "t"), [0, 3, 7])


def test_a_negative_row_is_refused():
    """It would wrap to the end of the bank rather than raise."""
    z = cache(bank_rows=np.array([0, -1, 7]))
    with pytest.raises(SystemExit, match="negative"):
        knnmeta.bank_rows(z, 10, "t")


def test_a_row_past_the_end_is_refused():
    z = cache(bank_rows=np.array([0, 3, 99]))
    with pytest.raises(SystemExit, match="different bank"):
        knnmeta.bank_rows(z, 10, "t")


def test_duplicate_rows_are_refused():
    """Harmless to a mask, but the cache is then not what it claims."""
    z = cache(bank_rows=np.array([0, 3, 3]))
    with pytest.raises(SystemExit, match="distinct"):
        knnmeta.bank_rows(z, 10, "t")


def test_a_cache_with_no_bank_rows_returns_none():
    assert knnmeta.bank_rows(cache(), 10, "t") is None


def test_an_empty_selection_does_not_crash_the_bounds_check():
    assert len(knnmeta.bank_rows(cache(bank_rows=np.array([], np.int64)),
                                 10, "t")) == 0


# --- the rest of the contract ------------------------------------------------

def test_a_different_split_mode_is_refused():
    z = cache(split_mode=np.array("cell8"))
    with pytest.raises(SystemExit, match="split"):
        knnmeta.check(z, "t", split_mode="sequence")


def test_a_different_street_file_is_refused():
    z = cache(street_file=np.array("other.f16.npy"))
    with pytest.raises(SystemExit, match="embedding space"):
        knnmeta.check(z, "t", street_file="mine.f16.npy")


def test_a_wrong_query_count_is_refused():
    with pytest.raises(SystemExit, match="query rows"):
        knnmeta.check(cache(), "t", n_release=11)


def test_too_few_neighbours_is_refused():
    with pytest.raises(SystemExit, match="neighbours per query"):
        knnmeta.check(cache(), "t", need_k=64)


def test_unknown_fields_are_simply_not_checked():
    """A consumer that cannot supply a field should still get the rest.

    The alternative is what the three bypassing consumers actually did, which
    is to check nothing at all.
    """
    z = cache(split_mode=np.array("sequence"), bank_rows=np.array([0, 1]))
    assert np.array_equal(knnmeta.check(z, "t", n_bank=5), [0, 1])


def test_a_split_hash_mismatch_is_refused_without_the_legacy_allowance():
    z = cache(split_hash=np.array("aaaaaaaaaaaa"))
    with pytest.raises(SystemExit, match="no longer that train side"):
        knnmeta.check(z, "t", split_hash="bbbbbbbbbbbb")


# --- REVIEW4 #1: the extension must match the embeddings ---------------------

def test_naming_an_extension_the_street_file_does_not_is_refused(monkeypatch):
    monkeypatch.setattr(knnmeta.prov, "exts_of", lambda p: [])
    with pytest.raises(SystemExit, match="names the bank extension"):
        knnmeta.check_ext(cache(bank_ext=np.array("bank_ext2")), "s", "t")


def test_a_street_file_with_an_extension_and_a_cache_without_is_refused(
        monkeypatch):
    monkeypatch.setattr(knnmeta.prov, "exts_of", lambda p: ["bank_ext"])
    with pytest.raises(SystemExit, match="names no bank extension"):
        knnmeta.check_ext(cache(), "s", "t")


def test_same_length_different_extension_is_refused(monkeypatch):
    """The scenario the review describes, and it passes every other guard.

    bank_ext and bank_ext2 both hold exactly 750,000 rows, so the lengths
    agree, the filename check passes and every neighbour index stays in range.
    """
    ids = {"bank_ext": np.arange(750000, dtype=np.int64),
           "bank_ext2": np.arange(750000, dtype=np.int64) + 9_000_000}
    monkeypatch.setattr(knnmeta.prov, "exts_of", lambda p: ["bank_ext"])
    monkeypatch.setattr(knnmeta.prov, "bank_ext",
                        lambda s, r: {"image_id": ids[s]})
    with pytest.raises(SystemExit, match="another photograph's address"):
        knnmeta.check_ext(cache(bank_ext=np.array("bank_ext2")), "s", "t")


def test_the_same_corpus_under_different_names_is_accepted(monkeypatch):
    """The test that protects the shipping path.

    A bank stacked from four parquets records four stems while the cache built
    over it names the single combined extension. Those strings differ and the
    corpora are identical, so a name comparison would reject production.
    Comparing the ordered id digest accepts it.
    """
    quarter = [np.arange(750000, dtype=np.int64) + i * 750000
               for i in range(4)]
    ids = {"bank_ext70": np.concatenate(quarter)}
    for i, q in enumerate(quarter):
        ids["bank_ext" + ("" if i == 0 else str(i + 1))] = q
    monkeypatch.setattr(
        knnmeta.prov, "exts_of",
        lambda p: ["bank_ext", "bank_ext2", "bank_ext3", "bank_ext4"])
    monkeypatch.setattr(knnmeta.prov, "bank_ext",
                        lambda s, r: {"image_id": ids[s]})
    knnmeta.check_ext(cache(bank_ext=np.array("bank_ext70")), "s", "t")


def test_extensions_in_the_wrong_order_are_refused(monkeypatch):
    """The digest is ordered, because a reordered stack is a different bank."""
    quarter = [np.arange(750000, dtype=np.int64) + i * 750000
               for i in range(4)]
    ids = {"bank_ext70": np.concatenate(quarter)}
    for i, q in enumerate(quarter):
        ids["bank_ext" + ("" if i == 0 else str(i + 1))] = q
    monkeypatch.setattr(
        knnmeta.prov, "exts_of",
        lambda p: ["bank_ext2", "bank_ext", "bank_ext3", "bank_ext4"])
    monkeypatch.setattr(knnmeta.prov, "bank_ext",
                        lambda s, r: {"image_id": ids[s]})
    with pytest.raises(SystemExit, match="another photograph's address"):
        knnmeta.check_ext(cache(bank_ext=np.array("bank_ext70")), "s", "t")


def test_no_extension_on_either_side_is_fine(monkeypatch):
    monkeypatch.setattr(knnmeta.prov, "exts_of", lambda p: [])
    knnmeta.check_ext(cache(), "s", "t")
