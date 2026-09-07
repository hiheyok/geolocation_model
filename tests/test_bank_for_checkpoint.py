"""A missing dependency is not permission to search everything (REVIEW8 #6, #7).

`eval_highres` and `multiquery` each resolved which bank rows to search
themselves, and each got it wrong the same two ways.

**A named-but-missing cache became the whole corpus.** Both validated the file
only `if it exists`, so moving it left `keep = None` and the fallback selected
every row -- including the ones the training bank excluded -- while printing
"checkpoint records none", which is the message for a *legacy* checkpoint that
never recorded a bank. The two cases were indistinguishable in the log.

**The validator's checks did not fire.** Both passed only `n_bank` and
`street_path`, leaving `split_mode`, `split_hash` and `street_file` at their
`None` defaults, so those comparisons never ran. A cache from a different split
over the same embeddings has a correct content digest and in-range rows, and
was accepted. The review's warning is the point of this file: *a call-presence
assertion is insufficient; test which metadata comparisons actually run.*
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import knnmeta

# Captured at import, before any fixture replaces it.
REAL_CHECK_BYTES = knnmeta.check_bytes  # noqa: E402


class Cache(dict):
    """An .npz stand-in with the `files` attribute knnmeta reads."""

    @property
    def files(self):
        return list(self)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A minimal but real on-disk world: a bank file, a cache, a parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    monkeypatch.setattr(knnmeta.config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(knnmeta.config, "DATASET_PARQUET", tmp_path / "d.parquet")
    labels = ["train", "train", "test", "train"]
    pq.write_table(pa.table({
        "image_id": pa.array([1, 2, 3, 4], pa.int64()),
        "split_sequence": pa.array(labels),
    }), tmp_path / "d.parquet")
    np.save(tmp_path / "bank.f16.npy", np.zeros((4, 8), np.float16))

    shash = knnmeta.sp.split_hash("sequence", np.asarray(labels, dtype=object))
    cache = Cache(idx=np.zeros((4, 32), np.int64),
                  split_mode=np.array("sequence"),
                  split_hash=np.array(shash),
                  street_file=np.array("bank.f16.npy"),
                  bank_rows=np.array([0, 1, 3]))
    np.savez(tmp_path / "k.npz", **cache)
    # check_ext / check_bytes are exercised in their own suites; isolating
    # argument dispatch is the whole point here, exactly as the review did.
    # These two read real files, and most tests here are about argument
    # dispatch. Stubbing them is deliberate -- but it also disabled the digest
    # comparison for every test in the file, which is how a bug in exactly
    # that comparison shipped with a passing test named after it. The two
    # tests that are about bytes put the real one back.
    monkeypatch.setattr(knnmeta, "check_ext", lambda *a, **k: None)
    monkeypatch.setattr(knnmeta, "check_bytes", lambda *a, **k: None)
    return tmp_path, {"split_mode": "sequence"}, cache


def test_the_declared_bank_is_returned(world):
    """Non-vacuity: the good path must work before a refusal means anything."""
    _, ck, _ = world
    rows = knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")
    assert np.array_equal(rows, [0, 1, 3])


def test_a_named_but_missing_cache_is_refused(world):
    """The bug: this used to return every row and call it 'records none'."""
    _, ck, _ = world
    with pytest.raises(SystemExit, match="not on disk"):
        knnmeta.bank_for_checkpoint(ck, "gone.npz", 4, "bank.f16.npy", "t")


def test_a_legacy_checkpoint_that_names_nothing_is_not_the_same_case(world,
                                                                     capsys):
    """Distinguishable in the log, which they were not."""
    _, ck, _ = world
    rows = knnmeta.bank_for_checkpoint(ck, None, 4, "bank.f16.npy", "t")
    assert np.array_equal(rows, [0, 1, 2, 3])
    assert "legacy checkpoint, not a missing artifact" in capsys.readouterr().out


def test_whole_corpus_must_be_asked_for_and_says_so(world, capsys):
    _, ck, _ = world
    rows = knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t",
                                       full_corpus=True)
    assert np.array_equal(rows, [0, 1, 2, 3])
    out = capsys.readouterr().out
    assert "WHOLE corpus" in out and "protocol differs" in out


# --- REVIEW8 #7: the checks must actually run, not merely be reachable ------

def test_a_cache_from_another_split_is_refused(world):
    """Same embeddings, same extension, different split. Its content digest is
    correct and its rows are in range, so only the split comparison can see
    it -- and that argument used to default to None."""
    tmp, ck, cache = world
    cache["split_mode"] = np.array("cell8")
    np.savez(tmp / "k.npz", **cache)
    with pytest.raises(SystemExit, match="built on split"):
        knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")


def test_a_stale_split_hash_is_refused(world):
    """The dataset's assignments changed while the cache kept the old ones."""
    tmp, ck, cache = world
    cache["split_hash"] = np.array("f" * 12)
    np.savez(tmp / "k.npz", **cache)
    with pytest.raises(SystemExit, match="no longer that train side"):
        knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")


def test_a_cache_over_another_street_file_is_refused(world):
    tmp, ck, cache = world
    cache["street_file"] = np.array("other.f16.npy")
    np.savez(tmp / "k.npz", **cache)
    with pytest.raises(SystemExit, match="embedding space"):
        knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")


def test_a_wrong_query_count_is_refused(world):
    """n_release was also left at None by both callers."""
    tmp, ck, cache = world
    cache["idx"] = np.zeros((9, 32), np.int64)
    np.savez(tmp / "k.npz", **cache)
    with pytest.raises(SystemExit, match="query rows"):
        knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")


def test_a_cache_with_no_bank_rows_is_refused(world):
    """"Which rows it searched is unknown" is not "it searched all of them".

    This was briefly recomputed from the split. It cannot be: every cache on
    disk lacking `bank_rows` carries only the pre-2026-09-03 digest, which
    hashes each label's first character -- "train" and "test" are both "t", so
    a swap leaves the digest AND `bank_n` unchanged while the derived bank
    gains the held-out rows. A rebuild stamps the rows and takes minutes,
    which is the correct price.
    """
    tmp, ck, cache = world
    del cache["bank_rows"]
    np.savez(tmp / "k.npz", **cache)
    with pytest.raises(SystemExit, match="records no bank rows"):
        knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")


def test_the_refusal_says_why_recomputing_is_not_an_option():
    """The reason belongs in the message: the next reader will otherwise try
    the same shortcut, since it looks obviously available."""
    import inspect

    src = inspect.getsource(knnmeta.bank_for_checkpoint)
    assert "first character" in src and "bank_n" in src


def test_the_legacy_digest_cannot_tell_train_from_test():
    """The property the refusal rests on, asserted directly rather than
    quoted -- if this ever became false, recomputing would become sound."""
    import splits as sp

    a = np.asarray(["train", "train", "test", "train"], dtype=object)
    b = np.asarray(["train", "test", "train", "train"], dtype=object)
    assert sp.split_hash_legacy("sequence", a) == sp.split_hash_legacy("sequence", b)
    assert sp.split_hash("sequence", a) != sp.split_hash("sequence", b)


def test_a_conditioned_bank_compares_against_its_retrieval_prefix(world, monkeypatch):
    """The k-NN is built on the retrieval prefix, not the joined file, so a
    joined bank must be validated against the prefix -- by NAME and by BYTES.

    The first version of this stamped no `street_digest`, so `check_bytes`
    warned and returned without comparing anything: it asserted the name half
    and was blind to the digest half, which is the half that was wrong. The
    joined file was digested against a stamp taken on the retrieval file, so
    every intact conditioned bank was refused at startup.
    """
    import json

    import safeio

    monkeypatch.setattr(knnmeta, "check_bytes", REAL_CHECK_BYTES)
    tmp, ck, cache = world
    # a stamp taken on the RETRIEVAL file, which is what build_knn records
    cache["street_digest"] = np.array(
        safeio.content_digest(tmp / "bank.f16.npy"))
    np.savez(tmp / "k.npz", **cache)

    # a joined bank: same rows, wider, so its own bytes differ from the prefix
    np.save(tmp / "joined.f16.npy", np.zeros((4, 12), np.float16))
    (tmp / "joined.f16.npy.prov.json").write_text(json.dumps(
        {"retrieval_file": "bank.f16.npy", "rows": 4}), encoding="utf-8")
    assert (safeio.content_digest(tmp / "joined.f16.npy")
            != safeio.content_digest(tmp / "bank.f16.npy")), "not a real test"

    rows = knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "joined.f16.npy", "t")
    assert np.array_equal(rows, [0, 1, 3])


def test_a_conditioned_bank_whose_prefix_changed_is_still_refused(world, monkeypatch):
    """The digest check must still fire -- fixing the path must not disable
    it. A retrieval cache rebuilt under its own name leaves the ids, lengths,
    row digest, split hash and filename all correct."""
    import json

    import safeio

    monkeypatch.setattr(knnmeta, "check_bytes", REAL_CHECK_BYTES)
    tmp, ck, cache = world
    cache["street_digest"] = np.array("0" * 12)      # not this file
    np.savez(tmp / "k.npz", **cache)
    np.save(tmp / "joined.f16.npy", np.zeros((4, 12), np.float16))
    (tmp / "joined.f16.npy.prov.json").write_text(json.dumps(
        {"retrieval_file": "bank.f16.npy", "rows": 4}), encoding="utf-8")
    with pytest.raises(SystemExit, match="digest"):
        knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "joined.f16.npy", "t")


def test_extension_identity_is_checked_against_the_joined_bank(world,
                                                               monkeypatch):
    """Two files, two questions, and they are not the same file.

    The joined bank is what records which extensions it was stacked from, so
    it is what extension identity must be asked of. Redirecting both checks to
    the retrieval prefix -- to fix the digest -- stopped `check_ext` seeing the
    joined bank at all, and a bank stacked from `bank_ext2` passed against a
    cache addressing `bank_ext`. Same row counts, so every index stays in
    range and each matched embedding simply gets another photograph's
    coordinates.
    """
    import json

    tmp, ck, cache = world
    np.save(tmp / "joined.f16.npy", np.zeros((4, 12), np.float16))
    (tmp / "joined.f16.npy.prov.json").write_text(json.dumps(
        {"retrieval_file": "bank.f16.npy", "rows": 4}), encoding="utf-8")

    seen = {}
    monkeypatch.setattr(knnmeta, "check_ext",
                        lambda z, path, what: seen.update(ext=Path(path).name))
    monkeypatch.setattr(knnmeta, "check_bytes",
                        lambda z, path, what, d=None: seen.update(
                            bytes=Path(path).name))
    knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "joined.f16.npy", "t")

    assert seen["ext"] == "joined.f16.npy", (
        "extension identity was asked of the retrieval prefix, which does not "
        "record what the joined bank was stacked from")
    assert seen["bytes"] == "bank.f16.npy", (
        "the digest was taken on the retrieval prefix, so that is the file it "
        "must be compared against")


def test_an_unconditioned_bank_asks_both_of_the_same_file(world, monkeypatch):
    """No sidecar, no split: `bytes_path` falls back to `street_path`."""
    tmp, ck, cache = world
    seen = {}
    monkeypatch.setattr(knnmeta, "check_ext",
                        lambda z, path, what: seen.update(ext=Path(path).name))
    monkeypatch.setattr(knnmeta, "check_bytes",
                        lambda z, path, what, d=None: seen.update(
                            bytes=Path(path).name))
    knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")
    assert seen == {"ext": "bank.f16.npy", "bytes": "bank.f16.npy"}
