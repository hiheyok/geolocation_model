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

import knnmeta  # noqa: E402


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


def test_a_cache_recording_no_bank_rows_is_refused_not_widened(world):
    """"Which rows it searched is unknown" is not "it searched all of them"."""
    tmp, ck, cache = world
    del cache["bank_rows"]
    np.savez(tmp / "k.npz", **cache)
    with pytest.raises(SystemExit, match="records no bank rows"):
        knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "bank.f16.npy", "t")


def test_a_conditioned_bank_compares_against_its_retrieval_prefix(world):
    """The k-NN is built on the retrieval prefix, not the joined file, so a
    joined bank must not be rejected for naming a different file."""
    import json

    tmp, ck, cache = world
    np.save(tmp / "joined.f16.npy", np.zeros((4, 12), np.float16))
    (tmp / "joined.f16.npy.prov.json").write_text(json.dumps(
        {"retrieval_file": "bank.f16.npy", "rows": 4}), encoding="utf-8")
    rows = knnmeta.bank_for_checkpoint(ck, "k.npz", 4, "joined.f16.npy", "t")
    assert np.array_equal(rows, [0, 1, 3])
