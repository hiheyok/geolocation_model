"""`gain_growth` must refuse caches that are not comparable.

The script reports a difference of two differences. Every term has to come
from the same queries, and each pair has to come from the same bank, or the
number is not a weaker result -- it is a meaningless one that still prints
"separated".

The first version checked none of it. Review fed it mismatched caches and got
**+100 pp, separated** out of a script whose entire purpose is to stop exactly
that, while `knn_gap` rejected the same inputs. These are that probe, driven
through `main`.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import gain_growth as G  # noqa: E402
import splits as sp  # noqa: E402

N, K = 8, 4
# Filled by the fixture once the synthetic dataset exists.
CURRENT_HASH = [None]


def write_cache(path, idx, bank_rows, split_mode="sequence", split_hash=None,
                drop=()):
    """A cache. `split_hash=None` means the one the dataset actually has.

    The first version of this file hard-coded "h1" for every cache, so the
    four agreed with each other and with nothing else -- which is precisely
    the hole review then found in the script: agreeing with each other is not
    agreeing with the split whose rows get sliced.
    """
    if split_hash is None:
        split_hash = CURRENT_HASH[0]
    fields = {"idx": idx, "bank_rows": bank_rows,
              "bank_n": np.array(len(bank_rows)),
              "split_mode": np.array(split_mode),
              "split_hash": np.array(split_hash)}
    for k in drop:
        fields.pop(k)
    np.savez(path, **fields)


@pytest.fixture
def caches(tmp_path, monkeypatch):
    """Four synthetic neighbour tables and the release they index."""
    monkeypatch.setattr(G.config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(G.K.config, "STREET_CACHE", tmp_path)
    table = pa.table({
        "image_id": pa.array(list(range(N)), pa.int64()),
        "lat": pa.array([float(i) for i in range(N)]),
        "lon": pa.array([0.0] * N),
        "split_sequence": pa.array(["test"] * N),
    })
    pq.write_table(table, tmp_path / "d.parquet")
    for mod in (G.config, G.K.config):
        monkeypatch.setattr(mod, "DATASET_PARQUET", tmp_path / "d.parquet")
    CURRENT_HASH[0] = str(sp.read(table, "sequence")[1])

    rows = np.arange(N)
    perfect = np.tile(np.arange(N).reshape(-1, 1), (1, K))     # self = 0 km
    wrong = np.full((N, K), N - 1)                             # far away
    names = {}
    for tag, idx in (("small_a", wrong), ("small_b", wrong),
                     ("big_a", wrong), ("big_b", perfect)):
        names[tag] = tag + ".npz"
        write_cache(tmp_path / names[tag], idx, rows)
    return tmp_path, names


def run(names, monkeypatch, split_mode="sequence"):
    argv = ["gain_growth.py", "--split-mode", split_mode]
    for k in ("small-a", "small-b", "big-a", "big-b"):
        argv += ["--" + k, names[k.replace("-", "_")]]
    monkeypatch.setattr(sys, "argv", argv)
    return G.main()


def test_comparable_caches_are_accepted(caches, monkeypatch):
    """The happy path, so the guards below are not passing vacuously."""
    _, names = caches
    assert run(names, monkeypatch) == 0


def test_a_pair_over_different_bank_rows_is_refused(caches, monkeypatch):
    """The reported defect: mismatched banks produced +100 pp, separated."""
    tmp, names = caches
    write_cache(tmp / names["big_b"], np.tile(np.arange(N).reshape(-1, 1),
                                              (1, K)),
                np.arange(N)[::-1])                    # same length, not same
    with pytest.raises(SystemExit) as e:
        run(names, monkeypatch)
    assert "different bank rows" in str(e.value)


def test_a_pair_with_a_different_bank_size_is_refused(caches, monkeypatch):
    tmp, names = caches
    write_cache(tmp / names["small_b"], np.zeros((N, K), int),
                np.arange(N - 1))
    with pytest.raises(SystemExit) as e:
        run(names, monkeypatch)
    assert "bank_n differs" in str(e.value)


def test_caches_from_a_different_split_are_refused(caches, monkeypatch):
    """--split-mode decides the test rows; the caches must agree with it.

    All four are changed together, because a pair that disagrees with each
    other is already refused by `check_pair`. The case this isolates is four
    mutually consistent caches that agree with each other and not with the
    split whose test rows the script is about to slice.
    """
    tmp, names = caches
    for tag in names:
        write_cache(tmp / names[tag], np.zeros((N, K), int), np.arange(N),
                    split_mode="cell8")
    with pytest.raises(SystemExit) as e:
        run(names, monkeypatch)
    assert "was built on split" in str(e.value)


def test_two_different_splits_of_the_same_mode_are_refused(caches,
                                                           monkeypatch):
    """Same name, different draw: 'the same queries' stops being true."""
    tmp, names = caches
    write_cache(tmp / names["big_a"], np.zeros((N, K), int), np.arange(N),
                split_hash="h2")
    write_cache(tmp / names["big_b"], np.zeros((N, K), int), np.arange(N),
                split_hash="h2")
    with pytest.raises(SystemExit) as e:
        run(names, monkeypatch)
    assert "not paired" in str(e.value)


def test_a_cache_too_old_to_record_its_bank_is_refused(caches, monkeypatch):
    """Five legacy tables on disk predate `bank_rows`."""
    tmp, names = caches
    write_cache(tmp / names["small_a"], np.zeros((N, K), int), np.arange(N),
                drop=("bank_rows",))
    with pytest.raises(SystemExit) as e:
        run(names, monkeypatch)
    assert "records no bank_rows" in str(e.value)


def test_caches_built_against_an_older_split_are_refused(caches, monkeypatch):
    """The reported defect: four caches can agree with each other and be stale.

    Nothing about the caches changes here -- they are mutually consistent, the
    same bank, the same split MODE. Only the dataset's assignment moved, which
    is what `sp.read`'s second return value exists to detect and what the
    script was discarding.
    """
    tmp, names = caches
    for tag in names:
        write_cache(tmp / names[tag], np.zeros((N, K), int), np.arange(N),
                    split_hash="a-split-from-last-week")
    with pytest.raises(SystemExit) as e:
        run(names, monkeypatch)
    assert "not the ones they indexed" in str(e.value)


def test_the_fixture_uses_the_dataset_own_hash(caches):
    """Guards the fixture: a hard-coded hash makes the test above vacuous."""
    tmp, names = caches
    assert CURRENT_HASH[0] and CURRENT_HASH[0] != "h1"
    z = np.load(tmp / names["small_a"])
    assert str(z["split_hash"]) == CURRENT_HASH[0]
