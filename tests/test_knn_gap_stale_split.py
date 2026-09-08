"""Two caches agreeing with each other is not agreeing with the split.

`knn_gap` compares `split_mode`, `split_hash` and `bank_rows` between its two
tables, then slices the test rows out of the CURRENT dataset. Both caches can
carry one old hash while the split has since been reassigned, and then every
query is scored against neighbours that were retrieved for a different query
set -- with no complaint, because the two agree.

`sp.read` returns the hash of the split it just computed, for exactly this.
The line took `[0]` and dropped it. Review found it in `scripts/gain_growth.py`,
which had inherited the line from here.
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

import knn_gap as K  # noqa: E402
import splits as sp  # noqa: E402

N, KK = 8, 4


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """Two comparable caches over a synthetic release, and its real hash."""
    monkeypatch.setattr(K.config, "STREET_CACHE", tmp_path)
    table = pa.table({
        "image_id": pa.array(list(range(N)), pa.int64()),
        "lat": pa.array([float(i) for i in range(N)]),
        "lon": pa.array([0.0] * N),
        "split_sequence": pa.array(["test"] * N),
    })
    pq.write_table(table, tmp_path / "d.parquet")
    monkeypatch.setattr(K.config, "DATASET_PARQUET", tmp_path / "d.parquet")
    current = str(sp.read(table, "sequence")[1])

    def write(name, split_hash):
        np.savez(tmp_path / name,
                 idx=np.tile(np.arange(N).reshape(-1, 1), (1, KK)),
                 bank_rows=np.arange(N), bank_n=np.array(N),
                 split_mode=np.array("sequence"),
                 split_hash=np.array(split_hash))

    return write, current


def run(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["knn_gap.py", "--a", "a.npz", "--b", "b.npz",
                         "--ranks", "1"])
    return K.main()


def test_caches_matching_the_current_split_are_accepted(bench, monkeypatch):
    """The happy path, so the guard below is not passing vacuously."""
    write, current = bench
    write("a.npz", current)
    write("b.npz", current)
    run(monkeypatch)


def test_two_caches_agreeing_on_a_stale_split_are_refused(bench, monkeypatch):
    """Both consistent with each other, neither with the dataset."""
    write, _ = bench
    write("a.npz", "a-split-from-last-week")
    write("b.npz", "a-split-from-last-week")
    with pytest.raises(SystemExit) as e:
        run(monkeypatch)
    assert "not the ones they indexed" in str(e.value)
