"""Two arms must address the same photographs, not merely the same count.

`knn_gap` scores BOTH arms against `--a`'s coordinates, which is correct only
if `--b` searched the same corpus. It compared array lengths -- and all four
bank extensions on disk hold exactly 750,000 rows, so two arms built over
different ones agree on every count while addressing different photographs.
Each of B's neighbours then gets another photograph's coordinates and the
difference is reported as a result.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import provenance as prov  # noqa: E402


@pytest.fixture
def release(tmp_path, monkeypatch):
    """A release parquet just large enough for `main` to reach the guard."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    import config
    import knn_gap

    pq.write_table(pa.table({
        "image_id": pa.array([1, 2, 3, 4], pa.int64()),
        "lat": pa.array([0.0, 1.0, 2.0, 3.0]),
        "lon": pa.array([0.0, 1.0, 2.0, 3.0]),
        "split_sequence": pa.array(["train", "train", "test", "train"]),
    }), tmp_path / "d.parquet")
    monkeypatch.setattr(config, "DATASET_PARQUET", tmp_path / "d.parquet")
    monkeypatch.setattr(knn_gap.config, "DATASET_PARQUET", tmp_path / "d.parquet")
    return knn_gap


def run(mod, ids_a, ids_b, monkeypatch):
    lat = np.zeros(len(ids_a))
    seq = iter([(lat, lat, np.asarray(ids_a)), (lat, lat, np.asarray(ids_b))])
    monkeypatch.setattr(mod, "coords", lambda ds, name: next(seq))
    monkeypatch.setattr(sys, "argv", ["knn_gap.py", "--a", "a.npz",
                                      "--b", "b.npz", "--ranks", "1"])
    mod.main()


def test_equal_length_but_different_corpora_is_refused(release, monkeypatch):
    """The reproduction: same counts, different photographs."""
    with pytest.raises(SystemExit, match="different corpora"):
        run(release, [10, 11, 12], [10, 11, 99], monkeypatch)


def test_a_reordered_corpus_is_refused(release, monkeypatch):
    """Order is identity here: bank row *k* is whatever id sits at position k,
    so a permutation pairs every neighbour with a different photograph."""
    with pytest.raises(SystemExit, match="different corpora"):
        run(release, [10, 11, 12], [10, 12, 11], monkeypatch)


def test_the_digest_distinguishes_equal_length_id_arrays():
    """The property the guard rests on, asserted rather than assumed."""
    a = np.asarray([10, 11, 12])
    assert prov.rows_digest(a) == prov.rows_digest(np.asarray([10, 11, 12]))
    assert prov.rows_digest(a) != prov.rows_digest(np.asarray([10, 11, 99]))
    assert prov.rows_digest(a) != prov.rows_digest(np.asarray([10, 12, 11]))
