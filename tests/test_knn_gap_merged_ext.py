"""A merged extension stem has no parquet, and its coordinates still exist.

`merge_bank_meta.py` makes several extensions look like one to `build_knn`, so
`bank_ext70` has a `_meta.npz` and no parquet -- it was never a harvest, it is
`bank_ext ++ bank_ext2 ++ bank_ext3 ++ bank_ext4`. `knn_gap` read
`{ext}.parquet` unconditionally and died on FileNotFoundError at the last
stage of a run whose two indexes were already built.

The join is by `image_id`, so what matters is that every id the metadata names
is covered exactly once -- not which file it came from, and not the order.
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

import knn_gap  # noqa: E402


def part(path, ids, lat0):
    pq.write_table(pa.table({
        "image_id": pa.array(list(ids), pa.int64()),
        "lat": pa.array([lat0 + i for i in range(len(ids))]),
        "lon": pa.array([-(lat0 + i) for i in range(len(ids))]),
    }), path)


@pytest.fixture
def processed(tmp_path, monkeypatch):
    monkeypatch.setattr(knn_gap.config, "PROCESSED", tmp_path)
    return tmp_path


def test_own_parquet_is_preferred(processed):
    part(processed / "bank_ext.parquet", [1, 2], 10.0)
    part(processed / "bank_ext2.parquet", [3, 4], 20.0)
    ids, lat, lon = knn_gap.ext_coords("bank_ext")
    assert sorted(ids.tolist()) == [1, 2]
    assert sorted(lat.tolist()) == [10.0, 11.0]


def test_merged_stem_falls_back_to_the_parts(processed):
    """bank_ext70.parquet does not exist; its four parts do."""
    part(processed / "bank_ext.parquet", [1, 2], 10.0)
    part(processed / "bank_ext2.parquet", [3, 4], 20.0)
    ids, lat, lon = knn_gap.ext_coords("bank_ext70")
    assert sorted(ids.tolist()) == [1, 2, 3, 4]
    # Coordinates travel with their id, whichever file held it.
    by_id = dict(zip(ids.tolist(), lat.tolist()))
    assert by_id[1] == 10.0 and by_id[3] == 20.0


def test_a_duplicated_id_is_refused(processed):
    """Two files naming one image would resolve to whichever sorted first."""
    part(processed / "bank_ext.parquet", [1, 2], 10.0)
    part(processed / "bank_ext2.parquet", [2, 3], 20.0)
    with pytest.raises(SystemExit) as e:
        knn_gap.ext_coords("bank_ext70")
    assert "more than once" in str(e.value)


def test_no_parquet_at_all_is_refused(processed):
    with pytest.raises(SystemExit) as e:
        knn_gap.ext_coords("bank_ext70")
    assert "no parquet" in str(e.value)


def test_coords_joins_by_id_not_by_position(processed, monkeypatch):
    """The metadata's order need not match any parquet's."""
    part(processed / "bank_ext.parquet", [1, 2], 10.0)
    part(processed / "bank_ext2.parquet", [3, 4], 20.0)

    ds = {"lat": [0.0], "lon": [0.0], "image_id": [99]}
    monkeypatch.setattr(knn_gap.prov, "bank_ext",
                        lambda ext, rel: {"image_id": np.array([4, 1, 3])})
    class Npz(dict):
        files = ["bank_ext"]

    monkeypatch.setattr(knn_gap.np, "load",
                        lambda *a, **k: Npz(bank_ext="bank_ext70"))
    lat, lon, ids = knn_gap.coords(ds, "whatever.npz")
    assert ids.tolist() == [99, 4, 1, 3]
    # id 4 is the second row of bank_ext2 (lat 21.0), id 1 the first of
    # bank_ext (10.0), id 3 the first of bank_ext2 (20.0).
    assert lat.tolist() == [0.0, 21.0, 10.0, 20.0]
