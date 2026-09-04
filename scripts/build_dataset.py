"""Shard metadata -> dataset.parquet + targets.parquet.

Two things matter here beyond plumbing:

  * OSV-5M images come in *sequences* -- consecutive captures along one road, so
    near-duplicate frames.  A random split leaks them across train and test and
    inflates every metric.  Splitting on sequence removes them; every split mode
    is written as its own column and chosen at read time, never here.
  * Targets are read straight off the final tile's base-g digits.  There is no
    per-step bounds arithmetic anywhere.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import provenance as prov
import splits as sp
import tile_math as tm


def shard_ids(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return [int(Path(n).stem) for n in z.namelist() if n.endswith(".jpg")]


def load_meta(csv_path, ids, shard_of):
    """One pass over the 2.9 GB csv for every shard at once.

    The id list is registered as an arrow table rather than inserted row by row:
    at ten shards that is 500k inserts, which dominates the csv scan itself.
    `shard_of` rides along so zip_name survives the join, which reorders rows.
    """
    con = duckdb.connect()
    want = pa.table({"id": pa.array([int(i) for i in ids], pa.int64()),
                     "shard": pa.array([shard_of[int(i)] for i in ids])})
    con.register("want", want)
    q = (
        "SELECT c.id, c.latitude AS lat, c.longitude AS lon, "
        "       c.country, c.sequence, w.shard "
        "FROM read_csv(?, header=true, sample_size=400000) c "
        "JOIN want w ON w.id = c.id "
        "WHERE c.latitude IS NOT NULL AND c.longitude IS NOT NULL"
    )
    res = con.execute(q, [csv_path.as_posix()])
    tbl = res.fetch_arrow_table()
    con.close()
    return tbl


def cell_ids(lat, lon, z):
    """z-cell id per image, reported only -- it no longer decides the split."""
    out = np.empty(len(lat), dtype=np.int64)
    for i in range(len(lat)):
        _, x, y = tm.tile_for(lat[i], lon[i], z)
        out[i] = x * (1 << z) + y
    return out


def build_targets(ids, lat, lon, g, steps):
    """Five rows per image: four policy steps plus the click."""
    cols = {k: [] for k in ("image_id", "step", "tile_z", "tile_x", "tile_y",
                            "target_action", "u", "v", "x0", "y0")}
    for i in range(len(ids)):
        acts = tm.target_actions(lat[i], lon[i], g, steps)
        for t in range(steps + 1):
            z, x, y = tm.prefix_tile(lat[i], lon[i], t, g, steps)
            nx, ny = tm.norm_corner(z, x, y)
            click = (t == steps)
            u, v = tm.target_uv(lat[i], lon[i], z, x, y) if click else (-1.0, -1.0)
            cols["image_id"].append(int(ids[i]))
            cols["step"].append(t)
            cols["tile_z"].append(z)
            cols["tile_x"].append(x)
            cols["tile_y"].append(y)
            cols["target_action"].append(-1 if click else acts[t])
            cols["u"].append(u)
            cols["v"].append(v)
            cols["x0"].append(nx)
            cols["y0"].append(ny)
    return pa.table({
        "image_id": pa.array(cols["image_id"], pa.int64()),
        "step": pa.array(cols["step"], pa.int8()),
        "tile_z": pa.array(cols["tile_z"], pa.int8()),
        "tile_x": pa.array(cols["tile_x"], pa.int32()),
        "tile_y": pa.array(cols["tile_y"], pa.int32()),
        "target_action": pa.array(cols["target_action"], pa.int16()),
        "u": pa.array(cols["u"], pa.float32()),
        "v": pa.array(cols["v"], pa.float32()),
        "x0": pa.array(cols["x0"], pa.float64()),
        "y0": pa.array(cols["y0"], pa.float64()),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=config.SHARD,
                    help="one shard, or a comma separated list: 00,01,02")
    ap.add_argument("--g", type=int, default=tm.G)
    ap.add_argument("--steps", type=int, default=tm.STEPS)
    a = ap.parse_args()

    shards = [s.strip() for s in a.shard.split(",") if s.strip()]
    ids, shard_of = [], {}
    for sh in shards:
        zp = config.TRAIN_ZIPS / (sh + ".zip")
        got = shard_ids(zp)
        for i in got:
            shard_of[int(i)] = sh
        ids.extend(got)
        print("shard      {}  {:,} images".format(zp.name, len(got)))
    print("images     {:,} across {} shard(s)".format(len(ids), len(shards)))

    tbl = load_meta(config.TRAIN_CSV, ids, shard_of)
    print("metadata   {:,} rows matched in train.csv".format(tbl.num_rows))

    image_id = np.asarray(tbl["id"])
    lat = np.asarray(tbl["lat"], dtype=np.float64)
    lon = np.asarray(tbl["lon"], dtype=np.float64)
    country = np.asarray(tbl["country"].to_pylist(), dtype=object)
    seq = np.asarray(tbl["sequence"].to_pylist(), dtype=object)
    row_shard = tbl["shard"].to_pylist()

    # Every mode is written as its own column and selected at read time; the
    # builder does not get to decide which one is the benchmark.  See splits.py.
    cell = cell_ids(lat, lon, config.SPLIT_CELL_ZOOM)
    modes = {m: sp.assign(m, lat, lon, seq, country) for m in sorted(sp.MODES)}
    split = modes[sp.PRIMARY]

    ds = pa.table({
        "image_id": pa.array(image_id, pa.int64()),
        "zip_name": pa.array([sh + "/" + str(i) + ".jpg"
                              for sh, i in zip(row_shard, image_id)]),
        "lat": pa.array(lat, pa.float64()),
        "lon": pa.array(lon, pa.float64()),
        "country": pa.array(country.tolist()),
        "sequence": pa.array(seq.tolist()),
        "cell_z8": pa.array(cell, pa.int64()),
        "split": pa.array(split.tolist()),
        **{sp.column(m): pa.array(v.tolist()) for m, v in modes.items()},
    })
    pq.write_table(ds, config.DATASET_PARQUET)
    # The authority for row order in this release. Everything downstream is
    # checked against this digest rather than against another derived file: a
    # chain of pairwise checks can be internally consistent and collectively
    # wrong.
    _ids = np.asarray(ds["image_id"])
    prov.write(config.DATASET_PARQUET, _ids, release=config.RELEASE,
               row_space="the release")

    tg = build_targets(image_id, lat, lon, a.g, a.steps)
    pq.write_table(tg, config.TARGETS_PARQUET)
    # Written in the same breath as the dataset, and that is the whole point:
    # the two are paired by row position downstream, and rebuilding one alone
    # gives every image another image's zoom path with no symptom.
    prov.write(config.TARGETS_PARQUET, _ids, release=config.RELEASE,
               row_space="the release")

    print("\nwrote      dataset.parquet  {:,} images".format(ds.num_rows))
    print("wrote      targets.parquet  {:,} rows ({} per image)".format(
        tg.num_rows, a.steps + 1))

    vals, cnt = np.unique(split, return_counts=True)
    print("\nsplit      " + "  ".join(
        "{} {:,} ({:.1f}%)".format(v, c, 100 * c / len(split))
        for v, c in zip(vals, cnt)))
    print("mode       {} (primary)   hash {}".format(
        sp.PRIMARY, sp.split_hash(sp.PRIMARY, split)))
    print("modes      " + ", ".join(
        "{} {}".format(m, sp.split_hash(m, v)) for m, v in sorted(modes.items())))
    print("cells      {:,} distinct z{}".format(
        len(np.unique(cell)), config.SPLIT_CELL_ZOOM))
    print("sequences  {:,}".format(len(set(seq.tolist()))))
    cv, cc = np.unique(country, return_counts=True)
    order = cc.argsort()[::-1]
    print("countries  {} distinct; top: ".format(len(cv)) +
          ", ".join("{} {:,}".format(cv[i], cc[i]) for i in order[:8]))


if __name__ == "__main__":
    main()
