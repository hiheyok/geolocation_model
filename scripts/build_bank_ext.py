"""Metadata for shards that join the retrieval bank without joining training.

The control in the s10 run said the bank is worth about six times the training
set, so the cheap axis to scale is the corpus.  A bank entry needs two things
and neither is a training row:

  * an embedding, so a query can find it;
  * its z16 address, so the retrieval prior knows which child cell it votes for.

The address is pure arithmetic on lat/lon, so **bank-only images need no tiles
fetched at all** -- which is what makes 750k extra images a 2.5 hour job rather
than a day.  They also never enter dataset.parquet, so the s10 benchmark, its
split hash and every checkpoint trained against it stay untouched.

Writes a parquet in the shape embed_street.py expects, plus the metadata the
kNN builder and the dataset need.
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
import tile_math as tm


def shard_ids(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return [int(Path(n).stem) for n in z.namelist() if n.endswith(".jpg")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default=",".join("{:02d}".format(i)
                                                 for i in range(10, 25)),
                    help="shards to add to the bank; must not overlap the release")
    ap.add_argument("--out", default="bank_ext")
    a = ap.parse_args()

    shards = [s.strip() for s in a.shards.split(",") if s.strip()]

    # never let a bank shard overlap the release: a training image appearing
    # twice in the bank would be retrievable as its own neighbour
    ds = pq.read_table(config.DATASET_PARQUET)
    have = set(np.asarray(ds["image_id"]).tolist())
    in_release = {n.split("/")[0] for n in ds["zip_name"].to_pylist()}
    clash = sorted(set(shards) & in_release)
    if clash:
        raise SystemExit("shards {} are already in release {}".format(
            clash, config.RELEASE))

    ids, shard_of = [], {}
    for sh in shards:
        got = shard_ids(config.TRAIN_ZIPS / (sh + ".zip"))
        for i in got:
            shard_of[int(i)] = sh
        ids.extend(got)
        print("shard      {}  {:,} images".format(sh, len(got)), flush=True)

    con = duckdb.connect()
    want = pa.table({"id": pa.array([int(i) for i in ids], pa.int64()),
                     "shard": pa.array([shard_of[int(i)] for i in ids])})
    con.register("want", want)
    tbl = con.execute(
        "SELECT c.id, c.latitude AS lat, c.longitude AS lon, c.country, "
        "       c.sequence, w.shard "
        "FROM read_csv(?, header=true, sample_size=400000) c "
        "JOIN want w ON w.id = c.id "
        "WHERE c.latitude IS NOT NULL AND c.longitude IS NOT NULL",
        [config.TRAIN_CSV.as_posix()]).fetch_arrow_table()
    con.close()

    image_id = np.asarray(tbl["id"])
    lat = np.asarray(tbl["lat"], dtype=np.float64)
    lon = np.asarray(tbl["lon"], dtype=np.float64)
    seq = np.asarray(tbl["sequence"].to_pylist(), dtype=object)
    row_shard = tbl["shard"].to_pylist()
    n = len(image_id)
    print("metadata   {:,} rows matched in train.csv".format(n), flush=True)

    dup = len(set(image_id.tolist()) & have)
    if dup:
        raise SystemExit("{:,} image ids already in the release".format(dup))

    x16 = np.empty(n, dtype=np.int64)
    y16 = np.empty(n, dtype=np.int64)
    for i in range(n):
        _, x, y = tm.tile_for(lat[i], lon[i], 4 * tm.STEPS)
        x16[i] = x
        y16[i] = y

    out_pq = config.PROCESSED / (a.out + ".parquet")
    pq.write_table(pa.table({
        "image_id": pa.array(image_id, pa.int64()),
        "zip_name": pa.array([sh + "/" + str(i) + ".jpg"
                              for sh, i in zip(row_shard, image_id)]),
        "lat": pa.array(lat, pa.float64()),
        "lon": pa.array(lon, pa.float64()),
        "country": pa.array(tbl["country"].to_pylist()),
        "sequence": pa.array(seq.tolist()),
    }), out_pq)

    meta = config.STREET_CACHE / (a.out + "_meta.npz")
    np.savez(meta, image_id=image_id, x16=x16, y16=y16,
             sequence=seq.astype(str), shards=np.array(shards),
             release=config.RELEASE)
    print("\nwrote      {}  {:,} images".format(out_pq.name, n))
    print("wrote      {}  (z16 addresses, sequences)".format(meta.name))
    print("sequences  {:,} distinct".format(len(set(seq.tolist()))))
    print("\nno tiles are needed for these rows: a bank entry votes with its z16")
    print("address, which is arithmetic on lat/lon, not a fetched view.")


if __name__ == "__main__":
    main()
