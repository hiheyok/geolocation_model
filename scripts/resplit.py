"""Write every split mode into dataset.parquet as its own column.

Row order in dataset.parquet is what aligns the street embedding memmaps, so
this rewrites columns and never rebuilds the table from the CSV.

Storing all modes side by side is the fix for the defect that prompted it: the
benchmark used to be whichever mode was generated last, so switching to the
stress test destroyed the primary split and every checkpoint's meaning with it.
Now a mode is *selected* at read time and none of them overwrite another.

The existing ``split`` column is treated as the benchmark of record.  If a
regenerated mode disagrees with it the script refuses to write, because that
disagreement is exactly the drift this is meant to catch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import splits as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(sorted(sp.MODES)),
                    help="comma separated; default writes all of them")
    ap.add_argument("--adopt", metavar="MODE",
                    help="point the legacy 'split' column at MODE; only needed "
                         "to change which mode is the benchmark of record")
    ap.add_argument("--force", action="store_true",
                    help="write even if a regenerated mode disagrees with the "
                         "existing 'split' column")
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    seq = ds["sequence"].to_pylist()
    country = np.asarray(ds["country"].to_pylist(), dtype=object)
    legacy = (np.asarray(ds["split"].to_pylist(), dtype=object)
              if "split" in ds.schema.names else None)

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    cols = {n: ds[n] for n in ds.schema.names}
    made = {}
    for m in modes:
        lab = sp.assign(m, lat, lon, seq, country)
        made[m] = lab
        cols[sp.column(m)] = pa.array(lab.tolist())

    agree = {m: float((made[m] == legacy).mean()) for m in made} if legacy is not None else {}
    match = [m for m, v in agree.items() if v == 1.0]

    print("rows       {:,}   row order preserved".format(ds.num_rows))
    print("\n mode        groups     train      val     test    hash        vs legacy 'split'")
    for m in modes:
        lab = made[m]
        n_g = len(np.unique(sp.group_ids(m, lat, lon, seq)))
        c = {s: int((lab == s).sum()) for s in ("train", "val", "test")}
        vs = "{:.1f}%".format(100 * agree[m]) if agree else "-"
        print("  {:<10} {:>7,}  {:>8,} {:>8,} {:>8,}   {}   {:>7}"
              .format(m, n_g, c["train"], c["val"], c["test"],
                      sp.split_hash(m, lab), vs))

    if legacy is not None:
        if match:
            print("\nbenchmark  the existing 'split' column is exactly {!r} -- every "
                  "checkpoint\n           trained so far stays valid and comparable."
                  .format(match[0]))
        elif not a.force and not a.adopt:
            raise SystemExit(
                "\nREFUSING TO WRITE: no regenerated mode reproduces the existing "
                "'split' column\n(best match {:.1f}%).  That means the benchmark on "
                "disk came from code that no\nlonger exists, and overwriting it would "
                "silently redefine every result.\nInspect first; pass --force only if "
                "you intend to replace the benchmark."
                .format(100 * max(agree.values())))

    if a.adopt:
        cols["split"] = pa.array(made[a.adopt].tolist())
        print("\nadopted    'split' now mirrors {!r}".format(a.adopt))

    pq.write_table(pa.table(cols), config.DATASET_PARQUET)
    print("\nwrote      {} split columns to {}"
          .format(len(modes), config.DATASET_PARQUET.name))


if __name__ == "__main__":
    main()
