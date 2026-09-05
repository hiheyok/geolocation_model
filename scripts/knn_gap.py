"""Is the retrieval gain actually inside the caches the ladders are training on?

`runs/RESMATCH.md` and `runs/XBANK.md` measured crops+tiles on vectors built
inside those scripts. `scripts/tiledrisk.py` trains on `knn_*.npz` files built
by the production path -- `pool_pyramid` -> `project_street` -> `build_knn` --
which involves a PCA to 768 that the probes never applied.

If the paired ladders come back flat, there are two very different reasons:

    the agent cannot use the gain          -- a real and interesting result
    the gain is not in these caches        -- a pipeline bug wearing its costume

This separates them **before** the ladders finish, from the neighbour tables
alone. No GPU, no encoder, no model: the `.npz` already holds 32 neighbour row
ids per query, so the top-1 error is a lookup and a great-circle.

It is scored on the same `sequence` test split the ladders select on, and the
neighbour ids are already same-sequence-excluded by `build_knn`.

    OSV_RELEASE=s10 py scripts/knn_gap.py --a knn_pyr768_l0_sequence_k32.npz \
                                          --b knn_pyr768_mix_sequence_k32.npz
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import splits as sp                              # noqa: E402
from tile_pool import paired                     # noqa: E402
from query_only import great_circle              # noqa: E402

THRESH = (1, 25, 200, 750, 2500)


def load(name, lat, lon, te, ranks):
    """Best great-circle error over the first `ranks` neighbours of each query."""
    z = np.load(config.STREET_CACHE / name)
    # `idx` is already release rows -- build_knn writes `bank_rows[best_j]`,
    # not `best_j` -- so it must not be indexed through `bank_rows` again.
    # `bank_rows` is kept here only to assert the two tables indexed the same
    # corpus.
    src = z["idx"][te][:, :ranks]
    e = great_circle(np.repeat(lat[te], ranks), np.repeat(lon[te], ranks),
                     lat[src].ravel(), lon[src].ravel()).reshape(len(te), ranks)
    return e.min(1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline knn npz")
    ap.add_argument("--b", required=True, help="treatment knn npz")
    ap.add_argument("--ranks", default="1,32",
                    help="comma-separated: top-1, and any-of-k")
    ap.add_argument("--split-mode", default="sequence")
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    te = np.flatnonzero(sp.read(ds, a.split_mode)[0] == "test")
    print("{:,} test queries, split {}\n".format(len(te), a.split_mode))

    for ranks in (int(r) for r in a.ranks.split(",")):
        err, meta = {}, {}
        for name in (a.a, a.b):
            err[name], z = load(name, lat, lon, te, ranks)
            meta[name] = z
        # The comparison is only meaningful if the two tables describe the same
        # bank and the same split; otherwise this measures the difference in
        # what was indexed, not in how it was encoded.
        za, zb = meta[a.a], meta[a.b]
        for k in ("split_mode", "split_hash", "bank_n"):
            if str(za[k]) != str(zb[k]):
                sys.exit("{} differs: {} vs {}".format(k, za[k], zb[k]))
        if not np.array_equal(za["bank_rows"], zb["bank_rows"]):
            sys.exit("the two tables index different bank rows")

        label = "top-1" if ranks == 1 else "any-of-{}".format(ranks)
        hdr = "%-34s %9s %s" % (label, "median km",
                                " ".join("%8s" % ("<%gkm" % t) for t in THRESH))
        print(hdr); print("-" * len(hdr))
        for name in (a.a, a.b):
            e = err[name]
            print("%-34s %9.1f %s" % (
                name.replace("knn_", "").replace("_sequence_k32.npz", ""),
                np.median(e),
                " ".join("%7.1f%%" % (100 * (e < t).mean()) for t in THRESH)))
        rng = np.random.default_rng(0)
        cells = []
        for t in THRESH:
            lo, hi = paired(err[a.a] < t, err[a.b] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[a.b] < t).mean() - (err[a.a] < t).mean()),
                lo, hi, " " if lo * hi > 0 else "~"))
        print("%-34s %9s %s\n" % ("b - a", "", " ".join(cells)))

    print("~ spans zero. {:,} bank rows, same rows in both tables."
          .format(int(za["bank_n"])))


if __name__ == "__main__":
    main()
