"""Does the tiles gain follow bank density *within* one bank?

The case for a bigger tiled corpus rests on an extrapolation. `runs/XBANK.md`
§3 subsampled the bank -- 25k, 50k, 100k, 200k, 400k rows -- and the matched
crops+tiles gain grew at every step, +1.80 pp to +2.60. The reading offered was
mechanistic: a sparse bank is limited by **coverage**, where no representation
can invent a match, and a dense bank by **discrimination**, which is what finer
features supply. That reading is what justifies expecting more at 3.4M than the
+2.03 pp measured at 400k.

But subsampling changes one thing globally, and four points along one axis is a
thin basis for a claim about mechanism. If density is really the variable, the
gain should also track density **inside a single fixed bank**: queries landing
in well-covered places should gain more than queries in empty ones, with the
bank, the split, the encoders and the projection all held constant.

That is a stronger test than the subsample series, because nothing varies
except where the query happens to be. It is also free -- the neighbour tables
already exist, so this is a lookup and a histogram, no GPU and no encoder.

Density is counted as bank rows sharing the query's cell on a coarse
equirectangular grid. The grid is deliberately crude: the question is which
order of magnitude of local coverage a query sits in, not a kernel estimate.

    OSV_RELEASE=s10 py scripts/gain_density.py \
        --a knn_pyr768_l0_sequence_k32.npz --b knn_pyr768_mix_sequence_k32.npz
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


def cell_of(lat, lon, deg):
    """Coarse equirectangular cell id. Crude on purpose -- see the docstring."""
    la = np.floor((lat + 90.0) / deg).astype(np.int64)
    lo = np.floor((lon + 180.0) / deg).astype(np.int64)
    return la * (int(360.0 / deg) + 1) + lo


def err_of(name, lat, lon, te, ranks):
    z = np.load(config.STREET_CACHE / name)
    src = z["idx"][te][:, :ranks]
    e = great_circle(np.repeat(lat[te], ranks), np.repeat(lon[te], ranks),
                     lat[src].ravel(), lon[src].ravel()).reshape(len(te), ranks)
    return e.min(1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--ranks", type=int, default=16,
                    help="the agent's retrieval window; 1 for pure top-1")
    ap.add_argument("--deg", type=float, default=1.0, help="grid cell degrees")
    ap.add_argument("--thresh", type=float, default=25.0)
    ap.add_argument("--split-mode", default="sequence")
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    te = np.flatnonzero(sp.read(ds, a.split_mode)[0] == "test")

    ea, za = err_of(a.a, lat, lon, te, a.ranks)
    eb, zb = err_of(a.b, lat, lon, te, a.ranks)
    if not np.array_equal(za["bank_rows"], zb["bank_rows"]):
        sys.exit("the two tables index different bank rows")
    bank = za["bank_rows"]

    cell = cell_of(lat, lon, a.deg)
    # How many bank rows share each query's cell. The query itself is a test
    # row and so is never in the bank, but its own sequence's frames may be --
    # build_knn excludes those from the neighbour search, not from the count,
    # so this is coverage of the place, not of the drive.
    counts = np.bincount(cell[bank], minlength=cell.max() + 1)
    dens = counts[cell[te]]

    edges = [0, 1, 10, 100, 1000, 10000, 10 ** 9]
    print("{:,} test queries, {:,} bank rows, {:g}-degree cells, "
          "any-of-{}, <{:g} km\n".format(
              len(te), len(bank), a.deg, a.ranks, a.thresh))
    hdr = "%-22s %8s %9s %9s %s" % ("bank rows in cell", "queries",
                                    "crops", "+tiles", "gain pp")
    print(hdr); print("-" * len(hdr))
    rng = np.random.default_rng(0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dens >= lo) & (dens < hi)
        if m.sum() < 50:
            continue
        ha = (ea[m] < a.thresh).mean()
        hb = (eb[m] < a.thresh).mean()
        clo, chi = paired(ea[m] < a.thresh, eb[m] < a.thresh, rng)
        print("%-22s %8s %8.1f%% %8.1f%% %+6.2f [%+.1f,%+.1f]%s" % (
            "{:,} - {:,}".format(lo, hi - 1) if hi < 10 ** 9
            else "{:,}+".format(lo),
            "{:,}".format(int(m.sum())), 100 * ha, 100 * hb,
            100 * (hb - ha), clo, chi, "" if clo * chi > 0 else " ~"))

    print("\n~ spans zero. If the gain tracks density inside one fixed bank, "
          "the mechanism behind the 25k->400k series is coverage giving way "
          "to discrimination, and extrapolating past 400k is reasonable. If "
          "it is flat, the subsample series was measuring something else and "
          "the extrapolation is unsupported.")


if __name__ == "__main__":
    main()
