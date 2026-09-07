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
import provenance as prov                        # noqa: E402
import splits as sp                              # noqa: E402
from tile_pool import paired                     # noqa: E402
from query_only import great_circle              # noqa: E402

THRESH = (1, 25, 200, 750, 2500)


def coords(ds, name):
    """(lat, lon, ids) over the bank a cache searched: release, then extension.

    The ids come back so the caller can compare CORPORA rather than lengths.
    Every extension on disk holds exactly 750,000 rows, so two arms built over
    different ones agree on every count while addressing different
    photographs.

    `idx` holds bank rows, and with a `--bank-ext` those run PAST the release:
    the caches this compares index 1,250,000 rows against a 500,000-row
    release, so every extension neighbour raised IndexError and the final
    measurement of a four-hour tile pass never ran. The cell8 gate passed only
    because it compares un-extended caches.

    The extension's true coordinates are in its parquet, so they are joined by
    `image_id` rather than recovered from the metadata's z16 addresses -- a
    z16 cell is 611 m across and half of that is a large fraction of the 1 km
    threshold this script reports.
    """
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    ids = np.asarray(ds["image_id"])
    z = np.load(config.STREET_CACHE / name, allow_pickle=True)
    ext = str(z["bank_ext"]) if "bank_ext" in z.files else ""
    if not ext:
        return lat, lon, ids
    m = prov.bank_ext(ext, config.RELEASE)
    want = np.asarray(m["image_id"])
    t = pq.read_table(config.PROCESSED / (ext + ".parquet"),
                      columns=["image_id", "lat", "lon"])
    have = np.asarray(t["image_id"])
    order = np.argsort(have)
    pos = order[np.searchsorted(have, want, sorter=order)]
    if not np.array_equal(have[pos], want):
        raise SystemExit(
            "{}.parquet does not contain every id {} names, so the extension's "
            "coordinates cannot be joined to its bank rows"
            .format(ext, config.bank_meta(ext).name))
    return (np.concatenate([lat, np.asarray(t["lat"], np.float64)[pos]]),
            np.concatenate([lon, np.asarray(t["lon"], np.float64)[pos]]),
            np.concatenate([ids, want]))


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
    ap.add_argument("--gate", action="store_true",
                    help="can any observable signal pick the better arm per "
                         "query? The oracle that picks correctly every time is "
                         "worth far more than the flat blend, so the question "
                         "is whether the choice is predictable without the "
                         "answer.")
    ap.add_argument("--flows", action="store_true",
                    help="split each net gain into the queries it wins and "
                         "the ones it loses. A net figure cannot tell a small "
                         "consistent shift from the residue of two large "
                         "opposing flows, and those imply very different "
                         "things about whether an agent can use the change.")
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    lat, lon, ids_a = coords(ds, a.a)
    _lat_b, _lon_b, ids_b = coords(ds, a.b)
    # Ordered identities, not lengths. Both arms are scored against `lat`/`lon`
    # -- A's -- so if B addressed a different corpus every one of its
    # neighbours would be given another photograph's coordinates and the
    # difference would be reported as a result. A length check cannot see it:
    # all four extensions on disk hold exactly 750,000 rows, which is the same
    # reason `knnmeta.check_ext` compares digests rather than counts.
    da, db = prov.rows_digest(ids_a), prov.rows_digest(ids_b)
    if da != db:
        raise SystemExit(
            "--a searched {:,} bank rows digesting {} and --b searched {:,} "
            "digesting {}. These are different corpora, and both arms are "
            "scored against --a's coordinates, so the comparison would be "
            "between one arm's neighbours and another arm's geography."
            .format(len(ids_a), da, len(ids_b), db))
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

    if a.gate:
        za_, zb_ = np.load(config.STREET_CACHE / a.a), np.load(
            config.STREET_CACHE / a.b)
        sa = np.asarray(za_["sim"][te][:, :4], np.float32)
        sb = np.asarray(zb_["sim"][te][:, :4], np.float32)
        ea, _ = load(a.a, lat, lon, te, 1)
        eb, _ = load(a.b, lat, lon, te, 1)
        t = 25.0
        ha, hb = ea < t, eb < t
        n = len(te)
        print("--- can a gate beat the flat blend? top-1, <{:g} km ---"
              .format(t))
        print("  baseline {:.1f}%   flat blend {:.1f}%   oracle {:.1f}%".format(
            100 * ha.mean(), 100 * hb.mean(), 100 * np.maximum(ha, hb).mean()))
        rng2 = np.random.default_rng(0)
        # Cosines from two different representations are not comparable in
        # scale, so the raw rule is joined by two scale-free ones: each arm's
        # own percentile, and the margin between its top-1 and its rank-4.
        ranks = lambda v: np.argsort(np.argsort(v)) / max(n, 1)
        rules = [("pick higher raw top-1 cosine", sb[:, 0] > sa[:, 0]),
                 ("pick higher within-arm percentile",
                  ranks(sb[:, 0]) > ranks(sa[:, 0])),
                 ("pick larger top1-minus-top4 margin",
                  (sb[:, 0] - sb[:, 3]) > (sa[:, 0] - sa[:, 3]))]
        for name, pick in rules:
            h = np.where(pick, eb, ea) < t
            lo, hi = paired(hb, h, rng2)
            print("  {:<38} {:6.1f}%  {:+.2f} [{:+.1f},{:+.1f}]{}".format(
                name, 100 * h.mean(), 100 * (h.mean() - hb.mean()), lo, hi,
                "" if lo * hi > 0 else " ~"))
        print()

    if a.flows:
        for ranks in (int(r) for r in a.ranks.split(",")):
            ea, _ = load(a.a, lat, lon, te, ranks)
            eb, _ = load(a.b, lat, lon, te, ranks)
            print("--- flows, any-of-{} ---".format(ranks))
            for t in THRESH[:3]:
                ha, hb = ea < t, eb < t
                win, loss = int((~ha & hb).sum()), int((ha & ~hb).sum())
                n = len(te)
                print("  <{:>4g} km  won {:>6,} ({:.2f}%)  lost {:>6,} "
                      "({:.2f}%)  net {:+.2f} pp  ratio {:.2f}x".format(
                          t, win, 100 * win / n, loss, 100 * loss / n,
                          100 * (win - loss) / n, win / max(loss, 1)))
            moved = (ea >= 25) & (eb < 25)
            if moved.any():
                # Are the wins rescues or refinements? A representation that
                # rescues 6,000 km errors is doing something different from one
                # that nudges 30 km errors under the line, and only the second
                # is what "finer features" is usually taken to mean.
                print("  of the {:,} queries crossing 25 km, the baseline's "
                      "error was median {:,.0f} km, p90 {:,.0f} km".format(
                          int(moved.sum()), float(np.median(ea[moved])),
                          float(np.percentile(ea[moved], 90))))
            print()

    print("~ spans zero. {:,} bank rows, same rows in both tables."
          .format(int(za["bank_n"])))


if __name__ == "__main__":
    main()
