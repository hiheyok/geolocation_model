"""Review item 61, measured: same-sequence exclusion never reached the bank.

`build_knn` drops bank rows sharing the query's sequence, because OSV-5M
captures run consecutively along a road and the benchmark splits on sequence --
so a same-drive frame in the bank is a near-duplicate of the query, and serving
it as a retrieval neighbour hands the model the answer.

The extension's sequence ids were offset into a disjoint namespace
(`ext_seq + seq_id.max() + 1`), which made the two corpora disjoint *by
construction* rather than by fact.  They are not disjoint: the extension is
more OSV-5M shards, with the same sequence naming.

This measures what that cost, using only artifacts already on disk.  The
corrected top-1 is simulated by walking each query's cached top-32 and taking
the first neighbour that a working exclusion would have left -- which is
exactly what the fixed `build_knn` will retrieve, since the ranking within the
surviving set does not change.

    OSV_RELEASE=s10 python scripts/seqleak.py
    OSV_RELEASE=s10 python scripts/seqleak.py --k 16 --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import provenance as prov
import splits as sp

R = 6371.0088


def great_circle(a1, o1, a2, o2):
    p = np.pi / 180
    d = (np.sin((a2 - a1) * p / 2) ** 2
         + np.cos(a1 * p) * np.cos(a2 * p) * np.sin((o2 - o1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


def ext_latlon(m):
    """The extension stores a z16 address, not coordinates; take cell centres.

    Worth naming: this rounds to at most ~300 m at the equator, which is the
    same order as the effect being measured, so it is a floor on the leaked
    neighbours' closeness rather than an exact figure. It cannot manufacture
    the effect -- rounding moves a point away from the truth, never toward it.
    """
    n16 = 1 << 16
    x = m["x16"].astype(np.float64) + 0.5
    y = m["y16"].astype(np.float64) + 0.5
    lon = x / n16 * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n16))))
    return lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn", default="knn_pca768_bank70_sequence_k32_"
                                     "bank_ext70.npz")
    ap.add_argument("--verify", action="store_true",
                    help="assert the cache is clean and exit non-zero if not; "
                         "for checking a rebuild rather than measuring a leak")
    ap.add_argument("--ext", default="bank_ext70")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    seq = np.asarray(ds["sequence"].to_pylist()).astype("U40")
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    labels, _ = sp.read(ds, "sequence")

    m = prov.bank_ext(a.ext, config.RELEASE)
    e_seq = np.asarray(m["sequence"]).astype("U40")
    e_lat, e_lon = ext_latlon(m)
    A_seq = np.concatenate([seq, e_seq])
    A_lat = np.concatenate([lat, e_lat])
    A_lon = np.concatenate([lon, e_lon])

    q = np.flatnonzero(labels == a.split)
    idx = np.load(config.STREET_CACHE / a.knn)["idx"][q]
    same = A_seq[idx] == seq[q][:, None]

    print("bank       {}  ({:,} release + {:,} extension rows)".format(
        a.ext, len(seq), len(e_seq)))
    print("queries    {:,} in {!r}, k={} of {} cached\n".format(
        len(q), a.split, a.k, idx.shape[1]))

    sk = same[:, :a.k]
    if a.verify:
        # A rebuild is only done when the number is zero. Checking the whole
        # cached width, not just the k a model happens to read: a leaked row
        # at rank 20 is still a leaked row the moment retr_k rises.
        n_bad = int(same.sum())
        print("verify: {:,} same-sequence neighbours across all {} cached "
              "ranks".format(n_bad, same.shape[1]))
        if n_bad:
            raise SystemExit(
                "cache is NOT clean: {:,} slots over {:,} queries still hold a "
                "same-sequence row".format(n_bad, int(same.any(1).sum())))
        print("verify: clean")
        return
    print("same-sequence neighbours that exclusion should have removed")
    print("  {:.2%} of neighbour slots".format(sk.mean()))
    print("  {:.2%} of queries have at least one".format(sk.any(1).mean()))
    print("  {:.2%} of queries have one as their TOP-1\n".format(sk[:, 0].mean()))

    d = great_circle(lat[q], lon[q], A_lat[idx[:, 0]], A_lon[idx[:, 0]])
    print("top-1 neighbour, distance to the query's true location")
    for name, msk in (("leaked (same sequence)", same[:, 0]),
                      ("legitimate", ~same[:, 0])):
        x = d[msk]
        if not len(x):
            continue
        print("  {:<24} n={:>7,}  median {:8.2f} km   <1km {:6.1%}   "
              "<25km {:6.1%}".format(name, len(x), float(np.median(x)),
                                     float((x < 1).mean()),
                                     float((x < 25).mean())))

    # The corrected top-1: the first cached neighbour a working exclusion
    # leaves. Ranking within the surviving set is unchanged, so this is what
    # the rebuilt cache retrieves -- for every query that still has one.
    first = np.argmax(~same, axis=1)
    none_left = same.all(1)
    ok = ~none_left
    cd = great_circle(lat[q][ok], lon[q][ok],
                      A_lat[idx[ok, first[ok]]], A_lon[idx[ok, first[ok]]])
    print("\ncorrected top-1 (first surviving neighbour of the cached {})"
          .format(idx.shape[1]))
    print("  {:>7,} queries  median {:8.2f} km   <1km {:6.1%}   <25km {:6.1%}"
          .format(len(cd), float(np.median(cd)), float((cd < 1).mean()),
                  float((cd < 25).mean())))
    if none_left.any():
        print("  {:,} queries have no surviving neighbour in the cached {} and "
              "would need a deeper search".format(int(none_left.sum()),
                                                  idx.shape[1]))
    print("\nas cached, all queries")
    print("  {:>7,} queries  median {:8.2f} km   <1km {:6.1%}   <25km {:6.1%}"
          .format(len(d), float(np.median(d)), float((d < 1).mean()),
                  float((d < 25).mean())))


if __name__ == "__main__":
    main()
