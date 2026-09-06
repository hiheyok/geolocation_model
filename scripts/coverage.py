"""Is the bottleneck coverage, retrieval, or ranking? (REVIEW8 R0)

`docs/PROJECT_BRIEF.md` asserted that the 57.5% / 13.4% gap between OSV-5M and
KartaView "is bank coverage". REVIEW8 objected, correctly: an aggregate gap
between two benchmarks does not causally exclude domain shift, geographic
composition, or preprocessing differences. The mechanism was asserted, not
decomposed.

This decomposes it. For each query, with radius `r`:

    class 1  COVERAGE   no eligible bank row lies within r of the query
    class 2  RETRIEVAL  one does, but no top-K candidate is within r
    class 3  RANKING    a top-K candidate is within r, but rank 1 is not
    solved              the rank-1 candidate is within r

The partition is exhaustive and every class names a different budget. Class 1
is answered by acquiring corpus; no reranker, no representation and no amount
of test-time compute can touch it. Class 2 is the representation's failure and
is what a better embedding or a larger K buys. Class 3 is ranking, and is the
only class a reranker over the cached shortlist can reach.

`A_r(K)`, the review's shortlist-coverage diagnostic, falls out of the same
computation: the fraction of queries with some top-K candidate within r. It is
an oracle ceiling for any method that must output one of those candidates'
coordinates -- NOT a bound on the agent, which emits a new coordinate, and not
a bound on an independent geographic proposal path.

**The predictor here is retrieval, not the agent.** Rank-1 retrieval scores
56.4% `<25 km` where the full agent scores 57.5%, so this measures the system
that supplies ~98% of the accuracy, and needs no checkpoint, no beam, and no
GPU. An agent-level version would have to join a per-image error export on an
identical cohort, which is a separate protocol question the review also raises.

**Same-sequence rows are excluded from the coverage test too.** They sit in the
bank as legitimate training rows but `build_knn` forbids retrieval from
choosing them, so counting a same-drive frame as available coverage would
report a candidate the system is not allowed to return -- and would make class
1 look far smaller than it is. That exclusion is the entire subject of §1 of
STATE.md.

    OSV_RELEASE=s10 py scripts/coverage.py
    OSV_RELEASE=s10 py scripts/coverage.py --knn knn_pyr768_mix_sequence_k32.npz
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
import knnmeta                                   # noqa: E402
import provenance as prov                        # noqa: E402
import splits as sp                              # noqa: E402
import tile_math as tm                           # noqa: E402

R_KM = 6371.0088
RADII = (1.0, 25.0, 200.0)
KS = (1, 2, 4, 8, 16, 32)


def unit(lat, lon):
    """Lat/lon to 3D unit vectors, so a Euclidean tree gives chord distance.

    Chord and great-circle are monotone in each other, so a k-nearest query on
    chords returns the same ordering; the conversion back is exact. Doing it
    this way avoids a haversine tree and the pole/antimeridian special cases
    that come with one.
    """
    la, lo = np.radians(lat), np.radians(lon)
    c = np.cos(la)
    return np.stack([c * np.cos(lo), c * np.sin(lo), np.sin(la)], 1)


def chord_to_km(d):
    return 2.0 * R_KM * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))


def bank_coords(z, ds, labels):
    """Lat/lon and sequence for every addressable row, release then extension.

    The extension has no lat/lon column -- it carries the z16 address the model
    is trained to predict -- so its coordinates are the centre of its z16 tile,
    which is 611 m wide. That is below the smallest radius reported here and is
    stated rather than hidden: at r=1 km an extension row's position carries up
    to ~430 m of quantisation.
    """
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    seq = np.asarray(ds["sequence"].to_pylist(), dtype=object).astype("U40")
    ext = str(z["bank_ext"]) if "bank_ext" in z.files else ""
    if not ext:
        return lat, lon, seq
    m = prov.bank_ext(ext, config.RELEASE)
    n16 = 1 << (4 * tm.STEPS)
    ex = (np.asarray(m["x16"], np.float64) + 0.5) / n16
    ey = (np.asarray(m["y16"], np.float64) + 0.5) / n16
    elon = ex * 360.0 - 180.0
    elat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * ey))))
    return (np.concatenate([lat, elat]), np.concatenate([lon, elon]),
            np.concatenate([seq, np.asarray(m["sequence"]).astype("U40")]))


def nearest_legal(tree, bank_seq, bank_rows, qv, qseq, probe=64):
    """Great-circle km to the nearest bank row the query is ALLOWED to match.

    Same-sequence rows are in the bank and are forbidden to retrieval, so they
    must not count as coverage. The tree is queried `probe` deep and the first
    neighbour on another drive is taken; a query whose whole neighbourhood is
    its own drive returns inf and is counted separately rather than silently
    treated as uncovered.
    """
    d, j = tree.query(qv, k=probe, workers=-1)
    rows = bank_rows[j]
    legal = bank_seq[rows] != qseq[:, None]
    d = np.where(legal, chord_to_km(d), np.inf)
    best = d.min(1)
    return best, int(np.isinf(best).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn",
                    default="knn_pca768_bank70_sequence_k32_bank_ext70.npz")
    ap.add_argument("--split", default="test")
    ap.add_argument("--probe", type=int, default=64,
                    help="tree depth searched for a different-sequence row")
    ap.add_argument("--out", default=str(ROOT / "runs" / "COVERAGE.md"))
    a = ap.parse_args()

    from scipy.spatial import cKDTree

    ds = pq.read_table(config.DATASET_PARQUET)
    z = np.load(config.STREET_CACHE / a.knn)
    mode = str(z["split_mode"])
    labels, _ = sp.read(ds, mode)
    lat, lon, seq = bank_coords(z, ds, labels)

    rows = knnmeta.bank_rows(z, len(lat), a.knn)
    if rows is None:
        sys.exit("{} records no bank_rows, so there is no eligible bank to "
                 "measure coverage against".format(a.knn))
    bad_rel, bad_ext = sp.ineligible(rows, len(labels), labels, mode,
                                     np.zeros(len(lat)), np.zeros(len(lat)))
    if len(bad_rel):
        sys.exit("{} banks {:,} rows this split holds out".format(
            a.knn, len(bad_rel)))

    q = np.flatnonzero(labels == a.split)
    idx = np.asarray(z["idx"], np.int64)[q]
    print("{}  {}  bank {:,}  queries {:,}  K={}".format(
        a.knn, mode, len(rows), len(q), idx.shape[1]), flush=True)

    print("building the geographic index over the eligible bank...", flush=True)
    tree = cKDTree(unit(lat[rows], lon[rows]))
    qv = unit(lat[q], lon[q])
    near, all_same = nearest_legal(tree, seq, rows, qv, seq[q], a.probe)
    if all_same:
        print("note: {:,} queries have no different-sequence bank row inside "
              "the {}-deep probe; they are counted as uncovered, which is a "
              "floor on class 1 rather than an exact value"
              .format(all_same, a.probe), flush=True)

    # distance to each cached candidate, in rank order
    cand = np.empty(idx.shape, np.float64)
    for s in range(0, len(q), 20000):
        e = min(s + 20000, len(q))
        cv = unit(lat[idx[s:e]].ravel(), lon[idx[s:e]].ravel())
        d = np.linalg.norm(cv - np.repeat(qv[s:e], idx.shape[1], 0), axis=1)
        cand[s:e] = chord_to_km(d).reshape(e - s, idx.shape[1])

    L = []
    L.append("# Where the errors live: coverage, retrieval, or ranking\n")
    L.append("`{}`, split `{}`, {:,} queries against a {:,}-row eligible "
             "bank.\n".format(a.knn, a.split, len(q), len(rows)))
    L.append("The predictor is **rank-1 retrieval**, which scores 56.4% "
             "`<25 km` where the full agent scores 57.5% -- so this measures "
             "the system supplying almost all of the accuracy, with no "
             "checkpoint and no beam. Same-sequence rows are excluded from "
             "the coverage test as well as from retrieval.\n")

    L.append("\n## A_r(K): oracle ceiling for returning a candidate's "
             "coordinates\n")
    L.append("| r | " + " | ".join("K={}".format(k) for k in KS) + " | any "
             "eligible bank row |")
    L.append("|---|" + "---|" * (len(KS) + 1))
    for r in RADII:
        cells = []
        for k in KS:
            cells.append("{:.1f}%".format(
                100.0 * (cand[:, :k].min(1) < r).mean()))
        L.append("| **{:g} km** | ".format(r) + " | ".join(cells)
                 + " | {:.1f}% |".format(100.0 * (near < r).mean()))

    L.append("\n## The partition\n")
    L.append("| r | coverage | retrieval | ranking | solved |")
    L.append("|---|---|---|---|---|")
    part = {}
    for r in RADII:
        top1 = cand[:, 0] < r
        anyk = cand.min(1) < r
        covered = near < r
        c1 = ~covered
        c2 = covered & ~anyk
        c3 = anyk & ~top1
        part[r] = (c1, c2, c3, top1)
        L.append("| **{:g} km** | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% |"
                 .format(r, 100 * c1.mean(), 100 * c2.mean(),
                         100 * c3.mean(), 100 * top1.mean()))
    L.append("\nclass 1 needs corpus; class 2 needs a better representation "
             "or a larger K; class 3 is the only one a reranker over the "
             "cached shortlist can reach.")

    L.append("\n## Conditional on being covered\n")
    for r in RADII:
        c1, c2, c3, top1 = part[r]
        cov = ~c1
        L.append("* **{:g} km** — {:.1f}% of queries have a legal bank row "
                 "within {:g} km. Of those, retrieval puts one in the top-32 "
                 "for {:.1f}%, and ranks it first for {:.1f}%."
                 .format(r, 100 * cov.mean(), r,
                         100 * (~c2[cov]).mean(), 100 * top1[cov].mean()))

    L.append("\n## Nearest legal bank row\n")
    fin = near[np.isfinite(near)]
    L.append("| percentile | km |")
    L.append("|---|---|")
    for p in (10, 25, 50, 75, 90, 99):
        L.append("| p{} | {:,.2f} |".format(p, np.percentile(fin, p)))

    txt = "\n".join(L) + "\n"
    Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    print("wrote {}".format(a.out))


if __name__ == "__main__":
    main()
