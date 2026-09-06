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

**The predictor here is retrieval, not the agent.** It needs no checkpoint, no
beam and no GPU. Every headline number is counted from the run and none is
written into the template: the first version of this file hardcoded "rank-1
scores 56.4%, the agent 57.5%, so the agent adds about a point" into its prose
and went on emitting it above a table reading 49.7%, because 56.4% turned out
to be a cohort artifact. A corrected computation that still ships the
conclusion it disproved is worse than no computation (REVIEW8 M1).

**No agent number is quoted here.** Subtracting an agent accuracy measured on
one cohort from a retrieval accuracy measured on another is not a treatment
effect, and the ranking bucket below belongs to rank-1 retrieval, not
automatically to the agent. The comparison that would mean something is a
paired evaluation of rank-1, the agent, and any reranker on these same ids.

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

    The extension's `_meta.npz` carries only the z16 address, and the first
    version of this script used tile centres for those rows. A z16 tile is
    611 m wide at the equator, so its half-diagonal is ~432 m -- material
    beside a 1 km threshold and beside a 340 m median nearest distance, and
    able to put a row inside the radius whose photograph is outside it, or the
    reverse (REVIEW8 M5).

    The source parquets do carry true `lat`/`lon`, so they are joined by
    `image_id` in the meta's own order. Every id must resolve; a partial join
    would silently mix true coordinates with tile centres, which is worse than
    consistently using either.
    """
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    seq = np.asarray(ds["sequence"].to_pylist(), dtype=object).astype("U40")
    ext = str(z["bank_ext"]) if "bank_ext" in z.files else ""
    if not ext:
        return lat, lon, seq
    m = prov.bank_ext(ext, config.RELEASE)
    want = np.asarray(m["image_id"])
    have_id, have_lat, have_lon = [], [], []
    # Every extension parquet on disk, not the stems the sidecar names: a bank
    # stacked from four shards records the single combined `bank_ext70`, for
    # which no parquet exists. The lookup is by id, so the order of this table
    # does not matter and a superset is harmless -- the join assertion below
    # is what establishes that the meta's rows were all found.
    for src in sorted(config.PROCESSED.glob("bank_ext*.parquet")):
        t = pq.read_table(src, columns=["image_id", "lat", "lon"])
        have_id.append(np.asarray(t["image_id"]))
        have_lat.append(np.asarray(t["lat"], np.float64))
        have_lon.append(np.asarray(t["lon"], np.float64))
    have_id = np.concatenate(have_id)
    order = np.argsort(have_id)
    pos = np.searchsorted(have_id[order], want)
    if pos.max() >= len(order) or not np.array_equal(
            have_id[order][np.clip(pos, 0, len(order) - 1)], want):
        raise SystemExit(
            "the extension's true coordinates could not be joined for every "
            "row: {:,} of {:,} ids resolve. A partial join would mix true "
            "coordinates with z16 tile centres, which is worse than "
            "consistently using either.".format(
                int((have_id[order][np.clip(pos, 0, len(order) - 1)]
                     == want).sum()), len(want)))
    j = order[pos]
    elat = np.concatenate(have_lat)[j]
    elon = np.concatenate(have_lon)[j]
    return (np.concatenate([lat, elat]), np.concatenate([lon, elon]),
            np.concatenate([seq, np.asarray(m["sequence"]).astype("U40")]))


def addresses(lat, lon):
    """z16 address of every row, derived the way `build_knn` derives it.

    Needed to ask a cell split which extension rows it holds out. Both halves
    of the address space go through the same projection here, which is only
    possible because the extension's true coordinates are now joined -- the
    tile-centre version would have round-tripped its own quantisation.
    """
    la = np.clip(np.asarray(lat, np.float64), -tm.MAX_LAT, tm.MAX_LAT)
    sin = np.sin(np.radians(la))
    px = (np.asarray(lon, np.float64) + 180.0) / 360.0
    py = 0.5 - np.log((1.0 + sin) / (1.0 - sin)) / (4.0 * np.pi)
    n16 = 1 << (4 * tm.STEPS)
    return (np.clip((px * n16).astype(np.int64), 0, n16 - 1),
            np.clip((py * n16).astype(np.int64), 0, n16 - 1))


def nearest_legal(tree, bank_seq, bank_rows, qv, qseq, probe=64):
    """Great-circle km to the nearest bank row the query is ALLOWED to match.

    Same-sequence rows are in the bank and are forbidden to retrieval, so they
    must not count as coverage. The tree is queried `probe` deep and the
    nearest neighbour on another drive is taken.

    A query whose whole probe is its own drive used to return inf and be
    counted as uncovered, described as "a floor on class 1". **That was
    backwards** (REVIEW8 M2): a legal row at rank probe+1 may well be inside
    the radius, so treating an exhausted probe as uncovered OVERstates class 1,
    making it an upper bound. Worse, it breaks the partition -- such a query
    can be counted uncovered while its rank-1 candidate is a legal row inside
    the radius, so it lands in two classes at once.

    So the probe is now widened for the unresolved queries until each finds a
    legal neighbour or the bank is exhausted. In practice this touches a
    handful of rows and terminates immediately; the point is that the result
    is exact rather than a bound in the wrong direction.
    """
    d, j = tree.query(qv, k=probe, workers=-1)
    legal = bank_seq[bank_rows[j]] != qseq[:, None]
    best = np.where(legal, chord_to_km(d), np.inf).min(1)

    widened = 0
    k = probe
    while np.isinf(best).any() and k < len(bank_rows):
        k = min(k * 8, len(bank_rows))
        u = np.flatnonzero(np.isinf(best))
        widened = max(widened, len(u))
        d2, j2 = tree.query(qv[u], k=k, workers=-1)
        lg = bank_seq[bank_rows[j2]] != qseq[u][:, None]
        best[u] = np.where(lg, chord_to_km(d2), np.inf).min(1)
    return best, widened


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

    # The same validation production uses, not a subset of it. Checking
    # bank_rows and release membership does not establish the cached idx, the
    # split hash, or the row-order identity (REVIEW8 M4).
    rows = knnmeta.check(z, a.knn, split_mode=mode, splits=labels,
                         split_hash=sp.split_hash(mode, labels),
                         n_release=len(labels), n_bank=len(lat),
                         street_path=config.STREET_CACHE / str(z["street_file"]))
    if rows is None:
        sys.exit("{} records no bank_rows, so there is no eligible bank to "
                 "measure coverage against".format(a.knn))
    # Real z16 addresses, and bad_ext acted on. Passing zeros meant a cell
    # split's geographic exclusion was computed against a single fake cell and
    # every extension row looked eligible (REVIEW8 M3).
    x16, y16 = addresses(lat, lon)
    bad_rel, bad_ext = sp.ineligible(rows, len(labels), labels, mode, x16, y16)
    if len(bad_rel) or len(bad_ext):
        sys.exit("{} banks {:,} release rows this split holds out and {:,} "
                 "extension rows inside its held-out cells; its coverage "
                 "would be measured against a bank it is not allowed to use."
                 .format(a.knn, len(bad_rel), len(bad_ext)))

    q = np.flatnonzero(labels == a.split)
    idx = np.asarray(z["idx"], np.int64)[q]
    print("{}  {}  bank {:,}  queries {:,}  K={}".format(
        a.knn, mode, len(rows), len(q), idx.shape[1]), flush=True)

    print("building the geographic index over the eligible bank...", flush=True)
    tree = cKDTree(unit(lat[rows], lon[rows]))
    qv = unit(lat[q], lon[q])
    near, widened = nearest_legal(tree, seq, rows, qv, seq[q], a.probe)
    exhausted = int(np.isinf(near).sum())
    print("probe  {:,} queries needed a search deeper than {}; {:,} have no "
          "legal bank row anywhere".format(widened, a.probe, exhausted),
          flush=True)

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
    # Every headline number is counted here, not written into the template.
    # The first version hardcoded "rank-1 scores 56.4%, the agent 57.5%, so
    # the agent adds ~1 pp" into the prose -- and kept emitting it above a
    # table that said 49.7%, because 56.4% turned out to be a cohort artifact.
    # That is exactly how a corrected computation goes on publishing the
    # conclusion it disproved (REVIEW8 M1).
    L.append("Predictor: **rank-1 retrieval**, scoring **{:.1f}% `<25 km`** on "
             "this cohort ({:,} queries, all of the `{}` split). No "
             "checkpoint, no beam, no GPU. The agent's number is measured on "
             "a different cohort and is deliberately not quoted here: the "
             "comparison that would mean anything is a paired one on these "
             "same ids. Same-sequence rows are excluded from the coverage "
             "test as well as from retrieval.\n"
             .format(100.0 * (cand[:, 0] < 25.0).mean(), len(q), a.split))
    L.append("Extension coordinates are the source parquets' true `lat`/`lon`, "
             "not z16 tile centres.\n")
    if exhausted:
        L.append("**{:,} queries have no legal bank row anywhere** and are "
                 "counted as uncovered at every radius.\n".format(exhausted))

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
        # The four classes must partition the queries exactly. They did not
        # when an exhausted probe could mark a query uncovered while its
        # rank-1 candidate was a legal row inside the radius -- it then landed
        # in class 1 and in "solved" at once, and the percentages summed past
        # 100 without anything saying so. Rounded aggregates cannot show this;
        # the per-query invariant can (REVIEW8 M2).
        tot = c1.astype(int) + c2 + c3 + top1
        if not (tot == 1).all():
            raise SystemExit(
                "the partition is not a partition at r={:g}: {:,} queries land "
                "in {} classes. covered/topK/top1 must be nested -- a legal "
                "candidate inside the radius implies the query is covered."
                .format(r, int((tot != 1).sum()),
                        sorted(set(tot[tot != 1].tolist()))))
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
