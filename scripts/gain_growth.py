"""Does a paired gain grow between two bank sizes? Test the difference.

`knn_gap` reports one contrast: treatment minus baseline at one bank size,
with a paired bootstrap CI. The question this answers is one level up -- is
the contrast at the LARGER bank bigger than the contrast at the smaller one --
and it needs its own test, which `runs/TILEFULL.md` originally did not give
it.

**Overlapping marginal intervals do not answer it.** The first version of that
report compared [+3.0, +3.6] at 1.15M with [+3.4, +4.0] at 3.4M, saw them
overlap, and concluded the growth had stopped. That is a standard error: the
difference of two estimates has its own sampling distribution, and here it is
much narrower than either, because the same test queries appear in both
measurements and most of the variance is common to them. Tested properly the
growth was +0.42 pp [+0.05, +0.79] -- separated, the opposite conclusion.

The four caches must be comparable, and this checks it rather than assuming
it: within each pair the same `bank_rows`, `bank_n`, `split_mode` and
`split_hash`, and the same corpus by ordered image ids; across the pairs the
same split, because a different bank is the whole point. The first version
checked none of it, and review fed it mismatched caches and got "+100 pp,
separated" out of a script whose entire job is to stop that.

So the statistic is a difference of differences, resampled over queries:

    d_q = (b_big[q] - a_big[q]) - (b_small[q] - a_small[q])

with each term an indicator of "within `--thresh` km". The four caches must
share a query set, which they do when they share a split: queries are release
rows and the bank extension only ever adds rows after them.

    OSV_RELEASE=s10 py scripts/gain_growth.py \\
        --small-a knn_pyr768_l0_b115_sequence_k32_bank_ext.npz \\
        --small-b knn_pyr768_l0l1_b115_sequence_k32_bank_ext.npz \\
        --big-a   knn_pyr768_l0_b340_sequence_k32_bank_ext70.npz \\
        --big-b   knn_pyr768_l0l1_b340_sequence_k32_bank_ext70.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config                                    # noqa: E402
import splits as sp                              # noqa: E402
import knn_gap as K                              # noqa: E402
import provenance as prov                        # noqa: E402


KEYS = ("split_mode", "split_hash", "bank_n", "bank_rows")


def cache(name):
    """One neighbour table, with the fields this comparison depends on.

    A cache too old to record them is refused rather than compared: five
    legacy tables on disk predate `bank_rows`, and without it "same bank"
    cannot be established at all.
    """
    z = np.load(config.STREET_CACHE / name)
    missing = [k for k in KEYS if k not in z.files]
    if missing:
        raise SystemExit(
            "{} records no {}, so it cannot be shown to describe the same "
            "bank and split as the caches it is being compared with; rebuild "
            "it with the current build_knn".format(name, ", ".join(missing)))
    return z


def check_pair(za, zb, na, nb):
    """Refuse two caches that did not index the same bank on the same split.

    This is `knn_gap`'s guard, and the first version of this script had none
    of it -- review fed it mismatched caches and got "+100 pp, separated"
    while `knn_gap` rejected the same inputs. A growth number is a difference
    of two differences, so an incomparable pair anywhere in it is not a
    weaker result, it is a meaningless one.
    """
    for k in ("split_mode", "split_hash", "bank_n"):
        if str(za[k]) != str(zb[k]):
            raise SystemExit("{} differs between {} and {}: {} vs {}".format(
                k, na, nb, za[k], zb[k]))
    if not np.array_equal(za["bank_rows"], zb["bank_rows"]):
        raise SystemExit(
            "{} and {} index different bank rows, so the contrast between "
            "them measures what was indexed rather than how it was encoded"
            .format(na, nb))


def gain(ds, a_name, b_name, te, thresh, rank):
    """Per-query gain, b minus a, as a 0/1 difference at `thresh` km.

    Coordinates come from `a`'s own cache: a bank extension changes which
    rows exist, so the two bank sizes need different coordinate arrays even
    though they share queries.
    """
    lat, lon, ids_a = K.coords(ds, a_name)
    _, _, ids_b = K.coords(ds, b_name)
    # Ordered identities, not lengths: every extension on disk holds exactly
    # 750,000 rows, so two arms over different ones agree on every count.
    prov.require_same_corpus(
        ids_a, ids_b, a_name + "'s bank", b_name + "'s bank",
        because="Both arms are scored against the first one's coordinates, "
                "so this would give one arm's neighbours another arm's "
                "geography and report the difference as a gain.")
    ea, _ = K.load(a_name, lat, lon, te, rank)
    eb, _ = K.load(b_name, lat, lon, te, rank)
    return (eb < thresh).astype(float) - (ea < thresh)


def main():
    ap = argparse.ArgumentParser()
    for k in ("small-a", "small-b", "big-a", "big-b"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--split-mode", default="sequence")
    ap.add_argument("--thresh", type=float, default=25.0)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5000)
    a = ap.parse_args()

    names = [getattr(a, k) for k in ("small_a", "small_b", "big_a", "big_b")]
    zs = {n: cache(n) for n in names}

    # Within a pair: same bank, same split -- otherwise that pair's gain is
    # not a gain. Across the pairs: same split only, because a different bank
    # is the entire point of the comparison.
    check_pair(zs[names[0]], zs[names[1]], names[0], names[1])
    check_pair(zs[names[2]], zs[names[3]], names[2], names[3])
    for n in names:
        if str(zs[n]["split_mode"]) != a.split_mode:
            raise SystemExit(
                "{} was built on split {!r} but --split-mode is {!r}; the "
                "test rows would not be the ones it indexed".format(
                    n, str(zs[n]["split_mode"]), a.split_mode))
    hashes = {str(zs[n]["split_hash"]) for n in names}
    if len(hashes) != 1:
        raise SystemExit(
            "the four caches were built against {} different {} splits, so "
            "'the same queries' is not true of them and the paired "
            "difference is not paired".format(len(hashes), a.split_mode))

    ds = pq.read_table(config.DATASET_PARQUET)
    labels, _ = sp.read(ds, a.split_mode)
    te = np.flatnonzero(labels == "test")

    gs = gain(ds, names[0], names[1], te, a.thresh, a.rank)
    gb = gain(ds, names[2], names[3], te, a.thresh, a.rank)

    d = gb - gs
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(d), (a.reps, len(d)))
    lo, hi = np.percentile(d[ix].mean(1) * 100, [2.5, 97.5])

    print("{:,} test queries, split {}, <{:g} km, any-of-{}".format(
        len(te), a.split_mode, a.thresh, a.rank))
    print("  gain at the smaller bank   {:+.2f} pp".format(100 * gs.mean()))
    print("  gain at the larger bank    {:+.2f} pp".format(100 * gb.mean()))
    print("  GROWTH                     {:+.2f} pp  95% CI [{:+.2f}, {:+.2f}]"
          "  -> {}".format(100 * d.mean(), lo, hi,
                           "separated" if lo * hi > 0 else "spans zero"))
    # Deliberately not a non-zero exit: "spans zero" is a finding, not a
    # failure, and a caller that treats it as one would be back to reading
    # an inconclusive result as a negative one.
    return 0


if __name__ == "__main__":
    sys.exit(main())
