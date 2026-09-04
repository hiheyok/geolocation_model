"""The one owner of what train/val/test mean.

This module exists because the split was implemented twice -- once in
``build_dataset.assign_splits`` and again in ``scripts/resplit.py`` -- and the
two drifted.  The parquet on disk carried a sequence split while the code
specified a z8-cell holdout, so re-running the builder would silently have
replaced the benchmark with the stress test.  Nothing outside this module may
decide what a split is.

Two splits are kept side by side in dataset.parquet, as ``split_sequence`` and
``split_cell8``, and are selected at read time rather than regenerated:

  sequence   the primary benchmark.  OSV-5M captures run consecutively along a
             road, so near-duplicate frames share a sequence; holding sequences
             out removes the real leakage vector.  Geographic proximity remains
             and must be quoted -- 10.2% of val sits within 1 km of a training
             image -- and coverage % is the ceiling memorisation can reach, so a
             model that beats it is demonstrably generalising.

  cell8      the unseen-region stress test.  Holding out whole z8 cells also
             removes the *answers*: only 2.3% of val (view, action) pairs at
             step 1 appear in training, and steps 2-3 cascade to 0.1% and 0.0%.
             A model is not expected to score well here; the question it
             answers is whether a mechanism transfers to regions whose
             identities were never seen, which is precisely what a tile-ID
             memory cannot do and a content-keyed memory might.

Both are versioned by a hash over the assignment itself, recorded in every
checkpoint, so a benchmark can never be swapped without it showing up.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import tile_math as tm

# mode -> holdout zoom, or None to group by capture sequence
MODES = {"sequence": None, "cell8": 8, "cell12": 12, "cell16": 16}

PRIMARY = "sequence"
STRESS = "cell8"


def column(mode):
    """Name of the parquet column holding this mode's assignment."""
    if mode not in MODES:
        raise ValueError("unknown split mode {!r}; have {}"
                         .format(mode, ", ".join(sorted(MODES))))
    return "split_" + mode


def group_ids(mode, lat, lon, seq):
    """The unit that is held out whole, one id per image."""
    if mode == "sequence":
        return np.asarray(seq, dtype=object)
    z = MODES[mode]
    out = np.empty(len(lat), dtype=np.int64)
    for i in range(len(lat)):
        _, x, y = tm.tile_for(lat[i], lon[i], z)
        out[i] = x * (1 << z) + y
    return out


def assign(mode, lat, lon, seq, country, seed=None, fractions=None):
    """Hold whole groups out, stratified by each group's majority country.

    Groups are filled in a shuffled order until each split's image quota is
    met, so a group never spans splits by construction -- which is the whole
    point, and the property that the drifted parquet had lost.
    """
    seed = config.SPLIT_SEED if seed is None else seed
    f_tr, f_va, _ = fractions or config.SPLIT_FRACTIONS
    groups = group_ids(mode, lat, lon, seq)
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))

    by_country = {}
    for gi in range(len(uniq)):
        vals, cnt = np.unique(country[inv == gi], return_counts=True)
        by_country.setdefault(vals[cnt.argmax()], []).append(gi)

    rng = np.random.default_rng(seed)
    label = np.empty(len(uniq), dtype=object)
    for _, gis in sorted(by_country.items()):
        gis = list(gis)
        rng.shuffle(gis)
        total = sum(int(counts[g]) for g in gis)
        seen = 0
        for g in gis:
            f = seen / total if total else 1.0
            label[g] = ("train" if f < f_tr else "val" if f < f_tr + f_va else "test")
            seen += int(counts[g])
    return label[inv]


def _digest(mode, labels, first_char_only):
    h = hashlib.sha256()
    h.update(mode.encode())
    h.update(repr(tuple(config.SPLIT_FRACTIONS)).encode())
    h.update(str(config.SPLIT_SEED).encode())
    if first_char_only:
        h.update("".join(str(x)[0] for x in labels).encode())
    else:
        h.update("\x00".join(str(x) for x in labels).encode())
    return h.hexdigest()[:12]


def split_hash_legacy(mode, labels):
    """The pre-2026-09-03 digest, kept only to recognise older checkpoints.

    It hashed `str(x)[0]`, so "train" and "test" both became "t" and the digest
    recorded little more than where the val rows were. Measured on the live
    s10 sequence split: swapping *every* train row with *every* test row -- an
    entirely different benchmark -- leaves the digest unchanged at 59e4b597281a,
    and so does moving a single row from train to test. Only val moves show up.
    """
    return _digest(mode, labels, True)


def split_hash(mode, labels):
    """Short digest of an assignment, in row order.

    Row order is what aligns the embedding memmaps, so the hash pins the
    benchmark and the alignment together -- which is the whole point, and which
    the previous version did not do: it hashed only the first character of each
    label, making train and test indistinguishable.

    Every guarantee built on this digest was correspondingly weaker than it
    read, including the bootstrap's "these arms were measured over the same
    rows" check. Checkpoints written before the fix carry the old value; see
    `split_hash_legacy` and the fallback in evaluate.check_split.
    """
    return _digest(mode, labels, False)


def read(table, mode):
    """Labels and hash for one mode out of an already-loaded parquet table."""
    col = column(mode)
    if col not in table.schema.names:
        raise SystemExit(
            "dataset.parquet has no {!r}.  Run scripts/resplit.py to write "
            "every split mode as its own column.".format(col))
    labels = np.asarray(table[col].to_pylist(), dtype=object)
    return labels, split_hash(mode, labels)
