"""Stamp the row-addressed artifacts already on disk with what they describe.

Every cache here is paired with every other cache by row position, and until
today nothing recorded which ordered rows a file actually holds.  `provenance`
closes that going forward: the builders write a sidecar.  This closes the other
half -- a corpus that took days of GPU time and cannot be rebuilt to get one.

**Be clear about what a backfilled sidecar is worth.**  It records the
alignment *as it stands today*, so it detects drift from now on and vouches for
nothing before that.  It is written with `basis="observed"` for exactly that
reason.  If two artifacts are already misaligned, this makes the misalignment
official rather than finding it.

What it can still find is a length disagreement or an unresolvable row space,
which is the loud half of the same failure, so it reports those instead of
stamping them.

Four row spaces exist here and the resolver tries them in order of how much
they prove:

    <stem>_meta.npz    the artifact names its own ids -- the strongest case,
                       and how every bank extension and pyramid cache is built
    <stem>_rows.i64.npy  a subset of the release, by row index
    the release        the full dataset.parquet order
    release ++ ext     a stacked bank: the release, then one extension

An artifact that fits none of them, or fits the last ambiguously, is reported
and left alone.  A guess here would be worse than nothing: it would put an
authoritative-looking sidecar on a file whose rows nobody actually knows.

    python scripts/backfill_prov.py             # report, write nothing
    python scripts/backfill_prov.py --write
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
import tile_math as tm


def artifact_stem(p):
    """`dual_c3.f16.npy` -> `dual_c3`, and `pyr33.f16.npy` -> `pyr33`."""
    n = p.name
    for suf in (".f16.npy", ".i64.npy", ".u8.npy", ".npy", ".parquet"):
        if n.endswith(suf):
            return n[:-len(suf)]
    return p.stem


def load_meta_ids(path):
    try:
        z = np.load(path, allow_pickle=True)
    except (OSError, ValueError):
        return None
    return np.asarray(z["image_id"]) if "image_id" in z.files else None


def resolve(p, n, ids, metas):
    """(ids for this artifact, how it was resolved) or (None, why not).

    Order matters more than it looks.  `pyr47_fuse_p05_rows.i64.npy` holds
    0..47645, which is a legal index into the 500,000-row release and means
    nothing of the kind -- it indexes `pyr47`, whose own rows are KartaView
    ids.  Reading it as a release index passes every bounds check and stamps
    the file with 47,646 ids that have nothing to do with it, which is worse
    than leaving it unstamped.  So a parent artifact's metadata is consulted
    before the release is ever assumed.
    """
    stem = artifact_stem(p)

    own = p.parent / (stem + "_meta.npz")
    got = load_meta_ids(own) if own.exists() else None
    if got is not None and len(got) == n:
        return got, "own metadata"

    rows_p = p.parent / (stem + "_rows.i64.npy")
    rows = np.load(rows_p) if rows_p.exists() else None

    # The longest metadata stem that prefixes this name: an export derived from
    # a cache is named after it, and that naming is a fact rather than an
    # inference from a length that several artifacts share.
    pref = sorted((m for m in metas
                   if metas[m] is not None
                   and stem.startswith(artifact_stem(m)[:-len("_meta")])),
                  key=lambda m: -len(m.name))
    for m in pref:
        par = metas[m]
        if rows is not None and len(rows) == n and rows.max(initial=-1) < len(par):
            return par[rows], "row index into " + m.name
        if len(par) == n:
            return par, "the rows of " + m.name

    if rows is not None and len(rows) == n and rows.max(initial=-1) < len(ids):
        return ids[rows], "row index into the release"

    if n == len(ids):
        return ids, "the release"

    # A stacked bank: the release followed by exactly one extension. Prefer a
    # metadata file whose stem prefixes this artifact's name, because that is
    # a fact about how it was named rather than an inference from its length.
    rest = n - len(ids)
    by_prefix = [m for m in metas
                 if stem.startswith(artifact_stem(m).replace("_meta", ""))
                 and metas[m] is not None and len(metas[m]) == rest]
    if len(by_prefix) == 1:
        return (np.concatenate([ids, metas[by_prefix[0]]]),
                "release ++ " + by_prefix[0].name)
    by_len = [m for m in metas if metas[m] is not None and len(metas[m]) == rest]
    if len(by_len) == 1:
        return (np.concatenate([ids, metas[by_len[0]]]),
                "release ++ " + by_len[0].name)
    if len(by_len) > 1:
        # Several extensions are the same size, so ask the bytes. The tail of
        # a stacked bank is the extension verbatim, so three rows settle it --
        # and if the widths differ (a pooled projection of an extension) they
        # cannot, which is a fact about the file rather than a failure here.
        hit = match_by_content(p, n, len(ids), by_len)
        if hit is not None:
            return (np.concatenate([ids, metas[hit]]),
                    "release ++ {} (matched on contents)".format(hit.name))
        sib = match_by_sibling(p, n, len(ids), by_len)
        if sib is not None:
            hit, why = sib
            return (np.concatenate([ids, metas[hit]]),
                    "release ++ {} ({})".format(hit.name, why))
        return None, "{} extensions have {:,} rows; the name does not say " \
                     "which and the contents do not match any of them" \
                     .format(len(by_len), rest)
    return None, "no row space of {:,} could be resolved".format(n)


def head_is_the_release(p, n_rel):
    """Which release-length artifact this bank's first rows are, verbatim.

    A stacked bank is written by copying the release-length embedding in
    first, so its head is that file byte for byte. When the head matches an
    artifact whose own row space is already resolved as the release, the
    release half of the stack is established by content rather than by
    assuming that "longer than the release" means "release first" -- which is
    exactly the assumption an extension-only cache defeats.
    """
    A = np.load(p, mmap_mode="r")
    if A.ndim != 2 or len(A) <= n_rel:
        return None
    head = np.asarray(A[:64])
    for c in sorted(p.parent.glob("*.f16.npy")):
        if c == p:
            continue
        B = np.load(c, mmap_mode="r")
        if B.ndim != 2 or len(B) != n_rel or B.shape[1] != A.shape[1]:
            continue
        rec = prov.read(c) or {}
        if not str(rec.get("row_space", "")).startswith("the release"):
            continue
        if np.array_equal(np.asarray(B[:64]), head):
            return c
    return None


def match_by_sibling(p, n, n_rel, cands):
    """Resolve a stack whose extension was never saved as a standalone file.

    `pool_bal_bank25` is the case: it is the release followed by a *pooled*
    `bank_ext`, and no `bank_ext_pool.f16.npy` was ever written, so matching
    the tail against extension files on disk cannot succeed no matter how
    many candidates share its length.

    Two independent facts settle it without guessing. The head must be a
    release-length artifact already resolved as the release, which fixes the
    stack's orientation; and another bank of the *same total length and the
    same release half* must already record which extension it carries. Both
    have to hold, and the row space says which sibling supplied the answer,
    so the inference is legible to whoever reads the sidecar later.
    """
    head = head_is_the_release(p, n_rel)
    if head is None:
        return None
    for c in sorted(p.parent.glob("*.f16.npy")):
        if c == p:
            continue
        B = np.load(c, mmap_mode="r")
        if B.ndim != 2 or len(B) != n:
            continue
        got = prov.exts_of(c)
        if len(got) != 1:
            continue
        hit = next((m for m in cands
                    if artifact_stem(m)[:-len("_meta")] == got[0]), None)
        if hit is not None:
            return hit, "head verified against {}, extension from {}".format(
                head.name, c.name)
    return None


def match_by_content(p, n, n_rel, cands):
    """Which extension is stacked under this bank, read off the bytes."""
    A = np.load(p, mmap_mode="r")
    if A.ndim != 2 or n_rel >= n:
        return None
    tail = np.asarray(A[n_rel:n_rel + 3])
    # Longest stem first: `bank_ext` prefixes `bank_ext2_dino` too, and taking
    # the first match would credit every numbered extension to the original.
    stems = sorted({artifact_stem(m)[:-len("_meta")] for m in cands},
                   key=len, reverse=True)
    for c in sorted(p.parent.glob("*.f16.npy")):
        stem = artifact_stem(c)
        base = next((s for s in stems if stem.startswith(s)), None)
        if base is None:
            continue
        B = np.load(c, mmap_mode="r")
        if B.ndim != 2 or B.shape[1] != A.shape[1] or len(B) != n - n_rel:
            continue
        if np.array_equal(np.asarray(B[0:3]), tail):
            return next(m for m in cands
                        if artifact_stem(m)[:-len("_meta")] == base)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the sidecars; without it, report only")
    ap.add_argument("--force", action="store_true",
                    help="overwrite sidecars that already exist")
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET, columns=["image_id"])
    ids = np.asarray(ds["image_id"])
    print("release    {}  {:,} rows  digest {}".format(
        config.RELEASE, len(ids), prov.rows_digest(ids)))
    print("authority  {}\n".format(config.DATASET_PARQUET.name))

    metas = {m: load_meta_ids(m)
             for m in sorted(config.STREET_CACHE.glob("*_meta.npz"))}

    cand = [config.DATASET_PARQUET, config.TARGETS_PARQUET]
    cand += sorted(config.STREET_CACHE.glob("*.f16.npy"))

    wrote = skipped = stuck = 0
    for p in cand:
        if p.suffix == ".parquet":
            n = len(pq.read_table(p, columns=["image_id"]))
            if p == config.TARGETS_PARQUET:
                n = n // (tm.STEPS + 1)      # steps+1 rows per image
            rows, how = ids, "the release"
            if n != len(ids):
                rows, how = None, "{:,} images against the release's {:,}".format(
                    n, len(ids))
        else:
            n = np.load(p, mmap_mode="r").shape[0]
            rows, how = resolve(p, n, ids, metas)

        if rows is None:
            print("  {:<34} {:>9,}   unresolved: {}".format(p.name, n, how))
            stuck += 1
            continue
        have = prov.read(p)
        if have and not a.force:
            ok = have.get("rows_digest") == prov.rows_digest(rows)
            print("  {:<34} {:>9,}   sidecar {} ({})".format(
                p.name, n, "matches" if ok else "MISMATCHES",
                have.get("basis", "?")))
            skipped += 1
            continue
        if a.write:
            prov.write(p, rows, basis="observed", release=config.RELEASE,
                       row_space=how, authority=config.DATASET_PARQUET.name)
        print("  {:<34} {:>9,}   {} [{}]".format(
            p.name, n, "stamped" if a.write else "would stamp", how))
        wrote += 1

    print("\n{} stamped, {} already had one, {} unresolved"
          .format(wrote, skipped, stuck))
    if not a.write:
        print("(dry run -- pass --write)")
    else:
        print("These say the alignment held on the day they were written. "
              "They catch drift after that and vouch for nothing before it.")


if __name__ == "__main__":
    main()
