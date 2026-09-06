"""One validator for a k-NN cache's metadata, shared by every consumer.

`GeoStepDataset` checks a cache's split mode, split hash, street filename,
query count, neighbour width and index range before using it. The direct
consumers never went through that path: `serve.py`, `eval_highres.py` and
`multiquery.py` each read `bank_rows` straight out of the `.npz` and turn it
into a search mask. So a stale or replaced cache of the right apparent name
selects a different split or a different bank while serving and evaluation
carry on normally (REVIEW4 #14).

Two checks live here that no consumer had at all.

**Negative rows wrap.** `keep[np.asarray(z["bank_rows"])] = True` with a
negative row marks a row from the *end* of the bank. NumPy is doing exactly
what it was asked; the result is a mask over images nobody selected, and every
shape still agrees.

**The extension is never cross-checked against the embeddings** (REVIEW4 #1).
A street file's sidecar records which extensions it was stacked from, and the
k-NN cache records which extension its neighbour ids address. Nothing compared
them, and all four extensions on disk hold exactly 750,000 rows -- so a
`release ++ bank_ext` street file beside a cache stamped `bank_ext2` passes the
filename check, the query count, and the index range, and then hands every
visually matched embedding a different photograph's z16 address.

That comparison is by **content, not name**. A bank stacked from four parquets
records four extension stems while its k-NN cache names the single combined
`bank_ext70`; those strings differ and the corpora are identical, so comparing
names would reject the shipping path. Comparing the ordered id digest accepts
it and still catches a swap.

**The embedding space is not bound to the cache either** (REVIEW6 #3). The ids
say *which photographs* the bank holds; nothing said which *vectors* were
searched. Rebuild a street cache under its own name -- a different pooling, a
retuned blend weight, a re-run of the same script -- and the k-NN's `idx` and
`sim` still describe neighbours chosen in the old space, while every name,
length, row digest and split hash agrees. The joined-prefix check added for
REVIEW6 #2 catches this for the *conditioned* file's retrieval block, but a
plain unconditioned rebuild has no prefix to check, so the cache is compared
against the bytes it was built from instead.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                      # noqa: E402
import provenance as prov          # noqa: E402
import safeio                      # noqa: E402
import splits as sp                # noqa: E402


def _ids_of(stems):
    """Ordered image ids of a list of extension stems, concatenated."""
    if not stems:
        return None
    parts = [np.asarray(prov.bank_ext(s, config.RELEASE)["image_id"])
             for s in stems]
    return np.concatenate(parts)


def check_ext(z, street_path, what):
    """The cache's extension and the street file's must be the same corpus.

    Compared by ordered id digest rather than by stem name: a bank stacked from
    `bank_ext` through `bank_ext4` records four stems while the cache built
    over it names `bank_ext70`, and those are the same 3,000,000 images in the
    same order.
    """
    ext = str(z["bank_ext"]) if "bank_ext" in z.files else ""
    street_stems = prov.exts_of(street_path) if street_path else []
    if not ext and not street_stems:
        return
    if bool(ext) != bool(street_stems):
        raise SystemExit(
            "{}: the k-NN cache {} while its street file {}. Neighbour ids "
            "and embedding rows would be indexed against different corpora."
            .format(what,
                    "names the bank extension {!r}".format(ext) if ext
                    else "names no bank extension",
                    "records {}".format(" ++ ".join(street_stems))
                    if street_stems else "records no bank extension"))
    a = prov.rows_digest(_ids_of([ext]))
    b = prov.rows_digest(_ids_of(street_stems))
    if a != b:
        raise SystemExit(
            "{}: the k-NN cache addresses extension {!r} (ids digest {}) but "
            "its street file was stacked from {} (digest {}). Every extension "
            "on disk holds 750,000 rows, so the lengths agree and every "
            "neighbour index stays in range -- each matched embedding would "
            "simply be given another photograph's address."
            .format(what, ext, a, " ++ ".join(street_stems), b))


def check_bytes(z, street_path, what, digest=None):
    """The neighbours in this cache were selected in *these* vectors.

    `check_ext` establishes that both sides address the same photographs.
    Nothing established that they address the same *embedding* of them, and
    the two are independent: rebuilding a street cache under its own name
    leaves the ids, the lengths, the row digest, the split hash and the
    filename all correct while `idx` and `sim` still describe a search through
    the vectors that file used to hold.

    Absence is a warning rather than a failure. All 19 k-NN caches on disk were
    built before the stamp existed, and refusing them would take the shipping
    path down to make a point about caches that are, as far as anything can
    tell, fine. The warning names what is unverified so a stale cache is at
    least visible in the log.

    They are deliberately not backfilled. Stamping an existing cache with the
    digest its street file has *today* would assert exactly the thing that
    cannot be checked -- that the file has not changed since the search -- and
    turn an honest "unverified" into a false "verified". The stamp is only
    worth anything when it is written by the process that did the search.

    `digest` lets a caller that has already digested the file pass it in.
    Re-reading is not free -- the street caches are 5-11 GB, and the first read
    of one comes off disk at 85-300 MB/s -- but it is not memoised, because the
    only key cheap enough to memoise on is the size-and-mtime stamp this
    function exists to distrust. Repeat reads land in the page cache at
    ~1,463 MB/s, which is the affordable half of that trade.
    """
    if "street_digest" not in z.files:
        print("warning: {} predates the street content stamp, so its "
              "neighbours are bound to a filename only -- a rebuild of that "
              "file under the same name would not be visible here."
              .format(what), flush=True)
        return
    want = str(z["street_digest"])
    got = digest if digest is not None else safeio.content_digest(street_path)
    # "absent" is the sentinel content_digest returns for a path that is not
    # there. It is a legal string, so a cache that somehow recorded it would
    # match a *missing* file and the check would pass on nothing at all.
    # Say what actually happened instead.
    if "absent" in (want, got):
        raise SystemExit(
            "{}: cannot compare the cache against {}, which is not on disk "
            "(recorded digest {!r}, found {!r}). The neighbours were selected "
            "in some embedding space; without the file there is no way to say "
            "it is the one this run reads."
            .format(what, Path(street_path).name, want, got))
    if got != want:
        raise SystemExit(
            "{}: the cache's neighbours were chosen in a {} whose bytes "
            "digest {}; that file now digests {}. It was rebuilt under the "
            "same name, so every recorded index and similarity describes a "
            "search through vectors this file no longer holds -- the ids, "
            "lengths and split hash all still agree. Rebuild the k-NN cache."
            .format(what, Path(street_path).name, want, got))


def bank_rows(z, n_bank, what):
    """Validated bank row ids from a cache, or exit.

    `bank_rows` is turned into a boolean mask by every direct consumer, and a
    raw fancy-index assignment accepts things it should not:

      * a **negative** row wraps to the end of the bank, marking a row nobody
        selected, silently and with the right shape;
      * a row **past the end** raises here rather than in the middle of a
        search, which is the difference between a message and a stack trace;
      * a **duplicate** row is not wrong for a mask but means the cache is not
        what it claims, so it is worth saying.
    """
    if "bank_rows" not in z.files:
        return None
    rows = np.asarray(z["bank_rows"], np.int64)
    if rows.size and rows.min() < 0:
        raise SystemExit(
            "{}: {:,} of the cache's bank rows are negative (lowest {}). "
            "NumPy would index those from the end of the bank and mark rows "
            "that were never selected."
            .format(what, int((rows < 0).sum()), int(rows.min())))
    if rows.size and rows.max() >= n_bank:
        raise SystemExit(
            "{}: the cache selects bank row {:,} but the embedding file holds "
            "{:,}. It was built over a different bank."
            .format(what, int(rows.max()), n_bank))
    if len(np.unique(rows)) != len(rows):
        raise SystemExit(
            "{}: the cache lists {:,} bank rows but only {:,} distinct ones."
            .format(what, len(rows), len(np.unique(rows))))
    return rows


def check(z, what, street_file=None, split_mode=None, split_hash=None,
          n_release=None, need_k=None, n_bank=None, street_path=None,
          splits=None, street_digest=None):
    """Every contract a k-NN cache carries. Returns validated bank rows.

    Each argument that is `None` is simply not checked, so a consumer that
    genuinely does not know a field (the server does not carry a split hash for
    a replaced cache) can still get the checks it *can* make, instead of the
    none it gets today.
    """
    if split_mode is not None and str(z["split_mode"]) != split_mode:
        raise SystemExit(
            "{}: cache was built on split {!r}, this run is {!r}; the bank "
            "must be that split's train side."
            .format(what, str(z["split_mode"]), split_mode))
    if (split_hash is not None and "split_hash" in z.files
            and str(z["split_hash"]) != split_hash):
        # Every cache on disk predates 2026-09-03, when split_hash still
        # hashed only each label's first character. That digest cannot tell a
        # train/test swap from the real assignment, so it is accepted with a
        # note rather than silently -- the same allowance `check_split` makes.
        legacy = (splits is not None and split_mode is not None
                  and sp.hash_matches(split_mode, splits,
                                      z["split_hash"]) == "legacy")
        if not legacy:
            raise SystemExit(
                "{}: cache was built against split hash {} but the data on "
                "disk hashes to {}. Its train side is no longer that train "
                "side.".format(what, str(z["split_hash"]), split_hash))
        print("note: {} carries the pre-2026-09-03 split digest, which pins "
              "the mode and the val positions only.".format(what), flush=True)
    if (street_file is not None and "street_file" in z.files
            and str(z["street_file"]) != street_file):
        raise SystemExit(
            "{}: cache was built over {!r} but this run reads {!r}. "
            "Neighbours found in one embedding space do not transfer to "
            "another.".format(what, str(z["street_file"]), street_file))
    if n_release is not None and z["idx"].shape[0] != n_release:
        raise SystemExit(
            "{}: cache holds {:,} query rows but the release has {:,} images. "
            "Both are indexed by row order in dataset.parquet, so every image "
            "would be given another image's neighbours."
            .format(what, z["idx"].shape[0], n_release))
    if need_k is not None and z["idx"].shape[1] < need_k:
        raise SystemExit(
            "{}: cache has {} neighbours per query, {} were asked for. "
            "Slicing would train on fewer neighbours than the run records."
            .format(what, z["idx"].shape[1], need_k))
    if street_path is not None:
        check_ext(z, street_path, what)
        check_bytes(z, street_path, what, street_digest)
    return bank_rows(z, n_bank, what) if n_bank is not None else None
