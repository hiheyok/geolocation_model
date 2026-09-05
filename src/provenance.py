"""What proves two row-addressed files describe the same ordered rows.

Nearly every artifact here is addressed by row position: street embeddings,
targets, bank extensions, PCA bases, pooled projections.  Row *i* of one is
paired with row *i* of another with nothing checking that they came from the
same build, and the failure that produces is the worst shape this project has:
every array is the right dtype and a plausible shape, every metric comes back
in a believable range, and each image has been scored against a different
image's embedding.  It cannot fail loudly, because there is nothing to fail on.

The review calls this one issue and it is right -- items 40, 43, 44, 46, 47, 62,
63 and 66 are the same hole seen from eight places.  So the fix is one thing: a
sidecar beside each artifact naming the ordered rows it describes.

Three deliberate choices.

**Digest the ids, do not store them.** 500,000 int64 ids hash in 2 ms and the
sidecar stays under a kilobyte, so the check is free enough to run on every
construction rather than behind a flag nobody sets.  Storing them again would
just be a second artifact to keep in step with the first.

**Absent is a warning, mismatched is a refusal.** Every artifact on disk today
predates this, and refusing them would strand a corpus that took days of GPU
time to build.  `scripts/backfill_prov.py` stamps them with the alignment as it
stands, which detects drift from that point forward and proves nothing about
what came before -- and says so, in the sidecar, in the `basis` field.

**The dataset parquet is the authority.** Its `image_id` column defines row
order for the release; every other artifact is checked against it, never
against another derived artifact.  A chain of pairwise checks can be internally
consistent and collectively wrong.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

VERSION = 1


def rows_digest(ids):
    """A 12-hex fingerprint of an ordered id sequence.

    Order-sensitive by construction: this exists to catch a reordering, which
    a set-based check (the one `build_knn` had) passes.  Integer ids are
    normalised to int64, so a column read back as int32 does not read as a
    different build; the external corpora key on strings instead, and those
    are hashed as NUL-joined UTF-8 rather than by their numpy dtype, whose
    width changes with the longest id in the array.
    """
    a = np.asarray(ids)
    h = hashlib.sha256()
    if a.dtype.kind in "iu":
        a = np.ascontiguousarray(a.astype(np.int64))
        h.update(b"prov%d|i|%d|" % (VERSION, a.size))
        h.update(a.tobytes())
    else:
        h.update(b"prov%d|s|%d|" % (VERSION, a.size))
        h.update(chr(0).join(str(x) for x in a.tolist()).encode("utf-8"))
    return h.hexdigest()[:12]


def bank_ext(stem, release):
    """Load a bank extension's metadata, refusing one from another release.

    The extension records the release it was harvested against and nothing
    ever read it (item 62). It matters because the *release* half of a stacked
    bank is addressed by row order in that release's dataset.parquet: pair an
    s01 extension with an s10 release and rows 0..n are one corpus while rows
    n.. are another, with a single coherent index space over the two and no
    way for anything downstream to notice.
    """
    import config

    p = config.bank_meta(stem)
    if not p.exists():
        raise SystemExit("bank extension {!r} has no metadata at {}"
                         .format(stem, p))
    m = np.load(p, allow_pickle=True)
    got = str(m["release"]) if "release" in m.files else None
    if got is None:
        print("warning: {} predates the release field, so nothing proves this "
              "extension was harvested against {!r}".format(p.name, release),
              flush=True)
    elif got != release:
        raise SystemExit(
            "{} was harvested against release {!r} but the run is {!r}. The "
            "release half of a stacked bank is addressed by row order in that "
            "release's dataset.parquet, so combining them gives one index "
            "space over two different corpora.".format(p.name, got, release))
    return m


def ext_stem_for(name):
    """`bank_ext_dual` -> `bank_ext`; `bank_ext70_pool` -> `bank_ext70`.

    The extension's embeddings are written one file per encoder scheme and all
    of them share a single metadata file, so the encoder suffix has to come
    off before the metadata can be found.
    """
    import config

    parts = name.split("_")
    for cut in range(len(parts), 0, -1):
        stem = "_".join(parts[:cut])
        if config.bank_meta(stem).exists():
            return stem
    return None


def check_stack(path, parts, what=None):
    """Verify a stacked artifact's rows are these parts *in this order*.

    Item 66: the merge checked the total length, which every wrong order also
    satisfies. Concatenating the parts and digesting the result is the check
    that distinguishes them, and it is the same digest the sidecar already
    holds.
    """
    ids = np.concatenate([np.asarray(p) for p in parts])
    return check(path, ids, what)


ENCODER_FIELDS = ("model", "crops", "size")


def carry(src, dst, n, **extra):
    """Give a row-preserving transform its source's row identity.

    Pooling, PCA projection and the like write one output row per input row in
    the same order, so the output describes exactly the rows the input did.
    Copying the digest rather than recomputing it is the point: the transform
    never sees the ids, and asking it to would mean threading the parquet
    through every utility that reshapes a matrix (item 46).
    """
    rec = read(src)
    if rec is None:
        print("warning: {} has no sidecar, so {} cannot inherit its rows"
              .format(Path(src).name, Path(dst).name), flush=True)
        return None
    if rec.get("rows") != n:
        raise SystemExit(
            "{} describes {:,} rows but the output has {:,}; this transform "
            "is supposed to preserve rows one for one"
            .format(Path(src).name, rec.get("rows", -1), n))
    out = {k: v for k, v in rec.items()
           if k not in ("version", "basis", "rows", "rows_digest")}
    out.update(extra)
    out["derived_from"] = Path(src).name
    import safeio

    d = {"version": VERSION, "rows": n, "rows_digest": rec["rows_digest"],
         "basis": "derived"}
    d.update(out)
    safeio.write_text(sidecar(dst), json.dumps(d, indent=1, sort_keys=True))
    return d


def exts_of(path):
    """Every bank extension a stacked artifact records, in stacked order.

    `ext_of` answers with one, which is all a single-extension stack has. The
    documented growth path appends to an already stacked base
    (`release ++ ext ++ ext2`), and a reader that sees only the first one
    cannot reconstruct the row space at all: it computes a shorter expected
    length and rejects a bank that is perfectly well formed.
    """
    rec = read(path) or {}
    return re.findall(r"(bank_ext\w*)_meta\.npz", str(rec.get("row_space", "")))


def stacked_on_release(path):
    """Whether a sidecar claims this artifact is the release followed by exts.

    The distinction the loader needs is not length. An extension-only cache
    (`bank_ext_bal.f16.npy`, 750,000 rows) is *longer* than the 500,000-row
    release and holds none of its images, so any rule that reads "longer than
    the release" as "release plus something" hands the first 500,000
    extension rows over as the release's images -- valid shapes, valid
    indices, every photograph paired with another photograph's embedding.
    Only the recorded row space separates the two.
    """
    rec = read(path) or {}
    return str(rec.get("row_space", "")).startswith("release ++")


def ext_of(path):
    """Which bank extension a stacked artifact records, or None.

    The consumers used to recover this by scanning the *filename* for "70",
    "55" or "40" (items 10 and 9). Renaming a bank then selected another
    extension's metadata while every dimensional check still passed, because
    the extensions differ in length but the check only compared the bank
    against whichever metadata the scan happened to pick. The sidecar records
    it instead, so the answer comes from what was built rather than from what
    it was called.
    """
    rec = read(path) or {}
    m = re.search(r"(bank_ext\w*)_meta\.npz", str(rec.get("row_space", "")))
    return m.group(1) if m else None


def ext_for_bank(path, name):
    """The extension stacked under a bank: recorded if possible, else guessed.

    Three call sites carried their own copy of a filename scan for "70", "55"
    and "40" -- and one of those copies was written on 2026-09-03 to fix a
    different bug, which is how a workaround becomes a third instance of the
    problem it was working around. Renaming a bank made every copy pick the
    wrong extension while every dimensional check still passed.

    Returns (stem, how) so the caller can say which answer it got.
    """
    got = ext_of(path)
    if got:
        return got, "recorded"
    stem = "bank_ext"
    for tag in ("70", "55", "40"):
        if "bank" + tag in name:
            stem = "bank_ext" + tag
            break
    return stem, "guessed from the filename; run scripts/backfill_prov.py"


def projection_of(path):
    """The PCA basis an artifact was projected with, or None if unrecorded."""
    return (read(path) or {}).get("projection") or None


def leak_blocklist(root):
    """Image ids `screen_leak.py` found with their own coordinates burned in.

    Written since the screen existed and read by nothing (item 14), so a
    dashcam frame with its GPS printed across it stayed in the evaluation set
    and the model could read the answer off the image. 1.45% of OSV-5M frames
    carry such an overlay and are worth +1.3 pp if exploited, so the size of
    the hole is known even where the count here is small.

    Returns a set of ids, empty when no screen has run. Empty is the honest
    answer -- it means nothing was screened, not that nothing leaks, and the
    callers say which.
    """
    p = Path(root) / "leak_blocklist.json"
    if not p.exists():
        return set(), 0
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return set(), 0
    return ({str(e["id"]) for e in d.get("detail", []) if "id" in e},
            int(d.get("screened", 0)))


def encoder_of(path):
    """What encoder produced an embedding cache, as far as its sidecar says."""
    rec = read(path) or {}
    return {k: rec[k] for k in ENCODER_FIELDS if k in rec}


def sidecar(path):
    return Path(str(path) + ".prov.json")


def write(path, ids, basis="built", **extra):
    """Record what rows an artifact describes, atomically.

    `basis` is the honest part: "built" means this was written by the process
    that produced the file, so the digest is a fact about it; "observed" means
    it was backfilled onto an existing file and only asserts the alignment as
    of that day.
    """
    import safeio

    rec = {"version": VERSION, "rows": int(len(ids)),
           "rows_digest": rows_digest(ids), "basis": basis}
    rec.update(extra)
    safeio.write_text(sidecar(path), json.dumps(rec, indent=1, sort_keys=True))
    return rec


def read(path):
    p = sidecar(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def check(path, ids, what=None, digest=None):
    """Refuse an artifact whose rows are not the rows given.

    Returns True when a sidecar was present and matched, False when there was
    none to check -- so a caller can count how much of a build is actually
    covered rather than assuming a silent pass means a verified one.
    """
    what = what or Path(path).name
    rec = read(path)
    if rec is None:
        print("warning: {} has no provenance sidecar, so nothing proves its "
              "rows are this release's rows in this order (run "
              "scripts/backfill_prov.py)".format(what), flush=True)
        return False
    want = digest if digest is not None else rows_digest(ids)
    if rec.get("rows") != len(ids) or rec.get("rows_digest") != want:
        raise SystemExit(
            "{} describes {:,} rows with digest {}, but the dataset it is "
            "being paired with has {:,} rows with digest {}. Row i of one is "
            "not row i of the other, and nothing downstream would notice: "
            "every metric would come back in a believable range with each "
            "image scored against another image's data. Rebuild it, or "
            "re-run scripts/backfill_prov.py if you are certain the file is "
            "correct and only the sidecar is stale."
            .format(what, rec.get("rows", -1), rec.get("rows_digest"),
                    len(ids), want))
    return True
