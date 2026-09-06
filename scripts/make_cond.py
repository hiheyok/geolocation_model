"""Join a retrieval cache and a conditioning cache into one street file.

The decoupled design needs the model to see more than the retrieval vector,
without the extra block ever reaching a cosine. `runs/PYR_LEVELS.md` measured
why that separation matters: pushed *into* the retrieval vector against a fixed
`L0L1` bank, an extra level at weight 0.10 replaces 8.2% of the top-16 and
flips the top-1 for 12.7% of queries, and the hit rate does not move at all
(59.6% -> 59.6%). It reorders without being about location.

The join is a concatenation, retrieval block first:

    [ retrieval 768 | conditioning 1536 ]  = 2304-d

and `StreetProj` splits it back apart. One tensor rather than a second dataset
field is deliberate: `fuse_flat` carries a comment recording that beam search
once inlined the fusion concatenation and drifted from training, so threading a
new argument through `fuse` / `policy_from` / `forward` / `beam.search` would
rebuild exactly that hazard. A wider tensor changes no signature anywhere.

**The sidecar records which prefix is the retrieval space**, because a name is
not enough for the consumer to check. `GeoStepDataset` compares the k-NN
cache's `street_file` against `retrieval_file` recorded here, and then verifies
the prefix *bytes* against that file on a sample of rows -- which catches a
rebuilt retrieval cache that kept its name, something no string comparison can.

    OSV_RELEASE=s10 py scripts/make_cond.py \
        --retrieval pyr768_mix.f16.npy --cond cond_native.f16.npy \
        --out pyr768_mix_cond.f16.npy
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import maskio                                    # noqa: E402
import safeio                                    # noqa: E402
import provenance as prov                        # noqa: E402


def _rows_of(path, what):
    """The sidecar's row identity, or exit.

    A missing sidecar is fatal here rather than a warning. `prov.carry` warns
    and returns `None` when the *retrieval* source has none, which leaves the
    join with no provenance at all -- and an unprovenanced artifact is exactly
    what the conditioning path must not produce, because the whole decoupled
    design rests on the consumer being able to check that the retrieval prefix
    is the space the k-NN was built in.
    """
    rec = prov.read(path)
    if rec is None:
        sys.exit(
            "{} has no provenance sidecar, so nothing establishes which "
            "images its rows are or what order they are in. Run "
            "scripts/backfill_prov.py, or rebuild it.".format(what))
    if "rows" not in rec or "rows_digest" not in rec:
        sys.exit("{}'s sidecar records no row identity: {}"
                 .format(what, sorted(rec)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True,
                    help="the cache the k-NN was built on; goes first")
    ap.add_argument("--cond", required=True,
                    help="conditioning cache, appended; never retrieved on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=50000)
    a = ap.parse_args()

    rp = config.STREET_CACHE / a.retrieval
    cp = config.STREET_CACHE / a.cond
    out = config.STREET_CACHE / a.out
    if out in (rp, cp):
        sys.exit("refusing to write over an input")

    R = np.load(rp, mmap_mode="r")
    C = np.load(cp, mmap_mode="r")
    if R.shape[0] != C.shape[0]:
        sys.exit("{} has {:,} rows and {} has {:,}; they must describe the "
                 "same images in the same order"
                 .format(a.retrieval, R.shape[0], a.cond, C.shape[0]))
    n, dr, dc = R.shape[0], R.shape[1], C.shape[1]

    # Equal row COUNTS were the only thing establishing that row i of one is
    # row i of the other, and a count cannot see a reordering or a same-length
    # replacement. The output then took the retrieval file's authoritative row
    # digest through `prov.carry` -- so a conditioning cache describing other
    # photographs became a fully provenanced artifact, and training read
    # another image's high-resolution features as if they were this row's
    # (REVIEW5 #5). Both sidecars, present and equal, checked here.
    rrec, crec = _rows_of(rp, a.retrieval), _rows_of(cp, a.cond)
    if (rrec["rows"], rrec["rows_digest"]) != (crec["rows"], crec["rows_digest"]):
        sys.exit(
            "{} describes {:,} rows with digest {} and {} describes {:,} with "
            "digest {}. Row i of one is not row i of the other. The counts "
            "agree, every block copies cleanly, and the join would inherit "
            "the retrieval file's provenance -- so nothing downstream would "
            "ever say the conditioning belongs to different photographs."
            .format(a.retrieval, rrec["rows"], rrec["rows_digest"],
                    a.cond, crec["rows"], crec["rows_digest"]))
    if rrec["rows"] != n:
        sys.exit("{} holds {:,} rows but its sidecar describes {:,}"
                 .format(a.retrieval, n, rrec["rows"]))

    # A conditioning pass is resumable, so it can be half finished. A zero row
    # is not an error downstream -- it is a well-formed vector of nothing, and
    # the adapter would train against it as though it were real detail.
    stem = a.cond.replace(".f16.npy", "")
    done_p = config.STREET_CACHE / (stem + "_done.u8.npy")
    meta_p = config.STREET_CACHE / (stem + "_meta.npz")
    if done_p.exists():
        d = maskio.load_mask(done_p, n, a.cond)
        if not maskio.is_complete(d, n):
            sys.exit("{} has {:,} of {:,} rows written. Joining now would give "
                     "the model zero-filled conditioning for the rest, which "
                     "is a well-formed vector of nothing and trains like real "
                     "detail. Finish the pass."
                     .format(a.cond, maskio.complete(d), n))
    else:
        print("warning: {} has no done-mask, so completeness cannot be "
              "checked".format(a.cond), flush=True)

    print("{}  {:,} x {}  (retrieval)".format(a.retrieval, n, dr))
    print("{}  {:,} x {}  (conditioning)".format(a.cond, n, dc))
    print("{}  {:,} x {}   {:.2f} GB".format(
        a.out, n, dr + dc, n * (dr + dc) * 2 / 1e9), flush=True)

    # Every check that can be made is made before the destination exists. An
    # output opened first and rejected second leaves new bytes beside whatever
    # sidecar the previous run wrote, which is the stale-sidecar hazard one
    # layer down (REVIEW5 #3). Drop that sidecar too: from here until
    # `prov.carry` succeeds, this file has no provenance and must not look as
    # though it does.
    stale = prov.sidecar(out)
    if stale.exists():
        stale.unlink()
    Y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(n, dr + dc))
    zeros = 0
    for s in range(0, n, a.block):
        e = min(s + a.block, n)
        Y[s:e, :dr] = R[s:e]
        blk = np.asarray(C[s:e], np.float32)
        zeros += int((np.abs(blk).sum(1) == 0).sum())
        Y[s:e, dr:] = C[s:e]
        print("  {:>9,}/{:,}".format(e, n), flush=True)
    Y.flush()
    # Close the mapping before touching the file as a file. Windows refuses to
    # unlink a path that still has a live mapping, so the zero-row guard below
    # raised PermissionError instead of its own message and left the
    # half-written join on disk -- the failure it exists to prevent, with a
    # worse diagnostic.
    mm = getattr(Y, "_mmap", None)
    if mm is not None:
        mm.close()
    del Y
    if zeros:
        out.unlink()
        sys.exit("{:,} conditioning rows are all zero".format(zeros))

    # The consumer needs to know where the retrieval block ends, and needs to
    # be able to *check* it rather than trust the name -- a retrieval cache
    # rebuilt under the same filename is exactly the failure a string
    # comparison cannot see.
    # The digest of the WHOLE retrieval file, not a sample of rows. A fixed
    # sample ignores the same rows forever, so a localized rebuild or a
    # corrupted block outside it passes every time rather than eventually
    # being caught. REVIEW6 #2.
    # The conditioning side's identity is published too, not just the
    # retrieval side's. `carry` copies the retrieval sidecar's row digest onto
    # the output by design -- that is what makes it a row-preserving transform
    # -- so without these fields the artifact records nothing whatsoever about
    # which conditioning cache went into it.
    prov.carry(rp, out, n, retrieval_file=a.retrieval, retrieval_dim=dr,
               retrieval_digest=safeio.content_digest(rp),
               cond_file=a.cond, cond_dim=dc,
               cond_digest=safeio.content_digest(cp),
               cond_rows_digest=crec["rows_digest"],
               cond_done_digest=safeio.content_digest(done_p),
               cond_meta_digest=safeio.content_digest(meta_p))
    print("\nwrote {}  retrieval block [0:{}), conditioning [{}:{})".format(
        out.name, dr, dr, dr + dc), flush=True)


if __name__ == "__main__":
    main()
