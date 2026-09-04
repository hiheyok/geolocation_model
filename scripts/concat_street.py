"""Join two street embedding caches side by side into one.

The dual encoder is not a fused model -- it is DINOv2 and SigLIP embedded
independently and concatenated, which is why it costs two passes over the
shards and no training change.  Row order is dataset.parquet order in both
inputs, so the join is positional.

Written in chunks through a memmap: at 500k images the output is 4.6 GB, which
is not something to hold in RAM alongside both inputs.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import provenance as prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="embeddings_c3")
    ap.add_argument("--b", default="siglip_c3")
    ap.add_argument("--out", default="dual_c3")
    ap.add_argument("--scale-b", type=float, default=1.0,
                    help="multiply the second block before joining. Only "
                         "the joined vector is L2-normalised downstream, so "
                         "each block's weight in the cosine is its raw "
                         "activation norm -- DINOv2 is 82.96 against SigLIP's "
                         "20.58, i.e. 81/19, which nobody chose. 4.03 makes "
                         "them equal-norm.")
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()

    pa = config.STREET_CACHE / (a.a + ".f16.npy")
    pb = config.STREET_CACHE / (a.b + ".f16.npy")
    A = np.load(pa, mmap_mode="r")
    B = np.load(pb, mmap_mode="r")
    if A.shape[0] != B.shape[0]:
        raise SystemExit("row counts differ: {} has {:,}, {} has {:,}".format(
            a.a, A.shape[0], a.b, B.shape[0]))
    if A.ndim != 2 or B.ndim != 2:
        raise SystemExit("both inputs must be [N, D]; got {} and {}".format(
            A.shape, B.shape))

    n, d = A.shape[0], A.shape[1] + B.shape[1]
    out = config.STREET_CACHE / (a.out + ".f16.npy")
    print("release    {}".format(config.RELEASE))
    if a.scale_b != 1.0:
        print("scale b    {:.3f}".format(a.scale_b))
    print("inputs     {} {}  +  {} {}".format(a.a, A.shape, a.b, B.shape))
    print("output     {}  ({:,}, {})  {:.2f} GB".format(
        out.name, n, d, n * d * 2 / 1e9), flush=True)

    C = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16, shape=(n, d))
    zeros = 0
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        C[lo:hi, :A.shape[1]] = A[lo:hi]
        C[lo:hi, A.shape[1]:] = (B[lo:hi] if a.scale_b == 1.0 else
                                 np.asarray(B[lo:hi], dtype=np.float32)
                                 * a.scale_b)
        # A zero row means an image the embedding pass never wrote, which would
        # otherwise show up as a mysteriously bad arm rather than as an error.
        zeros += int((np.abs(np.asarray(C[lo:hi], dtype=np.float32)).sum(1) == 0).sum())
        if (lo // a.chunk) % 5 == 0:
            print("  {:>7,}/{:,}".format(hi, n), flush=True)
    C.flush()

    if zeros:
        raise SystemExit("{:,} all-zero rows -- an embedding pass did not "
                         "finish; delete its cache and rerun it".format(zeros))
    # Row i of the output is row i of A beside row i of B, so the two inputs
    # have to be the same rows in the same order. Equal row counts do not show
    # that -- two encoders run over two different builds of the release agree
    # on length and on nothing else, and the joined vector would pair each
    # image's DINOv2 with another image's SigLIP. Nothing downstream could
    # notice: the width is right, the norms are ordinary, and the arm just
    # comes out mysteriously weak.
    ra, rb = prov.read(pa), prov.read(pb)
    if ra and rb and ra.get("rows_digest") != rb.get("rows_digest"):
        raise SystemExit(
            "{} describes rows {} and {} describes rows {}; they are "
            "different builds, so joining them positionally pairs each "
            "image's features with another image's."
            .format(a.a, ra.get("rows_digest"), a.b, rb.get("rows_digest")))
    if not (ra and rb):
        print("warning: {} has no provenance sidecar, so nothing proves the "
              "two inputs are the same rows in the same order (run "
              "scripts/backfill_prov.py)".format(a.a if not ra else a.b),
              flush=True)
    prov.carry(pa, out, n, joined_with=a.b, scale_b=a.scale_b)

    s = np.asarray(C[:512], dtype=np.float32)
    print("\nwrote {}  {:.2f} GB   mean L2 {:.2f}   zero rows 0".format(
        out.name, out.stat().st_size / 1e9, float(np.linalg.norm(s, axis=1).mean())))


if __name__ == "__main__":
    main()
