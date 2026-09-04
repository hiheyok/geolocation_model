"""Join two street embedding caches side by side into one.

The dual encoder is not a fused model -- it is DINOv2 and SigLIP embedded
independently and concatenated, which is why it costs two passes over the
shards and no training change.  Row order is dataset.parquet order in both
inputs, so the join is positional.

Written in chunks through a memmap: at 500k images the output is 4.6 GB, which
is not something to hold in RAM alongside both inputs.
"""

import argparse
import json
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

    # Validate BEFORE opening the destination. `mode="w+"` truncates, so a
    # check that runs after the copy does not prevent a bad publication -- it
    # destroys a good artifact and then fails, leaving the previous run's valid
    # sidecar sitting beside the new bad bytes. The next consumer reads that
    # sidecar, finds a matching digest, and accepts them.
    #
    # Both inputs must be provenanced, not just one. Refusing only when both
    # sidecars exist and disagree is asymmetric in a way that is easy to miss:
    # a missing sidecar on A leaves the output unstamped, while a missing one
    # on B launders the result through A's digest, so the combined file claims
    # rows nothing ever established B has.
    ra, rb = prov.read(pa), prov.read(pb)
    if not ra or not rb:
        raise SystemExit(
            "{} has no provenance sidecar, so nothing establishes that the two "
            "inputs are the same rows in the same order. A positional join "
            "cannot be checked by row count -- two encoders run over different "
            "builds agree on length and on nothing else, and every image would "
            "get another image's second half. Run scripts/backfill_prov.py."
            .format(a.a if not ra else a.b))
    if ra.get("rows_digest") != rb.get("rows_digest"):
        raise SystemExit(
            "{} describes rows {} and {} describes rows {}; they are different "
            "builds, so joining them positionally pairs each image's features "
            "with another image's."
            .format(a.a, ra.get("rows_digest"), a.b, rb.get("rows_digest")))

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
    # Both inputs were verified equal before the destination was opened, so
    # the output describes those same rows. The second encoder and the scale
    # are recorded structurally: `encoder_of` returns only model/crops/size, so
    # without these a basis fitted on [DINO | SigLIP x4.03] and one fitted on
    # [DINO | some other SigLIP] compare equal and are reused across spaces.
    prov.carry(pa, out, n, joined_with=a.b, scale_b=a.scale_b,
               combined=json.dumps({"a": prov.encoder_of(pa),
                                    "b": prov.encoder_of(pb),
                                    "scale_b": a.scale_b}, sort_keys=True))

    s = np.asarray(C[:512], dtype=np.float32)
    print("\nwrote {}  {:.2f} GB   mean L2 {:.2f}   zero rows 0".format(
        out.name, out.stat().st_size / 1e9, float(np.linalg.norm(s, axis=1).mean())))


if __name__ == "__main__":
    main()
