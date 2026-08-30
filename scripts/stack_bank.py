"""Stack a bank extension underneath the release's embeddings.

One file, one index space: rows 0..n-1 are the release in dataset.parquet order
(so a query still indexes itself correctly), and rows n.. are the bank-only
extension.  A neighbour index therefore addresses both corpora without anyone
downstream needing to know where the boundary is.

Written through a memmap in chunks: 1.25M x 4608 fp16 is 11.5 GB.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dual_c3", help="the release's embeddings")
    ap.add_argument("--ext", default="bank_ext_dual", help="the extension's")
    ap.add_argument("--out", default="dual_c3_bank25")
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()

    pb = config.STREET_CACHE / (a.base + ".f16.npy")
    pe = config.STREET_CACHE / (a.ext + ".f16.npy")
    B = np.load(pb, mmap_mode="r")
    E = np.load(pe, mmap_mode="r")
    if B.shape[1] != E.shape[1]:
        raise SystemExit("dim mismatch: {} is {}, {} is {}".format(
            a.base, B.shape, a.ext, E.shape))

    n, m, d = B.shape[0], E.shape[0], B.shape[1]
    out = config.STREET_CACHE / (a.out + ".f16.npy")
    print("release    {:,} rows".format(n))
    print("extension  {:,} rows".format(m))
    print("output     {}  ({:,}, {})  {:.2f} GB".format(
        out.name, n + m, d, (n + m) * d * 2 / 1e9), flush=True)

    C = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(n + m, d))
    zeros = 0
    for src, off in ((B, 0), (E, n)):
        for lo in range(0, src.shape[0], a.chunk):
            hi = min(lo + a.chunk, src.shape[0])
            blk = np.asarray(src[lo:hi])
            C[off + lo:off + hi] = blk
            zeros += int((np.abs(blk.astype(np.float32)).sum(1) == 0).sum())
            if (lo // a.chunk) % 10 == 0:
                print("  {:>9,}/{:,}".format(off + hi, n + m), flush=True)
    C.flush()
    if zeros:
        raise SystemExit("{:,} all-zero rows -- an embedding pass did not "
                         "finish".format(zeros))

    print("\nwrote {}  {:.2f} GB   rows 0..{:,} release, {:,}..{:,} bank-only"
          .format(out.name, out.stat().st_size / 1e9, n - 1, n, n + m - 1))


if __name__ == "__main__":
    main()
