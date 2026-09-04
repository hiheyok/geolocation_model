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
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import provenance as prov


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

    # Record the composed row space. Until now the merge was checked by total
    # length only, and every wrong order satisfies that too: an extension
    # stacked under the wrong base, or two extensions swapped, produces a file
    # of exactly the right shape. The digest of release-then-extension is the
    # thing that distinguishes them (item 66).
    ids = pq.read_table(config.DATASET_PARQUET, columns=["image_id"])
    base_ids = np.asarray(ids["image_id"])
    if len(base_ids) != n:
        raise SystemExit(
            "--base {} has {:,} rows but the release has {:,}; a stacked bank "
            "puts the release first, so this is not one".format(
                a.base, n, len(base_ids)))
    stem = prov.ext_stem_for(a.ext)
    if stem is None:
        raise SystemExit(
            "no metadata found for extension {!r}; it names the ids the "
            "second half of this file holds, and without it the stack cannot "
            "say what its rows are".format(a.ext))
    ext_ids = prov.bank_ext(stem, config.RELEASE)["image_id"]
    if len(ext_ids) != m:
        raise SystemExit(
            "{} describes {:,} images but {} holds {:,} rows".format(
                config.bank_meta(stem).name, len(ext_ids), a.ext, m))
    prov.write(out, np.concatenate([base_ids, ext_ids]),
               release=config.RELEASE,
               row_space="release ++ {}".format(config.bank_meta(stem).name),
               base=a.base, ext=a.ext)

    print("\nwrote {}  {:.2f} GB   rows 0..{:,} release, {:,}..{:,} bank-only"
          .format(out.name, out.stat().st_size / 1e9, n - 1, n, n + m - 1))


if __name__ == "__main__":
    main()
