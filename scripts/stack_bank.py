"""Stack a bank extension underneath the release's embeddings.

One file, one index space: rows 0..n-1 are the release in dataset.parquet order
(so a query still indexes itself correctly), and rows n.. are the bank-only
extension.  A neighbour index therefore addresses both corpora without anyone
downstream needing to know where the boundary is.

`--base` may itself be a stacked bank, which is how every bank past the first
is grown: `--base pool_bal_bank25 --ext bank_ext2_pool --out pool_bal_bank40`.
The output's row space then names each extension in stacked order, so a reader
can reconstruct the whole layout rather than just its last step.

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


def _row_space(a, pb, pe, n, m):
    """Reconstruct and verify what rows this stack will describe.

    Split out of main so it can run before the destination is
    truncated: every refusal below has to happen while the previous
    output is still intact.
    """
    # Record the composed row space. Until now the merge was checked by total
    # length only, and every wrong order satisfies that too: an extension
    # stacked under the wrong base, or two extensions swapped, produces a file
    # of exactly the right shape. The digest of release-then-extension is the
    # thing that distinguishes them (item 66).
    ids = pq.read_table(config.DATASET_PARQUET, columns=["image_id"])
    rel_ids = np.asarray(ids["image_id"])

    # The base may be the release *or* an already stacked bank. Requiring it
    # to be release-length made the documented growth path impossible: every
    # bank past the first is built by appending to the previous one
    # (`--base pool_bal_bank25 --ext bank_ext2_pool`), and pool_bal_bank25 has
    # 1,250,000 rows against the release's 500,000, so the guard rejected the
    # exact command the workflow prescribes. The 2.0M, 2.75M and 3.5M banks on
    # disk exist only because they predate it.
    parts, names = [rel_ids], []
    if n != len(rel_ids):
        if not prov.stacked_on_release(pb):
            raise SystemExit(
                "--base {} has {:,} rows, the release has {:,}, and its "
                "provenance does not say it is the release followed by bank "
                "extensions. A stack puts the release first; this cannot be "
                "shown to.".format(a.base, n, len(rel_ids)))
        for st in prov.exts_of(pb):
            parts.append(np.asarray(prov.bank_ext(st, config.RELEASE)["image_id"]))
            names.append(config.bank_meta(st).name)
        if sum(len(p) for p in parts) != n:
            raise SystemExit(
                "--base {} has {:,} rows but the release plus its recorded "
                "extensions {} come to {:,}".format(
                    a.base, n, names, sum(len(p) for p in parts)))
    # Verify the inputs BEFORE certifying the output. Writing a "built"
    # sidecar off unchecked inputs launders a reordered base into a file that
    # then reads as the intended order, and build_knn trusts that sidecar --
    # so the error becomes unrecoverable at exactly the point it is recorded.
    prov.check_stack(pb, parts, a.base + ".f16.npy")

    stem = prov.ext_stem_for(a.ext)
    if stem is None:
        raise SystemExit(
            "no metadata found for extension {!r}; it names the ids the "
            "second half of this file holds, and without it the stack cannot "
            "say what its rows are".format(a.ext))
    ext_ids = np.asarray(prov.bank_ext(stem, config.RELEASE)["image_id"])
    if len(ext_ids) != m:
        raise SystemExit(
            "{} describes {:,} images but {} holds {:,} rows".format(
                config.bank_meta(stem).name, len(ext_ids), a.ext, m))
    prov.check(pe, ext_ids, a.ext + ".f16.npy")
    if config.bank_meta(stem).name in names:
        raise SystemExit(
            "--base {} already carries {}; stacking it again would put the "
            "same rows in the index twice, and a query would retrieve its own "
            "duplicate as a neighbour".format(a.base, config.bank_meta(stem).name))
    names.append(config.bank_meta(stem).name)
    parts.append(ext_ids)
    return parts, names, ext_ids


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

    # Everything that can refuse this build runs BEFORE the destination is
    # opened. `mode="w+"` truncates, so a check placed after the copy cannot
    # prevent a bad publication -- it destroys whatever was at `out` and then
    # exits, leaving the previous run's still-valid sidecar beside the new
    # partial bytes. The next consumer compares that sidecar, finds a matching
    # digest, and accepts them.
    parts, names, ext_ids = _row_space(a, pb, pe, n, m)

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

    prov.write(out, np.concatenate(parts),
               release=config.RELEASE,
               row_space="release ++ " + " ++ ".join(names),
               base=a.base, ext=a.ext)

    print("\nwrote {}  {:.2f} GB   rows 0..{:,} release, {:,}..{:,} bank-only"
          .format(out.name, out.stat().st_size / 1e9, n - 1, n, n + m - 1))


if __name__ == "__main__":
    main()
