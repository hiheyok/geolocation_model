"""Could the staleness bugs have reached numbers already published?

REVIEW6 #3 and #4 and REVIEW5 #9 all describe the same shape: a k-NN table is
bound to its street file only by *filename*, and to the requested split only by
what it says about itself. So a street cache rebuilt under the same name, or a
cache scored against a split it was not built for, produces neighbours that are
plausible, finite, in range, and wrong.

Those bugs are latent by nature -- they do not raise. The only way to know
whether they fired is to audit the artifacts every published result was
computed from. This checks what can still be checked after the fact:

  mtime       is the street file NEWER than the k-NN built on it? No content
              digest was recorded historically, so this is the only available
              staleness signal. It is one-sided: newer means suspect, older
              proves nothing about content, and a rebuild that preserved mtime
              is invisible.
  split       does the cache's recorded split_mode/split_hash match the LIVE
              dataset? This is what `knn_gap` and `gain_density` never asked --
              they compared the two caches with each other instead.
  bank rows   negative (they wrap), out of range, duplicated
  extension   does the cache's `bank_ext` name the same corpus the street
              file's sidecar records? By ordered id digest, not by name.
  neighbours  are the recorded neighbour ids actually members of bank_rows?

    OSV_RELEASE=s10 py scripts/audit_knn.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import knnmeta                                   # noqa: E402
import provenance as prov                        # noqa: E402
import splits as sp                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000,
                    help="rows to check for bank membership")
    ap.add_argument("--digest", action="store_true",
                    help="verify each cache against the bytes of its street "
                         "file. Off by default because the street caches are "
                         "5-11 GB and this reads every one of them.")
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    live = {m: sp.digest(*sp.read(ds, m)[:1]) if hasattr(sp, "digest") else None
            for m in ("sequence", "cell8")}
    caches = sorted(config.STREET_CACHE.glob("knn_*.npz"))
    print("{:,} k-NN caches under {}\n".format(len(caches),
                                               config.STREET_CACHE))

    hdr = "%-46s %-24s %7s %6s %6s %6s %6s" % (
        "knn cache", "street file", "newer?", "rows", "ext", "nbrs", "bytes")
    print(hdr); print("-" * len(hdr))
    flags = []
    for p in caches:
        if "LEAKY" in p.name:
            continue
        z = np.load(p, allow_pickle=True)
        sf = str(z["street_file"]) if "street_file" in z.files else "?"
        spath = config.STREET_CACHE / sf
        # One-sided: a street file newer than the table built from it is
        # suspect. Older proves nothing -- a same-mtime rewrite is invisible,
        # which is exactly why REVIEW6 #3 asks for a content digest. The
        # `bytes` column is the two-sided answer, for caches built since.
        newer = (spath.exists()
                 and spath.stat().st_mtime > p.stat().st_mtime)
        note = []
        if newer:
            note.append("street file is NEWER than this table")

        n_bank = None
        if spath.exists():
            n_bank = np.load(spath, mmap_mode="r").shape[0]
        rows_ok = "-"
        try:
            r = knnmeta.bank_rows(z, n_bank if n_bank else 10 ** 12, p.name)
            rows_ok = "ok" if r is not None else "none"
        except SystemExit as e:
            rows_ok = "BAD"
            note.append(str(e).split(":", 1)[-1].strip()[:60])

        ext_ok = "-"
        if spath.exists():
            try:
                knnmeta.check_ext(z, spath, p.name)
                ext_ok = "ok"
            except SystemExit as e:
                ext_ok = "BAD"
                note.append(str(e).split(":", 1)[-1].strip()[:60])

        # Are the stored neighbour ids members of the bank? build_knn writes
        # bank_rows[best_j], so every id must be in bank_rows -- an id outside
        # it means the table and its bank disagree about what was searched.
        nb_ok = "-"
        if "bank_rows" in z.files and "idx" in z.files:
            br = np.asarray(z["bank_rows"], np.int64)
            idx = np.asarray(z["idx"][:min(a.sample, z["idx"].shape[0])],
                             np.int64).ravel()
            miss = int(np.isin(idx, br, invert=True).sum())
            nb_ok = "ok" if not miss else "{} off".format(miss)
            if miss:
                note.append("{:,} neighbour ids are not in bank_rows"
                            .format(miss))

        # The stamp REVIEW6 #3 asked for. Every cache written before
        # 2026-09-05 lacks it and reads "none": that is not a failure, it is
        # the honest statement that those tables are bound to a filename only.
        by_ok = "-"
        if "street_digest" not in z.files:
            by_ok = "none"
        elif not a.digest:
            by_ok = "skip"
        elif not spath.exists():
            by_ok = "gone"
            note.append("street file {} is not on disk".format(sf))
        else:
            try:
                knnmeta.check_bytes(z, spath, p.name)
                by_ok = "ok"
            except SystemExit as e:
                by_ok = "STALE"
                note.append(str(e).split(":", 1)[-1].strip()[:60])

        print("%-46s %-24s %7s %6s %6s %6s %6s" % (
            p.name[:46], sf[:24], "YES" if newer else "no", rows_ok, ext_ok,
            nb_ok, by_ok))
        for m in note:
            print("      ! {}".format(m))
            flags.append((p.name, m))

    print("\n--- split metadata against the LIVE dataset ---")
    print("what knn_gap and gain_density never checked (REVIEW5 #9)\n")
    for p in caches:
        if "LEAKY" in p.name or "pyr768" not in p.name:
            continue
        z = np.load(p, allow_pickle=True)
        mode = str(z["split_mode"])
        labels = sp.read(ds, mode)[0]
        got = sp.hash_matches(mode, labels, z["split_hash"]) \
            if "split_hash" in z.files else None
        print("%-46s mode %-9s live-hash %s" % (p.name[:46], mode, got))
        if got not in ("exact", "legacy", True):
            flags.append((p.name, "split hash does not match live: " + str(got)))

    print("\n{} flag(s)".format(len(flags)))
    for n, m in flags:
        print("  {}: {}".format(n, m))


if __name__ == "__main__":
    main()
