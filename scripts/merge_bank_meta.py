"""Join two or more bank-extension metadata files into one index space.

`build_knn.py --bank-ext` takes a single stem and reads a single `_meta.npz`,
because when it was written there was one extension.  There are now two --
shards 10-24 and 25-39, 750k images each -- and corpus size is the axis with
the largest measured effect in this project (+20.4 pp for 400k -> 1.15M).
Rather than teach the kNN builder about a list of extensions, this makes the
list look like one extension, which leaves that code path untouched.

**The order is a contract, not a preference.**  `stack_bank.py` lays the
embedding file out as release rows first, then extension rows in the order it
was given them, and `build_knn` addresses row `n_rel + i` with metadata row `i`.
So the parts here must be listed in exactly the order they were stacked:

    stack_bank.py  --base pool_bal_bank25 --ext bank_ext2_pool --out pool_bal_bank40
    merge_bank_meta.py --parts bank_ext,bank_ext2 --out bank_ext40

Getting that backwards does not fail loudly -- every row keeps a valid-looking
z16 address, just the wrong one -- so the row counts are checked against the
stacked embedding file when `--emb` is given, which is the only cheap way to
catch it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config

KEYS = ("image_id", "x16", "y16", "sequence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="bank_ext,bank_ext2",
                    help="extension stems, in the order stack_bank.py used")
    ap.add_argument("--out", default="bank_ext40")
    ap.add_argument("--emb", default=None,
                    help="stacked embeddings to check the row count against, "
                         "e.g. pool_bal_bank40.f16.npy")
    a = ap.parse_args()

    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    if len(parts) < 2:
        raise SystemExit("nothing to merge: {}".format(parts))
    if a.out in parts:
        raise SystemExit("refusing to write over an input")

    metas, release, shards = [], None, []
    for stem in parts:
        p = config.bank_meta(stem)
        if not p.exists():
            raise SystemExit("missing {} -- has that embedding pass finished?"
                             .format(p))
        z = np.load(p, allow_pickle=True)
        missing = [k for k in KEYS if k not in z.files]
        if missing:
            raise SystemExit("{} lacks {}".format(p.name, missing))
        r = str(z["release"])
        if release is None:
            release = r
        elif r != release:
            raise SystemExit("{} is release {}, expected {}".format(
                p.name, r, release))
        metas.append(z)
        shards.extend(np.asarray(z["shards"]).tolist())
        print("{:<16} {:>9,} images   shards {}".format(
            stem, len(z["image_id"]), ",".join(np.asarray(z["shards"]).tolist())))

    # A duplicate image would sit in the bank twice: retrievable as its own
    # neighbour, and voting twice for its own cell.  build_bank_ext.py already
    # excludes the release from each part; only part-against-part is left.
    ids = np.concatenate([m["image_id"] for m in metas])
    uniq = np.unique(ids)
    if len(uniq) != len(ids):
        raise SystemExit("{:,} duplicate image_ids across {}".format(
            len(ids) - len(uniq), parts))
    if len(set(shards)) != len(shards):
        raise SystemExit("a shard appears in more than one part: {}".format(
            sorted(shards)))

    out = {k: np.concatenate([m[k] for m in metas]) for k in KEYS}
    out["shards"] = np.array(shards)
    out["release"] = release

    if a.emb:
        pe = config.STREET_CACHE / a.emb
        n_ext = len(ids)
        n_emb = np.load(pe, mmap_mode="r").shape[0]
        # the release occupies the rows before the extension
        n_rel = n_emb - n_ext
        if n_rel <= 0:
            raise SystemExit("{} has {:,} rows, fewer than the {:,} extension "
                             "images alone".format(a.emb, n_emb, n_ext))
        print("\n{}  {:,} rows = {:,} release + {:,} extension".format(
            a.emb, n_emb, n_rel, n_ext))

    p = config.bank_meta(a.out)
    np.savez(p, **out)
    print("\nwrote {}  {:,} images from {} parts".format(
        p.name, len(ids), len(parts)))
    print("rows are {} in that order; pass --bank-ext {} to build_knn.py"
          .format(" then ".join(parts), a.out))


if __name__ == "__main__":
    main()
