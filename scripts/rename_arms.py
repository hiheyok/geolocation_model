"""Rename the ladder checkpoints to the self-describing scheme.

`s10_b55_c6` and `s10_w768_c4` differ in street width, bank size and epochs at
once, and neither name says any of it. A legend in the results table is a
workaround; the name should carry the facts. See `docs/NAMING.md`.

Only the 20 width/bank ladder arms are touched. The other 70-odd checkpoints are
architecture ablations on a different axis -- `ab_attn`, `mem_z4_seq`,
`retr_dual` -- and a width/bank name would be inventing facts about them.

Checkpoints are gitignored, so this is a local file rename with no effect on the
repository. What it *would* break is the old names in existing `BOOTSTRAP_*.md`
tables and commit messages, so `names.ALIASES` keeps mapping them and every
historical table still decodes.

Dry run by default. `--apply` performs it.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dir", default=str(ROOT / "checkpoints"))
    a = ap.parse_args()

    d = Path(a.dir)
    plan, clash, absent = [], [], []
    # ALIASES, not KNOWN: KNOWN is keyed by the new names, so iterating it
    # would look for files that by definition do not exist yet.
    for old, new in sorted(names.ALIASES.items()):
        src = d / (old + ".pt")
        dst = d / (new + ".pt")
        if not src.exists():
            absent.append(old)
            continue
        if dst.exists():
            clash.append((old, new))
            continue
        plan.append((src, dst, old, new))

    for old, new in clash:
        print("SKIP  {} -> {} already exists".format(old, new))
    if absent:
        print("absent, nothing to rename: {}".format(", ".join(absent)))
    for src, dst, old, new in plan:
        print("{:<26} -> {:<22} {:.0f} MB".format(
            old, new, src.stat().st_size / 1e6))

    if not a.apply:
        print("\n{} to rename. Dry run; pass --apply.".format(len(plan)))
        return
    for src, dst, old, new in plan:
        os.replace(src, dst)
    print("\nrenamed {} checkpoints".format(len(plan)))
    print("old names still decode through names.ALIASES, so existing "
          "BOOTSTRAP tables stay readable")


if __name__ == "__main__":
    main()
