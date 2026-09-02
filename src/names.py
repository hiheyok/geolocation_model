"""Turn a checkpoint tag into something a person can read.

Tags accreted rather than being designed. `s10_b55_c6` and `s10_w768_c4` differ
in two dimensions at once -- street-vector width and bank size -- and neither is
visible in the name, so every results table needed a key held in someone's head.
`s10_bal_bank25_c6` and `s10_pool_c6` are the same bank at different widths;
`s10_b55` and `s10_b40` are the same width at different banks. Nothing says so.

Renaming the checkpoints is not worth it -- they are named in every BOOTSTRAP
table and commit message already written, and renaming would quietly invalidate
that record. So this decodes instead, and tables print both: the tag that
identifies the file, and a label that says what it is.

**The table is explicit, not inferred.** A first attempt parsed the tags with
patterns and confidently labelled `s10_pool_c6` as a 0.40M bank when it was
trained on 1.15M -- the bank name simply is not in that tag. A wrong label in a
results table is worse than no label, so unknown tags return "" and the pattern
fallback only fires for names that follow the documented scheme.

**New runs should use the scheme in `docs/NAMING.md`**, which puts width, bank
and epochs in the tag itself. This exists for everything already on disk.
"""

import re

# tag -> (street width, bank in millions of images, epochs, split)
# Read off the runner that produced each arm, not guessed from the name.
KNOWN = {
    "d1536-b115-e2": (1536, 1.15, 2, 'sequence'),
    "d1536-b115-e2-cell8": (1536, 1.15, 2, 'cell8'),
    "d1536-b115-e4": (1536, 1.15, 4, 'sequence'),
    "d1536-b115-e4-cell8": (1536, 1.15, 4, 'cell8'),
    "d1536-b115-e6": (1536, 1.15, 6, 'sequence'),
    "d1536-b115-e6-cell8": (1536, 1.15, 6, 'cell8'),
    "d1536-b190-e2": (1536, 1.9, 2, 'sequence'),
    "d1536-b190-e4": (1536, 1.9, 4, 'sequence'),
    "d1536-b190-e6": (1536, 1.9, 6, 'sequence'),
    "d1536-b265-e2": (1536, 2.65, 2, 'sequence'),
    "d1536-b265-e4": (1536, 2.65, 4, 'sequence'),
    "d1536-b265-e6": (1536, 2.65, 6, 'sequence'),
    "d1536-b350-e2": (1536, 3.5, 2, 'sequence'),
    "d1536-b350-e4": (1536, 3.5, 4, 'sequence'),
    "d1536-b350-e6": (1536, 3.5, 6, 'sequence'),
    "d4608-b115-e2": (4608, 1.15, 2, 'sequence'),
    "d4608-b115-e2-cell8": (4608, 1.15, 2, 'cell8'),
    "d4608-b115-e4": (4608, 1.15, 4, 'sequence'),
    "d4608-b115-e4-cell8": (4608, 1.15, 4, 'cell8'),
    "d4608-b115-e6": (4608, 1.15, 6, 'sequence'),
    "d768-b265-e2": (768, 2.65, 2, 'sequence'),
    "d768-b265-e4": (768, 2.65, 4, 'sequence'),
    "d768-b265-e6": (768, 2.65, 6, 'sequence'),
}

# Names used before docs/NAMING.md existed. They appear in every
# BOOTSTRAP_*.md and commit message already written, so they keep
# decoding; the files themselves were renamed by
# scripts/rename_arms.py.
ALIASES = {
    "s10_b40": "d1536-b190-e2",
    "s10_b40_c4": "d1536-b190-e4",
    "s10_b40_c6": "d1536-b190-e6",
    "s10_b55": "d1536-b265-e2",
    "s10_b55_c4": "d1536-b265-e4",
    "s10_b55_c6": "d1536-b265-e6",
    "s10_b70": "d1536-b350-e2",
    "s10_b70_c4": "d1536-b350-e4",
    "s10_b70_c6": "d1536-b350-e6",
    "s10_bal_bank25": "d4608-b115-e2",
    "s10_bal_bank25_c4": "d4608-b115-e4",
    "s10_bal_bank25_c6": "d4608-b115-e6",
    "s10_cell8_bank25_lr1e4": "d4608-b115-e2-cell8",
    "s10_cell8_bank25_lr1e4_c": "d4608-b115-e4-cell8",
    "s10_pool": "d1536-b115-e2",
    "s10_pool_c4": "d1536-b115-e4",
    "s10_pool_c6": "d1536-b115-e6",
    "s10_pool_c8": "d1536-b115-e2-cell8",
    "s10_pool_c8_c4": "d1536-b115-e4-cell8",
    "s10_pool_c8_c6": "d1536-b115-e6-cell8",
    "s10_w768": "d768-b265-e2",
    "s10_w768_c4": "d768-b265-e4",
    "s10_w768_c6": "d768-b265-e6",
}

# the documented scheme: d<width>-b<bank in units of 0.01M>-e<epochs>[-cell8]
_SCHEME = re.compile(r"d(\d+)-b(\d+)-e(\d+)(?:-(\w+))?$")


def parts(tag):
    """(width, bank_millions, epochs, split) or None."""
    tag = ALIASES.get(tag, tag)
    if tag in KNOWN:
        return KNOWN[tag]
    m = _SCHEME.search(tag)
    if m:
        return (int(m.group(1)), int(m.group(2)) / 100.0, int(m.group(3)),
                m.group(4) or "sequence")
    return None


def describe(tag):
    """A short human label, or '' when the tag is not one we can name honestly."""
    p = parts(tag)
    if not p:
        return ""
    width, bank, ep, split = p
    # " / " not " | ": these strings land inside markdown table cells, where a
    # pipe silently splits one column into four.
    s = "{}-d / {:.2f}M bank / {} ep".format(width, bank, ep)
    return s if split == "sequence" else s + " / " + split


def label(tag, width=20):
    d = describe(tag)
    return "{:<{w}} {}".format(tag, "  " + d if d else "", w=width)


def new_tag(width, bank_m, epochs, split="sequence"):
    """Build a tag in the documented scheme, for runs from here on."""
    t = "d{}-b{:03.0f}-e{}".format(width, bank_m * 100, epochs)
    return t if split == "sequence" else t + "-" + split


if __name__ == "__main__":
    for t in sorted(KNOWN) + ["d768-b350-e6", "d1536-b350-e6-cell8",
                              "something_else"]:
        print("  {:<28} {}".format(t, describe(t) or "(not decodable)"))
    print("\n  new_tag(768, 3.50, 6) ->", new_tag(768, 3.50, 6))
