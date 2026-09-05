"""One validator for completion masks, because three producers disagreed.

A cache here records what it finished in a `done.u8.npy` beside the data, and
an unwritten row is a zero row -- which for a token histogram or an embedding
is a perfectly well-formed value. So the mask is the only thing standing
between an interrupted build and a model trained on blanks, and it is worth
exactly as much as the checks applied to it.

Those checks kept being added in one place at a time. `fetch_tiles` grew a
shape-and-values validator; `pyramid_cache` grew its own; the training consumer
in `dataset.py` defined "fetched" as `!= 0` until round three; `tile_cache`
loaded its resume mask with no checks at all, so a `2` counted as complete and
`fuse_head` then cast the mask to bool and agreed. Same guarantee, four
implementations, three of them wrong at some point.

Two properties are worth naming because both have been violated:

**Binary, not truthy.** `!= 0` is the natural spelling and it is the bug: a 2
or a 255 from a torn write or a half-migrated writer certifies a zero row.

**Counted, not summed.** `done.sum() >= n` looks like a completion test and is
not one -- a mask holding a single 2 and a single 0 passes it while a row is
still blank. Count the ones.
"""

import numpy as np


def load_mask(path, n, what="cache"):
    """Read a completion mask, refusing one that cannot mean what it says.

    A short mask belongs to a different build; a non-binary one is not a
    completion mask at all. Both are refusals rather than warnings, because
    the thing they protect -- "this row holds real data" -- has no other
    evidence behind it.
    """
    d = np.load(path)
    if d.shape != (n,):
        raise SystemExit(
            "{} completion mask is {} but this build wants {}; it belongs to "
            "a different cache. Delete it to start fresh."
            .format(what, d.shape, (n,)))
    check_mask(d, what)
    return d.astype(np.uint8)


def check_mask(d, what="cache"):
    """Refuse a mask holding anything but 0 and 1."""
    bad = np.unique(np.asarray(d)[(np.asarray(d) != 0) & (np.asarray(d) != 1)])
    if len(bad):
        raise SystemExit(
            "{} completion mask holds {} -- it is not a completion mask, so "
            "'done' cannot be read off it. Delete it to start fresh."
            .format(what, "values " + ", ".join(str(int(v)) for v in bad[:5])))
    return d


def complete(d):
    """How many rows are actually finished.

    `sum()` is not this. A mask with one 2 and one 0 sums to n and is missing
    a row, which is how an incomplete cache passed its own final check.
    """
    return int((np.asarray(d) == 1).sum())


def is_complete(d, n):
    return complete(d) == n
