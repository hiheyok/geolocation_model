"""An unfetched map tile is an all-zero token row, and that is a legal input.

`fetch_tiles` marks each completed row in `done.u8.npy` and leaves the rest of
the memmap at its zero fill.  Nothing read that mask, and `fetch_tiles` exited 0
even with fetches outstanding, so a run that lost tiles to a flaky server wrote
a success marker and then trained against uniform-zero histograms for every tile
it never got.  No loss can show you that: a zero histogram is well-formed.

These cover the reachability rule, which is the part that is easy to get wrong
in either direction -- too strict and every arm with a legitimately sparse z12
set refuses to start; too loose and the negatives path reads holes.  The wiring
was checked separately against the live s10 caches: both are complete (626,284
rows, 0 missing), construction is unchanged at 2.1 s for the 400,180-row train
split, and doctoring one on-path z12 row out of the mask makes it refuse.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import _check_fetched  # noqa: E402


def mask(path, n, holes=()):
    d = np.ones(n, np.uint8)
    for h in holes:
        d[h] = 0
    np.save(path, d)
    return path


# rows 0-3 are z4/z8, 4-7 are z12/z16; the split reads 0, 4, 5
ZS = np.array([4, 4, 8, 8, 12, 12, 16, 16], np.int64)
TOK_ROW = np.array([[0, 4, 5]], np.int64)


def test_a_complete_cache_passes(tmp_path):
    p = mask(tmp_path / "done.u8.npy", 8)
    _check_fetched(p, TOK_ROW, ZS, 0, "test")
    _check_fetched(p, TOK_ROW, ZS, 4, "train")


def test_a_hole_on_the_path_is_refused(tmp_path):
    p = mask(tmp_path / "done.u8.npy", 8, holes=[4])
    with pytest.raises(SystemExit, match="incomplete"):
        _check_fetched(p, TOK_ROW, ZS, 0, "test")


def test_an_unreachable_hole_is_ignored_until_negatives_reach_it(tmp_path):
    """Row 2 is a z8 tile this split never reads on-path.

    With negatives off it is unreachable and must not block a run.  With
    negatives on, `_negatives` descends into arbitrary z4/z8 siblings, so the
    same hole becomes reachable and must refuse -- the distinction the check
    exists to make.
    """
    p = mask(tmp_path / "done.u8.npy", 8, holes=[2])
    _check_fetched(p, TOK_ROW, ZS, 0, "test")
    with pytest.raises(SystemExit, match="incomplete"):
        _check_fetched(p, TOK_ROW, ZS, 4, "train")


def test_a_deep_hole_stays_unreachable_even_with_negatives(tmp_path):
    """Negatives are restricted to steps 1-2, whose inputs are z4 and z8.

    A missing z16 tile off the path is genuinely never read, and demanding it
    would refuse every arm whose fine levels are cached only on true paths.
    """
    p = mask(tmp_path / "done.u8.npy", 8, holes=[7])
    _check_fetched(p, TOK_ROW, ZS, 4, "train")


def test_a_stale_mask_is_refused_rather_than_reinterpreted(tmp_path):
    p = tmp_path / "done.u8.npy"
    np.save(p, np.ones(7, np.uint8))
    with pytest.raises(SystemExit, match="inconsistent"):
        _check_fetched(p, TOK_ROW, ZS, 0, "test")


def test_a_cache_without_a_mask_still_loads(tmp_path):
    """Caches predating the mask must keep working; this adds a check, not a
    migration."""
    _check_fetched(tmp_path / "absent.npy", TOK_ROW, ZS, 4, "train")
