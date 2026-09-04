"""Two consumers that read a cache without asking which of its rows are real.

An unwritten row is the zero fill.  After the per-token L2 normalisation the
fusion head applies, it becomes a unit-length nothing and trains as if it were
a real view -- so no loss, no metric and no assertion can show it.  The beam
search learned to refuse holes (`test_cache_completeness.py`); the fusion head
and the pyramid builder are separate consumers that the same fix never reached.

Each test here was checked by reverting its fix and watching it fail, which is
the discipline round two of the review found missing from three of round one's
forty.  The live s10 tile6 cache is complete -- 120,000 rows, 0 unwritten -- so
these have to construct the failure rather than wait for it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fuse_head import require_written  # noqa: E402


def mask(path, n, holes=()):
    d = np.ones(n, np.uint8)
    for h in holes:
        d[h] = 0
    np.save(path, d)
    return path


def test_a_complete_mask_passes(tmp_path):
    require_written(mask(tmp_path / "tile6_done.u8.npy", 16), 16, "tile6")


def test_an_unwritten_row_is_refused(tmp_path):
    p = mask(tmp_path / "tile6_done.u8.npy", 16, holes=[9])
    with pytest.raises(SystemExit, match="never written"):
        require_written(p, 16, "tile6")


def test_a_mask_from_another_build_is_refused(tmp_path):
    """A shorter mask is the dangerous case: it is all ones and describes a
    different file, so a count-based check reports full coverage."""
    p = mask(tmp_path / "tile6_done.u8.npy", 12)
    with pytest.raises(SystemExit, match="different builds"):
        require_written(p, 16, "tile6")


def test_a_missing_mask_warns_but_does_not_block(tmp_path, capsys):
    """Caches predating the mask exist; refusing them would strand real work."""
    require_written(tmp_path / "absent.u8.npy", 16, "tile6")
    assert "cannot prove" in capsys.readouterr().out


# --- the pyramid all-zero scan --------------------------------------------
#
# The check itself was never wrong; how it allocated was.  `X[ok][:, :, ei, :]`
# is fancy indexing, so it materialises BOTH encoders for every complete row as
# float32 and only then throws half away -- over 10 GB at full scale, the same
# shape of bug as the embed_street scan chunked earlier.  This pins the two
# forms to the same answer, so the cheap one can replace the expensive one.


def scan_chunked(X, ok, ei, step):
    z = 0
    for s in range(0, len(ok), step):
        blk = np.asarray(X[ok[s:s + step]][:, :, ei, :], np.float32)
        z += int((np.abs(blk).sum(-1) == 0).sum())
    return z


def test_the_chunked_scan_counts_what_the_whole_array_scan_counted():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((97, 33, 2, 8)).astype(np.float16)
    X[3, 5, 0] = 0                       # a hole in dinov2 only
    X[40, :, 1] = 0                      # every siglip token of one image
    ok = np.arange(97)
    for ei in (0, 1):
        whole = int((np.abs(np.asarray(X[ok][:, :, ei, :], np.float32)).sum(-1)
                     == 0).sum())
        for step in (1, 7, 96, 97, 500):
            assert scan_chunked(X, ok, ei, step) == whole
    assert scan_chunked(X, ok, 0, 20) == 1
    assert scan_chunked(X, ok, 1, 20) == 33


def test_the_chunked_scan_holds_on_an_empty_selection():
    X = np.ones((4, 33, 2, 8), np.float16)
    assert scan_chunked(X, np.zeros(0, np.int64), 0, 20000) == 0


# --- per-encoder masks -----------------------------------------------------
#
# The pyramid cache writes one column per encoder, so a row is real only when
# every encoder wrote it.  Checking column 0 alone passes an image that decoded
# for dinov2 and failed for siglip, leaving half its tokens zero behind a
# "done" flag.  The live pyr47 mask is (47,646, 2) and complete, so this had to
# be constructed too.

def test_a_two_column_mask_needs_every_encoder(tmp_path):
    p = tmp_path / "pyr_done.u8.npy"
    d = np.ones((16, 2), np.uint8)
    np.save(p, d)
    require_written(p, 16, "pyr")          # complete: fine

    d[9, 1] = 0                            # decoded for one encoder only
    np.save(p, d)
    with pytest.raises(SystemExit, match="never written"):
        require_written(p, 16, "pyr")


def test_a_two_column_mask_of_the_wrong_length_is_refused(tmp_path):
    p = tmp_path / "pyr_done.u8.npy"
    np.save(p, np.ones((12, 2), np.uint8))
    with pytest.raises(SystemExit, match="different builds"):
        require_written(p, 16, "pyr")


def test_both_mask_shapes_are_accepted(tmp_path):
    """tile6 writes (n,), the pyramid writes (n, 2); one helper serves both."""
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.ones(8, np.uint8))
    np.save(b, np.ones((8, 2), np.uint8))
    require_written(a, 8, "flat")
    require_written(b, 8, "per-encoder")
