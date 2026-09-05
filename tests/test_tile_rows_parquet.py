"""A tile cache's `rows` array is indices into an image list, not into reality.

`tile_cache.py` learned `--parquet` so the 3,000,000-row bank extension can be
tiled at all -- it had only ever read the release's `dataset.parquet`, which
covers 500,000 rows and none of the extension.  That flag introduces a way for
two caches to be shaped identically and mean different things.

The failure it opens is not a crash.  `rows_p` stores a sorted selection of
*positions in whatever list was read*, so resuming a release-built cache under
`--parquet bank_ext.parquet` writes new rows addressed in the extension's space
beside old rows addressed in the release's.  Both halves are real embeddings of
real photographs, the mask is complete, the memmap is the right shape, and
every downstream count agrees.  The only thing wrong is which image each row
is.  So the row source is recorded in the metadata and compared on resume, the
same way the grid already was.

The subtle case is the legacy one, and it is the reason `explicit_src` exists.
Caches built before the field existed carry no `rows_parquet`.  Silence there
means "release", because there was no other way to build one -- but only for a
run that is *also* asking for the release.  Reading silence as agreement for a
`--parquet` run would let through exactly the mixture the field was added to
catch, which is a guard that fails open.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tile_cache import check_resume  # noqa: E402

REL = "dataset.parquet"
EXT = "bank_ext.parquet"


def meta(tmp_path, **kw):
    p = tmp_path / "m.npz"
    np.savez(p, **kw)
    return np.load(p, allow_pickle=True)


# --- the grid check, which already existed and must keep working -------------

def test_same_grid_and_source_resumes(tmp_path):
    m = meta(tmp_path, grid=np.array([3, 2]), rows_parquet=REL)
    check_resume(m, (3, 2), REL, "tile6", explicit_src=False)


def test_a_transposed_grid_is_refused(tmp_path):
    """3x2 and 2x3 both hold six tiles; the product cannot tell them apart."""
    m = meta(tmp_path, grid=np.array([3, 2]), rows_parquet=REL)
    with pytest.raises(SystemExit, match="grid 3x2"):
        check_resume(m, (2, 3), REL, "tile6", explicit_src=False)


# --- the row-source check ----------------------------------------------------

def test_resuming_a_release_cache_as_an_extension_is_refused(tmp_path):
    m = meta(tmp_path, grid=np.array([3, 2]), rows_parquet=REL)
    with pytest.raises(SystemExit, match="row spaces"):
        check_resume(m, (3, 2), EXT, "tile6", explicit_src=True)


def test_resuming_an_extension_cache_as_the_release_is_refused(tmp_path):
    """The mirrored direction fails too -- neither half is privileged."""
    m = meta(tmp_path, grid=np.array([3, 2]), rows_parquet=EXT)
    with pytest.raises(SystemExit, match="row spaces"):
        check_resume(m, (3, 2), REL, "tileX", explicit_src=False)


def test_two_different_extensions_do_not_mix(tmp_path):
    """bank_ext2 and bank_ext3 are the same length and disjoint."""
    m = meta(tmp_path, grid=np.array([3, 2]), rows_parquet="bank_ext2.parquet")
    with pytest.raises(SystemExit, match="row spaces"):
        check_resume(m, (3, 2), "bank_ext3.parquet", "tileE",
                     explicit_src=True)


def test_matching_extension_source_resumes(tmp_path):
    m = meta(tmp_path, grid=np.array([3, 2]), rows_parquet=EXT)
    check_resume(m, (3, 2), EXT, "tileE", explicit_src=True)


# --- the legacy case, where the guard could fail open ------------------------

def test_legacy_metadata_resumes_for_a_release_run(tmp_path):
    """No `rows_parquet` predates the flag, so it can only be a release cache."""
    m = meta(tmp_path, grid=np.array([3, 2]))
    check_resume(m, (3, 2), REL, "tile6", explicit_src=False)


def test_legacy_metadata_is_refused_for_a_parquet_run(tmp_path):
    """Silence must not read as agreement -- this is the fail-open case."""
    m = meta(tmp_path, grid=np.array([3, 2]))
    with pytest.raises(SystemExit, match="pre-dates"):
        check_resume(m, (3, 2), EXT, "tile6", explicit_src=True)


def test_metadata_with_no_grid_still_checks_the_source(tmp_path):
    """A missing grid skips one check; it must not skip the other."""
    m = meta(tmp_path, rows_parquet=REL)
    with pytest.raises(SystemExit, match="row spaces"):
        check_resume(m, (3, 2), EXT, "tile6", explicit_src=True)


# --- the flag will not overwrite the release cache ---------------------------

def test_parquet_without_out_refuses(tmp_path):
    """`tile6` is what every pyramid number on record was measured against.

    Same guard `embed_street.py` carries, for the same reason: a run over a
    different image list that lands on the default stem leaves a file of the
    right name and the wrong rows.
    """
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "tile_cache.py"),
         "--parquet", EXT, "--n", "1"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "OSV_RELEASE": "s10"})
    assert r.returncode != 0
    assert "--out" in (r.stderr + r.stdout)
