"""A partly-tiled selection must be refused, not filled.

`pool_pyramid.py` writes the blended street vector `l2((1-w)*L0 + w*L1)`. The
failure it has to prevent is not a crash: if some requested rows have no tiles,
the natural thing to do is fall back to `L0` for those, and the result is a
bank whose rows do not mean the same thing. One cosine ranks all of them, so an
untiled row has the query's `L1` half scored against its `L0` -- cross-space,
systematically lower -- and it is silently demoted.

Nothing downstream can see that. Every row is a unit vector of the right width,
the file is the right shape, and every count agrees. So the check has to happen
here, and it has to be a refusal rather than a fallback.

Two ways a row can be untiled, and both matter: absent from the cache's `rows`
selection, or present in it but never written (`done == 0`, i.e. still the
memmap's zero fill). The second is the dangerous one -- a zero row
L2-normalises to a unit-length nothing that ranks like a real vector, which is
the trap that would have built a bank 76% zeros.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from pool_pyramid import levels, nrm, tile_positions  # noqa: E402

D = 768


def cache(tmp_path, monkeypatch, rows, done, stem="t6"):
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    np.save(tmp_path / (stem + "_rows.i64.npy"), np.asarray(rows, np.int64))
    np.save(tmp_path / (stem + "_done.u8.npy"), np.asarray(done, np.uint8))
    return stem


# --- the guard ---------------------------------------------------------------

def test_a_complete_cache_returns_positions(tmp_path, monkeypatch):
    s = cache(tmp_path, monkeypatch, [0, 1, 2, 3], [1, 1, 1, 1])
    got = tile_positions(s, np.arange(4))
    assert np.array_equal(got, np.arange(4))


def test_positions_follow_the_selection_not_the_row_id(tmp_path, monkeypatch):
    """`rows` is a selection: row 7 may live at position 1."""
    s = cache(tmp_path, monkeypatch, [3, 7, 11], [1, 1, 1])
    assert np.array_equal(tile_positions(s, np.array([7, 3])), [1, 0])


def test_a_row_outside_the_selection_is_refused(tmp_path, monkeypatch):
    s = cache(tmp_path, monkeypatch, [0, 1, 2], [1, 1, 1])
    with pytest.raises(SystemExit, match="partly-tiled"):
        tile_positions(s, np.arange(5))


def test_a_row_present_but_unwritten_is_refused(tmp_path, monkeypatch):
    """The dangerous case: in `rows`, so a length check passes, but zero-fill."""
    s = cache(tmp_path, monkeypatch, [0, 1, 2, 3], [1, 1, 0, 1])
    with pytest.raises(SystemExit, match="partly-tiled"):
        tile_positions(s, np.arange(4))


def test_a_two_column_done_mask_needs_every_column(tmp_path, monkeypatch):
    """One column per encoder: a row is done only when both halves are."""
    s = cache(tmp_path, monkeypatch, [0, 1, 2],
              [[1, 1], [1, 0], [1, 1]])
    with pytest.raises(SystemExit, match="partly-tiled"):
        tile_positions(s, np.arange(3))
    s2 = cache(tmp_path, monkeypatch, [0, 1, 2],
               [[1, 1], [1, 1], [1, 1]], stem="t6b")
    assert np.array_equal(tile_positions(s2, np.arange(3)), np.arange(3))


def test_a_cache_with_no_rows_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    with pytest.raises(SystemExit, match="which rows"):
        tile_positions("absent", np.arange(3))


# --- the pooling algebra -----------------------------------------------------

def toks(n, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, 2, D, generator=g)


def test_levels_are_unit_and_per_encoder_balanced():
    """Each 768-d half is separately normalised, so neither encoder dominates.

    Raw DINOv2 and SigLIP norms are 82.96 and 20.58, which hands DINOv2 81% of
    an unnormalised cosine. Normalising per half is what makes --scale-b
    unnecessary here.
    """
    L0, L1 = levels(toks(8, 3), toks(8, 6, seed=1))
    for V in (L0, L1):
        assert torch.allclose(V.norm(dim=-1), torch.ones(8), atol=1e-5)
        # each half carries exactly half the squared norm
        assert torch.allclose(V[:, :D].norm(dim=-1) ** 2,
                              torch.full((8,), 0.5), atol=1e-5)


def test_no_tiles_gives_no_second_level():
    L0, L1 = levels(toks(4, 3))
    assert L1 is None and L0.shape == (4, 2 * D)


def test_the_blend_endpoints_are_the_levels():
    """w=0 must be exactly L0 and w=1 exactly L1, or the sweep is off by a step."""
    L0, L1 = levels(toks(6, 3), toks(6, 6, seed=2))
    assert torch.allclose(nrm(1.0 * L0 + 0.0 * L1), L0, atol=1e-6)
    assert torch.allclose(nrm(0.0 * L0 + 1.0 * L1), L1, atol=1e-6)


def test_the_blend_is_a_unit_vector_at_every_weight():
    L0, L1 = levels(toks(6, 3), toks(6, 6, seed=3))
    for w in (0.0, 0.04, 0.5, 0.9, 1.0):
        V = nrm((1 - w) * L0 + w * L1)
        assert torch.allclose(V.norm(dim=-1), torch.ones(6), atol=1e-5)


def test_tile_order_does_not_matter_but_tile_content_does():
    """Equal averaging is order-free -- which is why GeM measured inert -- but
    the level must still depend on the tiles it was given."""
    T = toks(5, 6, seed=4)
    a, _ = levels(toks(5, 3), T)
    b, _ = levels(toks(5, 3), T[:, torch.randperm(6)])
    assert torch.allclose(a, b, atol=1e-6)
    _, l1a = levels(toks(5, 3), T)
    _, l1b = levels(toks(5, 3), toks(5, 6, seed=5))
    assert not torch.allclose(l1a, l1b, atol=1e-3)
