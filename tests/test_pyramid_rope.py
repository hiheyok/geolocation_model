"""Pooling by mean is permutation-invariant, and that is the defect.

`l2((L0+L1)/2)` averages 3 crops and 6 tiles into one vector, so a feature in
the left crop and the same feature in the right crop produce the *same* pooled
vector and the retrieval cosine cannot separate them. It is the street-side
version of what `map-pooling-is-spatially-blind` found on the map side, where
replacing a mean pool with rotary positions plus attention pooling moved the
median from 375 to 351 km.

The fix here is a fixed rotation, not a learned one, which is what makes it
affordable: no training signal is needed, so the whole 3.4M-row corpus can be
re-pooled from vectors already on disk. Rotating view `k` by `R(p_k)` before
summing makes the pooled inner product

    <sum_k R(p_k) x_k, sum_l R(p_l) y_l> = sum_kl <x_k, R(p_l - p_k) y_l>

which depends only on *relative* position: same-position matches keep full
credit, cross-position matches are attenuated.

The load-bearing property is the last group of tests. `rope_max = 0` must
reproduce the unrotated pooling **exactly**, so the shipping bank is the
zero point of the new parameterisation rather than a separate code path. An
arm ladder whose baseline is only approximately the shipping arm measures the
approximation as well as the treatment.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pool_pyramid as P  # noqa: E402

CROPS, GRID = 3, (3, 2)          # dual_c3 and tile6, the shipping pair
PAIRS = P.D_ENC // 2


def views(b=4, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, 2, P.D_ENC, generator=g)


# ------------------------------------------------------- the coordinates ----

def test_a_crop_sits_at_the_centre_of_the_column_it_covers():
    """A full-height crop covers one tile column and both of its rows."""
    p0, p1 = P.view_positions(CROPS, GRID)
    assert p0 == [(0.0, 0.5, 0.0), (0.0, 0.5, 1.0), (0.0, 0.5, 2.0)]


def test_tiles_are_row_major_over_the_grid():
    """`tile_cache` writes axis 1 row-major; the positions must agree."""
    _, p1 = P.view_positions(CROPS, GRID)
    assert p1 == [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, 2.0),
                  (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 2.0)]


def test_a_transposed_grid_gives_different_positions():
    """3x2 and 2x3 hold six tiles either way, so the grid cannot be inferred
    from the count -- which is why `tile_grid` reads it rather than deriving
    it. If these agreed, reading it would be pointless."""
    assert P.view_positions(2, (2, 3))[1] != P.view_positions(3, (3, 2))[1]


def test_a_crop_and_its_children_share_a_column():
    p0, p1 = P.view_positions(CROPS, GRID)
    for c in range(CROPS):
        kids = [p for p in p1 if p[2] == c]
        assert len(kids) == GRID[1]
        assert all(k[2] == p0[c][2] for k in kids)


# ------------------------------------------------------------ the angles ----

def test_every_axis_gets_the_full_frequency_band():
    """Round-robin, not contiguous sections. With contiguous sections the axis
    holding the slow end barely rotates and contributes nothing, and the
    coordinates here span 0 to 2 -- far too short to recover from that."""
    ang = P.rope_angles([(1.0, 1.0, 1.0)], PAIRS, ["depth", "row", "col"],
                        1.0, 1000.0)[0]
    for i in range(3):
        own = ang[i::3]
        assert own.max() == pytest.approx(1.0)
        assert own.min() == pytest.approx(1e-3, rel=1e-6)


def test_an_axis_that_is_not_listed_never_turns():
    ang = P.rope_angles([(1.0, 1.0, 1.0)], PAIRS, ["col"], 1.0, 1000.0)[0]
    assert np.count_nonzero(ang) == PAIRS  # every pair belongs to the one axis
    ang2 = P.rope_angles([(1.0, 0.0, 0.0)], PAIRS, ["col"], 1.0, 1000.0)[0]
    assert not np.any(ang2), "col rotated on a view whose col is 0"


def test_the_angle_is_proportional_to_the_coordinate():
    """Relative position is what the inner product sees, so the map from
    coordinate to angle has to be linear."""
    a1 = P.rope_angles([(0.0, 0.0, 1.0)], PAIRS, ["col"], 0.4, 100.0)
    a2 = P.rope_angles([(0.0, 0.0, 2.0)], PAIRS, ["col"], 0.4, 100.0)
    assert np.allclose(a2, 2 * a1)


def test_a_language_model_base_would_be_nearly_a_no_op_here():
    """Why this is parameterised by an angle and not by a base period.

    Standard RoPE is `theta_j = base^(-2j/d)`. Over the whole coordinate range
    this pyramid has -- 0 to 2 -- base 10,000 turns the median pair 1.16
    degrees and leaves 48% of pairs under one degree. That is within rounding
    of the identity this is an alternative to.
    """
    th = 10000.0 ** (-2 * np.arange(PAIRS) / P.D_ENC)
    deg = np.degrees(2 * th)
    assert np.median(deg) < 2.0
    assert np.mean(deg < 1.0) > 0.4


def test_the_default_spread_rotates_the_whole_vector():
    """The language-model spread is the wrong one over three positions.

    A geometric spread leaves most dimensions slow, which pays over thousands
    of positions and does not here: at spread 1000, 41% of pairs turn under a
    degree per unit and carry no position at any coordinate this pyramid has.
    The default has to keep the slowest pair meaningfully turning.
    """
    th = P.rope_angles([(0.0, 0.0, 1.0)], PAIRS, ["col"], 1.0, P.ROPE_SPREAD)[0]
    assert np.degrees(th.min()) > 3.0, "the slowest pair is nearly identity"
    assert np.degrees(np.median(th)) > 10.0

    slow = P.rope_angles([(0.0, 0.0, 1.0)], PAIRS, ["col"], 1.0, 1000.0)[0]
    assert np.mean(np.degrees(slow) < 1.0) > 0.4, "1000 was not the problem"


# ---------------------------------------------------------- the rotation ----

def test_the_rotation_preserves_every_norm():
    """It is orthogonal, which is why it commutes with `nrm` and changes only
    the angles *between* views."""
    V = views(n=6)
    ang = torch.from_numpy(P.rope_angles(
        P.view_positions(CROPS, GRID)[1], PAIRS,
        ["depth", "row", "col"], 0.7, 1000.0))
    R = P.apply_rope(V, ang)
    assert torch.allclose(R.norm(dim=-1), V.norm(dim=-1), atol=1e-4)


def test_the_pooled_vector_stops_being_permutation_invariant():
    """The whole point. Swap two views and the mean is unchanged; the rotated
    sum is not."""
    V = views(n=3)
    S = V[:, [2, 1, 0]]
    assert torch.allclose(V.mean(1), S.mean(1), atol=1e-6)
    ang = torch.from_numpy(P.rope_angles(
        P.view_positions(CROPS, GRID)[0], PAIRS, ["col"], 0.7, 1000.0))
    assert not torch.allclose(P.apply_rope(V, ang).mean(1),
                              P.apply_rope(S, ang).mean(1), atol=1e-3)


def test_a_same_position_match_keeps_more_credit_than_a_moved_one():
    """The property the retrieval cosine is being given: two images sharing a
    feature in the same view should score above two sharing it in different
    views. Under a plain mean those are identical."""
    b, feat = 1, views(b=1, n=1)[:, 0]
    ang = torch.from_numpy(P.rope_angles(
        P.view_positions(CROPS, GRID)[0], PAIRS, ["col"], 0.7, 1000.0))

    def pooled(slot):
        V = torch.zeros(b, CROPS, 2, P.D_ENC)
        V[:, slot] = feat
        return P.apply_rope(V, ang).mean(1).flatten()

    q, same, moved = pooled(0), pooled(0), pooled(2)
    aligned = torch.dot(q, same) / (q.norm() * same.norm())
    across = torch.dot(q, moved) / (q.norm() * moved.norm())
    assert aligned > across + 0.05, (float(aligned), float(across))


# ------------------------------------------- the zero point, exactly --------

def test_zero_rotation_reproduces_the_unrotated_pooling_bit_for_bit():
    """The load-bearing test.

    The shipping bank has to be the `rope_max = 0` point of this
    parameterisation, not a neighbouring code path. If it is merely close, an
    arm ladder measures the approximation alongside the treatment, and this
    project has twice mistaken a baseline difference for a result
    (`tiles-gain-grows-with-corpus`, `pyramid-fusion-head-is-negative`).
    """
    C, T = views(n=CROPS, seed=1), views(n=6, seed=2)
    p0, p1 = P.view_positions(CROPS, GRID)
    z0, z1 = (torch.from_numpy(P.rope_angles(p, PAIRS,
                                             ["depth", "row", "col"],
                                             0.0, 1000.0)) for p in (p0, p1))
    base = P.levels(C, T)
    zero = P.levels(C, T, z0, z1)
    for a, b in zip(base, zero):
        assert torch.equal(a, b), (a - b).abs().max()


def test_no_axes_leaves_the_pooling_untouched():
    """The default path must not even enter the rotation."""
    C, T = views(n=CROPS, seed=3), views(n=6, seed=4)
    for a, b in zip(P.levels(C, T), P.levels(C, T, None, None)):
        assert torch.equal(a, b)


def test_a_nonzero_rotation_actually_changes_the_output():
    """So the two tests above are not passing because nothing is wired up."""
    C, T = views(n=CROPS, seed=5), views(n=6, seed=6)
    p0, p1 = P.view_positions(CROPS, GRID)
    a0, a1 = (torch.from_numpy(P.rope_angles(p, PAIRS, ["depth", "row", "col"],
                                             0.7, 1000.0))
              for p in (p0, p1))
    base, rot = P.levels(C, T), P.levels(C, T, a0, a1)
    assert not torch.allclose(base[0], rot[0], atol=1e-3)
    assert not torch.allclose(base[1], rot[1], atol=1e-3)


def test_the_levels_stay_unit_length():
    """Downstream reads one cosine; a level that is not normalised silently
    reweights the blend (`blend-weight-is-set-by-block-norms`)."""
    C, T = views(n=CROPS, seed=7), views(n=6, seed=8)
    p0, p1 = P.view_positions(CROPS, GRID)
    a0, a1 = (torch.from_numpy(P.rope_angles(p, PAIRS, ["row", "col"],
                                             0.9, 1000.0))
              for p in (p0, p1))
    for L in P.levels(C, T, a0, a1):
        assert torch.allclose(L.norm(dim=-1), torch.ones(L.shape[0]),
                              atol=1e-5)
