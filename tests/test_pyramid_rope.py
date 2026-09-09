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

import inspect
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
COLS = None                      # filled in below, from the real geometry
PAIRS = P.D_ENC // 2
COLS = P.crop_columns(CROPS, GRID[0], P.FRAME)


def views(b=4, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, 2, P.D_ENC, generator=g)


# ------------------------------------------------------- the coordinates ----

def test_the_crops_sit_where_the_preprocessor_actually_puts_them():
    """Reported on #64. They are NOT at columns 0, 1, 2.

    `embed_street.preprocess` scales the short side to 224 and slides a
    224-wide window across the width, so on the 910x512 OSV-5M frame `w` is
    398 and the windows land at 0, 87, 174 -- 56% of the width each, sharing
    61% with a neighbour. Their centres are 0.344, 1.000 and 1.656 tile
    columns: the outer two are 1.31 columns apart, not 2.

    Putting them at 0 and 2 would claim two views that mostly show the same
    pixels are as far apart as the outer tile columns, and would align each
    crop with a tile centre it does not sit on -- the exact cross-level
    matching the rotation exists to get right.
    """
    cols = P.crop_columns(CROPS, GRID[0], P.FRAME)
    assert cols == pytest.approx([0.34422, 1.0, 1.65578], abs=1e-4)
    p0, _ = P.view_positions(CROPS, GRID, cols)
    assert [r for _, r, _ in p0] == [0.5, 0.5, 0.5]


def test_the_crop_positions_come_from_the_preprocessor_itself():
    """One definition of where a crop sits. A second copy of the arithmetic
    would be a second answer, and the two would drift apart silently."""
    import embed_street
    w = max(224, round(910 * 224 / 512))
    assert embed_street.crop_lefts(w, 224, 3) == [0, 87, 174]
    assert P.crop_columns(3, 3, (910, 512)) == pytest.approx(
        [3 * (l + 112) / w - 0.5 for l in (0, 87, 174)])


def test_a_different_frame_moves_the_crops():
    """The positions are a function of the frame, so `--frame` is load-bearing
    rather than decorative. A square frame gives one crop, dead centre."""
    assert P.crop_columns(3, 3, (512, 512)) == pytest.approx([1.0, 1.0, 1.0])
    wide = P.crop_columns(3, 3, (2048, 512))
    assert wide[0] < 0.34422 and wide[2] > 1.65578


def test_tiles_are_row_major_over_the_grid():
    """`tile_cache` writes axis 1 row-major; the positions must agree."""
    _, p1 = P.view_positions(CROPS, GRID, COLS)
    assert p1 == [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, 2.0),
                  (1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 2.0)]


def test_a_transposed_grid_gives_different_positions():
    """3x2 and 2x3 hold six tiles either way, so the grid cannot be inferred
    from the count -- which is why `tile_grid` reads it rather than deriving
    it. If these agreed, reading it would be pointless."""
    assert P.view_positions(2, (2, 3))[1] != P.view_positions(3, (3, 2))[1]


def test_a_crop_is_not_the_parent_of_a_tile_column():
    """The model this replaced. A crop is 56% of the width and shares 61% of
    itself with its neighbour, so it is not the parent of one column -- only
    the middle crop lands on a column centre, and the outer two do not."""
    p0, p1 = P.view_positions(CROPS, GRID, COLS)
    centres = {c for _, _, c in p1}
    on_centre = [c for _, _, c in p0 if c in centres]
    assert on_centre == [1.0], on_centre


def test_the_crops_span_less_than_the_tiles_do():
    """Because they overlap. The rotation should separate them less than it
    separates the tile columns, and it only does that if their coordinates
    say so."""
    p0, p1 = P.view_positions(CROPS, GRID, COLS)
    spread = lambda ps: max(c for _, _, c in ps) - min(c for _, _, c in ps)
    assert spread(p0) == pytest.approx(1.3116, abs=1e-3)
    assert spread(p1) == 2.0


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
        P.view_positions(CROPS, GRID, COLS)[1], PAIRS,
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
        P.view_positions(CROPS, GRID, COLS)[0], PAIRS, ["col"], 0.7, 1000.0))
    assert not torch.allclose(P.apply_rope(V, ang).mean(1),
                              P.apply_rope(S, ang).mean(1), atol=1e-3)


def _one_view(pos, n, axes, slot, mx=0.7, spread=None):
    """Pool a single feature placed in view `slot`, and nothing else."""
    feat = views(b=1, n=1)[:, 0]
    ang = torch.from_numpy(P.rope_angles(
        pos, PAIRS, axes, mx, P.ROPE_SPREAD if spread is None else spread))
    V = torch.zeros(1, n, 2, P.D_ENC)
    V[:, slot] = feat
    v = P.apply_rope(V, ang).mean(1).flatten()
    return v / v.norm()


def test_a_same_position_match_keeps_more_credit_than_a_moved_one():
    """The property the retrieval cosine is being given: two images sharing a
    feature in the same view score above two sharing it in different views.
    Under a plain mean those are identical."""
    _, p1 = P.view_positions(CROPS, GRID, COLS)
    q = _one_view(p1, 6, ["col"], 0)
    same = float(torch.dot(q, _one_view(p1, 6, ["col"], 0)))
    moved = float(torch.dot(q, _one_view(p1, 6, ["col"], 2)))
    assert same == pytest.approx(1.0, abs=1e-5)
    assert moved < 0.85, moved


def test_the_overlapping_crops_separate_less_than_the_tiles():
    """The geometry showing through the metric, which is the point of getting
    it right: two crops sharing 61% of their pixels are held closer than two
    disjoint tile columns, instead of being forced 2 columns apart."""
    p0, p1 = P.view_positions(CROPS, GRID, COLS)
    c = float(torch.dot(_one_view(p0, 3, ["col"], 0),
                        _one_view(p0, 3, ["col"], 2)))
    t = float(torch.dot(_one_view(p1, 6, ["col"], 0),
                        _one_view(p1, 6, ["col"], 2)))
    assert c > t, (c, t)


def test_the_default_spread_separates_positions_and_1000_barely_does():
    """Ties the spread default to the thing it exists for. At 1000 the outer
    crops are 0.97 apart -- indistinguishable in a cosine that ranks 3.4M
    rows."""
    p0, _ = P.view_positions(CROPS, GRID, COLS)
    far = lambda sp: float(torch.dot(_one_view(p0, 3, ["col"], 0, spread=sp),
                                     _one_view(p0, 3, ["col"], 2, spread=sp)))
    assert far(1000.0) > 0.96
    assert far(P.ROPE_SPREAD) < 0.93


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
    p0, p1 = P.view_positions(CROPS, GRID, COLS)
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
    p0, p1 = P.view_positions(CROPS, GRID, COLS)
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
    p0, p1 = P.view_positions(CROPS, GRID, COLS)
    a0, a1 = (torch.from_numpy(P.rope_angles(p, PAIRS, ["row", "col"],
                                             0.9, 1000.0))
              for p in (p0, p1))
    for L in P.levels(C, T, a0, a1):
        assert torch.allclose(L.norm(dim=-1), torch.ones(L.shape[0]),
                              atol=1e-5)


# ------------------------------------ the guard is the rotation's, not ------

def test_the_geometry_guard_does_not_fire_without_rotation():
    """Reported on #64.

    The crop/grid check ran unconditionally, so a 3-crop cache against a 2x3
    grid -- which pooled fine before this flag existed -- started exiting. A
    flag must not restrict the inputs of the path that does not use it.
    """
    src = Path(inspect.getsourcefile(P.main)).read_text(encoding="utf-8")
    body = src[src.index("def main("):]
    i, j = body.index("grid = tile_grid("), body.index("p0, p1 = view_positions")
    assert "if axes:" in body[i:j], "geometry checks run outside `if axes`"


def test_a_frame_at_or_below_square_collapses_the_crops():
    """What the frame guard is actually for.

    A range check could never fire -- a crop is inside its own image, so the
    column is always within (-0.5, gc-0.5). What can go wrong is the frame
    collapsing the crops onto each other: at or below square the short side is
    the width, the window cannot slide, and all three crops are the same
    pixels. Rotating those on `col` would separate identical views.
    """
    assert len(set(P.crop_columns(3, 3, (512, 512)))) == 1
    assert len(set(P.crop_columns(3, 3, (400, 512)))) == 1
    assert len(set(P.crop_columns(3, 3, P.FRAME))) == 3


def test_the_collapse_guard_is_reachable_from_main():
    src = Path(inspect.getsourcefile(P.main)).read_text(encoding="utf-8")
    assert "cannot slide" in src
