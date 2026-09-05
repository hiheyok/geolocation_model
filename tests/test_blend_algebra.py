"""The identity the pyramid blend rests on, and the trap it explains.

Concatenating two unit blocks and taking a cosine is not a neutral "fusion" --
it is an *exactly* weighted sum of the two blocks' cosines, with the weights
set by the block norms. That is why `combo = concat([l2(mean), l2(head)])`
silently means "average the two similarities 50/50", and why the same shape
once let DINOv2 take 81% of the cosine against SigLIP.

Fixing the weight is therefore not a new mechanism, it is naming a constant
that was already there.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _unit(rng, n, d):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _cos(A, B):
    return (A / np.linalg.norm(A, axis=1, keepdims=True)) @ \
           (B / np.linalg.norm(B, axis=1, keepdims=True)).T


@pytest.mark.parametrize("w", [0.0, 0.05, 0.25, 0.5, 0.8, 1.0])
def test_weighted_concat_cosine_is_the_weighted_sum_of_cosines(w):
    """concat[sqrt(1-w)*a, sqrt(w)*b] gives cos = (1-w)*cos_a + w*cos_b."""
    rng = np.random.default_rng(0)
    qa, qb = _unit(rng, 7, 16), _unit(rng, 7, 12)
    ba, bb = _unit(rng, 11, 16), _unit(rng, 11, 12)

    Q = np.concatenate([np.sqrt(1 - w) * qa, np.sqrt(w) * qb], axis=1)
    B = np.concatenate([np.sqrt(1 - w) * ba, np.sqrt(w) * bb], axis=1)

    want = (1 - w) * _cos(qa, ba) + w * _cos(qb, bb)
    assert np.allclose(_cos(Q, B), want, atol=1e-10)


def test_a_plain_concat_of_two_unit_blocks_is_exactly_fifty_fifty():
    """What `fuse_head` builds. The 0.5 was never chosen -- it fell out of
    concatenating two unit blocks, and the sweep shows the optimum is near
    0.05-0.15, so the accidental constant was roughly ten times too large."""
    rng = np.random.default_rng(1)
    qa, qb = _unit(rng, 5, 8), _unit(rng, 5, 8)
    ba, bb = _unit(rng, 9, 8), _unit(rng, 9, 8)
    Q = np.concatenate([qa, qb], axis=1)
    B = np.concatenate([ba, bb], axis=1)
    want = 0.5 * _cos(qa, ba) + 0.5 * _cos(qb, bb)
    assert np.allclose(_cos(Q, B), want, atol=1e-10)


def test_unequal_block_norms_silently_set_the_weight():
    """The DINOv2/SigLIP failure, in miniature: leave the blocks un-normalised
    and the louder one decides the ranking. Raw activation norms were 82.96
    against 20.58, i.e. 81/19, which nobody chose."""
    rng = np.random.default_rng(2)
    qa, qb = _unit(rng, 6, 8) * 82.96, _unit(rng, 6, 8) * 20.58
    ba, bb = _unit(rng, 10, 8) * 82.96, _unit(rng, 10, 8) * 20.58
    Q = np.concatenate([qa, qb], axis=1)
    B = np.concatenate([ba, bb], axis=1)
    w = 20.58 ** 2 / (82.96 ** 2 + 20.58 ** 2)
    want = (1 - w) * _cos(qa, ba) + w * _cos(qb, bb)
    assert np.allclose(_cos(Q, B), want, atol=1e-10)
    assert w < 0.06, "the quiet block gets almost no say"


def test_pca_destroys_the_block_structure_the_weight_needs():
    """Why the exported `pyr47_fuse_p05` arm cannot be re-weighted.

    PCA mixes both blocks into every component, so after it there is no
    'first half' to scale. The published arm PCA'd the concat down to 1536-d
    for an equal-bytes comparison and, in doing so, welded the 50/50 in.
    """
    rng = np.random.default_rng(3)
    X = np.concatenate([_unit(rng, 200, 8), _unit(rng, 200, 8)], axis=1)
    mu = X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:8].T                                    # 16 -> 8
    # every retained component draws on both halves
    load_a = np.abs(P[:8]).sum(0)
    load_b = np.abs(P[8:]).sum(0)
    assert (load_a > 1e-6).all() and (load_b > 1e-6).all()


def test_the_level_mean_is_itself_an_unweighted_choice():
    """`FuseHead.baseline` averages the per-level means equally. The sweep
    finds 1:1:2 beats 1:1:1 by +1.17 pp at 25 km, separated -- so the deepest
    level, the one that only exists on high-resolution source, wants the most
    weight of the three."""
    rng = np.random.default_rng(4)
    lv = [_unit(rng, 4, 6) for _ in range(3)]
    equal = np.stack(lv).mean(0)
    weighted = (lv[0] + lv[1] + 2 * lv[2]) / 4.0
    assert not np.allclose(equal, weighted)
