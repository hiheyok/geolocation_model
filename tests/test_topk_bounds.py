"""Top-k against a disjoint bank, at the boundaries that broke twice.

The original code used `min(K, n-1)`, which assumes the query is inside the
bank -- true when the two sets are the same, false here, so it dropped one
legal neighbour and gave k=0 on a one-row bank. Fixing it to `min(K, n)`
introduced a worse bug: argpartition takes an index, so `kth=k` is out of
bounds exactly when K reaches the bank size. Both versions shipped.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def topk(S, K):
    """The shared shape of the three call sites."""
    k = max(1, min(K, S.shape[1]))
    j = np.argpartition(-S, k - 1, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, j, 1), axis=1)
    return np.take_along_axis(j, o, 1)


@pytest.mark.parametrize("n_bank,K", [(1, 1), (1, 32), (2, 2), (5, 5),
                                      (5, 32), (32, 32), (40, 32)])
def test_no_out_of_bounds_at_any_boundary(n_bank, K):
    rng = np.random.default_rng(0)
    S = rng.random((4, n_bank))
    got = topk(S, K)
    assert got.shape == (4, min(K, n_bank))


def test_every_bank_row_is_reachable():
    """min(K, n-1) silently made the last row unreachable."""
    S = np.eye(6)[:3]                      # query i matches bank row i exactly
    got = topk(S, 6)
    assert got.shape[1] == 6, "a legal neighbour was dropped"


def test_a_one_row_bank_returns_that_row():
    S = np.array([[0.5], [0.2]])
    got = topk(S, 32)
    assert got.shape == (2, 1) and (got == 0).all()


def test_the_top_result_is_actually_the_best():
    rng = np.random.default_rng(1)
    S = rng.random((8, 20))
    got = topk(S, 5)
    assert (got[:, 0] == S.argmax(1)).all()
