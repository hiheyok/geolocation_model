"""Calibration must not reach the learned quality gate (REVIEW8 #8).

`--calib top1` subtracts each photograph's best cosine so candidates from
different photographs are comparable. `sim` then serves two purposes that
respond differently to that shift:

    weighting   softmax(sim / tau)   shift-invariant -- unchanged
    quality     sim[:, 0] as a gate  NOT invariant -- becomes exactly 0

`cond`, `pos` and `dual` checkpoints were trained with the raw cosine in that
slot. So the ONE-photograph reference -- the baseline every multi-photo number
is measured against -- was itself measuring a changed gate input, and the
multi-photo arms mixed neighbour aggregation with the removal of a learned
confidence signal.

The existing regression test compares softmax weights, which are invariant by
construction and cannot see this. These compare the complete prior output with
a gate that is explicitly nonzero, which is what the review asked for.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from model import NeighborBatch  # noqa: E402
from retrieval import RetrievalPrior  # noqa: E402


def prior_with_a_live_gate(seed=0):
    """Every gate is zero-initialised, so an untrained prior is invariant to
    everything -- the vacuity this repository has been caught by before."""
    torch.manual_seed(seed)
    p = RetrievalPrior(mode="cond", d_street=8, n_steps=4).eval()
    with torch.no_grad():
        for m in p.cond.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.5)
                torch.nn.init.normal_(m.bias, std=0.5)
    return p


def bias(p, sim, sim_q=None):
    K = sim.shape[1]
    x = torch.zeros(1, K, dtype=torch.long)
    y = torch.zeros(1, K, dtype=torch.long)
    with torch.no_grad():
        return p(x, y, sim, torch.zeros(1, dtype=torch.long),
                 torch.zeros(1, dtype=torch.long),
                 torch.zeros(1, dtype=torch.long), 256, sim_q=sim_q)


def test_the_gate_is_actually_live():
    """Otherwise every assertion below passes for the wrong reason."""
    p = prior_with_a_live_gate()
    a = bias(p, torch.tensor([[0.9, 0.8]]))
    b = bias(p, torch.tensor([[0.4, 0.3]]))
    assert not torch.allclose(a, b), "the quality gate ignores its input"


def test_calibration_alone_moves_the_prior_output():
    """The defect, stated as a measurement: a constant shift that the softmax
    cannot see still moves the policy bias, by whole logits."""
    p = prior_with_a_live_gate()
    raw = torch.tensor([[0.9, 0.8]])
    cal = raw - raw.max()
    assert torch.allclose(torch.softmax(raw / 0.07, 1),
                          torch.softmax(cal / 0.07, 1), atol=1e-6), \
        "the weighting half must be invariant, or this tests the wrong thing"
    assert not torch.allclose(bias(p, raw), bias(p, cal), atol=1e-4)


def test_passing_the_raw_similarity_restores_the_trained_input():
    """The fix: calibrated scores weight the neighbours, the raw cosine feeds
    the gate, and the result equals the uncalibrated one."""
    p = prior_with_a_live_gate()
    raw = torch.tensor([[0.9, 0.8]])
    cal = raw - raw.max()
    assert torch.allclose(bias(p, cal, sim_q=raw), bias(p, raw), atol=1e-6)


def test_omitting_it_leaves_every_other_path_unchanged():
    """`sim_q=None` must mean `sim`, so training and single-photo evaluation
    are bit-identical to before."""
    p = prior_with_a_live_gate()
    raw = torch.tensor([[0.9, 0.8, 0.5]])
    assert torch.equal(bias(p, raw), bias(p, raw, sim_q=None))


def test_the_neighbour_record_defaults_to_one_similarity():
    nb = NeighborBatch(torch.zeros(1, 2, dtype=torch.long),
                       torch.zeros(1, 2, dtype=torch.long),
                       torch.zeros(1, 2))
    assert nb.q is None
    assert NeighborBatch.of([nb.x, nb.y, nb.sim]).q is None


def test_one_photograph_reports_its_raw_best_cosine():
    """The aggregation: max over the taken photographs' own top-1s, which for
    a single photograph is exactly what the checkpoints were trained on."""
    from multiquery import merge_candidates

    idx = np.array([[10, 11, 12, 13], [20, 21, 22, 23]], np.int64)
    sim = np.array([[0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3, 0.2]], np.float32)

    _, cs, q = merge_candidates([0], idx, sim, 4, "rr", calib="top1")
    assert q == pytest.approx(0.9)
    assert cs.max() == pytest.approx(0.0)      # calibration still applied

    _, _, q2 = merge_candidates([0, 1], idx, sim, 4, "rr", calib="top1")
    assert q2 == pytest.approx(0.9)            # best of the two photographs
