"""Training and inference must score the same network (REVIEW8, pipeline).

`GeoAgent.forward` (training), `GeoAgent.policy_from` (off-path negatives) and
`beam.search` (inference) each assemble the same four calls -- fuse, retrieval
prior, geo prior, policy logits -- and each does it in its own code. Nothing
asserted they agree.

They have not agreed, twice, and both times the divergence was silent:

* `beam.search` ignored the neighbour embeddings entirely, so `pos` and `dual`
  models decoded as if unconditioned. Fixing it moved the median from 81.6 km
  to 55.8 km. Nothing raised; the arm simply scored badly and the cause looked
  like the architecture.
* beam search once inlined the fusion concatenation, so adding a fourth block
  to the fusion raised a shape error there while training passed. That one at
  least crashed. `fuse_flat` exists because of it.

The review's recommendation is a shared forward path with parity tests. The
shared path is largely there; the parity test was not. This is it.

The test is deliberately at the level the bugs occurred: it does not check that
each function is correct, only that the three callers **compose them
identically**. A wrong `retr_prior` passes here. A caller that forgets to pass
`nbrs`, or drops `step`, or reorders the two priors, does not.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tile_math as tm                       # noqa: E402
from model import GeoAgent, NeighborBatch    # noqa: E402

D_S, B, K = 768, 3, 4
A = tm.actions()


def build(**kw):
    """A model whose retrieval prior actually contributes.

    This matters more than it looks. Every gate on the prior -- `g_cell`,
    `g_sink`, `g_neg`, `w_pos` -- is **zero-initialised**, so a freshly built
    model scores identically with and without its neighbours. Parity tests on
    such a model pass whatever the callers do with `nbrs`, which is precisely
    the bug they exist to catch. The first version of this file did exactly
    that and its own non-vacuity check caught it.

    So the zero gates are given values, the way training would. Same trap as
    the zero-gate deadlock: a zero gate times anything is an honest-looking
    null.
    """
    torch.manual_seed(0)
    m = GeoAgent(d_street=D_S, n_actions=A, n_steps=tm.STEPS + 1, **kw)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.detach().abs().max() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    return m.eval()


def batch(model, g=None):
    g = g or torch.Generator().manual_seed(1)
    S = tm.STEPS + 1
    n_cls = 12
    b = {
        "street": torch.randn(B, D_S, generator=g),
        "tokens": torch.rand(B, S, A, n_cls, generator=g),
        "x0": torch.rand(B, S, generator=g),
        "y0": torch.rand(B, S, generator=g),
        "step": torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
    }
    if model.retr is not None:
        n16 = 1 << (4 * tm.STEPS)
        b["nbr_x"] = torch.randint(0, n16, (B, K), generator=g)
        b["nbr_y"] = torch.randint(0, n16, (B, K), generator=g)
        b["nbr_sim"] = torch.rand(B, K, generator=g)
        b["nbr_emb"] = torch.randn(B, K, D_S, generator=g)
    return b


def beam_style(model, b, t):
    """Exactly the call sequence `beam.search` uses, for one step.

    Written out here rather than imported so that a change to either side has
    to be reflected deliberately in the other. If this stops matching
    `beam.search`, that is the test asking to be updated, not a licence to
    delete it.
    """
    S = b["step"].shape[1]
    st = b["street"]
    tok = b["tokens"][:, t]
    x0, y0 = b["x0"][:, t], b["y0"][:, t]
    sp = b["step"][:, t]
    f, keys = model.fuse_flat(st, tok, x0, y0, sp)
    n_logits = keys.shape[1] + (1 if model.sink is not None else 0)
    nbrs = None
    if "nbr_x" in b:
        nbrs = NeighborBatch(b["nbr_x"], b["nbr_y"], b["nbr_sim"],
                             b.get("nbr_emb"))
    prior = model.retr_prior(nbrs, st, x0, y0, sp, 1, n_logits)
    prior = model._add_geo(prior, f, x0, y0, sp, n_logits)
    return model.policy_logits(f, keys, prior, sp)


ARMS = [
    pytest.param({}, id="plain"),
    pytest.param({"retr": True, "retr_mode": "scalar"}, id="retr-scalar"),
    pytest.param({"retr": True, "retr_mode": "dual", "d_key": 128},
                 id="retr-dual"),
    pytest.param({"retr": True, "retr_mode": "pos", "d_key": 128},
                 id="retr-pos"),
    pytest.param({"retr": True, "retr_mode": "dual", "d_key": 128,
                  "sink": True}, id="retr-dual-sink"),
]


@pytest.mark.parametrize("kw", ARMS)
def test_beam_scores_the_same_network_as_training(kw):
    """The headline invariant: one image, one tile, one step, two code paths."""
    m = build(**kw)
    b = batch(m)
    with torch.no_grad():
        train_logits, _ = m(b)
        for t in range(tm.STEPS):
            got = beam_style(m, b, t)
            assert got.shape == train_logits[:, t].shape
            assert torch.allclose(got, train_logits[:, t], atol=1e-5), (
                "step {}: beam and training disagree by {:.3e}".format(
                    t, (got - train_logits[:, t]).abs().max().item()))


@pytest.mark.parametrize("kw", ARMS)
def test_policy_from_scores_the_same_network_as_training(kw):
    """The third caller. Off-path negatives are where the retrieval prior is
    supposed to matter most, so a divergence here is not cosmetic."""
    m = build(**kw)
    b = batch(m)
    n = tm.STEPS
    nbrs = None
    if "nbr_x" in b:
        nbrs = NeighborBatch(b["nbr_x"], b["nbr_y"], b["nbr_sim"],
                             b.get("nbr_emb"))
    with torch.no_grad():
        train_logits, _ = m(b)
        got = m.policy_from(b["street"], b["tokens"][:, :n], b["x0"][:, :n],
                            b["y0"][:, :n], b["step"][:, :n], nbrs)
    assert torch.allclose(got.view(B, n, -1), train_logits, atol=1e-5)


def test_the_test_can_fail():
    """Non-vacuity, and the specific historical bug.

    Dropping `nbrs` from the beam path is what actually happened, and it moved
    the median 81.6 -> 55.8 km when fixed. If a conditioned model scores
    identically with and without its neighbours, this test proves nothing.
    """
    m = build(retr=True, retr_mode="dual", d_key=128)
    b = batch(m)
    with torch.no_grad():
        train_logits, _ = m(b)
        f, keys = m.fuse_flat(b["street"], b["tokens"][:, 0], b["x0"][:, 0],
                              b["y0"][:, 0], b["step"][:, 0])
        n_logits = keys.shape[1]
        blind = m.policy_logits(
            f, keys, m._add_geo(m.retr_prior(None, b["street"], b["x0"][:, 0],
                                             b["y0"][:, 0], b["step"][:, 0],
                                             1, n_logits),
                                f, b["x0"][:, 0], b["y0"][:, 0],
                                b["step"][:, 0], n_logits),
            b["step"][:, 0])
    assert not torch.allclose(blind, train_logits[:, 0], atol=1e-5), (
        "a model that ignores its neighbours scores identically, so the "
        "parity assertions above cannot detect the bug they exist for")
