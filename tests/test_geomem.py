"""GeoMem must start inert AND trainable, which are easy to confuse.

A zero-initialised gate multiplying a zero-initialised table is inert in
exactly the way you want and untrainable in a way you cannot see: the
contribution is `gate * emb`, so the gradient with respect to the table is the
gate and the gradient with respect to the gate is the table. With both at zero
neither ever moves, the table stays bit-for-bit zero for the whole run, and the
metrics are indistinguishable from an honest null. That happened once, and it
cost two training arms before the table was inspected rather than the hit rate.

So the invariant is two-sided and both halves are asserted here:

  inert     the additive term is exactly 0 at init, so --init from a checkpoint
            without a GeoMem is lossless
  trainable the table receives nonzero gradient on the very first backward pass
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model import GeoAgent


def _agent(mode):
    return GeoAgent(d_street=4608, n_steps=5, sink=True, geo=mode, d_geo=128)


def _batch(n=4):
    f = torch.randn(n, 512)
    x0 = torch.rand(n)
    y0 = torch.rand(n)
    step = torch.arange(n) % 4
    return f, x0, y0, step


def test_inert_at_init():
    """No GeoMem contribution before it has learned anything."""
    for mode in ("bias", "key"):
        b = _agent(mode).geo_bias(*_batch(), 257)
        assert b.abs().max().item() == 0.0, mode


def test_table_receives_gradient():
    """The deadlock test. If this fails, every arm using it is void."""
    for mode in ("bias", "key"):
        m = _agent(mode)
        m.geo_bias(*_batch(), 257).sum().backward()
        g = m.geo.emb.weight.grad
        assert g is not None, mode
        assert int((g.abs().sum(1) > 0).sum()) > 0, (
            "{}: no table row received gradient -- the table can never "
            "train and the arm would report a false null".format(mode))


def test_sink_is_never_biased():
    """The 257th action is a verdict, not a place, so it has no tile."""
    for mode in ("bias", "key"):
        m = _agent(mode)
        with torch.no_grad():
            m.geo.emb.weight.normal_(0, 1)
        b = m.geo_bias(*_batch(), 257)
        assert b.shape[1] == 257
        assert b[:, -1].abs().max().item() == 0.0, mode


def test_rows_land_in_the_right_level_block():
    """Steps past the covered levels must fall through to UNK, not alias."""
    m = _agent("key")
    x0 = torch.tensor([0.0, 0.5, 0.25, 0.75])
    y0 = torch.tensor([0.0, 0.5, 0.25, 0.75])
    rows = m.geo.rows(x0, y0, torch.tensor([0, 1, 2, 3]))
    assert rows.shape == (4, 256)
    assert bool(((rows[0] >= 0) & (rows[0] < 256)).all()), "step 0 -> z4 block"
    assert bool(((rows[1] >= 256) & (rows[1] < 65792)).all()), "step 1 -> z8"
    assert bool((rows[2] == m.geo.unk).all()), "step 2 -> UNK"
    assert bool((rows[3] == m.geo.unk).all()), "step 3 -> UNK"
    assert int(rows[0].unique().numel()) == 256, "children must be distinct"
