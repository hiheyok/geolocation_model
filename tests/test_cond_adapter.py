"""Conditioning rides beside the retrieval vector without entering the cosine.

`runs/PYR_LEVELS.md` measured what happens when extra detail is put *into* the
retrieval vector: against a fixed `L0L1` bank, an extra level at weight 0.10
replaces 8.2% of the top-16 and flips the top-1 for 12.7% of queries while the
hit rate does not move at all (59.6% -> 59.6%). It reorders the ranking without
being about location. So the only place left for extra pixels is a path that is
not a cosine -- the model's own conditioning input, which is compared to
nothing.

Two properties have to hold, and both are the kind that fail silently.

**Identity at initialisation.** The goal is to fine-tune from trained weights,
so a checkpoint loaded into the conditioned model must behave *exactly* as it
did before the adapter existed -- not approximately. A zero gate gives that,
and `test_the_adapter_is_exactly_identity_at_init` asserts bit equality rather
than a tolerance.

**The gate must still train.** A zero gate times a zero table never trains and
looks exactly like an honest null, which this project has already been caught
by. Here the table is randomly initialised, so the gate has gradient from the
first step and the projection starts once the gate has moved. Both halves of
that are asserted.

The prefix check is the other silent failure: the k-NN cache addresses the
retrieval file's embedding space, and a joined cache whose first columns came
from a *different* build of that file leaves every neighbour the right row of
the wrong geometry, with every shape and every index still valid.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config                              # noqa: E402
import tile_math as tm                     # noqa: E402
from dataset import _check_retrieval_prefix  # noqa: E402
from encoders import StreetProj            # noqa: E402
from model import GeoAgent                 # noqa: E402

D_R, D_C, D = 768, 1536, 512


# --- the adapter --------------------------------------------------------------

def test_the_adapter_is_exactly_identity_at_init():
    """Bit equality, not a tolerance: a fine-tune must start *at* the trained
    point, not near it."""
    torch.manual_seed(0)
    plain = StreetProj(D_R, D)
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    x, c = torch.randn(4, D_R), torch.randn(4, D_C)
    with torch.no_grad():
        a, b = plain(x), cond(torch.cat([x, c], -1))
    assert torch.equal(a, b)


def test_the_conditioning_is_ignored_at_init_whatever_it_contains():
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    x = torch.randn(4, D_R)
    with torch.no_grad():
        a = cond(torch.cat([x, torch.zeros(4, D_C)], -1))
        b = cond(torch.cat([x, torch.randn(4, D_C) * 100], -1))
    assert torch.equal(a, b)


def test_the_gate_has_gradient_at_init():
    """Not the zero-gate deadlock: the table is random, so the gate moves."""
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    out = cond(torch.cat([torch.randn(4, D_R), torch.randn(4, D_C)], -1))
    out.sum().backward()
    assert cond.cond_gate.grad.abs().item() > 0
    # and the projection waits its turn -- zero while the gate is zero
    assert cond.cond_proj.weight.grad.abs().max() == 0


def test_the_projection_trains_once_the_gate_has_moved():
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    with torch.no_grad():
        cond.cond_gate.fill_(0.1)
    out = cond(torch.cat([torch.randn(4, D_R), torch.randn(4, D_C)], -1))
    out.sum().backward()
    assert cond.cond_proj.weight.grad.abs().max() > 0


def test_the_conditioning_changes_the_output_once_gated():
    torch.manual_seed(0)
    cond = StreetProj(D_R, D, d_cond=D_C)
    with torch.no_grad():
        cond.cond_gate.fill_(1.0)
    x = torch.randn(4, D_R)
    with torch.no_grad():
        a = cond(torch.cat([x, torch.zeros(4, D_C)], -1))
        b = cond(torch.cat([x, torch.randn(4, D_C)], -1))
    assert not torch.allclose(a, b)


def test_an_unconditioned_cache_is_refused():
    """Silently splitting a 768-d vector would take retrieval as conditioning."""
    cond = StreetProj(D_R, D, d_cond=D_C)
    with pytest.raises(ValueError, match="retrieval"):
        cond(torch.randn(4, D_R))


# --- the checkpoint path ------------------------------------------------------

def agent(**kw):
    return GeoAgent(d_street=D_R, n_actions=tm.actions(),
                    n_steps=tm.STEPS + 1, **kw)


def test_an_old_checkpoint_loads_and_only_the_adapter_is_new():
    torch.manual_seed(0)
    old = agent()
    torch.manual_seed(0)
    new = agent(d_cond=D_C)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert set(missing) == {
        "street.cond_gate", "street.cond_proj.weight", "street.cond_proj.bias",
        "street.cond_norm.weight", "street.cond_norm.bias"}
    assert not unexpected


def test_the_retrieval_prior_never_sees_the_conditioning_block():
    """The learned keys live in the bank's space, which has no conditioning."""
    m = agent(d_cond=D_C, retr=True, retr_mode="dual")
    assert m.retr.__class__ is not None
    # the key projection was built for the retrieval width
    w = [p for n, p in m.retr.named_parameters() if n.endswith("weight")]
    assert any(p.shape[-1] == D_R for p in w), \
        "no retrieval-side projection is {}-d".format(D_R)


# --- the prefix guard ---------------------------------------------------------

def joined(tmp_path, monkeypatch, retr, cond, retr_name="r.f16.npy"):
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    # np.save only appends .npy when the name lacks it, and these names end
    # in .f16.npy already, so the file lands exactly where the loader looks.
    np.save(tmp_path / retr_name, retr.astype(np.float16))
    return np.concatenate([retr, cond], 1).astype(np.float16)


def test_a_matching_prefix_passes(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((50, 8)).astype(np.float16)
    c = rng.standard_normal((50, 4)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r, c)
    _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8)


def test_a_rebuilt_retrieval_cache_under_the_same_name_is_caught(
        tmp_path, monkeypatch):
    """The failure a string comparison cannot see.

    Same filename, same shape, same row count -- different vectors. The k-NN
    addresses one embedding space and the model's retrieval block is in
    another, and every neighbour index is still perfectly in range.
    """
    rng = np.random.default_rng(0)
    r = rng.standard_normal((50, 8)).astype(np.float16)
    c = rng.standard_normal((50, 4)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r, c)
    # the retrieval cache is rebuilt in place, keeping its name
    other = rng.standard_normal((50, 8)).astype(np.float16)
    np.save(tmp_path / "r.f16.npy", other)
    with pytest.raises(SystemExit, match="different space under the same name"):
        _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 8)


def test_a_missing_retrieval_cache_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    s = np.zeros((10, 12), np.float16)
    with pytest.raises(SystemExit, match="not on"):
        _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "gone.f16.npy", 8)


def test_a_wrong_shaped_retrieval_cache_is_refused(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((50, 8)).astype(np.float16)
    s = joined(tmp_path, monkeypatch, r,
               rng.standard_normal((50, 4)).astype(np.float16))
    with pytest.raises(SystemExit, match="declares"):
        _check_retrieval_prefix(tmp_path / "j.f16.npy", s, "r.f16.npy", 6)


def test_an_undeclared_retrieval_file_warns_rather_than_failing(
        tmp_path, monkeypatch, capsys):
    """Allowed, but never silent: without it nothing ties this file's
    retrieval half to the bank the neighbours came from."""
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    _check_retrieval_prefix(tmp_path / "j.f16.npy", np.zeros((5, 12),
                                                             np.float16),
                            None, 8)
    assert "cannot be verified" in capsys.readouterr().out
