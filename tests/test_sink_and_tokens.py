"""Train/inference parity for the pieces added on 2026-09-02.

Both bugs these cover were live, and neither raised: `--sink-k > 1` trained one
network and scored another because beam search called `policy_logits` without
`step`, and a sub-patch map cache could be read by a model built for 12-d
tokens. Quietly evaluating a different model than the one that trained is the
failure mode this project keeps hitting.

The first version of this file did not actually catch either. It called
`policy_logits` directly, so reverting the beam fix left it green, and it
appended the sink key by hand on top of the one `policy_logits` appends -- an
A+2 shape that never occurs. The guard now lives in the model: omitting `step`
while extra sink keys exist raises, so the caller cannot get it wrong.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tile_math as tm  # noqa: E402
import tiles as T  # noqa: E402
from model import GeoAgent  # noqa: E402


def build(sink_k=1, n_classes=12):
    torch.manual_seed(0)
    return GeoAgent(d_street=768, n_actions=tm.actions(), n_steps=tm.STEPS + 1,
                    map_layers=1, pool="attn", n_pool_q=4, pos="both",
                    sink=True, sink_k=sink_k, n_classes=n_classes,
                    retr=True, retr_mode="dual", d_key=128)


def inputs(B=8, seed=1):
    """Exactly the shapes production passes: keys WITHOUT the sink appended."""
    torch.manual_seed(seed)
    return (torch.randn(B, 512),
            torch.randn(B, tm.actions(), 256),
            torch.randint(0, tm.STEPS, (B,)))


def test_extra_sink_keys_start_neutral():
    """A sink_k=4 model loaded with sink_k=1 weights must score identically."""
    a, b = build(1), build(4)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert set(missing) == {"sink_ext", "sink_ext_b", "sink_ext_g"}
    assert not unexpected
    a.eval(); b.eval()
    f, keys, step = inputs()
    with torch.no_grad():
        la = a.policy_logits(f, keys, None, step)
        lb = b.policy_logits(f, keys, None, step)
    assert la.shape == lb.shape == (len(f), tm.actions() + 1)
    assert torch.allclose(la, lb, atol=1e-6)


def test_omitting_step_raises_when_extras_exist():
    """The regression itself, guarded where the caller cannot skip it.

    beam.search omitted `step`, which silently disabled the extra keys. A test
    that calls policy_logits directly cannot catch that, so the model refuses
    instead: any caller that forgets `step` fails loudly.
    """
    m = build(4)
    m.eval()
    f, keys, _ = inputs()
    with pytest.raises(ValueError, match="step"):
        m.policy_logits(f, keys, None)
    # a sink_k=1 model has no extras and must stay callable without step
    m1 = build(1)
    m1.eval()
    with torch.no_grad():
        assert m1.policy_logits(f, keys, None).shape[-1] == tm.actions() + 1


def test_sink_gate_can_actually_learn():
    """Neutral must not mean dead.

    The first design gave each extra a -20 bias, which is neutral and also
    gradient-starved: about e^-20 of the log-sum-exp gradient, so the keys
    never move and sink_k reads as a null result. The gate must start at zero
    and receive a real gradient.
    """
    m = build(4)
    m.train()
    f, keys, step = inputs()
    loss = m.policy_logits(f, keys, None, step)[:, -1].sum()
    loss.backward()
    g = m.sink_ext_g.grad
    assert g is not None, "sink gate has no gradient"
    # one gate per step; every step present in the batch must receive one
    seen = torch.unique(step).tolist()
    assert g.abs()[seen].min().item() > 1e-3, (
        "sink gate gradient {} at steps {}; the extras cannot learn"
        .format(g.tolist(), seen))


def test_sub_tokens_contain_the_original():
    """sub>1 must be a strict superset: averaging the cells restores sub=1."""
    rng = np.random.default_rng(0)
    mask = (rng.integers(0, T.N_CLASSES, (512, 512))
            * T.CLASS_STEP).astype(np.uint8)
    base = T.to_tokens(mask)
    assert base.shape == (256, 12)
    for sub in (2, 4):
        t = T.to_tokens(mask, sub=sub)
        assert t.shape == (256, 12 * sub * sub)
        cells = t.reshape(256, sub * sub, T.N_CLASSES)
        assert np.allclose(cells.sum(-1), 1.0, atol=1e-5)
        assert np.allclose(cells.mean(1), base, atol=1e-6)


def test_token_source_rejects_a_width_mismatch(tmp_path):
    """A 12-d cache must not be read as if it were a sub=2 cache.

    Checking that Linear(48,...) rejects a 12-wide tensor tested pytorch, not
    this code. This builds a real cache directory and asserts TokenSource
    refuses it, which is what protects an arm from being scored against the
    wrong map representation.
    """
    from beam import TokenSource

    np.save(tmp_path / "tokens.f16.npy", np.zeros((4, 256, 12), np.float16))
    # No index file: the width check runs before the index is read, which is
    # deliberate -- refusing a mismatched cache should not depend on anything
    # else being loadable. (Writing one here also faults pyarrow under pytest
    # on this machine.)
    with pytest.raises(ValueError, match="48-d"):
        TokenSource(tm.G, cache=str(tmp_path), sub=2)
    # The matching-width path is deliberately not exercised here: completing
    # the constructor builds a tile client and a thread pool, which faults in
    # native code under pytest on Windows. That path runs on every evaluation
    # anyway; the rejection is the part that had no coverage.
