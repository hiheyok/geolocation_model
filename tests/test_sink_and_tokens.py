"""Train/inference parity for the pieces added on 2026-09-02.

Both bugs these cover were live: `--sink-k > 1` trained one network and scored
another because beam search called `policy_logits` without `step`, and a
sub-patch map cache could be read by a model built for 12-d tokens. Neither
raises -- they quietly evaluate a different model than the one that trained,
which is the failure mode this project keeps hitting.
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


def _logits(m, B=8, seed=1):
    torch.manual_seed(seed)
    f = torch.randn(B, 512)
    keys = torch.randn(B, tm.actions(), 256)
    step = torch.randint(0, tm.STEPS, (B,))
    k = torch.cat([keys, m.sink.expand(B, 1, -1)], 1)
    with torch.no_grad():
        return m.policy_logits(f, k, None, step), m.policy_logits(f, k, None), step


def test_extra_sink_keys_start_neutral():
    """A sink_k=4 model loaded with sink_k=1 weights must score identically."""
    a, b = build(1), build(4)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
    assert set(missing) == {"sink_ext", "sink_ext_b"}
    assert not unexpected
    a.eval(); b.eval()
    la, _, _ = _logits(a)
    lb, _, _ = _logits(b)
    assert torch.allclose(la, lb, atol=1e-6)


def test_sink_keys_are_dead_without_step():
    """The regression itself: omitting `step` silently disables the extras.

    Once the extras carry weight, passing step must change the sink logit.
    If this ever stops differing, the extras are inert and sink_k is a no-op.
    """
    m = build(4)
    m.eval()
    with torch.no_grad():
        m.sink_ext_b.fill_(0.0)          # switch the extra keys on
    with_step, without_step, _ = _logits(m)
    assert not torch.allclose(with_step[:, -1], without_step[:, -1], atol=1e-4)


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


def test_tokenizer_width_must_match_the_cache():
    """A model built for 12-d tokens must not silently accept 48-d ones."""
    m = build(1, n_classes=48)
    assert m.map.proj.in_features == 48
    with pytest.raises(RuntimeError):
        m.map.proj(torch.randn(4, 256, 12))
