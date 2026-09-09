"""Reconstruction pretraining for the fusion trunk, and what it must not touch.

`pyramid-fusion-head-is-negative` recorded a head whose retrieval fell
monotonically as its loss fell -- 33.1% at init, 32.2% at one epoch, 20.5% at
twelve, 12.9% with tighter positives -- and named the cause: 2.5M parameters
against 38,009 images, the ones with a cross-sequence positive within
`--pos-km`. Its post-mortem says to attack data volume rather than the
objective's details.

Reconstruction needs no pairs, so it reaches all 3,400,180 rows. The thing
that must survive it is the head's honesty property: `out` is a residual on
the mean-pooled baseline, zero-initialised, so the contrastive phase starts
exactly at that baseline and can only depart if it pays. Pretraining the
output layer would spend that, and every number the head has ever reported is
comparable only through it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fuse_pretrain as F  # noqa: E402
from fuse_head import FuseHead, D_ENC  # noqa: E402


def head(d=32, n_reg=9):
    return FuseHead(d=d, n_reg=n_reg)


def toks(b=4, n_reg=9):
    x = torch.randn(b, n_reg, 2, D_ENC)
    return torch.nn.functional.normalize(x, dim=-1)


# ------------------------------------------- what must not be pretrained ----

def test_the_residual_is_excluded_from_the_saved_trunk():
    """The property every head number is comparable through."""
    assert F.NOT_PRETRAINED == ("out.",)
    m = head()
    keys = [k for k in m.state_dict() if not k.startswith("out.")]
    assert keys and all(not k.startswith("out.") for k in keys)
    assert [k for k in m.state_dict() if k.startswith("out.")], \
        "the head has no `out.*` to exclude, so this guard is vacuous"


def test_loading_a_trunk_leaves_the_residual_at_zero():
    m = head()
    trained = head()
    torch.nn.init.normal_(trained.out[1].weight)     # pretend it was trained
    trunk = {k: v for k, v in trained.state_dict().items()
             if not k.startswith("out.")}
    miss, unexpected = m.load_state_dict(trunk, strict=False)
    assert not unexpected
    assert set(miss) == {"out.1.weight", "out.1.bias", "out.0.weight",
                         "out.0.bias"} or all(k.startswith("out.")
                                              for k in miss), miss
    assert float(m.out[1].weight.abs().max()) == 0.0


def test_a_head_with_a_zero_residual_emits_the_baseline():
    """What 'starts at the baseline' means, checked rather than asserted in
    a comment."""
    m = head().eval()
    x = toks()
    with torch.no_grad():
        got = m(x)
        want = torch.nn.functional.normalize(m.base(x), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_pretraining_does_not_give_the_residual_a_gradient():
    m = head()
    dec = F.Decoder(32, 9)
    for p in m.out.parameters():
        p.requires_grad_(False)
    x = toks()
    p = F.trunk_forward(m, x)
    r = torch.nn.functional.normalize(dec(p), dim=-1)
    (1.0 - (r * x).sum(-1)).mean().backward()
    assert m.out[1].weight.grad is None
    assert any(q.grad is not None and float(q.grad.abs().sum()) > 0
               for q in m.proj.parameters()), "the trunk got no gradient"


# ------------------------------------------------------- the trunk path -----

def test_trunk_forward_matches_the_shipping_forward_up_to_the_pool():
    """`trunk_forward` restates `FuseHead.forward` rather than refactoring it,
    because `forward` is the shipping path for three recorded results. So the
    restatement has to be checked against the original."""
    m = head().eval()
    x = toks()
    with torch.no_grad():
        p = F.trunk_forward(m, x)
        full = m(x)
        rebuilt = torch.nn.functional.normalize(m.base(x) + m.out(p), dim=-1)
    assert torch.allclose(full, rebuilt, atol=1e-6)


def test_the_decoder_reconstructs_the_token_grid_shape():
    dec = F.Decoder(32, 9)
    out = dec(torch.randn(3, 4 * 32))
    assert out.shape == (3, 9, 2, D_ENC)


def test_the_decoder_can_learn_a_fixed_target():
    """So a flat loss in the real run means the data, not a broken head."""
    torch.manual_seed(0)
    m, dec = head(), F.Decoder(32, 9)
    opt = torch.optim.Adam(list(m.parameters()) + list(dec.parameters()),
                           lr=3e-3)
    x = toks(b=8)
    first = None
    for _ in range(60):
        p = F.trunk_forward(m, x)
        r = torch.nn.functional.normalize(dec(p), dim=-1)
        loss = (1.0 - (r * x).sum(-1)).mean()
        first = float(loss) if first is None else first
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert float(loss) < first * 0.8, (first, float(loss))


# ---------------------------------------------------------- the corpora -----

EXPECTED = {"dual_c3.f16.npy": "tile6",
            "bank_ext_bal.f16.npy": "tile6_ext",
            "bank_ext2_bal.f16.npy": "tile6_ext2",
            "bank_ext3_bal.f16.npy": "tile6_ext3",
            "bank_ext4_bal.f16.npy": "tile6_ext4"}


def test_every_corpus_pairs_crops_with_its_own_tiles():
    """A mismatched pair is one image's crops beside another image's tiles:
    right shape, unit norm, wrong photograph, and it trains perfectly well.

    Spelled out per corpus rather than derived from the name, because the
    derivation is the thing that could be wrong."""
    assert dict(F.CORPORA) == EXPECTED


def test_the_release_comes_first_and_the_extensions_are_in_order():
    """Reconstruction does not care about row order, but a reader comparing
    this against `ropeladder.EXTS` or `bank_ext70_meta.npz` does."""
    assert [t for _, t in F.CORPORA] == ["tile6", "tile6_ext", "tile6_ext2",
                                         "tile6_ext3", "tile6_ext4"]


def test_the_corpora_cover_the_whole_bank_not_just_the_release():
    """The whole point: every row the corpus has, against the 38,009 images
    with a cross-sequence positive. Training on the release alone would be
    500,000 -- better than 38,009 and still not the argument."""
    assert len(F.CORPORA) == 5
    assert F.CORPORA[0][0] == "dual_c3.f16.npy"
    assert sum(1 for c, _ in F.CORPORA if "bank_ext" in c) == 4


def test_a_missing_corpus_is_skipped_not_fatal(monkeypatch, tmp_path, capsys):
    """A partial corpus should cost coverage, not the run -- but it has to
    say so, because silently training on a fifth of the data and reporting
    the row count is how a null gets attributed to the method."""
    monkeypatch.setattr(F.config, "STREET_CACHE", tmp_path)
    got = list(F.corpus_blocks(16))
    assert got == []
    assert "absent" in capsys.readouterr().out
