"""A checkpoint is a contract between train.py and everything that reads one.

The most expensive bugs in this project all live in that gap, and none of them
raised: `--sink-k > 1` trained one network and scored another because a caller
dropped an argument; `eval_highres` hardcoded a bank and printed one arm's
number under another's name; the k-NN row restriction was taken from the wrong
cache. Component tests cannot see any of that. The defect is never inside a
piece -- it is a consumer reading fewer fields than the producer wrote.

So there are two tests here, and the second is the one with teeth. The first
checks that load_model rebuilds the architecture a checkpoint describes. The
second checks that every field train.py writes is actually read by somebody:
that is the check that would have caught `--save-opt` saving optimizer state
nothing loads, and it fails the moment a new field is added without a reader.
"""

import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tile_math as tm  # noqa: E402
from model import GeoAgent  # noqa: E402


# How a GeoAgent constructor argument is spelled in a checkpoint. load_model
# has to perform this translation, and getting one wrong is invisible: the
# model builds, loads and scores, just not as the architecture that trained.
CK_NAME = {
    "map_layers": lambda v: {"map_layers": v},
    "pool": lambda v: {"pool": v},
    "n_pool_q": lambda v: {"pool_q": v},
    "pos": lambda v: {"pos": v},
    "sink": lambda v: {"neg": 4 if v else 0},
    "sink_k": lambda v: {"sink_k": v},
    "n_classes": lambda v: {"map_sub": int(round((v / 12) ** 0.5))},
    "mem": lambda v: {"mem": v},
    "retr": lambda v: {"retr": v},
    "retr_mode": lambda v: {"retr_mode": v},
    "d_key": lambda v: {"d_key": v},
    "geo": lambda v: {"geo": v},
}

# Each case is a full non-default architecture. If load_model drops any part of
# it the parameters differ and the strict load fails -- the mechanism that
# caught sink_k, generalised to every field that shapes the network.
ARCH_CASES = [
    ("map_layers", {"map_layers": 2}),
    ("pool", {"pool": "attn"}),
    ("pool_q", {"pool": "attn", "n_pool_q": 8}),
    ("pos", {"pos": "both"}),
    ("sink_k", {"sink": True, "sink_k": 4}),
    ("map_sub", {"n_classes": 12 * 4}),
    ("mem", {"mem": "content"}),
    ("retr_mode", {"retr": True, "retr_mode": "dual"}),
    ("d_key", {"retr": True, "retr_mode": "dual", "d_key": 64}),
    ("geo", {"geo": "key"}),
]

DEFAULT_CK = {
    "g": tm.G, "steps": tm.STEPS, "map_layers": 0, "pool": "mean",
    "pool_q": 4, "pos": "learned", "neg": 0, "sink_k": 1, "map_sub": 1,
    "mem": "none", "d_mem": 64, "retr": False, "retr_mode": "scalar",
    "d_key": 128, "retr_tau": 0.07, "map_loop": False, "enc_gate": False,
    "geo": "none", "d_geo": 128,
}


def build(**kw):
    torch.manual_seed(0)
    base = dict(d_street=768, n_actions=tm.actions(), n_steps=tm.STEPS + 1)
    base.update(kw)
    return GeoAgent(**base)



def logits_of(model, n_classes=12, seed=3):
    """A fixed forward, for comparing two models that hold the same weights."""
    torch.manual_seed(seed)
    B, S, A = 2, tm.STEPS + 1, tm.actions()
    batch = {"street": torch.randn(B, 768),
             "tokens": torch.randn(B, S, A, n_classes),
             "x0": torch.rand(B, S), "y0": torch.rand(B, S),
             "step": torch.arange(S).unsqueeze(0).expand(B, S).contiguous()}
    model.eval()
    with torch.no_grad():
        lg, _ = model(batch)
    return lg


@pytest.mark.parametrize("name,kw", ARCH_CASES, ids=[c[0] for c in ARCH_CASES])
def test_load_model_rebuilds_what_the_checkpoint_describes(
        name, kw, tmp_path, monkeypatch):
    """Save a non-default architecture, read it back, compare the parameters."""
    import config
    import evaluate as E

    m = build(**kw)
    ck = dict(DEFAULT_CK, model=m.state_dict())
    for arg, val in kw.items():
        ck.update(CK_NAME[arg](val))

    monkeypatch.setattr(config, "CHECKPOINTS", tmp_path)
    torch.save(ck, tmp_path / "contract.pt")
    loaded, _, d_street = E.load_model("contract", "cpu")

    assert d_street == 768
    a, b = m.state_dict(), loaded.state_dict()
    assert set(a) == set(b), "{}: parameter set differs".format(name)
    for k in a:
        assert a[k].shape == b[k].shape, "{}: {} shape differs".format(name, k)

    nc = kw.get("n_classes", 12)
    assert torch.allclose(logits_of(m, nc), logits_of(loaded, nc), atol=1e-6), (
        "{}: the reloaded model scores differently".format(name))

    # A case only proves something if the field actually changes the model.
    # geo="tile" was silently not a value GeoAgent recognises, and the
    # comparison passed because both sides built the same default. Where the
    # parameter sets differ, a strict load already proves it; where they match,
    # the difference has to be behavioural -- pos="both" adds rotary, which has
    # no parameters at all, so a reader that dropped it would load cleanly and
    # score a different network in silence.
    d = build()
    ds = d.state_dict()
    if set(ds) == set(a) and all(ds[k].shape == a[k].shape for k in ds):
        d.load_state_dict(a)
        assert not torch.allclose(logits_of(d, nc), logits_of(m, nc), atol=1e-6), (
            "{}: same parameters AND same outputs as the default, so this case "
            "cannot detect a reader that ignores the field".format(name))


def test_the_round_trip_would_notice_a_dropped_field(tmp_path, monkeypatch):
    """The round-trip test is only worth having if it fails when it should.

    A checkpoint that says sink_k=1 while its weights hold four sink keys is
    exactly the shape of the bug that shipped: the reader builds the smaller
    network and scores something the training never produced.
    """
    import config
    import evaluate as E

    m = build(sink=True, sink_k=4)
    ck = dict(DEFAULT_CK, model=m.state_dict(), neg=4, sink_k=1)
    monkeypatch.setattr(config, "CHECKPOINTS", tmp_path)
    torch.save(ck, tmp_path / "contract.pt")
    with pytest.raises(RuntimeError):
        E.load_model("contract", "cpu")


def saved_fields():
    """The keys train.py writes, read out of the source rather than restated.

    Restating them here is the mistake this file exists to catch: the list
    would go stale exactly when a field is added, which is the moment it
    matters.

    Parsed from the AST rather than matched against the call, because the call
    keeps changing shape and the test kept failing for the wrong reason -- once
    when the write moved from `torch.save` to `safeio.save_torch`, and again
    when the dict was lifted into a local. The checkpoint dict is identifiable
    on its own terms: it is by far the largest string-keyed dict literal in the
    file.
    """
    import ast

    tree = ast.parse((ROOT / "src" / "train.py").read_text(encoding="utf-8"))
    best = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if len(keys) > len(best):
            best = keys
    return best


# Fields written for provenance -- a human or a report reads them, nothing
# reconstructs behaviour from them. Anything not on this list must have a
# reader, or it is dead weight that will be mistaken for a live setting.
PROVENANCE = {
    "g", "steps", "epoch", "epochs_total", "val_loss", "val_km", "val_hit",
    "select", "sel_n", "sel_k", "init_from", "soft", "retr_drop", "model",
}

# Orphans that are known and recorded, so a NEW one still fails the test.
# Listing them here is not forgiveness: each is on the backlog, and the test
# also fails once one is fixed, so the entry cannot outlive the defect.
KNOWN_ORPHANS = {
    # --save-opt writes optimizer state that nothing loads, and neither the
    # scheduler nor the RNG is saved at all. So a 2+2+2 ladder is three
    # optimizer restarts, not six continuous epochs, which is how "e4 vs e6 is
    # inside noise" should be read. Deferred deliberately: fixing it changes
    # training, and every arm on file was trained the current way.
    "opt",
}


def test_every_saved_field_has_a_reader():
    """A written field with no reader is a setting that silently does nothing.

    `--save-opt` is the live example: it writes optimizer state that nothing
    loads, so a 2+2+2 ladder is three optimizer restarts rather than six
    continuous epochs -- which changes how "e4 vs e6 is inside noise" should be
    read. Recorded in the backlog; this test names any future one immediately.
    """
    fields = saved_fields()
    assert "retr_k" in fields and "knn_file" in fields, (
        "the parser stopped matching train.py's save block")

    readers = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in list((ROOT / "src").glob("*.py"))
        + list((ROOT / "scripts").glob("*.py"))
        if p.name != "train.py")

    # An actual read expression -- ck.get("x") or ck["x"] -- not the name
    # appearing anywhere. A bare substring search calls `steps` and `init_from`
    # read because they occur inside unrelated strings, which is how a check
    # like this quietly stops checking.
    orphans = {f for f in fields - PROVENANCE
               if not re.search(
                   r'\w*(?:\.get\(|\[)\s*"' + re.escape(f) + '"', readers)}

    new = sorted(orphans - KNOWN_ORPHANS)
    assert not new, (
        "written to every checkpoint and read by nothing: {}. Either wire it "
        "up, stop writing it, or record it in PROVENANCE or KNOWN_ORPHANS "
        "with a reason.".format(", ".join(new)))

    fixed = sorted(KNOWN_ORPHANS - orphans)
    assert not fixed, (
        "{} now has a reader -- remove it from KNOWN_ORPHANS so the list keeps "
        "meaning what it says.".format(", ".join(fixed)))


def test_the_training_forward_and_the_search_forward_agree():
    """Two code paths score the same model, and they must not drift apart.

    `forward` assembles a (B, S) batch; beam search assembles (image, beam)
    rows and calls fuse_flat / retr_prior / _add_geo / policy_logits itself.
    Every bug in this file's docstring is a divergence between those two, and
    the sink one was exactly this: search omitted `step`, so the extra sink
    keys were live in training and dead at inference, and nothing raised.

    Every optional additive term is switched on. That matters: with the default
    geo="none" the `_add_geo` line is a no-op, and a test that exercises a
    no-op proves the two paths agree about nothing.
    """
    torch.manual_seed(0)
    B, S, A = 3, tm.STEPS + 1, tm.actions()
    m = build(sink=True, sink_k=4, retr=True, retr_mode="dual",
              pool="attn", pos="both", geo="key")
    # An all-zero parameter makes its whole branch vacuous, and this model has
    # several by design: the retrieval gates start at zero, and GeoMem's table
    # starts at zero behind a gate of one. Comparing the two paths at init
    # would therefore hold even if both ignored the additive terms entirely --
    # the zero-gate trap that once made a dead feature look like an honest
    # null. So give every zeroed parameter something to say.
    with torch.no_grad():
        for _, q in m.named_parameters():
            if not q.any():
                q.copy_(torch.randn_like(q) * 0.05 + 0.1)
    m.eval()

    batch = {
        "street": torch.randn(B, 768),
        "tokens": torch.randn(B, S, A, 12),
        "x0": torch.rand(B, S), "y0": torch.rand(B, S),
        "step": torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
        "nbr_x": torch.randint(0, 1 << 16, (B, 8)),
        "nbr_y": torch.randint(0, 1 << 16, (B, 8)),
        "nbr_sim": torch.rand(B, 8),
        "nbr_emb": torch.randn(B, 8, 768),
    }
    with torch.no_grad():
        train_logits, _ = m(batch)

    # The same rows, assembled the way search assembles them: one row per
    # (image, live beam) rather than one per (image, step).
    n = S - 1
    st, x0 = batch["street"], batch["x0"][:, :n].reshape(-1)
    y0, sp = batch["y0"][:, :n].reshape(-1), batch["step"][:, :n].reshape(-1)
    with torch.no_grad():
        f, keys = m.fuse_flat(
            st.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1),
            batch["tokens"][:, :n].reshape(B * n, A, 12), x0, y0, sp)
        n_logits = keys.shape[1] + 1
        prior = m.retr_prior(
            (batch["nbr_x"], batch["nbr_y"], batch["nbr_sim"], batch["nbr_emb"]),
            st, x0, y0, sp, n, n_logits)
        prior = m._add_geo(prior, f, x0, y0, sp, n_logits)
        search_logits = m.policy_logits(f, keys, prior, sp)

    assert search_logits.shape == (B * n, A + 1)
    assert prior is not None and prior.abs().sum() > 0, (
        "the additive terms are all zero; this test would pass with both "
        "paths ignoring them")
    assert torch.allclose(train_logits.reshape(B * n, -1), search_logits,
                          atol=1e-5), "the two forwards disagree"


def test_dropping_step_raises_rather_than_diverging():
    """The sink regression itself: a caller that forgets `step` must fail."""
    m = build(sink=True, sink_k=4)
    m.eval()
    with pytest.raises(ValueError, match="step"):
        m.policy_logits(torch.randn(4, 512),
                        torch.randn(4, tm.actions(), 256), None)
