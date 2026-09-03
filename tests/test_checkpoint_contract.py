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
    ("geo", {"geo": "tile"}),
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
    """
    src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
    i = src.index("torch.save({")
    j = src.index("config.CHECKPOINTS", i)
    return set(re.findall(r'"([a-z_0-9]+)":', src[i:j]))


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
