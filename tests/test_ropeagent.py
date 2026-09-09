"""The retrain that turns a retrieval null into an agent number.

`knn_gap` said the fixed rotation does nothing: net +0.06 to +0.14 pp, spanning
zero everywhere. That closes the retrieval question and not the agent one --
`pool_pyramid`'s docstring is explicit that only a retrain converts a
retrieval probe into an agent number, "and this project has twice had
inference-level reasoning predict the wrong sign".

What makes the experiment clean is the pair of facts either side of that null:
the projected banks are mostly *different* (mean cosine to arm 0 of 0.326,
0.384, 0.238, so the PCA did not discard the rotation) while the neighbour
ordering is unchanged. Same neighbours, different vectors -- so a difference
here is the agent reading the vectors, not the retrieval moving under it.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import overnight as O  # noqa: E402
import ropeagent as R  # noqa: E402


def argv(arm):
    return R.rung(R.tag(arm), R.street(arm), "sequence", R.EPOCHS)


# ------------------------------------------------------ one flag apart ------

def test_an_arm_differs_from_the_baseline_only_in_its_bank():
    """The whole design. Anything else that moved would be measured as the
    rotation."""
    import json
    ship = json.load(open(ROOT / "runs/marks/night-c6-pyrL0L1.done"))["argv"]
    got = argv("ropeDR")
    assert len(ship) == len(got)
    diff = [(a, b) for a, b in zip(ship, got) if a != b]
    assert [a for a, _ in diff] == ["pyrL0L1-b340-c6",
                                    "pyr768_l0l1_b340.f16.npy",
                                    "knn_pyr768_l0l1_b340_sequence_k32_"
                                    "bank_ext70.npz"]


def test_every_arm_uses_the_same_recipe():
    base = [t for t in argv("rope0") if "rope0" not in t]
    for arm in R.ARMS[1:]:
        assert [t for t in argv(arm) if arm not in t] == base


def test_the_knn_file_follows_the_street_file():
    """A bank paired with another bank's neighbour table is the failure that
    has no symptom: right shape, right count, wrong photographs."""
    for arm in R.ARMS:
        a = argv(arm)
        assert a[a.index("--knn-file") + 1] == config_knn(arm)


def config_knn(arm):
    import config
    return config.knn_name(R.street(arm), "sequence", ext="bank_ext70")


def test_the_ladder_arms_and_the_agent_arms_are_the_same_set():
    import ropeladder
    assert R.ARMS == [a for a, _, _ in ropeladder.ARMS]


def test_arm_zero_is_trained_rather_than_assumed():
    """It should equal the shipping arm -- its bank is bit-identical on 2.65M
    of 3.4M rows and fp16 noise on the rest -- but "should" is the word that
    has cost this project the most, and the gap it measures is what a chain
    rebuild costs, which every other arm inherits."""
    assert "rope0" in R.ARMS


def test_one_cosine_not_a_chain():
    """`night0908` showed the chained `--init` sawtooth costs 0.38 to 2.18 pp
    against a single cosine."""
    for arm in R.ARMS:
        a = argv(arm)
        assert "--init" not in a
        assert a[a.index("--epochs") + 1] == str(R.EPOCHS)


# ---------------------------------------------------------- the window ------

class Sampler:
    def __init__(self):
        self.stop_flag = type("F", (), {"set": lambda self: None})()

    def start(self):
        pass


@pytest.fixture
def window(tmp_path, monkeypatch):
    cache = tmp_path / "street"
    ckpt = tmp_path / "ckpt"
    cache.mkdir()
    ckpt.mkdir()
    for name in ("RUNS", "LOGS", "MARKS"):
        monkeypatch.setattr(O, name, tmp_path / name)
    monkeypatch.setattr(R.config, "STREET_CACHE", cache)
    monkeypatch.setattr(R.config, "CHECKPOINTS", ckpt)
    monkeypatch.setattr(O, "Sampler", Sampler)
    monkeypatch.setattr(O, "log", lambda *a, **k: None)
    monkeypatch.setattr(R, "log", lambda *a, **k: None)
    monkeypatch.setattr(R, "load_state", lambda: {"done": {}, "failed": {}})
    monkeypatch.setattr(sys, "argv", ["ropeagent.py", "10"])
    for arm in R.ARMS:
        (cache / R.street(arm)).write_bytes(b"bank")
        (cache / config_knn(arm)).write_bytes(b"neighbours")
    return cache, ckpt


def runner(monkeypatch, ckpt, fails=()):
    seen = []

    def fake(st, state, deadline):
        seen.append(st.name)
        if st.name not in fails:
            (ckpt / (st.name + ".pt")).write_bytes(b"w")
            return True
        return False

    monkeypatch.setattr(O, "run_stage", fake)
    return seen


def test_a_missing_bank_is_refused_before_anything_trains(window, monkeypatch):
    """Otherwise it is four retries over twenty minutes with the reason
    buried in a training log."""
    cache, ckpt = window
    (cache / R.street("ropeDR")).unlink()
    seen = runner(monkeypatch, ckpt)
    assert R.main() == 1
    assert seen == []


def test_a_missing_neighbour_table_is_refused_too(window, monkeypatch):
    cache, ckpt = window
    (cache / config_knn("ropeD")).unlink()
    seen = runner(monkeypatch, ckpt)
    assert R.main() == 1 and seen == []


def test_a_complete_window_compares_every_arm_and_the_shipping_one(
        window, monkeypatch):
    cache, ckpt = window
    got = []
    monkeypatch.setattr(O, "run_stage",
                        lambda st, *a: (got.append(st.argv),
                                        (ckpt / (st.name + ".pt")
                                         ).write_bytes(b"w"), True)[-1])
    assert R.main() == 0
    boot = [a for a in got if a[0].endswith("bootstrap.py")][0]
    tags = boot[boot.index("--tags") + 1].split(",")
    assert tags == [R.tag(a) for a in R.ARMS] + [R.SHIP]


def test_a_failed_arm_still_leaves_a_comparison(window, monkeypatch):
    """Arms are independent, so three of them plus the shipping reference is
    still a result."""
    cache, ckpt = window
    seen = runner(monkeypatch, ckpt, fails=("ropeDRC-c6",))
    assert R.main() == 1                      # incomplete, but measured
    assert "ropeagent-boot" in seen


def test_one_arm_alone_is_not_compared(window, monkeypatch):
    """One rotated arm against the shipping one cannot separate the rotation
    from the chain rebuild -- the shape `runs/TILESHIP.md` got wrong."""
    cache, ckpt = window
    seen = runner(monkeypatch, ckpt,
                  fails=tuple(R.tag(a) for a in R.ARMS[1:]))
    assert R.main() == 1
    assert "ropeagent-boot" not in seen


def test_an_arm_without_a_checkpoint_is_not_compared(window, monkeypatch):
    cache, ckpt = window
    got = []
    monkeypatch.setattr(O, "run_stage",
                        lambda st, *a: (got.append(st.name), True)[1])
    R.main()                                   # nothing writes a .pt
    assert "ropeagent-boot" not in got
