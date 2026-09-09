"""Bounding the instrument, before trusting it on sub-1 pp gaps again.

`rope0-c6` and `pyrL0L1-b340-c6` are the same experiment -- same recipe, same
seed, neighbour tables differing on 84 of 500,000 top-1 queries -- and at
20,000 test images they separate at [+0.12, +1.02] pp. They disagree on 10.1%
of images; the separation is the +111 residual of 2,029 flips.

The paired bootstrap resamples test images. It has no training-run variance in
it, so it will separate two identical runs given enough images. The cause is
`--neg-random`, which twelve call sites pass: it overrides the seeded sink
negatives with OS entropy drawn inside each `__getitem__`, so no two runs see
the same off-path tiles. Dropping it is one token, and costs an image seeing
the same negatives every epoch -- which is why this measures both directions.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import overnight as O  # noqa: E402
import negseed as N  # noqa: E402


def reference_argv():
    from night0908 import rung
    return rung(N.REF, N.BANK, "sequence", N.EPOCHS)


# ---------------------------------------------------- exactly one token -----

def test_the_arm_is_the_reference_minus_neg_random():
    """Removed, not rebuilt, so the contrast cannot carry anything else."""
    ref = reference_argv()
    got = N.argv_for(N.REF)
    assert "--neg-random" in ref
    assert got == [t for t in ref if t != "--neg-random"]
    assert len(ref) - len(got) == 1


def test_neg_random_is_a_bare_flag_so_removing_the_token_is_the_whole_edit():
    """If it took a value, dropping the token would strand that value as a
    positional argument -- which `train.py` would either reject, or worse,
    silently absorb into another flag."""
    ref = reference_argv()
    i = ref.index("--neg-random")
    assert ref[i + 1].startswith("--"), ref[i:i + 2]
    src = Path(ROOT / "src/train.py").read_text(encoding="utf-8")
    j = src.index('"--neg-random"')
    assert 'action="store_true"' in src[j:j + 120]


def test_both_arms_run_the_same_command_but_for_the_tag():
    a, b = (N.argv_for(t) for t in N.ARMS)
    assert [t for t in a if t not in N.ARMS] == [t for t in b
                                                 if t not in N.ARMS]
    assert a != b


def test_both_arms_share_the_reference_bank():
    """Both contrasts are against `rope0-c6`, so its bank has to be the one
    under test -- otherwise the pair measures the bank too."""
    for t in N.ARMS:
        a = N.argv_for(t)
        assert a[a.index("--street-file") + 1] == N.BANK


def test_the_seed_is_pinned():
    for t in N.ARMS:
        a = N.argv_for(t)
        assert a[a.index("--seed") + 1] == "0"


# ---------------------------------------------------------- the window ------

class Sampler:
    def __init__(self):
        self.stop_flag = type("F", (), {"set": lambda self: None})()

    def start(self):
        pass


@pytest.fixture
def window(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / (N.REF + ".pt")).write_bytes(b"reference weights")
    for name in ("RUNS", "LOGS", "MARKS"):
        monkeypatch.setattr(O, name, tmp_path / name)
    monkeypatch.setattr(N.config, "CHECKPOINTS", ckpt)
    monkeypatch.setattr(O, "Sampler", Sampler)
    monkeypatch.setattr(O, "log", lambda *a, **k: None)
    monkeypatch.setattr(N, "log", lambda *a, **k: None)
    monkeypatch.setattr(N, "load_state", lambda: {"done": {}, "failed": {}})
    monkeypatch.setattr(sys, "argv", ["negseed.py", "10"])
    return ckpt


def runner(monkeypatch, ckpt, fails=()):
    seen = []

    def fake(st, state, deadline):
        seen.append(st.name)
        if st.name in fails:
            return False
        (ckpt / (st.name + ".pt")).write_bytes(b"w")
        return True

    monkeypatch.setattr(O, "run_stage", fake)
    return seen


def test_a_missing_reference_stops_the_window(window, monkeypatch):
    (window / (N.REF + ".pt")).unlink()
    seen = runner(monkeypatch, window)
    assert N.main() == 1 and seen == []


def test_both_arms_or_no_comparison(window, monkeypatch):
    """One seeded arm against the reference measures the seeding change
    confounded with the run-to-run spread this exists to bound -- that is the
    question, not an answer to it."""
    seen = runner(monkeypatch, window, fails=("negseed-b-c6",))
    assert N.main() == 1
    assert "negseed-boot" not in seen


def test_a_complete_window_compares_both_arms_and_the_reference(
        window, monkeypatch):
    got = []
    monkeypatch.setattr(
        O, "run_stage",
        lambda st, *a: (got.append(st.argv),
                        (window / (st.name + ".pt")).write_bytes(b"w"),
                        True)[-1])
    assert N.main() == 0
    boot = [a for a in got if a[0].endswith("bootstrap.py")][0]
    assert boot[boot.index("--tags") + 1].split(",") == N.ARMS + [N.REF]


def test_the_comparison_uses_the_larger_sample(window, monkeypatch):
    """5,000 images could not resolve this: the contrast being bounded was
    inside noise at 5,000 and separated at 20,000."""
    got = []
    monkeypatch.setattr(
        O, "run_stage",
        lambda st, *a: (got.append(st.argv),
                        (window / (st.name + ".pt")).write_bytes(b"w"),
                        True)[-1])
    N.main()
    boot = [a for a in got if a[0].endswith("bootstrap.py")][0]
    assert boot[boot.index("--n") + 1] == "20000"
