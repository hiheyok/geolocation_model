"""The window loop, once, with every rule the copies had learned separately.

Four scripts had grown their own copy of "run these stages in order": the
`overnight` arm loop, `tilefull`, `tilefull_ladder` and `night0908`. Each copy
knew a different subset of the rules, and each one learned its subset from a
separate review finding -- `tilefull` that a deadline break must not exit 0
(#59), `night0908` that a bootstrap whose training stage failed will happily
compare whatever checkpoint is on disk. Neither knew what the other knew, and
`Stage.critical` is still documented-but-unread in two of them.

That is `shared-leaf-duplicated-loop` exactly: the leaf (`run_stage`) was
shared from the start; the loop around it was copied. These tests are about
the loop, so they live with the loop.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import overnight as O  # noqa: E402


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(O, "log", lambda *a, **k: None)


def plan(*names):
    return [O.Stage(n, ["x"], est=1) for n in names]


def runner(monkeypatch, fails=()):
    """Record the stages that ran; fail the named ones."""
    seen = []
    def fake(st, state, deadline):
        seen.append(st.name)
        return st.name not in fails
    monkeypatch.setattr(O, "run_stage", fake)
    return seen


def test_everything_runs_when_nothing_goes_wrong(quiet, monkeypatch):
    """So the tests below are not passing vacuously."""
    seen = runner(monkeypatch)
    res = O.run_plan(plan("a", "b", "c"), {}, 10 ** 12)
    assert seen == ["a", "b", "c"]
    assert res == {"done": ["a", "b", "c"], "failed": [], "skipped": [],
                   "unrun": []}


def test_a_non_critical_failure_does_not_stop_the_window(quiet, monkeypatch):
    """A window survives one dead arm; that is what non-critical means."""
    seen = runner(monkeypatch, fails=("a",))
    res = O.run_plan(plan("a", "b"), {}, 10 ** 12)
    assert seen == ["a", "b"]
    assert res["failed"] == ["a"] and res["done"] == ["b"]


def test_a_critical_failure_drops_the_rest_and_names_them(quiet, monkeypatch):
    """`Stage.critical` is documented in three runners and read by one."""
    p = plan("a", "b", "c")
    p[0].critical = True
    seen = runner(monkeypatch, fails=("a",))
    res = O.run_plan(p, {}, 10 ** 12)
    assert seen == ["a"]
    assert res["unrun"] == ["b", "c"], "dropped stages were not recorded"


def test_a_deadline_records_what_did_not_run(quiet, monkeypatch):
    """The #59 defect: the loop broke and the caller saw a completed window."""
    seen = runner(monkeypatch)
    res = O.run_plan(plan("a", "b", "c"), {}, -1.0)
    assert seen == []
    assert res["unrun"] == ["a", "b", "c"]


def test_a_comparison_is_skipped_when_its_subject_failed(quiet, monkeypatch):
    """A bootstrap reads checkpoints by tag, and a failed tag usually still
    has a file -- the previous rung's, or an older run's. Running it anyway
    produces a clean-looking report of the wrong thing."""
    p = plan("train", "boot")
    p[1].prereqs = ("train",)
    seen = runner(monkeypatch, fails=("train",))
    res = O.run_plan(p, {}, 10 ** 12)
    assert seen == ["train"]
    assert res["skipped"] == ["boot"]


def test_a_skip_propagates_like_a_failure(quiet, monkeypatch):
    """A stage skipped for a dead prerequisite did not run either, so anything
    depending on IT is in exactly the same position. The copy this was taken
    from checked only the failed list, so the second link in a chain ran."""
    p = plan("train", "boot", "report")
    p[1].prereqs = ("train",)
    p[2].prereqs = ("boot",)
    seen = runner(monkeypatch, fails=("train",))
    res = O.run_plan(p, {}, 10 ** 12)
    assert seen == ["train"]
    assert res["skipped"] == ["boot", "report"]


def test_a_stage_with_no_prereqs_attribute_is_fine(quiet, monkeypatch):
    """`Stage` does not define `prereqs`; the callers attach it."""
    seen = runner(monkeypatch)
    O.run_plan(plan("a"), {}, 10 ** 12)
    assert seen == ["a"]


# ------------------------------------------------------------ the window ----

import lrlong  # noqa: E402


class Sampler:
    def __init__(self):
        self.stop_flag = type("F", (), {"set": lambda self: None})()

    def start(self):
        pass


@pytest.fixture
def window(tmp_path, monkeypatch, quiet):
    for name in ("RUNS", "LOGS", "MARKS"):
        monkeypatch.setattr(O, name, tmp_path / name)
    monkeypatch.setattr(O, "Sampler", Sampler)
    monkeypatch.setattr(lrlong, "load_state", lambda: {"done": {}, "failed": {}})
    monkeypatch.setattr(lrlong, "log", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["lrlong.py", "10"])
    monkeypatch.setattr(lrlong.config, "CHECKPOINTS", tmp_path)
    return tmp_path


def test_the_window_compares_only_arms_that_trained(window, monkeypatch):
    """The bootstrap tag list is assembled from what ran, not from the plan.

    Both halves matter: a stage that failed must not be compared, and neither
    must a stage that succeeded in a previous attempt of this script but left
    no checkpoint behind.
    """
    seen = runner(monkeypatch, fails=("pyrL0L1-b340-c10",))
    for tag in ("pyrL0L1-b340-c6x2-lr5e-5", "pyrL0L1-b340-c6-lr15"):
        (window / (tag + ".pt")).write_bytes(b"x")
    lrlong.main()
    boot = [s for s in seen if s == "lrlong-boot"]
    assert boot, "the comparison never ran"


def test_the_window_skips_the_comparison_when_no_arm_trained(window,
                                                             monkeypatch):
    """Otherwise it re-reports c6 against e4 -- a real-looking table of a
    measurement this window did not make."""
    seen = runner(monkeypatch, fails=tuple(
        t for t in ("pyrL0L1-b340-c6x2-lr5e-5", "pyrL0L1-b340-c6-lr15",
                    "pyrL0L1-b340-c6-lr6", "pyrL0L1-b340-c10")))
    lrlong.main()
    assert "lrlong-boot" not in seen


def test_an_arm_that_ran_without_leaving_a_checkpoint_is_not_compared(
        window, monkeypatch):
    """A stage can be recorded done by an earlier attempt of this script and
    still have no checkpoint -- an interrupted save, or a cleaned directory.
    Comparing it reads whatever `bootstrap` finds under that tag, or nothing.
    """
    argv = []
    monkeypatch.setattr(O, "run_stage",
                        lambda st, *a: (argv.append(st.argv), True)[1])
    lrlong.main()                       # every stage "succeeds", no .pt files
    assert not [a for a in argv if a and a[0].endswith("bootstrap.py")]


def test_an_incomplete_window_does_not_exit_zero(window, monkeypatch):
    runner(monkeypatch, fails=("pyrL0L1-b340-c10",))
    for tag in ("pyrL0L1-b340-c6x2-lr5e-5", "pyrL0L1-b340-c6-lr15",
                "pyrL0L1-b340-c6-lr6"):
        (window / (tag + ".pt")).write_bytes(b"x")
    assert lrlong.main() == 1
