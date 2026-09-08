"""A failed critical stage must stop the plan and make the run exit non-zero.

`Stage.critical` is documented as "if it fails, dependents are dropped".
Nothing in this repository read it: `overnight.py` sets it on six stages and
`tilebig.py` on four, and both loops call `run_stage` and discard the result.
It reads as protection and provided none.

What that costs here is specific. The plan is pool -> stack -> project ->
build_knn per arm, then `knn_gap`. Each stage writes a path the next one
reads, and those paths usually already exist from an earlier run. So a failed
pool does not stop anything: the stack stacks the previous pool's output, the
projection projects that, the index indexes it, and `knn_gap` publishes a
comparison in which one arm is silently a different experiment.

Review found it by simulating a failure of the first critical stage and
watching all 20 stages get scheduled anyway, with a zero exit. These are that
probe.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import overnight as O  # noqa: E402
import tilefull  # noqa: E402


class Sampler:
    """`main` starts a sampler thread and sets a flag on it at the end."""

    def __init__(self):
        self.stop_flag = type("F", (), {"set": lambda self: None})()

    def start(self):
        pass


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """`tilefull.main`, with the disk, the threads and the stages stubbed.

    Returns the list every `run_stage` call appends to, so a test can see
    exactly which stages were scheduled rather than only the exit code.
    """
    for name in ("RUNS", "LOGS", "MARKS"):
        monkeypatch.setattr(O, name, tmp_path / name)
    monkeypatch.setattr(O, "Sampler", Sampler)
    monkeypatch.setattr(tilefull, "load_state", lambda: {"done": {}, "failed": {}})
    monkeypatch.setattr(tilefull, "log", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["tilefull.py", "10"])
    return []


def stub(seen, fail_names=()):
    def run_stage(st, state, deadline):
        seen.append(st.name)
        return st.name not in fail_names
    return run_stage


def test_a_clean_run_schedules_every_stage_and_exits_zero(runner, monkeypatch):
    monkeypatch.setattr(tilefull, "run_stage", stub(runner))
    assert tilefull.main() == 0
    assert len(runner) == 20


def test_a_failed_critical_stage_stops_the_plan(runner, monkeypatch):
    """The reported defect: the first pool fails and nothing else should run."""
    first = "tilefull-pool-pyrL0-bank_ext2"
    monkeypatch.setattr(tilefull, "run_stage", stub(runner, {first}))
    rc = tilefull.main()
    assert rc == 1, "a run whose first critical stage failed exited zero"
    assert runner == [first], (
        "stages ran after a failed critical prerequisite: " + str(runner[1:]))


def test_no_comparison_is_published_after_a_critical_failure(runner,
                                                            monkeypatch):
    """The consequence that matters: knn_gap must not run on a stale arm."""
    monkeypatch.setattr(tilefull, "run_stage",
                        stub(runner, {"tilefull-stack-pyrL0L1-b340"}))
    tilefull.main()
    assert not [s for s in runner if "knngap" in s]


def test_a_non_critical_failure_does_not_stop_the_plan(runner, monkeypatch):
    """`knn_gap` failing is how this run actually went; the rest still ran."""
    monkeypatch.setattr(tilefull, "run_stage",
                        stub(runner, {"tilefull-knngap"}))
    rc = tilefull.main()
    assert rc == 1, "a failed stage must still be visible in the exit code"
    assert [s for s in runner if "cell8" in s], (
        "a non-critical failure dropped the stages after it")


def test_every_stage_that_writes_an_input_is_marked_critical(runner,
                                                            monkeypatch):
    """The flag is only protection if it is actually set on the producers.

    Guards against the flag being dropped from a stage later: pool, stack,
    project and build_knn all write a path a later stage reads.
    """
    stages = []
    monkeypatch.setattr(tilefull, "run_stage",
                        lambda st, *a: (stages.append(st), True)[1])
    tilefull.main()
    producers = [s for s in stages
                 if any(k in s.name for k in ("pool", "stack", "proj"))
                 or s.name in ("tilefull-knn-pyrL0", "tilefull-knn-pyrL0L1")]
    assert producers
    assert all(s.critical for s in producers), \
        [s.name for s in producers if not s.critical]
