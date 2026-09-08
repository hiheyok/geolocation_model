"""A window that closes early is not a completed measurement.

Both runners stop when the deadline passes. Neither recorded that anything was
left unrun, so the exit code was 0: all six training rungs could succeed, the
comparison never happen, and the process still tell a caller, a chain script
and a monitor that the result was ready.

It is the same shape as the `Stage.critical` defect these runners already fix
-- an outcome that reads as success because nothing wrote down that it was
not. Found by review on #59; the identical loop is in `tilefull.py`, which was
already merged, so both are fixed together under the "same lines, two files"
exception in AGENTS.md §1.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import overnight as O  # noqa: E402
import tilefull  # noqa: E402
import tilefull_ladder  # noqa: E402


class Sampler:
    def __init__(self):
        self.stop_flag = type("F", (), {"set": lambda self: None})()

    def start(self):
        pass


class Clock:
    """`O.now()` that jumps past the deadline on a chosen call.

    `main` calls it once to set the deadline, then once per loop iteration, so
    call `n` guards stage `n - 2`.
    """

    def __init__(self, jump_at):
        self.n, self.jump_at = 0, jump_at

    def __call__(self):
        self.n += 1
        return 10 ** 9 if self.n >= self.jump_at else 0.0


@pytest.fixture(params=[tilefull, tilefull_ladder],
                ids=["tilefull", "tilefull_ladder"])
def mod(request, tmp_path, monkeypatch):
    """Either runner, with disk, threads and stages stubbed."""
    m = request.param
    for name in ("RUNS", "LOGS", "MARKS"):
        monkeypatch.setattr(O, name, tmp_path / name)
    monkeypatch.setattr(O, "Sampler", Sampler)
    monkeypatch.setattr(m, "load_state", lambda: {"done": {}, "failed": {}})
    monkeypatch.setattr(m, "log", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [m.__name__ + ".py", "10"])
    return m


def seen_stages(m, monkeypatch):
    out = []
    monkeypatch.setattr(m, "run_stage",
                        lambda st, *a: (out.append(st.name), True)[1])
    return out


def test_a_complete_run_exits_zero(mod, monkeypatch):
    """So the deadline tests below are not passing vacuously."""
    seen_stages(mod, monkeypatch)
    assert mod.main() == 0


def test_a_deadline_that_stops_the_last_stage_is_not_success(mod, monkeypatch):
    """The reported defect: every stage but the comparison ran, exit was 0."""
    seen = seen_stages(mod, monkeypatch)
    mod.main()                       # real clock: the whole plan runs
    total = len(seen)
    assert total > 1
    # Re-run with a clock that expires one stage before the end. `main` calls
    # `now` once for the deadline and once per iteration, so call `total + 1`
    # guards the last stage.
    seen.clear()
    monkeypatch.setattr(O, "now", Clock(jump_at=total + 1))
    rc = mod.main()
    assert len(seen) == total - 1, seen
    assert rc == 1, "a run that skipped its final stage exited zero"


def test_a_deadline_before_anything_runs_is_not_success(mod, monkeypatch):
    seen = seen_stages(mod, monkeypatch)
    monkeypatch.setattr(O, "now", Clock(jump_at=2))
    assert mod.main() == 1
    assert seen == []


def test_the_comparison_is_the_stage_that_gets_cut(mod, monkeypatch):
    """Worth naming: the plan puts the measurement last, so it goes first."""
    seen = seen_stages(mod, monkeypatch)
    mod.main()
    assert "knngap" in seen[-1] or "boot" in seen[-1], seen[-1]
