"""`mutate.py` must not read a broken invocation as a caught defect.

It treated every nonzero pytest exit as "the test failed", so a mistyped path
-- exit 4, "file or directory not found" -- reported the defect caught and
exited 0. So did exit 5, "no tests ran". A tool whose whole job is to catch
tests that cannot fail must not itself pass on a suite that never ran.

Only exit 1 means a test failed. And two more ways a run can lie about what
it measured, both reported and both reproduced:

  * an unrelated test that was ALREADY failing makes any mutation look caught,
    even one the selected tests never import;
  * a `.pyc` is reused when the source's size and its mtime truncated to
    SECONDS both match, so a same-length edit -- `VALUE = 1` to `VALUE = 2` --
    within the same second imports the old bytecode and the test passes, and
    the tool condemns a test that works.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MUTATE = ROOT / "scripts" / "mutate.py"

# built without escapes: a heredoc has eaten them here before
FAILING = "def test_broken():" + chr(10) + "    assert False" + chr(10)


@pytest.fixture
def target(tmp_path):
    """A module with a flag, and a test that reads it."""
    src = tmp_path / "m.py"
    src.write_text("VALUE = 1\n", encoding="utf-8")
    t = tmp_path / "test_m.py"
    t.write_text(
        "import sys\n"
        "sys.path.insert(0, r'{}')\n".format(tmp_path.as_posix())
        + "import m\n\n\ndef test_value():\n    assert m.VALUE == 1\n",
        encoding="utf-8")
    return src, t


def run(*args):
    return subprocess.run([sys.executable, str(MUTATE), *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_a_caught_defect_succeeds(target):
    src, t = target
    r = run("--file", str(src), "--test", str(t), "--old", "1", "--new", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "defect was caught" in r.stdout


def test_a_missing_test_target_is_not_a_caught_defect(target):
    """The report: exit 4 read as success."""
    src, _ = target
    r = run("--file", str(src), "--test", str(ROOT / "tests" / "nope.py"),
            "--old", "1", "--new", "2")
    assert r.returncode != 0
    # Caught at the baseline now, which is earlier and names the same code.
    assert "exited 4" in (r.stdout + r.stderr)
    assert "defect was caught" not in r.stdout


def test_no_tests_collected_is_not_a_caught_defect(target, tmp_path):
    """Exit 5, the other false green: a file pytest reads and finds nothing in."""
    src, _ = target
    empty = tmp_path / "test_empty.py"
    empty.write_text("# collected, and holds no tests", encoding="utf-8")
    r = run("--file", str(src), "--test", str(empty), "--old", "1", "--new", "2")
    assert r.returncode != 0
    assert "exited 5" in (r.stdout + r.stderr)
    assert "defect was caught" not in r.stdout


def test_a_test_that_does_not_notice_is_reported(target):
    """The tool's actual purpose: a live suite that stays green."""
    src, t = target
    src.write_text("VALUE = 1\nUNUSED = 3\n", encoding="utf-8")
    r = run("--file", str(src), "--test", str(t), "--old", "3", "--new", "4")
    assert r.returncode != 0
    assert "THE TEST DID NOT FAIL" in (r.stdout + r.stderr)


def test_the_file_is_restored_whatever_happened(target):
    src, _ = target
    before = src.read_bytes()
    run("--file", str(src), "--test", str(ROOT / "tests" / "nope.py"),
        "--old", "1", "--new", "2")
    assert src.read_bytes() == before


def test_a_baseline_failure_is_not_a_caught_defect(target, tmp_path):
    """Reported: an already-failing test that never imports the mutated module
    reported the defect caught."""
    src, _ = target
    unrelated = tmp_path / "test_unrelated.py"
    unrelated.write_text(FAILING, encoding="utf-8")
    r = run("--file", str(src), "--test", str(unrelated),
            "--old", "1", "--new", "2")
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "do not pass before the mutation" in out
    assert "defect was caught" not in r.stdout


def test_the_file_is_untouched_when_the_baseline_fails(target, tmp_path):
    """The baseline runs first, so a bad selection never edits your source."""
    src, _ = target
    before = src.read_bytes()
    unrelated = tmp_path / "test_broken.py"
    unrelated.write_text(FAILING, encoding="utf-8")
    run("--file", str(src), "--test", str(unrelated), "--old", "1", "--new", "2")
    assert src.read_bytes() == before


def test_a_private_cache_does_not_reuse_stale_bytecode(tmp_path):
    """A `.pyc` is reused when the source's SIZE and its mtime TRUNCATED TO
    SECONDS both match. `VALUE = 1` -> `VALUE = 2` is the same length and the
    edit lands within the same second, so both match and the OLD code is
    imported -- the test passes and the tool condemns a test that works.

    Driven through `pytest_exit` rather than the whole script, because the
    hazard needs the cache warmed and the mtime pinned between two runs, and
    an end-to-end version cannot reach in between. An earlier version of this
    test did go end to end, and `mutate.py` reported it did not fail when the
    isolation was removed -- it was not testing what it was named after.
    """
    import os
    import time

    sys.path.insert(0, str(ROOT / "scripts"))
    from mutate import pytest_exit

    mod = tmp_path / "m.py"
    t = tmp_path / "test_m.py"
    mod.write_text("VALUE = 1" + chr(10), encoding="utf-8")
    t.write_text(
        "import sys" + chr(10)
        + "sys.path.insert(0, r'" + tmp_path.as_posix() + "')" + chr(10)
        + "import m" + chr(10) + chr(10) + chr(10)
        + "def test_v():" + chr(10) + "    assert m.VALUE == 1" + chr(10),
        encoding="utf-8")

    shared = tmp_path / "cache"
    assert pytest_exit([str(t)], shared) == 0, "the module should start passing"

    stamp = time.time()
    os.utime(mod, (stamp, stamp))
    assert pytest_exit([str(t)], shared) == 0
    mod.write_text("VALUE = 2" + chr(10), encoding="utf-8")   # same length
    os.utime(mod, (stamp, stamp))                             # same second

    # Reusing the warmed cache is the hazard itself: the edit is invisible.
    assert pytest_exit([str(t)], shared) == 0, (
        "this fixture no longer demonstrates the stale-bytecode hazard, so "
        "the assertion below proves nothing")
    # A private cache has nothing stale to find.
    assert pytest_exit([str(t)], tmp_path / "fresh") == 1
