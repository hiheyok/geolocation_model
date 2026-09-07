"""`mutate.py` must not read a broken invocation as a caught defect.

It treated every nonzero pytest exit as "the test failed", so a mistyped path
-- exit 4, "file or directory not found" -- reported the defect caught and
exited 0. So did exit 5, "no tests ran". A tool whose whole job is to catch
tests that cannot fail must not itself pass on a suite that never ran.

Only exit 1 means a test failed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MUTATE = ROOT / "scripts" / "mutate.py"


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
    assert "nothing was measured" in (r.stdout + r.stderr)
    assert "defect was caught" not in r.stdout


def test_no_tests_collected_is_not_a_caught_defect(target, tmp_path):
    """Exit 5, the other false green: a file pytest reads and finds nothing in."""
    src, _ = target
    empty = tmp_path / "test_empty.py"
    empty.write_text("# collected, and holds no tests", encoding="utf-8")
    r = run("--file", str(src), "--test", str(empty), "--old", "1", "--new", "2")
    assert r.returncode != 0
    assert "nothing was measured" in (r.stdout + r.stderr)
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
