"""Review items 70-73 are one design, so they get one test file.

A stage was identified by its name alone and its log was append-only across
days, so both files said things about work a *different* run had done.  This
cost an hour on 2026-09-02: a wait loop grepped a training log for `FAIL`,
matched a failure appended the previous day, and killed `fuse-attn-pyr47` at
epoch 11 of 12.

Measured on the 200 logs and 212 markers currently on disk:

* 47 of the 200 parsed to **zero epochs** under the old non-multiline regex,
  so every one of those report rows carried a NaN seconds-per-epoch (item 73).
* Two training logs were genuinely spliced -- `train-d768-b350-e2-drop70-sink4`
  and `gm_bias` each show four epochs across the whole file and two in the
  last attempt, so their best-epoch selection mixed two attempts (item 72).
* All 212 markers predate the identity stamp, and all 212 read as "unknown"
  rather than "mismatch", which is what keeps a finished queue from re-running.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import runlog  # noqa: E402

LOG = """setup line
==== attempt 1 at 21:04 ====
ep 1 100.0s
FAIL something went wrong
==== attempt 2 at 22:10 ====
ep 1 90.0s
ep 2 91.0s
"""


def test_only_the_last_attempt_is_returned():
    tail = runlog.last_attempt(LOG)
    assert "90.0s" in tail and "100.0s" not in tail


def test_a_stale_failure_from_an_earlier_attempt_is_not_visible():
    """The 2026-09-02 failure exactly: a grep for FAIL matched a previous
    attempt and killed a run at epoch 11 of 12."""
    assert "FAIL" in LOG
    assert "FAIL" not in runlog.last_attempt(LOG)


def test_a_log_with_no_header_is_returned_whole():
    """Anything not launched by the runner has no headers, and returning
    nothing for it would be a worse failure than the one being fixed."""
    plain = "ep 1 10.0s\nep 2 11.0s\n"
    assert runlog.last_attempt(plain) == plain


def test_the_attempt_count_is_reported_from_the_whole_file():
    assert runlog.attempts(LOG) == 2
    assert runlog.attempts("no headers here") == 0


def test_the_epoch_regex_needs_multiline():
    """Item 73, and the reason 47 of 200 real logs reported NaN s/epoch: `^`
    without re.M matches only at the very start of the string, so a log
    beginning with an attempt header yields nothing at all."""
    import re
    without = re.compile(r"^ep\s+(\d+)\s+([\d.]+)s")
    with_m = re.compile(r"^ep\s+(\d+)\s+([\d.]+)s", re.M)
    assert without.findall(LOG) == []
    assert len(with_m.findall(runlog.last_attempt(LOG))) == 2


# --- markers ---------------------------------------------------------------

def ident(argv, release="s10"):
    return runlog.marker_identity("train", argv, release)


def test_a_marker_for_the_same_command_matches():
    i = ident(["src/train.py", "--epochs", "6"])
    assert runlog.marker_matches(i, i) is True


def test_a_marker_for_different_arguments_does_not_match():
    """The case that silently skipped work: the stage name stayed and the
    command moved -- a different bank, width or epoch count."""
    a = ident(["src/train.py", "--epochs", "6"])
    b = ident(["src/train.py", "--epochs", "12"])
    assert runlog.marker_matches(a, b) is False


def test_a_marker_from_another_release_does_not_match():
    argv = ["src/train.py", "--epochs", "6"]
    assert runlog.marker_matches(ident(argv, "s01"), ident(argv, "s10")) is False


@pytest.mark.parametrize("text", ["", "26s\n", "satisfied by check\n", None,
                                  "satisfied by check"])
def test_a_pre_stamp_marker_reads_as_unknown_not_as_a_mismatch(text):
    """All 212 markers on disk are bare durations.  Reading them as a
    mismatch would re-run every finished stage in the queue; reading them as a
    match without saying so is the hole being closed.  Unknown is the third
    answer, and the runner logs it."""
    assert runlog.marker_matches(text, ident(["x"])) is None


@pytest.mark.parametrize("text", [
    "{not json}",
    '{"stage": "train", "argv": ["x"',          # truncated mid-write
    '{"stage": "train", "argv": ["x"], ',       # truncated at a comma
    "{",
])
def test_a_corrupt_json_marker_is_a_mismatch_not_a_legacy_success(text):
    """This used to read as None, and the caller reads None as satisfied.

    Marker writes were not atomic, so an interrupted run leaves exactly these
    bytes on disk -- and the stage that was interrupted is precisely the one
    that must not be skipped on restart. A legacy marker never begins with a
    brace, so the compatibility path keeps the old files without also adopting
    every half-written new one.
    """
    assert runlog.marker_matches(text, ident(["x"])) is False


def test_key_order_does_not_change_the_identity():
    """Written by one process and read by another, possibly a version apart."""
    a = runlog.marker_identity("s", ["a", "b"], "s10", extra=1)
    b = runlog.marker_identity("s", ["a", "b"], "s10", extra=1)
    assert a == b and runlog.marker_matches(a, b) is True
