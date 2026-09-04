"""The runner's log and marker conventions, in one place.

Four review items (70-73) are one design: a stage is identified by its name
alone, and its log is append-only across days.  Both make a file say something
about work that a *different* run did.

This cost an hour on 2026-09-02.  A wait loop grepped a training log for
`FAIL`, matched a failure appended the previous day, and killed
`fuse-attn-pyr47` at epoch 11 of 12.  Nothing was wrong with the grep; the file
simply held two runs and did not distinguish them.

Two conventions follow, and both are about scoping a question to one run:

* **Read the last attempt, never the file.**  `last_attempt` slices off
  everything before the final header.  Every parser here wants the current
  attempt -- a per-epoch curve mixing two attempts is not a curve, and the
  calibration parser took the *first* match in the file, which after a retry
  is the timing of the run that failed.
* **A marker records what it marked.**  A stage's marker is a filename, so a
  changed command, a changed release or a rebuilt input all leave it looking
  finished.  Writing the identity into the marker and comparing on read is
  what makes it a record rather than a flag.
"""

import json
import re

HEADER = "==== attempt"
ATTEMPT = re.compile(r"^==== attempt \d+ at .*$", re.M)


def last_attempt(text):
    """Everything after the final attempt header, or all of it if there is none.

    A log with no header is a single run and needs no slicing -- that is the
    common case for anything not launched by the runner, and returning nothing
    for it would be a worse failure than the one this fixes.
    """
    hits = list(ATTEMPT.finditer(text))
    return text[hits[-1].end():] if hits else text


def attempts(text):
    """How many attempt headers the log holds. 0 means it was never retried."""
    return len(ATTEMPT.findall(text))


def marker_identity(name, argv, release=None, **extra):
    """What a completion marker is a completion *of*.

    Keyed on the command rather than the stage name: renaming is handled
    separately by `legacy_check`, but re-pointing a stage at different
    arguments -- a different bank, a different width, a different number of
    epochs -- is the case that silently reused a finished marker and skipped
    the work.
    """
    rec = {"stage": name, "argv": list(argv), "release": release}
    rec.update(extra)
    return json.dumps(rec, sort_keys=True)


def marker_matches(text, identity):
    """Whether a marker's contents describe this stage: True, False or None.

    None means the marker predates the identity convention, which every marker
    already on disk does. That is a third answer on purpose: treating those as
    a mismatch would re-run every finished stage in the queue, and treating
    them as a match without saying so is the hole being closed.
    """
    text = (text or "").strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text) == json.loads(identity)
    except ValueError:
        return None
