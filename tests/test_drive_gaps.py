"""The per-drive minimum gap must hold against every frame kept, not the last.

`--per-sequence` caps how many frames one drive contributes, but a cap does
not make them distinct views: at the old default the three frames arrived a
median 2.0 s apart. `--min-gap-s` is what makes them distinct, and it only
does so if a candidate is compared with all of them.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kartaview_harvest import DriveGaps, shot_seconds  # noqa: E402


def take(gaps, sid, times):
    """The accept/reject decision for each candidate, in order."""
    out = []
    for t in times:
        ok = gaps.allows(sid, t)
        out.append(ok)
        if ok:
            gaps.record(sid, t)
    return out


def test_a_candidate_is_compared_with_every_frame_not_the_latest():
    """0 s, 60 s, 1 s at a 30 s gap: the third was measured against 60 s alone
    and accepted, one second from the first."""
    assert take(DriveGaps(30.0), "d", [0, 60, 1]) == [True, True, False]


def test_the_gap_still_admits_a_genuinely_separated_frame():
    assert take(DriveGaps(30.0), "d", [0, 60, 120]) == [True, True, True]


def test_drives_are_independent():
    g = DriveGaps(30.0)
    assert take(g, "a", [0, 1]) == [True, False]
    assert take(g, "b", [0, 1]) == [True, False]


def test_a_resumed_harvest_honours_frames_already_on_disk():
    """Seeding kept `max()` over the existing records, so a new frame near an
    earlier one -- but far from the newest -- was accepted."""
    g = DriveGaps(30.0)
    g.seed([{"sequence_id": "d", "shot_date": "2017-10-31 15:53:00.000"},
            {"sequence_id": "d", "shot_date": "2017-10-31 15:54:00.000"}])
    base = shot_seconds("2017-10-31 15:53:00.000")
    assert g.allows("d", base + 1) is False        # 1 s from the FIRST
    assert g.allows("d", base + 120) is True


def test_an_unknown_timestamp_cannot_satisfy_the_gap():
    """A missing shot_date must not silently pass; the cap decides alone."""
    g = DriveGaps(30.0)
    assert g.allows("d", None) is True
    g.record("d", None)
    assert g.times.get("d", []) == []
    assert take(DriveGaps(30.0), "d", [0, 1]) == [True, False]


def test_a_copy_does_not_commit_to_the_original():
    """Candidates are provisional: only downloads that succeed are kept."""
    g = DriveGaps(30.0)
    g.record("d", 0)
    prov = g.copy()
    prov.record("d", 60)
    assert g.allows("d", 60) is True
    assert prov.allows("d", 60) is False
