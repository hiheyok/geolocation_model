"""`runs/FINAL.md` ranks arms by hit rate, so it must say which are comparable.

Training against the pre-2026-09-04 leaky neighbour cache is worth 2 to 4 pp
(`runs/LEAKTRAIN.md`) -- larger than most gaps this file ranks. Without a
provenance column the ranking is partly a ranking of which cache an arm
happened to train against, presented as a ranking of ideas.

The sharper problem is the headline. Ranking by hit rate alone put
`s10_b55_c6` first at 73.8%: no checkpoint on disk, so no split, no street
file and no cache, and an error file from 2026-09-01 -- three days before the
leak was fixed. The arms nothing can be checked about sort to the top
precisely because they were measured before the fixes, and the previous
version of this file carried a hand-written banner saying so, which
regenerating silently removed.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import digest  # noqa: E402
import knnmeta  # noqa: E402


def test_a_verified_cache_is_marked_clean():
    assert digest.cache_mark(knnmeta.CACHE_OK) == "clean"


def test_a_proven_rebuild_is_marked_stale():
    assert "STALE" in digest.cache_mark(knnmeta.CACHE_REBUILT)


def test_a_newer_cache_is_marked_suspect_not_stale():
    m = digest.cache_mark(knnmeta.CACHE_NEWER)
    assert "suspect" in m and "STALE" not in m


def test_a_missing_checkpoint_is_named_not_a_question_mark():
    """`?` reads as a formatting gap, not as "this cannot be checked"."""
    assert digest.cache_mark(None) == digest.NO_CKPT


def test_an_arm_with_no_checkpoint_is_not_verifiable():
    """The headline defect: 20 such arms, and they sort to the top."""
    assert not digest.verifiable({})
    assert not digest.verifiable({"cache": None})


def test_a_proven_stale_arm_is_not_verifiable():
    assert not digest.verifiable({"cache": knnmeta.CACHE_REBUILT})


def test_a_suspect_arm_is_still_comparable():
    """`suspect` is a warning to read, not grounds to hide the arm.

    Only a proven mismatch or a missing checkpoint disqualifies. Excluding
    everything unproven would drop almost the whole archive.
    """
    assert digest.verifiable({"cache": knnmeta.CACHE_NEWER})
    assert digest.verifiable({"cache": knnmeta.CACHE_PLAUSIBLE})
    assert digest.verifiable({"cache": knnmeta.CACHE_OK})


def test_the_note_is_silent_when_every_arm_is_verified():
    rows = [("a", None, {"cache": knnmeta.CACHE_OK}, 0)]
    assert digest.cache_note(rows) == []


def test_the_note_appears_when_any_arm_is_not_verified():
    rows = [("a", None, {"cache": knnmeta.CACHE_OK}, 0),
            ("b", None, {"cache": knnmeta.CACHE_NEWER}, 0)]
    out = digest.cache_note(rows)
    assert out and "2 to 4 pp" in out[0]


def test_the_note_appears_for_arms_with_no_checkpoint():
    """None is not "verified"; the first version's set test let it through."""
    rows = [("a", None, {}, 0)]
    assert digest.cache_note(rows)


def _row(tag, hit, cache="__absent__"):
    import numpy as np
    e = np.full(100, 1.0 if hit else 1000.0)
    meta = {} if cache == "__absent__" else {"cache": cache}
    return (tag, e, meta, 0)


def test_the_headline_pool_drops_arms_with_no_checkpoint():
    """The reported defect, at the point that decides what the file claims."""
    rows = [_row("no-ckpt", True), _row("good", False, knnmeta.CACHE_OK)]
    ok, hidden = digest.headline_pool(rows)
    assert hidden == 1
    assert [r[0] for r in ok] == ["good"]


def test_the_headline_pool_drops_proven_stale_arms():
    rows = [_row("stale", True, knnmeta.CACHE_REBUILT),
            _row("good", False, knnmeta.CACHE_OK)]
    ok, hidden = digest.headline_pool(rows)
    assert hidden == 1 and [r[0] for r in ok] == ["good"]


def test_the_headline_pool_keeps_everything_verifiable():
    rows = [_row("a", True, knnmeta.CACHE_OK),
            _row("b", False, knnmeta.CACHE_NEWER),
            _row("c", False, knnmeta.CACHE_PLAUSIBLE)]
    ok, hidden = digest.headline_pool(rows)
    assert hidden == 0 and len(ok) == 3


# ------------------------------------------- a mark is not a comparison -----

def test_the_note_does_not_claim_matching_marks_are_comparable():
    """Reported on #62.

    A mark says whether a checkpoint trained against the bytes now sitting at
    the path it recorded. That is a check on one arm. `clean` does not even
    mean leak-free -- an arm that trained on the leaky cache still verifies as
    `clean` if nothing overwrote that file since -- so two arms sharing a mark
    have established nothing about each other. They also need the same split
    mode, the same protocol, and the same side of the 2026-09-04 rebuild.
    """
    out = "\n".join(digest.cache_note([("a", None, {}, 0)]))
    assert "sharing a mark are comparable" not in out
    assert "not a comparison between two" in out
    assert "does not mean leak-free" in out
    assert "same split mode" in out


# ------------------------------------------------ the exclusion sentence ----

BEFORE = digest.LEAK_FIXED - 86400
AFTER = digest.LEAK_FIXED + 86400


def _arm(tag, pct, cache="__absent__", when=AFTER):
    """An arm whose `<25 km` rate is `pct`, with a chosen provenance."""
    import numpy as np
    e = np.concatenate([np.full(pct, 1.0), np.full(100 - pct, 1000.0)])
    meta = {} if cache == "__absent__" else {"cache": cache}
    return (tag, e, meta, when)


def test_nothing_excluded_says_nothing():
    rows = [_arm("a", 90, knnmeta.CACHE_OK)]
    assert digest.exclusion_note(rows, rows) == []


def test_a_losing_excluded_arm_does_not_discredit_the_winner():
    """The reported defect, reproduced exactly: a verified 90% winner and a
    missing-checkpoint 10% loser. The old text fired on any exclusion at all
    and announced that the top score was excluded and predated the leak fix,
    which here is false twice over."""
    win, lose = _arm("win", 90, knnmeta.CACHE_OK), _arm("lose", 10)
    out = "\n".join(digest.exclusion_note([win, lose], [win]))
    assert "1 of 2 arms are excluded" in out
    assert "best raw hit rate" not in out
    assert "leak" not in out
    assert "not a result" not in out


def test_an_excluded_leader_is_named_with_both_numbers():
    """When it IS true, say it -- and say what it beat, so the reader can see
    the size of what is being set aside."""
    top, ok = _arm("top", 90), _arm("ok", 60, knnmeta.CACHE_OK)
    out = "\n".join(digest.exclusion_note([top, ok], [ok]))
    assert "`top` at 90.0%" in out and "`ok` at 60.0%" in out
    assert "not a result" in out


def test_the_leak_clause_needs_the_dates_to_support_it():
    """Measured after the rebuild, the arm is still unverifiable -- but not
    for that reason, and saying so would be a second false claim."""
    late = _arm("top", 90, when=AFTER)
    ok = _arm("ok", 60, knnmeta.CACHE_OK)
    assert "leak" not in "\n".join(digest.exclusion_note([late, ok], [ok]))
    early = _arm("top", 90, when=BEFORE)
    assert "leak" in "\n".join(digest.exclusion_note([early, ok], [ok]))


def test_the_reason_counts_come_from_the_rows():
    rows = [_arm("a", 10), _arm("b", 10),
            _arm("c", 10, knnmeta.CACHE_REBUILT), _arm("d", 90,
                                                       knnmeta.CACHE_OK)]
    out = "\n".join(digest.exclusion_note(rows, [rows[3]]))
    assert "3 of 4 arms" in out
    assert "2 with no checkpoint" in out
    assert "1 proven by digest" in out


def test_a_reason_with_no_rows_behind_it_is_not_printed():
    rows = [_arm("a", 10, knnmeta.CACHE_REBUILT), _arm("b", 90,
                                                       knnmeta.CACHE_OK)]
    out = "\n".join(digest.exclusion_note(rows, [rows[1]]))
    assert "no checkpoint" not in out
    assert "1 proven by digest" in out


def test_every_arm_excluded_still_reports():
    """No verifiable arm to compare against, so the comparison clause has to
    drop out rather than crash or invent one."""
    rows = [_arm("a", 90), _arm("b", 10)]
    out = "\n".join(digest.exclusion_note(rows, []))
    assert "2 of 2 arms" in out
    assert "`a` at 90.0%" in out
    assert "above the best verifiable" not in out


# ------------------------------- what the report does NOT establish ---------

def test_the_note_does_not_offer_paired_intervals_as_proof_of_comparability():
    """Rechecked on #62.

    The first correction traded one false authority for another: it sent the
    reader to the paired intervals "which measure it rather than assume it".
    They do not. `BOOTSTRAP_ship.md` is in this file, was separated at 2.06 to
    4.00 pp, and is retracted -- the arms differed in the cache they trained
    against, so the interval measured that. Tightness is not provenance.
    """
    out = "\n".join(digest.cache_note([("a", None, {}, 0)]))
    assert "which measure it rather than assume it" not in out
    assert "does not establish it either" in out
    assert "BOOTSTRAP_ship.md" in out
    assert "establish independently" in out


# ------------------------------------- a missing cache proves nothing -------

def test_a_missing_cache_is_not_reported_as_a_proven_rebuild():
    """Reported on #62. `verifiable` rejects CACHE_REBUILT and CACHE_MISSING
    alike, and the note counted "everything with a checkpoint" as the former.
    The table said `gone` in the same breath as the note claimed proof by
    digest."""
    gone = _arm("gone", 10, knnmeta.CACHE_MISSING)
    ok = _arm("ok", 90, knnmeta.CACHE_OK)
    out = "\n".join(digest.exclusion_note([gone, ok], [ok]))
    assert "proven by digest" not in out
    assert "no longer on disk" in out
    assert "nothing about them can be checked" in out


def test_the_three_reasons_are_counted_separately():
    rows = [_arm("a", 10), _arm("b", 10, knnmeta.CACHE_REBUILT),
            _arm("c", 10, knnmeta.CACHE_MISSING),
            _arm("d", 90, knnmeta.CACHE_OK)]
    out = "\n".join(digest.exclusion_note(rows, [rows[3]]))
    assert "1 with no checkpoint" in out
    assert "1 proven by digest" in out
    assert "1 whose neighbour cache is no longer on disk" in out
    assert "does not name" not in out


def test_a_new_exclusion_reason_is_noticed_rather_than_absorbed():
    """The shape of the defect, not just this instance. Subtracting one known
    category from the total is how an unknown one gets misdescribed; an
    unlisted state has to surface as unlisted."""
    rows = [_arm("a", 10, "some future state"), _arm("b", 90,
                                                     knnmeta.CACHE_OK)]
    out = "\n".join(digest.exclusion_note(rows, [rows[1]]))
    assert "1 for a reason this note does not name" in out
    assert "proven by digest" not in out
