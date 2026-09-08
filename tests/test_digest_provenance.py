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
