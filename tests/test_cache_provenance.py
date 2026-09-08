"""A checkpoint must be bound to its cache's CONTENT, not to its filename.

Every `knn_*` cache was rebuilt on 2026-09-04 04:05 after the same-sequence
bank leak was found, and
`knn_pca768_bank70_sequence_k32_bank_ext70.npz` went on meaning something
different under the same name. `d768-b350-e6-drop70` and `wd29-fix-e4` record
that filename and trained on the leaky bytes; `pyrL0*` records it and trained
on the clean ones. Comparing across that boundary measured the cache, not the
arms -- the same street file scores 57.5% leak-trained and 54.6% clean,
separated (`runs/LEAKTRAIN.md`). AGENTS.md §7 is written about exactly this.

Two paths, because the archive predates the fix: an exact digest going
forward, and an mtime comparison for the ~100 checkpoints without one. Only
the digest can prove anything. A newer mtime means "cannot be ruled out" -- a
copy, a restore or a `touch` moves it without changing a byte -- so it reports
CACHE_NEWER, not CACHE_REBUILT.

The case that matters most is the last one here: *unverifiable must not read
as fine*. That is what `check_merged` learned about shallow clones and what
`gain_growth` learned about split hashes, in the same week.
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import knnmeta  # noqa: E402
import safeio  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A neighbour cache on disk, and a checkpoint dir beside it."""
    monkeypatch.setattr(knnmeta.config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(knnmeta.config, "CHECKPOINTS", tmp_path)
    p = tmp_path / "knn_x.npz"
    p.write_bytes(b"the clean cache")
    return tmp_path, p


def test_a_matching_digest_verifies(cache):
    tmp, p = cache
    ck = {"knn_file": "knn_x.npz", "knn_digest": safeio.content_digest(p)}
    assert knnmeta.cache_provenance(ck)[0] == knnmeta.CACHE_OK


def test_a_rebuilt_cache_is_caught_by_digest(cache):
    """The exact check: same name, same size, different bytes."""
    tmp, p = cache
    ck = {"knn_file": "knn_x.npz", "knn_digest": safeio.content_digest(p)}
    p.write_bytes(b"the leaky cache")          # same length, new content
    state, why = knnmeta.cache_provenance(ck)
    assert state == knnmeta.CACHE_REBUILT
    assert "digests" in why


def test_a_newer_cache_without_a_digest_is_suspect_not_proven(cache):
    """The retroactive check, for checkpoints that predate `knn_digest`.

    This is how the leak was found by hand. But a newer mtime is NOT proof:
    a copy, a restore or a `touch` moves it without changing a byte. The
    first version returned CACHE_REBUILT here, so a report generated after an
    innocent copy would have asserted the arms trained on different data.
    """
    tmp, p = cache
    ck = {"knn_file": "knn_x.npz", "saved_at": time.time() - 3600}
    state, why = knnmeta.cache_provenance(ck)
    assert state == knnmeta.CACHE_NEWER
    assert state != knnmeta.CACHE_REBUILT, "mtime must not claim proof"
    assert "written after" in why


def test_only_a_digest_can_prove_a_mismatch(cache):
    """CACHE_REBUILT is reserved for the check that actually establishes it."""
    tmp, p = cache
    ck = {"knn_file": "knn_x.npz", "knn_digest": safeio.content_digest(p),
          "saved_at": time.time() - 3600}
    p.write_bytes(b"the leaky cache")
    assert knnmeta.cache_provenance(ck)[0] == knnmeta.CACHE_REBUILT


def test_an_older_cache_without_a_digest_is_plausible_not_verified(cache):
    """It cannot see an in-place rewrite, so it must not claim to."""
    tmp, p = cache
    ck = {"knn_file": "knn_x.npz", "saved_at": time.time() + 3600}
    state, _ = knnmeta.cache_provenance(ck)
    assert state == knnmeta.CACHE_PLAUSIBLE
    assert state != knnmeta.CACHE_OK


def test_no_timestamp_and_no_digest_is_unknown_not_fine(cache):
    """The whole point: unverifiable must not read as verified."""
    tmp, p = cache
    state, _ = knnmeta.cache_provenance({"knn_file": "knn_x.npz"})
    assert state == knnmeta.CACHE_UNKNOWN
    assert state != knnmeta.CACHE_OK


def test_a_missing_cache_is_reported(cache):
    tmp, _ = cache
    ck = {"knn_file": "gone.npz", "knn_digest": "abc"}
    assert knnmeta.cache_provenance(ck)[0] == knnmeta.CACHE_MISSING


def test_an_arm_without_retrieval_is_not_flagged(cache):
    """No cache is not a suspect cache."""
    assert knnmeta.cache_provenance({})[0] == knnmeta.CACHE_OK


def test_the_checkpoint_file_supplies_the_timestamp(cache):
    """Checkpoints written before `saved_at` still get the mtime check."""
    tmp, p = cache
    ckpt = tmp / "arm.pt"
    ckpt.write_bytes(b"x")
    import os
    old = time.time() - 7200
    os.utime(ckpt, (old, old))
    state, why = knnmeta.cache_provenance({"knn_file": "knn_x.npz"}, "arm")
    assert state == knnmeta.CACHE_NEWER, why


# ------------------------------------------- the report, not the check ------

sys.path.insert(0, str(ROOT / "scripts"))


def _lines(states):
    import bootstrap
    tags = list(states)
    return bootstrap.cache_lines(tags, {t: (s, t + ".npz")
                                        for t, s in states.items()})


def test_all_verified_says_nothing():
    assert _lines({"a": knnmeta.CACHE_OK, "b": knnmeta.CACHE_OK}) == []


def test_uniformly_unverified_arms_are_still_reported():
    """The reported defect: identical statuses suppressed the whole table.

    Two arms that are equally unverifiable are not two safe arms. Requiring
    them to DISAGREE was the weight-decay idiom applied where it does not
    hold -- there, sameness IS safety; here it is not.
    """
    out = _lines({"a": knnmeta.CACHE_NEWER, "b": knnmeta.CACHE_NEWER})
    assert out, "two equally unverifiable arms produced no warning"
    assert "cannot be shown" in out[0]


def test_uniformly_unknown_arms_are_still_reported():
    out = _lines({"a": knnmeta.CACHE_UNKNOWN, "b": knnmeta.CACHE_UNKNOWN})
    assert out and "2 of these 2 arms" in out[0]


def test_a_proven_mismatch_is_named_as_proven():
    out = _lines({"a": knnmeta.CACHE_REBUILT, "b": knnmeta.CACHE_OK})
    assert "PROVEN by digest" in out[0] and "`a`" in out[0]


def test_an_unproven_mismatch_is_not_called_proven():
    """CACHE_NEWER must not be reported as established."""
    out = _lines({"a": knnmeta.CACHE_NEWER, "b": knnmeta.CACHE_OK})
    assert out and "PROVEN" not in out[0]
