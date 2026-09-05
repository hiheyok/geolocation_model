"""The two provenance primitives the suite never exercised.

A follow-up review pointed out that `split_hash` and the error-cache resolver
had no tests, and both turned out to be wrong. `split_hash` hashed only the
first character of each label, so "train" and "test" were the same character
and swapping every train row with every test row left the digest unchanged --
which invalidated the "these arms were measured over the same rows" guarantee
built on it. The resolver returned the newest file for a tag, which after
retraining is the *previous* model's errors.

Both are the same failure: a guarantee stated in a docstring and never checked.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import splits as sp  # noqa: E402


LABELS = np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 2)


def test_hash_distinguishes_train_from_test():
    """The defect itself: a full train/test swap must not be invisible."""
    swapped = np.where(LABELS == "train", "test",
                       np.where(LABELS == "test", "train", LABELS))
    assert sp.split_hash("sequence", LABELS) != sp.split_hash("sequence", swapped)


def test_hash_notices_a_single_row_moving():
    one = LABELS.copy()
    one[0] = "test"
    assert sp.split_hash("sequence", LABELS) != sp.split_hash("sequence", one)


def test_hash_notices_a_val_row_moving():
    one = LABELS.copy()
    one[6] = "train"
    assert sp.split_hash("sequence", LABELS) != sp.split_hash("sequence", one)


def test_hash_is_stable_for_an_unchanged_assignment():
    assert sp.split_hash("sequence", LABELS) == sp.split_hash("sequence", LABELS.copy())


def test_hash_depends_on_the_mode():
    assert sp.split_hash("sequence", LABELS) != sp.split_hash("cell8", LABELS)


def test_the_legacy_hash_is_kept_and_is_the_weak_one():
    """Kept deliberately, so 122 existing checkpoints still load.

    Asserting its weakness is the point: it documents why the fallback in
    check_split prints a caveat rather than treating a legacy match as proof.
    """
    swapped = np.where(LABELS == "train", "test",
                       np.where(LABELS == "test", "train", LABELS))
    assert sp.split_hash_legacy("sequence", LABELS) == \
        sp.split_hash_legacy("sequence", swapped)


def test_resolver_prefers_the_current_checkpoints_stamp(tmp_path, monkeypatch):
    """After retraining, the newest file for a tag is the OLD model's errors."""
    import config
    import bootstrap as B

    ckdir, cache = tmp_path / "ck", tmp_path / "errs"
    ckdir.mkdir(); cache.mkdir()
    monkeypatch.setattr(config, "CHECKPOINTS", ckdir)
    monkeypatch.setattr(B, "CACHE", cache)
    monkeypatch.setattr(B.config, "CHECKPOINTS", ckdir)

    (ckdir / "arm.pt").write_bytes(b"x" * 16)
    stamp = B.ckpt_stamp("arm")
    tail = "test_5000r_k2_d3.npy"

    stale = cache / "arm_deadbeef0000_{}".format(tail)
    np.save(stale, np.zeros(3))
    current = cache / "arm_{}_{}".format(stamp, tail)
    np.save(current, np.ones(3))
    # make the stale one newer, which is the situation after a retrain
    import os, time
    t = time.time() + 60
    os.utime(stale, (t, t))

    assert B.find_err_cache("arm").name == current.name


def test_resolver_refuses_a_cache_older_than_the_checkpoint(tmp_path, monkeypatch):
    import os
    import time
    import config
    import bootstrap as B

    ckdir, cache = tmp_path / "ck", tmp_path / "errs"
    ckdir.mkdir(); cache.mkdir()
    monkeypatch.setattr(config, "CHECKPOINTS", ckdir)
    monkeypatch.setattr(B, "CACHE", cache)
    monkeypatch.setattr(B.config, "CHECKPOINTS", ckdir)

    legacy = cache / "arm_test_5000r_k2_d3.npy"
    np.save(legacy, np.zeros(3))
    (ckdir / "arm.pt").write_bytes(b"x" * 16)     # written after the cache
    t = time.time() + 60
    os.utime(ckdir / "arm.pt", (t, t))

    assert B.find_err_cache("arm") is None


def test_resolver_returns_none_rather_than_a_path_that_cannot_be_used():
    """marathon called .exists() on the result; None has to be the contract."""
    import bootstrap as B
    assert B.find_err_cache("definitely-not-an-arm") is None


@pytest.mark.parametrize("name,tag", [
    ("d768-b350-e6_7a7b42da0864_test_5000r_k2_d3.npy", "d768-b350-e6"),
    ("d768-b350-e6_test_5000r_k2_d3.npy", "d768-b350-e6"),
    ("s10_cell8_bank25_lr1e4_c_test_5000r_k2_d3.npy", "s10_cell8_bank25_lr1e4_c"),
    ("arm_val_1000r_k1_d2.npy", "arm"),
])
def test_tag_recovery_survives_underscores_and_stamps(name, tag):
    import bootstrap as B
    assert B.tag_of_err_cache(name) == tag


# --- the release stamp ------------------------------------------------------
#
# `check_split` has enforced this for the internal evaluators since the day a
# wrong-release pairing was found, but the external ones -- eval_highres and
# multiquery -- never call it: they score against a foreign image set, so there
# is no split to check.  They still index s10 row-addressed artifacts, and they
# force OSV_RELEASE rather than reading it.  Moving the check into `load_model`
# covers every consumer, so these pin the three cases it distinguishes.

def test_a_matching_release_passes():
    import evaluate
    evaluate.check_release({"release": config_release()}, "arm")


def test_a_foreign_release_is_refused():
    import evaluate
    with pytest.raises(SystemExit, match="release mismatch"):
        evaluate.check_release({"release": "s99"}, "arm")


def test_a_checkpoint_predating_the_stamp_warns_but_loads(capsys):
    """44 of the 122 checkpoints on disk carry no release field.

    Inventing one for them would refuse real arms on a guess, so the missing
    case is deliberately weaker than the mismatched case.
    """
    import evaluate
    evaluate.check_release({}, "old-arm")
    assert "predates the release stamp" in capsys.readouterr().out


def config_release():
    import config
    return config.RELEASE


# --- one three-way comparison, used by all three artifact families ----------
#
# Checkpoints, kNN caches and bootstrap provenance each store a split digest,
# and each needs the same answer.  Getting it separately in three places is how
# the kNN cache came to refuse the entire shipping retrieval path while
# check_split accepted the very same checkpoints.

LABELS = np.array(["train"] * 6 + ["val"] * 2 + ["test"] * 2, dtype=object)


def test_the_current_digest_reads_as_exact():
    assert sp.hash_matches("sequence", LABELS,
                           sp.split_hash("sequence", LABELS)) == "exact"


def test_the_pre_fix_digest_reads_as_legacy_not_as_a_pass():
    """A real answer, not a pass: it means the assignment agrees as far as a
    digest that cannot see a train/test swap can tell."""
    assert sp.hash_matches("sequence", LABELS,
                           sp.split_hash_legacy("sequence", LABELS)) == "legacy"


def test_a_train_test_swap_still_reads_as_legacy_which_is_the_known_limit():
    """The weak digest cannot distinguish this, and pretending otherwise would
    be worse than recording the limit."""
    swapped = np.where(LABELS == "train", "test",
                       np.where(LABELS == "test", "train", LABELS))
    assert sp.hash_matches("sequence", swapped,
                           sp.split_hash_legacy("sequence", LABELS)) == "legacy"
    assert sp.hash_matches("sequence", swapped,
                           sp.split_hash("sequence", LABELS)) is None


def test_an_unrelated_digest_matches_nothing():
    assert sp.hash_matches("sequence", LABELS, "000000000000") is None


def test_a_missing_digest_matches_nothing():
    assert sp.hash_matches("sequence", LABELS, None) is None


# --- the error cache must depend on its retrieval inputs, not just the arm ---
#
# `ckpt_stamp` identifies the weights.  The per-image errors also depend on the
# kNN cache and the bank those weights are scored against, and the key covered
# neither -- so rebuilding the kNN cache left every cached array valid-looking
# and stale.  Not hypothetical: on 2026-09-04 the cache was rebuilt to remove
# same-sequence leakage and the re-measurement returned in six seconds with the
# pre-rebuild numbers, to four significant figures.

def _arm(tmp_path, monkeypatch, knn_bytes=b"a", street_bytes=b"b"):
    import torch
    import bootstrap as B
    ck_dir, street = tmp_path / "ck", tmp_path / "street"
    ck_dir.mkdir(exist_ok=True)
    street.mkdir(exist_ok=True)
    (street / "k.npz").write_bytes(knn_bytes)
    (street / "s.f16.npy").write_bytes(street_bytes)
    torch.save({"knn_file": "k.npz", "street_file": "s.f16.npy"},
               ck_dir / "arm.pt")
    monkeypatch.setattr(B.config, "CHECKPOINTS", ck_dir)
    monkeypatch.setattr(B.config, "STREET_CACHE", street)
    return B, street


def test_the_key_changes_when_the_knn_cache_is_rebuilt(tmp_path, monkeypatch):
    B, street = _arm(tmp_path, monkeypatch)
    before = B.inputs_stamp("arm")
    (street / "k.npz").write_bytes(b"rebuilt, different contents")
    assert B.inputs_stamp("arm") != before, \
        "a rebuilt kNN cache must invalidate the cached errors"


def test_the_key_changes_when_the_bank_changes(tmp_path, monkeypatch):
    B, street = _arm(tmp_path, monkeypatch)
    before = B.inputs_stamp("arm")
    (street / "s.f16.npy").write_bytes(b"a different bank entirely")
    assert B.inputs_stamp("arm") != before


def test_the_key_is_stable_when_nothing_moved(tmp_path, monkeypatch):
    B, _ = _arm(tmp_path, monkeypatch)
    assert B.inputs_stamp("arm") == B.inputs_stamp("arm")


def test_an_arm_with_no_retrieval_keeps_its_existing_key(tmp_path, monkeypatch):
    """Invalidating non-retrieval arms would recompute them for no reason."""
    import torch
    import bootstrap as B
    ck_dir = tmp_path / "ck2"
    ck_dir.mkdir()
    torch.save({"epoch": 1}, ck_dir / "arm.pt")
    monkeypatch.setattr(B.config, "CHECKPOINTS", ck_dir)
    assert B.inputs_stamp("arm") == ""


def test_a_tag_survives_a_round_trip_through_the_new_key(tmp_path, monkeypatch):
    """The resolver parses the tag back out of the filename, and the key just
    grew a field. A parser that missed it would fold the stamp into the tag."""
    import bootstrap as B
    name = "d768-b350-e6-drop70_a0fa1e3331d4_i6669982f_test_5000r_k2_d3.npy"
    assert B.tag_of_err_cache(name) == "d768-b350-e6-drop70"
