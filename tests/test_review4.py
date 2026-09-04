"""Round four: the seams between two guarantees that were each closed alone.

The shape that recurs is a check placed where it cannot do its job -- after the
destination is already truncated, or on one of the two inputs it is supposed to
be comparing. A check in the wrong place is worse than no check, because it
reads as coverage.

Only the items whose files the running clean-bank chain does not import are
covered here; the rest wait, for the reason in STATE.md 11.
"""

import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("OSV_RELEASE", "s10")

import provenance as prov  # noqa: E402


def _src(name):
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


# ------------------------- 4. validate before truncating the destination --

@pytest.mark.parametrize("name", ["concat_street.py", "stack_bank.py"])
def test_inputs_are_validated_before_the_destination_is_opened(name):
    """`mode="w+"` truncates, so a check after the copy cannot prevent a bad
    publication -- it destroys a good artifact and then exits, leaving the
    previous run's valid sidecar beside the new partial bytes. The next
    consumer compares that sidecar, finds a match, and accepts them."""
    s = _src(name)
    open_at = s.index('open_memmap(out, mode="w+"')
    checks = [s.index("prov.read(pa)")] if name == "concat_street.py" else \
             [s.index("parts, names, ext_ids = _row_space(")]
    for c in checks:
        assert c < open_at, "{}: validation runs after the destructive open".format(name)


def test_stack_bank_row_space_is_reconstructible_without_writing():
    """The refusals have to be reachable while the old output is still intact,
    which means they cannot live in the middle of the copy loop."""
    import stack_bank

    assert hasattr(stack_bank, "_row_space")


# ------------------- 5. concatenation must not certify a one-sided join --

def test_concat_refuses_when_either_input_lacks_provenance():
    """The asymmetry is what made this easy to miss: a missing sidecar on A
    left the output unstamped, while a missing one on B laundered the result
    through A's digest -- so the combined file claimed rows nothing had ever
    established B has."""
    s = _src("concat_street.py")
    assert "if not ra or not rb:" in s
    # and the old both-must-exist-to-refuse form is gone
    assert "if ra and rb and ra.get" not in s


def test_concat_records_both_encoders_and_the_scale():
    """`encoder_of` returns only model/crops/size, so without a structured
    combined identity a basis fitted on [DINO | SigLIP x4.03] and one fitted on
    [DINO | a different SigLIP] compare equal and get reused across spaces."""
    s = _src("concat_street.py")
    assert "combined=json.dumps(" in s
    assert '"scale_b": a.scale_b' in s


def test_a_one_sided_digest_would_have_certified_a_mismatched_join(tmp_path):
    """The bug, reproduced: carrying A's digest onto the output says nothing
    about B, and the two can be different builds of the same length."""
    a_ids = np.arange(100)
    b_ids = np.arange(100)[::-1]          # same rows, different order
    pa, pb = tmp_path / "a.npy", tmp_path / "b.npy"
    for p in (pa, pb):
        np.save(p, np.zeros((100, 2), np.float16))
    prov.write(pa, a_ids, basis="observed")
    prov.write(pb, b_ids, basis="observed")
    ra, rb = prov.read(pa), prov.read(pb)
    assert ra["rows_digest"] != rb["rows_digest"], "fixture must differ"
    # carrying A alone would have stamped the output with a_ids' digest
    assert prov.rows_digest(a_ids) == ra["rows_digest"]


# ------------------------- 11 & 12. one completion-mask validator, shared --

def test_a_non_binary_mask_is_refused(tmp_path):
    """`!= 0` is the natural spelling and it is the bug: a 2 or a 255 from a
    torn write certifies a zero-filled row as complete."""
    import maskio

    p = tmp_path / "done.u8.npy"
    np.save(p, np.array([1, 1, 2, 1], np.uint8))
    with pytest.raises(SystemExit) as e:
        maskio.load_mask(p, 4, "test")
    assert "not a completion mask" in str(e.value)


def test_a_mask_of_the_wrong_length_is_refused(tmp_path):
    import maskio

    p = tmp_path / "done.u8.npy"
    np.save(p, np.ones(3, np.uint8))
    with pytest.raises(SystemExit):
        maskio.load_mask(p, 4, "test")


def test_sum_is_not_a_completion_test():
    """A mask with one 2 and one 0 sums to n while a row is still blank --
    which is how an incomplete cache passed its own final check."""
    import maskio

    d = np.array([1, 1, 2, 0], np.uint8)
    assert int(d.sum()) >= len(d)            # the old test would pass
    assert maskio.complete(d) == 2           # the real count
    assert maskio.is_complete(d, len(d)) is False


def test_a_fully_written_mask_is_complete():
    import maskio

    d = np.ones(6, np.uint8)
    assert maskio.is_complete(d, 6) is True


@pytest.mark.parametrize("name,needle", [
    ("tile_cache.py", "maskio.load_mask(done_p"),
    ("fetch_tiles.py", 'maskio.load_mask(sdone_p'),
    ("fuse_head.py", "maskio.check_mask(d"),
])
def test_every_mask_consumer_uses_the_shared_validator(name, needle):
    """Four producers grew four copies of this check and three were wrong at
    some point; the copies are why nobody noticed when one drifted."""
    assert needle in _src(name)


def test_no_consumer_still_reads_a_mask_as_truthy():
    for name in ("tile_cache.py", "fetch_tiles.py", "fuse_head.py"):
        s = _src(name)
        assert "d.astype(bool)" not in s, name
        assert "if sdone[sr]:" not in s, name


def test_seed_from_refuses_a_negative_source_row():
    """A negative row indexes from the END in numpy, so it passes any max()
    bound check and copies some other tile's perfectly valid tensor."""
    s = _src("fetch_tiles.py")
    assert "srow.min() < 0" in s
    assert "wraps from the end" in s


def test_seed_from_refuses_duplicate_rows_or_addresses():
    s = _src("fetch_tiles.py")
    assert "len(np.unique(srow)) != len(srow)" in s
    assert "duplicate tile addresses" in s


def test_tile_cache_no_work_branch_still_verifies():
    """Returning early meant a resumed-but-complete cache skipped both the
    verification and the metadata write."""
    s = _src("tile_cache.py")
    i = s.index("if len(todo) == 0:")
    j = s.index("nothing to do;")
    assert "maskio.is_complete(done, n)" in s[i:j]
    assert "np.savez(meta_p" in s[i:j]


# ----------------------------- 20. the external cohort is frozen, not ranked --

def test_taking_the_n_smallest_hashes_is_not_append_stable():
    """The claim the source used to make. Membership depends on the global nth
    value, so a newly harvested id hashing low joins and evicts someone."""
    old = ["img{:05d}".format(i) for i in range(2000)]
    new = old + ["extra{:05d}".format(i) for i in range(3000)]

    def pick(ids, n):
        k = np.array([zlib.crc32(("0" + i).encode()) for i in ids])
        return {ids[j] for j in np.argsort(k)[:n]}

    a, b = pick(old, 500), pick(new, 500)
    assert a != b, "the growth must move the cohort for this to be a bug"
    # and the size of the move is large, not marginal
    assert len(a & b) < len(a)


def test_eval_highres_freezes_and_replays_the_cohort():
    s = _src("eval_highres.py")
    assert "cohort_seed" in s and "cohort_digest" in s
    assert "replayed from" in s
    # the false claim is gone
    assert "stable as the manifest grows" not in s


def test_the_frozen_cohort_round_trips_through_the_digest(tmp_path):
    ids = ["img{:05d}".format(i) for i in range(50)]
    p = tmp_path / "cohort.json"
    p.write_text(json.dumps(ids), encoding="utf-8")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back == ids
    assert prov.rows_digest(np.asarray(back)) == prov.rows_digest(np.asarray(ids))
    # order-sensitive, so a reshuffled cohort is a different benchmark
    assert prov.rows_digest(np.asarray(back[::-1])) != \
        prov.rows_digest(np.asarray(ids))
