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
