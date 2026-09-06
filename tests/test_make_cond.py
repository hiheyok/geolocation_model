"""A row count is not a row identity (REVIEW5 #5).

`make_cond` joins a retrieval cache and a conditioning cache into one street
file, retrieval block first. The only thing establishing that row i of one is
row i of the other was that the two arrays had the same number of rows -- and
then `prov.carry` copied the *retrieval* file's authoritative row digest onto
the join.

So a conditioning cache describing other photographs, or the same ones in
another order, produced a fully provenanced artifact. Every shape agrees, the
completion mask is complete, the copy loop runs cleanly, and training reads
another image's high-resolution features as though they were this row's. No
consumer downstream can tell, because the published sidecar says the rows are
the retrieval file's rows and that part is true.

These tests drive `main()` through `sys.argv`, not the helpers, because the
argument parsing and the ordering of the checks against the destination open
are both part of what is being asserted.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import provenance as prov  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A retrieval cache and a conditioning cache that legitimately join."""
    import config
    import make_cond

    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(make_cond.config, "STREET_CACHE", tmp_path)
    ids = np.arange(6, dtype=np.int64)
    r = np.arange(6 * 4, dtype=np.float16).reshape(6, 4) + 1
    c = np.arange(6 * 2, dtype=np.float16).reshape(6, 2) + 100
    np.save(tmp_path / "r.f16.npy", r)
    np.save(tmp_path / "c.f16.npy", c)
    prov.write(tmp_path / "r.f16.npy", ids)
    prov.write(tmp_path / "c.f16.npy", ids)
    np.save(tmp_path / "c_done.u8.npy", np.ones(6, np.uint8))
    return tmp_path, ids, r, c


def run(monkeypatch, out="j.f16.npy"):
    import make_cond

    monkeypatch.setattr(sys, "argv",
                        ["make_cond", "--retrieval", "r.f16.npy",
                         "--cond", "c.f16.npy", "--out", out, "--block", "4"])
    make_cond.main()


def test_a_legitimate_join_succeeds(cache, monkeypatch):
    """Non-vacuity: the fixture must join before a refusal means anything."""
    tmp, ids, r, c = cache
    run(monkeypatch)
    j = np.load(tmp / "j.f16.npy")
    assert j.shape == (6, 6)
    assert np.array_equal(j[:, :4], r) and np.array_equal(j[:, 4:], c)
    rec = json.loads((tmp / "j.f16.npy.prov.json").read_text())
    assert rec["rows_digest"] == prov.rows_digest(ids)
    assert rec["retrieval_file"] == "r.f16.npy" and rec["retrieval_dim"] == 4


def test_a_reordered_conditioning_cache_is_refused(cache, monkeypatch):
    """The failure the review describes: same count, complete mask, different
    photographs. Nothing else in the script can see it."""
    tmp, ids, _, c = cache
    prov.write(tmp / "c.f16.npy", ids[::-1])
    with pytest.raises(SystemExit, match="Row i of one is not row i"):
        run(monkeypatch)


def test_a_same_length_replacement_is_refused(cache, monkeypatch):
    tmp, ids, _, _ = cache
    prov.write(tmp / "c.f16.npy", ids + 10_000)
    with pytest.raises(SystemExit, match="Row i of one is not row i"):
        run(monkeypatch)


def test_a_conditioning_cache_with_no_sidecar_cannot_be_laundered(
        cache, monkeypatch):
    """`prov.carry` warns on a missing source sidecar and returns None; that
    allowance must not extend to the conditioning side, whose identity is the
    thing being asserted."""
    tmp, _, _, _ = cache
    prov.sidecar(tmp / "c.f16.npy").unlink()
    with pytest.raises(SystemExit, match="no provenance sidecar"):
        run(monkeypatch)


def test_a_corrupt_sidecar_is_refused_rather_than_read_as_empty(cache,
                                                                monkeypatch):
    """`prov.read` returns None for unparseable JSON, which is the same signal
    as absent -- both must refuse, neither may pass."""
    tmp, _, _, _ = cache
    prov.sidecar(tmp / "c.f16.npy").write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="no provenance sidecar"):
        run(monkeypatch)


def test_a_sidecar_without_a_row_digest_is_refused(cache, monkeypatch):
    tmp, _, _, _ = cache
    prov.sidecar(tmp / "c.f16.npy").write_text(
        json.dumps({"version": 1, "rows": 6}), encoding="utf-8")
    with pytest.raises(SystemExit, match="records no row identity"):
        run(monkeypatch)


def test_the_conditioning_identity_reaches_the_published_sidecar(cache,
                                                                 monkeypatch):
    """`carry` copies the retrieval row digest by design, so without these
    fields the artifact records nothing about which conditioning cache it
    holds."""
    tmp, ids, _, _ = cache
    run(monkeypatch)
    rec = json.loads((tmp / "j.f16.npy.prov.json").read_text())
    assert rec["cond_file"] == "c.f16.npy" and rec["cond_dim"] == 2
    assert rec["cond_rows_digest"] == prov.rows_digest(ids)
    assert rec["cond_digest"] not in (None, "absent")
    assert rec["cond_done_digest"] not in (None, "absent")


def test_a_refusal_before_the_open_leaves_the_previous_join_intact(
        cache, monkeypatch):
    """Checking everything before opening the destination has a second
    benefit worth pinning: a rejected re-run does not destroy the good
    artifact that was already there."""
    tmp, ids, _, _ = cache
    run(monkeypatch)
    before = (tmp / "j.f16.npy").read_bytes()
    prov.write(tmp / "c.f16.npy", ids[::-1])
    with pytest.raises(SystemExit, match="Row i of one is not row i"):
        run(monkeypatch)
    assert (tmp / "j.f16.npy").read_bytes() == before
    assert prov.sidecar(tmp / "j.f16.npy").exists()


def test_a_failure_after_the_open_leaves_no_orphaned_sidecar(cache,
                                                             monkeypatch):
    """The stale-sidecar hazard proper (REVIEW5 #3, one layer down).

    The zero-row guard fires *after* the copy loop and unlinks the output. The
    previous run's sidecar sat beside it and would have survived, describing a
    join that no longer exists -- and would then be inherited by whatever was
    written to that name next.
    """
    tmp, ids, _, c = cache
    run(monkeypatch)
    assert prov.sidecar(tmp / "j.f16.npy").exists()

    bad = c.copy()
    bad[2] = 0                      # marked done, written as nothing
    np.save(tmp / "c.f16.npy", bad)
    prov.write(tmp / "c.f16.npy", ids)
    with pytest.raises(SystemExit, match="all zero"):
        run(monkeypatch)
    assert not (tmp / "j.f16.npy").exists()
    assert not prov.sidecar(tmp / "j.f16.npy").exists(), (
        "a sidecar describing a join that was deleted")


def test_an_incomplete_conditioning_pass_is_still_refused(cache, monkeypatch):
    """The pre-existing guard, kept working by the reordering of the checks."""
    tmp, _, _, _ = cache
    np.save(tmp / "c_done.u8.npy", np.array([1, 1, 1, 0, 0, 0], np.uint8))
    with pytest.raises(SystemExit, match="rows written"):
        run(monkeypatch)
