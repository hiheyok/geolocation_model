"""Nothing proved that two row-addressed files described the same rows.

This is review item 40 and the seven it subsumes.  The failure it guards is the
worst shape in this project: every array is the right dtype and a plausible
shape, every metric lands in a believable range, and each image has been scored
against another image's data.  There is no loud version of it to catch.

The digest is order-sensitive on purpose -- a set-based check is what
`build_knn` already had, and a reordering passes it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import provenance as prov  # noqa: E402


def test_the_digest_notices_a_reordering():
    a = np.arange(1000)
    b = a.copy()
    b[[3, 700]] = b[[700, 3]]
    assert prov.rows_digest(a) != prov.rows_digest(b)


def test_the_digest_ignores_the_integer_width():
    """A column read back as int32 is the same rows, not a different build."""
    a = np.arange(50)
    assert prov.rows_digest(a.astype(np.int32)) == prov.rows_digest(
        a.astype(np.int64))


def test_the_digest_separates_string_ids_that_share_a_prefix():
    """The external corpora key on strings, whose numpy width tracks the
    longest element -- so the raw buffer is not a safe thing to hash."""
    assert (prov.rows_digest(np.array(["a", "b"]))
            != prov.rows_digest(np.array(["a", "bb"])))


def test_the_digest_separates_a_truncation():
    a = np.arange(100)
    assert prov.rows_digest(a) != prov.rows_digest(a[:99])


def test_a_matching_sidecar_passes(tmp_path):
    p = tmp_path / "emb.f16.npy"
    ids = np.arange(64)
    np.save(p, np.zeros((64, 4), np.float16))
    prov.write(p, ids)
    assert prov.check(p, ids) is True


def test_a_reordered_artifact_is_refused(tmp_path):
    """The case with no other symptom: same file, same length, wrong order."""
    p = tmp_path / "emb.f16.npy"
    ids = np.arange(64)
    prov.write(p, ids)
    shuffled = ids.copy()
    shuffled[[1, 2]] = shuffled[[2, 1]]
    with pytest.raises(SystemExit, match="Row i of one is not row i"):
        prov.check(p, shuffled)


def test_a_length_change_is_refused(tmp_path):
    p = tmp_path / "emb.f16.npy"
    prov.write(p, np.arange(64))
    with pytest.raises(SystemExit, match="describes"):
        prov.check(p, np.arange(65))


def test_a_missing_sidecar_warns_and_reports_that_it_checked_nothing(
        tmp_path, capsys):
    """Every artifact on disk predates this, so absent cannot be fatal -- but
    the caller has to be able to tell a verified pass from an unverified one."""
    p = tmp_path / "emb.f16.npy"
    assert prov.check(p, np.arange(8)) is False
    assert "no provenance sidecar" in capsys.readouterr().out


def test_a_corrupt_sidecar_reads_as_absent_rather_than_raising(tmp_path):
    p = tmp_path / "emb.f16.npy"
    prov.sidecar(p).write_text("{not json", encoding="utf-8")
    assert prov.read(p) is None
    assert prov.check(p, np.arange(8)) is False


def test_the_basis_records_whether_the_builder_wrote_it(tmp_path):
    """A backfilled sidecar asserts the alignment as of the day it was
    written and vouches for nothing before it.  That distinction only
    survives if it is recorded."""
    p = tmp_path / "emb.f16.npy"
    prov.write(p, np.arange(8))
    assert json.loads(prov.sidecar(p).read_text())["basis"] == "built"
    prov.write(p, np.arange(8), basis="observed")
    assert json.loads(prov.sidecar(p).read_text())["basis"] == "observed"


def test_extra_fields_survive_the_round_trip(tmp_path):
    p = tmp_path / "emb.f16.npy"
    prov.write(p, np.arange(8), release="s10", row_space="the release")
    rec = prov.read(p)
    assert rec["release"] == "s10" and rec["row_space"] == "the release"


def test_the_digest_holds_no_null_bytes_of_its_own(tmp_path):
    """Regression: the string branch joins on NUL, and an earlier edit put a
    real NUL into the source file rather than the escape."""
    src = (ROOT / "src" / "provenance.py").read_bytes()
    assert b"\x00" not in src
