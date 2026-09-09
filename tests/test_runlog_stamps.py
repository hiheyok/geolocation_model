"""What a marker has to stamp before "already done" is trustworthy.

`stage_inputs` exists because argv is not identity: the command naming
`knn_pca768_bank70_..._npz` is character for character the same whether that
file is the leaky cache or the clean rebuild. Two flag families were outside
it, and both were found by review on #65.

* **A comparison's inputs.** `knn_gap --a X --b Y` names two neighbour tables
  and nothing else. Rebuild an arm and the argv does not move, so the gap
  stage stayed satisfied and the runner reported a comparison it had not
  made -- the exact failure `stage_inputs` was written to prevent, in the one
  kind of stage whose whole output is a claim about two other files.
* **A basis as an output.** `project_street --out X.f16.npy` also writes
  `X_pca.npz`, and every later extension is projected with it. Only the
  projection was recorded, so deleting the basis left the fitting stage
  satisfied and the next projection failed with nothing pointing at the stage
  that should have re-run.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import runlog  # noqa: E402
import safeio  # noqa: E402


@pytest.fixture
def cache(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(config, "DATASET_PARQUET", tmp_path / "dataset.parquet")
    return tmp_path


def gap(a, b):
    return ["scripts/knn_gap.py", "--a", a, "--b", b, "--ranks", "1,16"]


# --------------------------------------------------- a comparison's inputs --

def test_a_comparison_stamps_both_tables(cache):
    (cache / "x.npz").write_bytes(b"one")
    (cache / "y.npz").write_bytes(b"two")
    got = runlog.stage_inputs(gap("x.npz", "y.npz"))
    assert "--a x.npz" in got and "--b y.npz" in got
    assert got["--a x.npz"] != "absent"


def test_rebuilding_one_side_changes_the_stamp(cache):
    """The reported defect. Same argv, different bytes: without this the gap
    stage stays satisfied and the old comparison is reported as current."""
    (cache / "x.npz").write_bytes(b"one")
    (cache / "y.npz").write_bytes(b"two")
    before = runlog.stage_inputs(gap("x.npz", "y.npz"))
    (cache / "x.npz").write_bytes(b"REBUILT")
    assert runlog.stage_inputs(gap("x.npz", "y.npz")) != before


def test_a_marker_from_the_old_table_no_longer_matches(cache):
    """End to end through the identity a marker actually stores."""
    (cache / "x.npz").write_bytes(b"one")
    (cache / "y.npz").write_bytes(b"two")
    argv = gap("x.npz", "y.npz")
    marker = runlog.marker_identity("g", argv, "s10",
                                    inputs=runlog.stage_inputs(argv))
    (cache / "x.npz").write_bytes(b"REBUILT")
    fresh = runlog.marker_identity("g", argv, "s10",
                                   inputs=runlog.stage_inputs(argv))
    assert runlog.marker_matches(marker, fresh) is False


def test_a_stem_valued_flag_is_stamped_not_recorded_absent(cache):
    """`concat_street --a embeddings_c3` names a stem; `knn_gap --a x.npz`
    names a file. Stamping the bare name only would leave every stem
    permanently `absent`, which reads like a stamp and is not one."""
    (cache / "embeddings_c3.f16.npy").write_bytes(b"vectors")
    got = runlog.stage_inputs(["scripts/concat_street.py",
                               "--a", "embeddings_c3"])
    assert got["--a embeddings_c3"] != "absent"


def test_a_genuinely_missing_input_is_still_absent(cache):
    got = runlog.stage_inputs(gap("gone.npz", "gone2.npz"))
    assert got["--a gone.npz"] == "absent"


# ------------------------------------------------------- the basis output ---

def proj(out):
    return ["scripts/project_street.py", "--src", "s.f16.npy", "--out", out,
            "--dim", "768"]


def test_the_fitted_basis_is_recorded_as_an_output(cache):
    (cache / "p768.f16.npy").write_bytes(b"projected")
    (cache / "p768_pca.npz").write_bytes(b"basis")
    got = runlog.stage_outputs(proj("p768.f16.npy"))
    assert "p768_pca.npz" in got and "p768.f16.npy" in got


def test_deleting_the_basis_makes_the_fit_stale(cache):
    """The reported defect. With the intermediates kept, a resume skipped the
    fit and the projection that needed the basis then had nothing to use."""
    (cache / "p768.f16.npy").write_bytes(b"projected")
    (cache / "p768_pca.npz").write_bytes(b"basis")
    argv = proj("p768.f16.npy")
    rec = runlog.marker_identity("fit", argv, "s10",
                                 inputs=runlog.stage_inputs(argv))
    import json
    d = json.loads(rec)
    d["outputs"] = runlog.stage_outputs(argv)
    text = json.dumps(d, sort_keys=True)
    assert runlog.outputs_intact(text)[0]
    (cache / "p768_pca.npz").unlink()
    ok, why = runlog.outputs_intact(text)
    assert not ok and "p768_pca.npz" in why


def test_a_stage_that_writes_no_basis_records_none(cache):
    """Recorded only if it exists, so this costs nothing for the stages that
    do not fit one -- and never becomes an output that can never be found."""
    (cache / "q.f16.npy").write_bytes(b"x")
    got = runlog.stage_outputs(["scripts/pool_pyramid.py", "--out",
                                "q.f16.npy"])
    assert list(got) == ["q.f16.npy"]


def test_a_reused_basis_is_an_input_of_the_projection_that_uses_it(cache):
    """The other half: `--basis` was already stamped, which is what makes the
    fit and the projection a chain rather than two independent stages."""
    (cache / "p768_pca.npz").write_bytes(b"basis")
    got = runlog.stage_inputs(["scripts/project_street.py", "--src",
                               "big.f16.npy", "--basis", "p768_pca.npz"])
    assert got["--basis p768_pca.npz"] != "absent"
