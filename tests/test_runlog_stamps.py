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


# ----------------------------------------------- the chain, end to end ------

CHAIN = ("--src", "--tiles", "--base", "--ext")


def test_every_link_of_the_pool_stack_project_chain_is_stamped():
    """Each of these names a file an earlier stage wrote, under a name that
    does not change when its contents do."""
    for flag in CHAIN:
        assert runlog.INPUT_FLAGS.get(flag) == "street", flag


def test_a_rebuilt_pool_invalidates_the_stage_that_reads_it(cache):
    """Reported on #65. Change an arm's angle and its pools re-run -- their
    argv carries `--rope-max` -- but the fit reads `--src` and the stacks read
    `--base`/`--ext`, none of which were stamped. So the whole downstream
    chain stayed satisfied and the runner exited 0 on an experiment whose
    first stage had been replaced."""
    (cache / "pyr_ropeD.f16.npy").write_bytes(b"rotated one way")
    argv = ["scripts/project_street.py", "--src", "pyr_ropeD.f16.npy",
            "--out", "pyr768_ropeD.f16.npy", "--dim", "768"]
    before = runlog.marker_identity("fit", argv, "s10",
                                    inputs=runlog.stage_inputs(argv))
    (cache / "pyr_ropeD.f16.npy").write_bytes(b"rotated another way")
    after = runlog.marker_identity("fit", argv, "s10",
                                   inputs=runlog.stage_inputs(argv))
    assert runlog.marker_matches(before, after) is False


def test_a_rebuilt_extension_invalidates_the_stack(cache):
    (cache / "pyr_ropeD.f16.npy").write_bytes(b"base")
    (cache / "bank_ext2_ropeD.f16.npy").write_bytes(b"ext")
    argv = ["scripts/stack_bank.py", "--base", "pyr_ropeD",
            "--ext", "bank_ext2_ropeD", "--out", "pyr_ropeD_b190"]
    before = runlog.stage_inputs(argv)
    (cache / "bank_ext2_ropeD.f16.npy").write_bytes(b"ext rebuilt")
    assert runlog.stage_inputs(argv) != before


def test_a_rebuilt_tile_cache_invalidates_the_pool(cache):
    (cache / "dual_c3.f16.npy").write_bytes(b"crops")
    (cache / "tile6.f16.npy").write_bytes(b"tiles")
    argv = ["scripts/pool_pyramid.py", "--src", "dual_c3.f16.npy",
            "--tiles", "tile6", "--out", "pyr_x.f16.npy"]
    before = runlog.stage_inputs(argv)
    (cache / "tile6.f16.npy").write_bytes(b"tiles rebuilt")
    assert runlog.stage_inputs(argv) != before


# ------------------------------------------------ the naming conventions ----

def test_a_flag_naming_a_corpus_resolves_to_its_metadata(cache):
    """`--ext bank_ext70` names a corpus whose file is
    `bank_ext70_meta.npz`. Two of the three conventions in this cache would
    otherwise stamp `absent` forever."""
    (cache / "bank_ext70_meta.npz").write_bytes(b"ids")
    got = runlog.stage_inputs(["scripts/seqleak.py", "--ext", "bank_ext70"])
    assert got["--ext bank_ext70"] != "absent"


def test_the_conventions_are_tried_in_order_and_a_miss_stays_absent(cache):
    C = runlog.street_path
    (cache / "x.npz").write_bytes(b"1")
    assert C(cache, "x.npz").name == "x.npz"
    (cache / "y.f16.npy").write_bytes(b"1")
    assert C(cache, "y").name == "y.f16.npy"
    (cache / "z_meta.npz").write_bytes(b"1")
    assert C(cache, "z").name == "z_meta.npz"
    assert C(cache, "nothing").name == "nothing"


# ------------------------------------------- a comparison's checkpoints -----

def boot(tags):
    return ["scripts/bootstrap.py", "--tags", tags, "--split", "test",
            "--n", "5000"]


def test_a_bootstrap_stamps_every_checkpoint_it_compares(cache, monkeypatch):
    """Reported on #67. `--tags a,b,c` names the checkpoints a comparison is
    a comparison OF, and its argv does not move when any of them is
    retrained -- so a rebuilt arm re-ran its training, found the comparison
    satisfied, kept the previous report and exited 0."""
    import config
    monkeypatch.setattr(config, "CHECKPOINTS", cache)
    for t in ("a", "b"):
        (cache / (t + ".pt")).write_bytes(b"weights " + t.encode())
    got = runlog.stage_inputs(boot("a,b"))
    assert "--tags a" in got and "--tags b" in got
    assert got["--tags a"] != "absent"


def test_retraining_one_arm_invalidates_the_comparison(cache, monkeypatch):
    import config
    monkeypatch.setattr(config, "CHECKPOINTS", cache)
    for t in ("a", "b"):
        (cache / (t + ".pt")).write_bytes(b"weights " + t.encode())
    before = runlog.stage_inputs(boot("a,b"))
    (cache / "b.pt").write_bytes(b"retrained")
    assert runlog.stage_inputs(boot("a,b")) != before


def test_the_tags_are_stamped_one_by_one_not_as_a_blob(cache, monkeypatch):
    """So the record says which arm moved, not merely that something did."""
    import config
    monkeypatch.setattr(config, "CHECKPOINTS", cache)
    for t in ("a", "b", "c"):
        (cache / (t + ".pt")).write_bytes(b"w")
    got = runlog.stage_inputs(boot(" a , b ,c"))
    assert {"--tags a", "--tags b", "--tags c"} <= set(got)
