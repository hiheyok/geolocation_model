"""The third review round: ten bugs, each of which reopened a closed guarantee.

The pattern worth naming is that every one of them sits at a *seam*. Round one
fixed a producer, round two fixed a consumer, and the hole was in whichever of
the two nobody had looked at that day -- `file_stamp` claims to include a size
and then slices it off; the completion mask is validated on write and read back
as "nonzero means done"; row provenance is checked on the one cache length that
happens to be the release's. So these tests are written against the *contract*
of each seam rather than against the line that was wrong, which is the only
version of them that would have failed before the fix and keeps failing if the
fix is later narrowed.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("OSV_RELEASE", "s10")

import provenance as prov  # noqa: E402
import runlog  # noqa: E402
import safeio  # noqa: E402


# ----------------------------------------------------- 7. file_stamp size --

def test_file_stamp_separates_equal_mtime_different_size(tmp_path):
    """The reproduction from the review, as a test.

    `"{:x}{:x}".format(size, mtime_ns)[-12:]` keeps 12 hex characters and a
    contemporary nanosecond mtime is 16 of them, so the size was concatenated
    and then sliced straight back off. Two files of wildly different length
    stamped identically whenever their mtimes matched -- which a restore, a
    `touch -r`, or any copy that preserves timestamps produces.
    """
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"x")
    b.write_bytes(b"y" * 999_999)
    st = a.stat()
    os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert a.stat().st_mtime_ns == b.stat().st_mtime_ns, "fixture needs equal mtimes"
    assert safeio.file_stamp(a) != safeio.file_stamp(b)


def test_file_stamp_still_notices_a_rewrite(tmp_path):
    p = tmp_path / "c"
    p.write_bytes(b"one")
    first = safeio.file_stamp(p)
    time.sleep(0.01)
    p.write_bytes(b"two")
    assert safeio.file_stamp(p) != first


def test_file_stamp_is_stable_for_an_untouched_file(tmp_path):
    """A key that changed on every read would invalidate every cache always."""
    p = tmp_path / "d"
    p.write_bytes(b"stable")
    assert safeio.file_stamp(p) == safeio.file_stamp(p)


# ------------------------------------------- 5. markers vs inputs/outputs --

def _ident(argv, inputs=None):
    return runlog.marker_identity("s", argv, "s10",
                                  inputs=inputs if inputs is not None else {})


def test_a_rebuilt_input_makes_a_marker_stale_with_argv_unchanged():
    """The dangerous case for the same-sequence fix.

    A clean k-NN cache replaced by an older one under the same name leaves the
    argv character for character identical, so an identity of name+argv+release
    goes on reporting the rebuild as satisfied.
    """
    argv = ["scripts/build_knn.py", "--street-file", "b.f16.npy"]
    before = _ident(argv, {"--street-file b.f16.npy": "aaaaaaaaaaaa"})
    after = _ident(argv, {"--street-file b.f16.npy": "bbbbbbbbbbbb"})
    assert runlog.marker_matches(before, after) is False


def test_a_deleted_output_makes_a_marker_stale():
    rec = json.loads(_ident(["src/train.py", "--tag", "t"]))
    rec["outputs"] = {"definitely_not_here_9e3f.pt": "aaaaaaaaaaaa"}
    ok, why = runlog.outputs_intact(json.dumps(rec))
    assert ok is False and "gone" in why


def test_an_output_swapped_under_its_own_name_makes_a_marker_stale(tmp_path,
                                                                   monkeypatch):
    p = tmp_path / "artifact.npz"
    p.write_bytes(b"first")
    rec = json.loads(_ident(["scripts/build_knn.py"]))
    rec["outputs"] = {p.name: safeio.file_stamp(p)}
    monkeypatch.setattr(runlog, "ROOT", tmp_path)
    assert runlog.outputs_intact(json.dumps(rec))[0] is True
    time.sleep(0.01)
    p.write_bytes(b"second, longer")
    assert runlog.outputs_intact(json.dumps(rec))[0] is False


def test_outputs_are_not_part_of_the_compared_identity():
    """They are stamped after the stage ran, so they cannot be in an identity
    computed before it runs; they are verified separately instead."""
    a = json.loads(_ident(["x"]))
    b = dict(a)
    b["outputs"] = {"anything.pt": "0123456789ab"}
    assert runlog.marker_matches(json.dumps(b, sort_keys=True),
                                 json.dumps(a, sort_keys=True)) is True


def test_a_marker_without_input_stamps_still_satisfies_but_is_flagged():
    """Refusing the 200+ markers on disk would re-run a finished queue."""
    argv = ["scripts/build_knn.py"]
    old = json.dumps({"stage": "s", "argv": argv, "release": "s10"},
                     sort_keys=True)
    new = _ident(argv, {"dataset.parquet": "aaaaaaaaaaaa"})
    assert runlog.marker_matches(old, new) is True
    assert runlog.marker_is_pre_inputs(old) is True
    assert runlog.marker_is_pre_inputs(new) is False


def test_stage_inputs_always_stamps_the_row_authority_and_the_code():
    got = runlog.stage_inputs(["scripts/build_knn.py", "--k", "32"])
    assert "dataset.parquet" in got
    assert "code:scripts/build_knn.py" in got


def test_stage_inputs_does_not_guess_ambiguous_flags():
    """`--tag` is an output under train.py and an input under bootstrap.py.

    Resolving it one way for both would either strand a stage forever or
    certify one that never ran, so it is deliberately not an input flag.
    """
    got = runlog.stage_inputs(["scripts/bootstrap.py", "--tag", "whatever"])
    assert not any("--tag" in k for k in got)


def test_stage_outputs_records_only_what_exists(tmp_path):
    """A flag resolved to the wrong path must go unrecorded rather than become
    an output that can never be found and a stage that re-runs forever."""
    assert runlog.stage_outputs(
        ["src/train.py", "--tag", "no_such_tag_7c1a"]) == {}


# ------------------------------------------------ 8. corrupt JSON markers --

def test_an_interrupted_marker_write_does_not_read_as_a_legacy_success():
    half = '{"argv": ["src/train.py"], "rele'
    assert runlog.marker_matches(half, _ident(["src/train.py"])) is False


# ---------------------------------------- 1. street cache row-space guard --

class _FakeConfig:
    pass


def _write_cache(path, n, d=4):
    a = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                  shape=(n, d))
    a[:] = 1.0
    a.flush()
    del a


def test_an_extension_only_cache_is_refused_as_a_street_file(tmp_path,
                                                             monkeypatch):
    """The hazard is not hypothetical and length cannot catch it.

    `bank_ext_bal.f16.npy` holds 750,000 rows and none of them are release
    rows, so it is *longer* than the 500,000-row release. Any rule of the form
    "longer than the release means release-plus-something" hands its first
    500,000 rows over as the release's images: valid shapes, in-range indices,
    every photograph paired with another photograph's embedding.
    """
    import dataset as ds

    ids = np.arange(500)
    p = tmp_path / "bank_ext_bal.f16.npy"
    _write_cache(p, 750)
    prov.write(p, np.arange(10_000, 10_750), basis="observed",
               row_space="the rows of bank_ext_meta.npz")
    with pytest.raises(SystemExit) as e:
        ds._check_street_rows(p, p.name, ids, 750, prov.rows_digest(ids))
    assert "not release-addressed" in str(e.value) \
        or "cannot be a --street-file" in str(e.value)


def test_a_stacked_cache_whose_rows_are_right_is_accepted(tmp_path, monkeypatch):
    import dataset as ds

    ids = np.arange(500)
    ext = np.arange(10_000, 10_750)
    p = tmp_path / "stack.f16.npy"
    _write_cache(p, 1250)
    prov.write(p, np.concatenate([ids, ext]), basis="built",
               row_space="release ++ bank_ext_meta.npz")
    monkeypatch.setattr(prov, "bank_ext", lambda stem, rel: {"image_id": ext})
    ds._check_street_rows(p, p.name, ids, 1250, prov.rows_digest(ids))


def test_a_stack_in_the_wrong_order_is_refused(tmp_path, monkeypatch):
    """Total length is satisfied by every wrong order too, which is why the
    digest is of the concatenation rather than of the parts."""
    import dataset as ds

    ids = np.arange(500)
    ext = np.arange(10_000, 10_750)
    p = tmp_path / "stack.f16.npy"
    _write_cache(p, 1250)
    prov.write(p, np.concatenate([ext, ids]), basis="built",   # reversed
               row_space="release ++ bank_ext_meta.npz")
    monkeypatch.setattr(prov, "bank_ext", lambda stem, rel: {"image_id": ext})
    with pytest.raises(SystemExit):
        ds._check_street_rows(p, p.name, ids, 1250, prov.rows_digest(ids))


def test_a_release_length_cache_is_still_checked_against_release_ids(tmp_path):
    import dataset as ds

    ids = np.arange(500)
    p = tmp_path / "rel.f16.npy"
    _write_cache(p, 500)
    prov.write(p, ids[::-1], basis="observed", row_space="the release")
    with pytest.raises(SystemExit):
        ds._check_street_rows(p, p.name, ids, 500, prov.rows_digest(ids))


def test_exts_of_reads_every_extension_in_a_multi_extension_stack(tmp_path):
    """A reader that saw only the first one computed a shorter expected length
    and rejected a bank that is perfectly well formed."""
    p = tmp_path / "multi.f16.npy"
    p.write_bytes(b"")
    prov.write(p, np.arange(3), basis="built",
               row_space="release ++ bank_ext_meta.npz ++ bank_ext2_meta.npz")
    assert prov.exts_of(p) == ["bank_ext", "bank_ext2"]
    assert prov.stacked_on_release(p) is True


def test_stacked_on_release_is_false_for_an_extension_only_row_space(tmp_path):
    p = tmp_path / "ext.f16.npy"
    p.write_bytes(b"")
    prov.write(p, np.arange(3), basis="built",
               row_space="the rows of bank_ext_meta.npz")
    assert prov.stacked_on_release(p) is False
    # it still names an extension, which is exactly why the name is not enough
    assert prov.exts_of(p) == ["bank_ext"]


# ------------------------------------------------ 9. binary completion mask --

def test_a_non_binary_completion_mask_is_refused(tmp_path):
    """A 2 or a 255 used to certify an unwritten all-zero token row as fetched.

    Beam inference already required `== 1`, so the same cache could train on a
    blank tile and fetch that tile live at evaluation -- the two paths
    disagreeing, and neither of them complaining.
    """
    import dataset as ds

    done = np.array([1, 1, 2, 1], dtype=np.uint8)
    p = tmp_path / "done.u8.npy"
    np.save(p, done)
    zs = np.array([4, 4, 8, 8])
    with pytest.raises(SystemExit) as e:
        ds._check_fetched(p, np.array([0, 1, 2, 3]), zs, 0, "train")
    assert "neither 0 nor 1" in str(e.value)


def test_a_fully_fetched_binary_mask_passes(tmp_path):
    import dataset as ds

    np.save(tmp_path / "done.u8.npy", np.ones(4, dtype=np.uint8))
    ds._check_fetched(tmp_path / "done.u8.npy", np.array([0, 1, 2, 3]),
                      np.array([4, 4, 8, 8]), 0, "train")


def test_an_unfetched_row_is_still_caught(tmp_path):
    import dataset as ds

    np.save(tmp_path / "done.u8.npy", np.array([1, 0, 1, 1], dtype=np.uint8))
    with pytest.raises(SystemExit) as e:
        ds._check_fetched(tmp_path / "done.u8.npy", np.array([0, 1, 2, 3]),
                          np.array([4, 4, 8, 8]), 0, "train")
    assert "incomplete" in str(e.value)


# ----------------------------------------------- 2. seeded sink negatives --

class _Neg:
    """The parts of GeoStepDataset that negative sampling actually reads."""

    neg_rng = None      # bound below

    def __init__(self, split, seed, n_actions=256, g=16, n=8):
        import hashlib

        self.neg_seed, self.neg_random, self.n_actions, self.g = \
            seed, False, n_actions, g
        self._split_entropy = int.from_bytes(
            hashlib.sha256(str(split).encode()).digest()[:4], "big")
        rng = np.random.default_rng(0)
        self.tile = rng.integers(0, 64, size=(n, 4, 3))
        self.tile[:, :, 0] = 4          # z, so descend() stays in range
        self.action = rng.integers(0, n_actions, size=(n, 4))
        self.lut = _AllTiles()


class _AllTiles(dict):
    def __getitem__(self, k):
        return int(k) % 1000


def _bind():
    import dataset as ds

    _Neg.neg_rng = ds.GeoStepDataset.neg_rng
    _Neg.sample_negatives = ds.GeoStepDataset.sample_negatives


def test_the_same_seed_gives_the_same_negatives():
    """Two commands recording one seed must see the same off-path tiles.

    They did not: the train set took `neg_random=True`, so each __getitem__
    built `default_rng(None)` and seeded from OS entropy -- independent of
    np.random.seed and torch.manual_seed both, and with workers also dependent
    on scheduling. The recorded seed described strictly less of the run than
    it appeared to, for the shipping configuration (--neg 4).
    """
    _bind()
    a, b = _Neg("train", 0), _Neg("train", 0)
    for i in range(4):
        ra = a.sample_negatives(i, 4, a.neg_rng(i))
        rb = b.sample_negatives(i, 4, b.neg_rng(i))
        for x, y in zip(ra, rb):
            assert np.array_equal(x, y), "index {} differs".format(i)


def test_the_negative_stream_does_not_depend_on_the_worker_or_the_epoch():
    """Drawn afresh per index, so any traversal order gives the same draws --
    which is what makes the result independent of --workers."""
    _bind()
    d = _Neg("train", 0)
    fwd = [d.sample_negatives(i, 4, d.neg_rng(i)) for i in range(6)]
    rev = {i: d.sample_negatives(i, 4, d.neg_rng(i)) for i in reversed(range(6))}
    for i in range(6):
        for x, y in zip(fwd[i], rev[i]):
            assert np.array_equal(x, y)


def test_different_seeds_give_different_negatives():
    _bind()
    a, b = _Neg("train", 0), _Neg("train", 1)
    same = all(np.array_equal(x, y)
               for i in range(6)
               for x, y in zip(a.sample_negatives(i, 4, a.neg_rng(i)),
                               b.sample_negatives(i, 4, b.neg_rng(i))))
    assert not same


def test_train_and_val_draw_different_streams_at_the_same_seed():
    """Otherwise the val negatives, which are fixed on purpose so sink accuracy
    is comparable across arms, would coincide with the training ones."""
    _bind()
    a, b = _Neg("train", 11), _Neg("val", 11)
    same = all(np.array_equal(x, y)
               for i in range(6)
               for x, y in zip(a.sample_negatives(i, 4, a.neg_rng(i)),
                               b.sample_negatives(i, 4, b.neg_rng(i))))
    assert not same


def test_neg_random_still_available_and_is_not_reproducible():
    """Kept as an explicit opt-out, so the old behaviour is a choice that gets
    recorded rather than a default nobody noticed."""
    _bind()
    d = _Neg("train", 0)
    d.neg_random = True
    draws = {tuple(d.neg_rng(0).integers(0, 10**9, size=8)) for _ in range(6)}
    assert len(draws) > 1


def test_train_py_seeds_its_negatives_and_records_whether_it_did():
    src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
    assert "neg_random=a.neg_random, neg_seed=a.seed" in src
    assert '"neg_random": a.neg_random' in src


# --------------------------------------- 3. PCA basis fitted on train only --

def test_the_pca_basis_is_fitted_on_the_training_split():
    """`--fit-from 500000` bounds the sample to the release, which reads like a
    split and is not one: on s10 the release is the whole 80/10/10 split, so
    20.0% of the 200,000-row fit sample was val and test (39,903 rows, measured).
    """
    src = (ROOT / "scripts" / "project_street.py").read_text(encoding="utf-8")
    assert 'labels[:hi] == "train"' in src
    assert "--fit-all-rows" in src
    assert "fit_split" in src and "fit_split_hash" in src


def test_held_out_rows_cannot_change_a_training_only_basis():
    """The property the fix has to have, checked on the arithmetic itself."""
    rng = np.random.default_rng(0)
    labels = np.array(["train"] * 80 + ["val"] * 10 + ["test"] * 10)
    X = rng.normal(size=(100, 6)).astype(np.float32)
    pool = np.flatnonzero(labels == "train")
    pick = np.sort(rng.choice(pool, 40, replace=False))
    mu_a = X[pick].mean(0)

    Y = X.copy()
    Y[labels != "train"] = 1e3          # make held-out rows wildly distinctive
    assert np.allclose(mu_a, Y[pick].mean(0))
    # and the old rule, which sampled the whole range, does move
    old = np.sort(np.random.default_rng(1).choice(100, 40, replace=False))
    assert not np.allclose(X[old].mean(0), Y[old].mean(0))


# ------------------------------------ 4. stacking onto an already-stacked base --

def test_stack_bank_accepts_a_stacked_base():
    """The documented growth path is `--base pool_bal_bank25 --ext
    bank_ext2_pool`, and the guard required the base to be release-length --
    so the command the workflow prescribes always exited. The banks on disk
    exist only because they predate it."""
    src = (ROOT / "scripts" / "stack_bank.py").read_text(encoding="utf-8")
    assert "prov.stacked_on_release(pb)" in src
    assert "prov.check_stack(pb, parts" in src


def test_a_multi_extension_row_space_round_trips(tmp_path):
    """Both the data and the digest must describe release ++ ext1 ++ ext2."""
    rel = np.arange(0, 100)
    e1 = np.arange(1000, 1050)
    e2 = np.arange(2000, 2025)
    p = tmp_path / "chain.f16.npy"
    p.write_bytes(b"")
    prov.write(p, np.concatenate([rel, e1, e2]), basis="built",
               row_space="release ++ bank_ext_meta.npz ++ bank_ext2_meta.npz")
    assert prov.exts_of(p) == ["bank_ext", "bank_ext2"]
    prov.check_stack(p, [rel, e1, e2], p.name)
    with pytest.raises(SystemExit):
        prov.check_stack(p, [rel, e2, e1], p.name)     # swapped
    with pytest.raises(SystemExit):
        prov.check_stack(p, [rel, e1], p.name)         # base's ext omitted


def test_stacking_the_same_extension_twice_is_refused():
    src = (ROOT / "scripts" / "stack_bank.py").read_text(encoding="utf-8")
    assert "already carries" in src


# --------------------------------------- 10. extension release enforcement --

def test_bank_ext_refuses_metadata_from_another_release(tmp_path, monkeypatch):
    """`serve.py` and `occupancy_probe.py` both bypassed this loader.

    The checkpoint is validated against the release; the extension metadata is
    a separately replaceable file, and a mismatched one attaches another
    release's z16 addresses to valid bank rows -- plausible coordinates, wrong
    place, nothing anywhere to notice.
    """
    import config

    p = tmp_path / "bank_ext_meta.npz"
    np.savez(p, image_id=np.arange(5), x16=np.arange(5), y16=np.arange(5),
             release="s01")
    monkeypatch.setattr(config, "bank_meta", lambda stem: p)
    with pytest.raises(SystemExit) as e:
        prov.bank_ext("bank_ext", "s10")
    assert "s01" in str(e.value)


def test_the_two_bypassing_consumers_now_call_the_loader():
    """A grep test, because the bug was not in the loader but in not using it."""
    for name in ("serve.py", "occupancy_probe.py"):
        src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "prov.bank_ext(" in src, name
        assert "np.load(config.bank_meta(" not in src, name
