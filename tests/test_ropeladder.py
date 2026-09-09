"""The chain that turns a rotation into a neighbour table, and the sweep.

Four arms at 3,400,180 rows do not fit on the disk at once, so this runner
deletes each arm's intermediates once its bank exists. That makes the delete
list load-bearing: too wide and it removes the corpus everything is built
from, too narrow and the fourth arm runs out of space at 90% of a stack.

The other thing worth pinning is the ORDER of the extensions. Stacking them in
any other order builds a file of the right length whose rows name the wrong
photographs, and every check downstream passes: right dtype, right count,
every row a unit vector.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import overnight as O  # noqa: E402
import ropeladder as R  # noqa: E402


def argv_of(plan, frag):
    return [st.argv for st in plan if frag in st.name]


# ------------------------------------------------------------- the arms ----

def test_arm_zero_is_the_identity_and_comes_first():
    """Every other arm is measured against it. `--rope-max 0` reproduces the
    unrotated POOLING bit for bit (#64); it does not reproduce the shipping
    bank, because that one's basis was fitted with about a fifth of its
    sample from val and test. Arm 0 is the baseline in this chain's own
    coordinate system, which is the only one the treated arms share."""
    assert R.ARMS[0][0] == "rope0"
    assert R.ARMS[0][2] == 0.0


def test_the_ladder_adds_one_axis_at_a_time():
    """The point of the ladder: `col` is the risky axis -- mirrored by
    heading -- so it has to be separable from the two that are not."""
    axes = [a[1].split(",") for a in R.ARMS[1:]]
    assert axes == [["depth"], ["depth", "row"], ["depth", "row", "col"]]
    for a in R.ARMS[1:]:
        assert a[2] > 0, "a treated arm with no rotation is arm 0 again"


def test_every_arm_passes_the_same_rotation_to_every_pool():
    """A bank pooled at two different angles is not a bank."""
    for arm, axes, mx in R.ARMS:
        pools = argv_of(R.arm_plan(arm, axes, mx), "pool")
        assert len(pools) == 1 + len(R.EXTS)
        flags = {tuple(a[a.index("--rope-axes"):]) for a in pools}
        assert len(flags) == 1, flags


def test_every_pool_carries_its_tiles():
    """A partly-tiled bank ranks blended rows against L0 rows on one cosine
    and silently demotes the untiled ones."""
    for arm, axes, mx in R.ARMS:
        for a in argv_of(R.arm_plan(arm, axes, mx), "pool"):
            assert "--tiles" in a


# -------------------------------------------------------- the extensions ----

def test_the_extensions_are_stacked_in_the_recorded_order():
    assert [e for e, _, _ in R.EXTS] == ["bank_ext", "bank_ext2",
                                         "bank_ext3", "bank_ext4"]
    assert [s for _, _, s in R.EXTS] == ["b115", "b190", "b265", "b340"]


def test_each_stack_builds_on_the_previous_one():
    plan = R.arm_plan("ropeD", "depth", 1.0)
    bases = [a[a.index("--base") + 1] for a in argv_of(plan, "stack")]
    outs = [a[a.index("--out") + 1] for a in argv_of(plan, "stack")]
    assert bases == ["pyr_ropeD"] + outs[:-1]
    assert outs[-1] == R.stack_at("ropeD", R.FINAL)


def test_a_pooled_extension_is_named_extension_first():
    """`provenance.ext_stem_for` strips suffixes from the RIGHT, so
    `ropeD_ext2` resolves to no metadata and `stack_bank` refuses the arm."""
    assert R.pooled_ext("bank_ext2", "ropeD") == "bank_ext2_ropeD"


# --------------------------------------------------------------- the fit ----

def test_the_basis_is_fitted_on_the_release_and_reused_for_the_stack():
    """One basis for every row. Fitting per file would put the release and the
    corpus in different spaces, and it would look like a weaker bank rather
    than an error."""
    plan = R.arm_plan("ropeD", "depth", 1.0)
    fit = argv_of(plan, "rope-fit-")[0]
    assert "--basis" not in fit, "the fitting run must not reuse a basis"
    proj = argv_of(plan, "rope-proj-")[0]
    assert proj[proj.index("--basis") + 1] == R.basis("ropeD")


def test_each_arm_fits_its_own_basis():
    """The rotation changes the vectors, so a basis fitted on the unrotated
    ones does not describe them."""
    assert len({R.basis(a) for a, _, _ in R.ARMS}) == len(R.ARMS)


def test_the_fit_runs_before_the_extensions_are_pooled():
    names = [st.name for st in R.arm_plan("ropeD", "depth", 1.0)]
    assert names.index("rope-fit-ropeD") < names.index(
        "rope-pool-bank_ext-ropeD")


# -------------------------------------------------------------- the sweep ---

def test_the_sweep_never_lists_a_shipping_file():
    """The delete list is built from names this script constructs, but the
    consequence of a mistake is the corpus, so it is checked as well."""
    for arm, _, _ in R.ARMS:
        assert not (set(R.sweepable(arm)) & R.KEEP)


def test_the_sweep_keeps_the_bank_and_the_basis():
    """What the arm is FOR has to survive it."""
    for arm, _, _ in R.ARMS:
        got = set(R.sweepable(arm))
        assert R.projected(arm) not in got
        assert R.basis(arm) not in got


def test_the_sweep_removes_every_intermediate_the_chain_wrote():
    """Derived from the plan, not hand-listed -- a stage that starts writing a
    new intermediate must show up here rather than quietly filling the disk."""
    plan = R.arm_plan("ropeD", "depth", 1.0)
    wrote = {a[a.index("--out") + 1] for a in (st.argv for st in plan)
             if "--out" in a}
    wrote = {w if w.endswith(".npy") else w + ".f16.npy" for w in wrote}
    keep = {R.projected("ropeD")}
    assert wrote - keep == set(R.sweepable("ropeD"))


def test_the_corpus_is_protected_by_name_as_well():
    for e, t, _ in R.EXTS:
        assert e + "_bal.f16.npy" in R.KEEP
        assert t + ".f16.npy" in R.KEEP
    assert R.BASE_SRC in R.KEEP and R.SHIPPING in R.KEEP


def test_keep_disables_the_sweep(monkeypatch, tmp_path):
    monkeypatch.setattr(R.config, "STREET_CACHE", tmp_path)
    p = tmp_path / R.sweepable("ropeD")[0]
    p.write_bytes(b"x")
    assert R.sweep("ropeD", keep=True) == 0
    assert p.exists()


def test_the_sweep_deletes_only_what_it_listed(monkeypatch, tmp_path):
    monkeypatch.setattr(R.config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(R, "log", lambda *a, **k: None)
    doomed = tmp_path / R.sweepable("ropeD")[0]
    doomed.write_bytes(b"xx")
    for name in (R.SHIPPING, R.projected("ropeD"), "bank_ext2_bal.f16.npy"):
        (tmp_path / name).write_bytes(b"y")
    R.sweep("ropeD", keep=False)
    assert not doomed.exists()
    for name in (R.SHIPPING, R.projected("ropeD"), "bank_ext2_bal.f16.npy"):
        assert (tmp_path / name).exists(), name


# ------------------------------------------------------------ the window ----

class Sampler:
    def __init__(self):
        self.stop_flag = type("F", (), {"set": lambda self: None})()

    def start(self):
        pass


@pytest.fixture
def window(tmp_path, monkeypatch):
    for name in ("RUNS", "LOGS", "MARKS"):
        monkeypatch.setattr(O, name, tmp_path / name)
    monkeypatch.setattr(O, "Sampler", Sampler)
    monkeypatch.setattr(O, "log", lambda *a, **k: None)
    monkeypatch.setattr(R, "log", lambda *a, **k: None)
    monkeypatch.setattr(R, "load_state", lambda: {"done": {}, "failed": {}})
    monkeypatch.setattr(R, "sweep", lambda *a, **k: 0)
    monkeypatch.setattr(R, "free_gb", lambda: 10 ** 6)
    monkeypatch.setattr(sys, "argv", ["ropeladder.py", "10"])
    return tmp_path


def runner(monkeypatch, fails=()):
    seen = []

    def fake(st, state, deadline):
        seen.append(st.name)
        return st.name not in fails

    monkeypatch.setattr(O, "run_stage", fake)
    return seen


def test_a_complete_ladder_exits_zero(window, monkeypatch):
    seen = runner(monkeypatch)
    assert R.main() == 0
    assert len([s for s in seen
                if s.startswith("rope-gap-")]) == len(R.ARMS) - 1


def test_a_failed_arm_stops_the_ladder_but_keeps_the_gaps_it_earned(
        window, monkeypatch):
    """An arm that fails is not a window that measured nothing: the arms
    before it built, and each of those is still a comparison against arm 0."""
    seen = runner(monkeypatch, fails=("rope-knn-ropeDR",))
    assert R.main() == 1
    assert "rope-gap-ropeD" in seen
    assert "rope-gap-ropeDR" not in seen
    assert not [s for s in seen if "ropeDRC" in s], "kept going after a failure"


def test_no_baseline_means_no_comparison_at_all(window, monkeypatch):
    """Without arm 0 the remaining arms could still be ranked against each
    other, and that ranking would have no zero point."""
    seen = runner(monkeypatch, fails=("rope-pool-rel-rope0",))
    assert R.main() == 1
    assert not [s for s in seen if s.startswith("rope-gap-")]


def test_every_gap_is_measured_against_arm_zero(window, monkeypatch):
    got = []
    monkeypatch.setattr(O, "run_stage",
                        lambda st, *a: (got.append(st.argv), True)[1])
    R.main()
    base = R.config.knn_name(R.projected("rope0"), "sequence", ext=R.MERGED)
    for a in [g for g in got if g[0].endswith("knn_gap.py")]:
        assert a[a.index("--a") + 1] == base


def test_a_disk_too_small_is_refused_before_anything_runs(window, monkeypatch):
    seen = runner(monkeypatch)
    monkeypatch.setattr(R, "free_gb", lambda: 20.0)
    assert R.main() == 1
    assert seen == [], "started a chain that cannot fit"


def test_the_basis_refit_is_measured_against_the_shipping_bank(window,
                                                               monkeypatch):
    """Arm 0 is the identity at the pooling step but NOT the shipping bank:
    `pyr768_mix_pca.npz` was fitted with about a fifth of its sample from val
    and test, and this chain fits on training rows only. Unmeasured, that
    difference sits inside every arm as an unlabelled gap from everything
    already published."""
    monkeypatch.setattr(Path, "exists", lambda self: True)
    seen = runner(monkeypatch)
    R.main()
    assert "rope-gap-basis" in seen


def test_a_missing_shipping_table_does_not_stop_the_ladder(window,
                                                           monkeypatch):
    """It is a comparison this ladder did not need in order to be a ladder."""
    monkeypatch.setattr(Path, "exists", lambda self: False)
    seen = runner(monkeypatch)
    assert R.main() == 0
    assert "rope-gap-basis" not in seen
    assert "rope-gap-ropeDRC" in seen
