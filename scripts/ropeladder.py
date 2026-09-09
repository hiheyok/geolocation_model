"""Does a fixed rotation give the pooled pyramid a usable position?

`pool_pyramid` can now rotate each view by `R(depth, row, col)` before summing
(#64), which turns the pooled inner product into

    <sum_k R(p_k) x_k, sum_l R(p_l) y_l> = sum_kl <x_k, R(p_l - p_k) y_l>

-- same-position matches keep full credit, cross-position matches are
attenuated. Nothing is learned, so the whole 3,400,180-row corpus can be
re-pooled from per-view vectors already on disk and scored by `knn_gap`
**before any retrain**. That is the entire reason this is affordable, and it
is why the ladder stops at retrieval: an agent number costs a day per arm and
this costs half an hour.

**The axis ladder is the experiment, not a formality.**

* `depth` is partly redundant already: `levels()` normalises L0 and L1
  separately and blends them 50/50, so the levels are not cross-crediting at
  full strength to begin with.
* `row` is heading-invariant. Sky is up in every photograph.
* `col` is the risky one. It is left/right in the frame, and two cars passing
  the same building in opposite directions put it in mirrored columns, so
  position-aligned matching would *penalise* a true match. The crops also
  overlap 61% and their centres span only 1.31 columns, so most of what `col`
  can do it must do on the tiles.

Read the arms against each other, not against a hope: if `depth+row` gains and
`+col` gives it back, heading is the reason and the fix is a heading-invariant
encoding rather than a bigger rotation.

**Arm 0 is built through this same chain, and it is not the shipping bank.**
`--rope-max 0` is the identity at the POOLING step, so the 1536-d vectors do
match. The projection does not: `pyr768_mix_pca.npz` was fitted the old way,
with about a fifth of its sample drawn from val and test, and `project_street`
now fits on training rows only. Every arm here therefore fits its own basis
the same way, which is what keeps the ladder internally clean -- and it is why
arm 0 has to be built rather than substituted for by the bank already on disk,
which lives in a different coordinate system.

That difference is measured rather than waved at: the last comparison is arm 0
against the shipping bank, which is the "what does refitting the basis on
train-only rows do" experiment `project_street` describes as untried. It costs
one `knn_gap` because both neighbour tables already exist by then.

    OSV_RELEASE=s10 py scripts/ropeladder.py 6
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

REL = "s10"
if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = REL

import config                                   # noqa: E402
import overnight as O                           # noqa: E402
from overnight import Stage, log, load_state    # noqa: E402

MERGED = "bank_ext70"
BASE_SRC = "dual_c3.f16.npy"        # the release's 3 crops, 500,000 rows
BASE_TILES = "tile6"                # its 6 tiles
SHIPPING = "pyr768_l0l1_b340.f16.npy"

# (crop cache, tile cache, the stack size it produces) in the order
# `bank_ext70_meta.npz` records them. Stacking in any other order builds a
# file of the right length whose rows name the wrong photographs.
EXTS = [("bank_ext", "tile6_ext", "b115"),
        ("bank_ext2", "tile6_ext2", "b190"),
        ("bank_ext3", "tile6_ext3", "b265"),
        ("bank_ext4", "tile6_ext4", "b340")]
FINAL = "b340"

# (arm, --rope-axes, --rope-max). Cheapest question first is not the rule
# here; arm 0 is first because every other arm is measured against it.
ARMS = [("rope0", "depth,row,col", 0.0),
        ("ropeD", "depth", 1.0),
        ("ropeDR", "depth,row", 1.0),
        ("ropeDRC", "depth,row,col", 1.0)]

# Never delete these, whatever the cleanup list says. Everything the shipping
# bank is built from lives here, and this script writes nothing with these
# names -- but a cleanup bug that only shows up as "the bank is gone" is not
# one worth risking on that reasoning alone.
KEEP = {BASE_SRC, SHIPPING, "pyr768_l0_b340.f16.npy",
        "pyr768_mix_pca.npz", "pyr768_l0_pca.npz"}
KEEP |= {e + "_bal.f16.npy" for e, _, _ in EXTS}
KEEP |= {t + ".f16.npy" for _, t, _ in EXTS} | {BASE_TILES + ".f16.npy"}

# Peak disk for one arm: 1.5 GB pooled release + 4 x 2.3 GB pooled extensions
# + 3.5 + 5.8 + 8.1 + 10.4 GB of stacks + 5.2 GB projected. Only the last
# survives cleanup, so `n` finished arms cost 5.4 GB each and the one being
# built costs the rest.
PEAK_GB = 44
RESIDENT_GB = 5.4


def pooled_rel(arm):
    return "pyr_{}.f16.npy".format(arm)


def pooled_ext(ext, arm):
    """`bank_ext2_ropeD` -- the extension stem must come FIRST.

    `provenance.ext_stem_for` strips suffixes from the right to find the
    metadata naming an extension's images, so `ropeD_ext2` resolves to None
    and `stack_bank` refuses the arm.
    """
    return "{}_{}".format(ext, arm)


def stack_at(arm, size):
    return "pyr_{}_{}".format(arm, size)


def basis(arm):
    return "pyr768_{}_pca.npz".format(arm)


def projected(arm):
    return "pyr768_{}_{}.f16.npy".format(arm, FINAL)


def rope_flags(axes, rope_max):
    return ["--rope-axes", axes, "--rope-max", str(rope_max)]


def arm_plan(arm, axes, rope_max):
    """Pool, project-to-fit, pool the extensions, stack, project, index.

    Every stage is `critical`: each one reads the file the previous wrote, and
    a previous run's file is usually still at that path, so a failure here
    does not stop the chain -- it changes which experiment the chain is of.
    """
    st = []
    pool = ["scripts/pool_pyramid.py", "--src", BASE_SRC,
            "--tiles", BASE_TILES, "--out", pooled_rel(arm)]
    st.append(Stage("rope-pool-rel-" + arm, pool + rope_flags(axes, rope_max),
                    release=REL, est=6 * 60, retries=2, critical=True))
    # Fit the basis on the RELEASE, before any extension exists, and reuse it
    # for the stack below. One basis for every row is the point: fitting per
    # file would put the release and the corpus in different spaces, and the
    # failure would look like a weaker bank rather than an error.
    st.append(Stage(
        "rope-fit-" + arm,
        ["scripts/project_street.py", "--src", pooled_rel(arm),
         "--out", "pyr768_{}.f16.npy".format(arm), "--dim", "768"],
        release=REL, est=10 * 60, retries=2, critical=True))
    for ext, tstem, _size in EXTS:
        st.append(Stage(
            "rope-pool-{}-{}".format(ext, arm),
            ["scripts/pool_pyramid.py", "--src", ext + "_bal.f16.npy",
             "--tiles", tstem, "--out", pooled_ext(ext, arm) + ".f16.npy"]
            + rope_flags(axes, rope_max),
            release=REL, est=8 * 60, retries=2, critical=True))
    base = "pyr_" + arm
    for ext, _tstem, size in EXTS:
        out = stack_at(arm, size)
        st.append(Stage(
            "rope-stack-{}-{}".format(arm, size),
            ["scripts/stack_bank.py", "--base", base,
             "--ext", pooled_ext(ext, arm), "--out", out],
            release=REL, est=12 * 60, retries=2, critical=True))
        base = out
    st.append(Stage(
        "rope-proj-" + arm,
        ["scripts/project_street.py", "--src", stack_at(arm, FINAL) + ".f16.npy",
         "--out", projected(arm), "--dim", "768", "--basis", basis(arm)],
        release=REL, est=15 * 60, retries=2, critical=True))
    st.append(Stage(
        "rope-knn-" + arm,
        ["scripts/build_knn.py", "--street-file", projected(arm),
         "--split-mode", "sequence", "--k", "32", "--bank-ext", MERGED],
        release=REL, est=25 * 60, retries=2, critical=True))
    return st


def sweepable(arm):
    """The files this arm's chain wrote that nothing downstream still needs.

    Deleting is what makes four arms fit in the free space, so it is not
    optional -- but it is scoped to names this script constructs, checked
    against `KEEP`, and it is safe to resume across: `Stage.satisfied` calls
    `runlog.outputs_intact`, so a marker whose output has gone is stale and
    the stage re-runs.
    """
    names = [pooled_rel(arm), "pyr768_{}.f16.npy".format(arm)]
    names += [pooled_ext(e, arm) + ".f16.npy" for e, _, _ in EXTS]
    names += [stack_at(arm, s) + ".f16.npy" for _, _, s in EXTS]
    return [n for n in names if n not in KEEP]


def sweep(arm, keep):
    """Delete this arm's intermediates. Returns bytes freed."""
    if keep:
        return 0
    freed = 0
    for name in sweepable(arm):
        p = config.STREET_CACHE / name
        if name in KEEP or not p.exists():
            continue
        freed += p.stat().st_size
        p.unlink()
        for side in (p.with_suffix(p.suffix + ".prov.json"),):
            if side.exists():
                side.unlink()
    log("swept  {}  ({:.1f} GB reclaimed; the projected bank and its basis "
        "stay)".format(arm, freed / 2 ** 30))
    return freed


def free_gb():
    return shutil.disk_usage(str(config.STREET_CACHE)).free / 2 ** 30


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return 0
    hours = float(arg) if arg else 6.0
    keep = "--keep" in sys.argv
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    # Refused up front rather than discovered at 90% of a stack. A part-written
    # bank is a file of the right dtype and the wrong length, and the stage
    # that consumes it reports a shape error a long way from the cause.
    need = PEAK_GB + (RESIDENT_GB * (len(ARMS) - 1) if not keep else
                      PEAK_GB * (len(ARMS) - 1))
    if free_gb() < need:
        return log("{:.0f} GB free, and this needs about {:.0f}. {}".format(
            free_gb(), need,
            "Drop --keep, or free space." if keep else "Free some space.")) or 1

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("ropeladder: does a fixed rotation give the pooled pyramid a position?")
    log("deadline {}  ({:.1f} h),  {:.0f} GB free".format(
        O.hhmm(deadline), hours, free_gb()))

    res = {"done": [], "failed": [], "skipped": [], "unrun": []}
    built = []
    for arm, axes, rope_max in ARMS:
        r = O.run_plan(arm_plan(arm, axes, rope_max), state, deadline)
        for k, v in r.items():
            res[k] += v
        if r["failed"] or r["unrun"]:
            log("{} did not finish; leaving its files in place to look at"
                .format(arm))
            break
        built.append(arm)
        sweep(arm, keep)
        log("{:.0f} GB free after {}".format(free_gb(), arm))

    # The comparisons, each against arm 0. `knn_gap` reads neighbour tables
    # only -- under a minute, no GPU -- so every arm that built gets one even
    # if a later arm did not.
    if "rope0" in built:
        a = config.knn_name(projected("rope0"), "sequence", ext=MERGED)
        for arm in built[1:]:
            r = O.run_plan([Stage(
                "rope-gap-" + arm,
                ["scripts/knn_gap.py", "--a", a,
                 "--b", config.knn_name(projected(arm), "sequence",
                                        ext=MERGED),
                 "--ranks", "1,16,32", "--flows"],
                release=REL, est=5 * 60, retries=1)], state, deadline)
            for k, v in r.items():
                res[k] += v
        # Arm 0 against the shipping bank. NOT a check that they agree -- they
        # should not. The pooling is identical, but `pyr768_mix_pca.npz` was
        # fitted with roughly a fifth of its sample drawn from val and test,
        # and this chain fits on training rows only. So this measures the
        # basis refit, which `project_street` describes as untried and which
        # would otherwise sit inside every arm above as an unlabelled
        # difference from everything already published.
        ship_knn = config.knn_name(SHIPPING, "sequence", ext=MERGED)
        if (config.STREET_CACHE / ship_knn).exists():
            r = O.run_plan([Stage(
                "rope-gap-basis",
                ["scripts/knn_gap.py", "--a", ship_knn, "--b", a,
                 "--ranks", "1,16,32", "--flows"],
                release=REL, est=5 * 60, retries=1)], state, deadline)
            for k, v in r.items():
                res[k] += v
        else:
            log("no shipping neighbour table at {}, so the basis refit is "
                "unmeasured; the ladder above is unaffected".format(ship_knn))
    else:
        log("arm 0 never built, so there is no baseline to compare against; "
            "skipping every gap rather than ranking arms against each other")
        res["skipped"].append("rope-gap-all")

    O.SAMPLER.stop_flag.set()
    log("ropeladder done. {} built: {}".format(len(built), ", ".join(built)))
    for k in ("failed", "skipped", "unrun"):
        if res[k]:
            log("  {}: {}".format(k, ", ".join(res[k])))
    return 1 if (res["failed"] or res["skipped"] or res["unrun"]) else 0


if __name__ == "__main__":
    sys.exit(main())
