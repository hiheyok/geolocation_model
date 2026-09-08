"""Carry the paired tiles experiment from 1.15M bank rows to the whole corpus.

`scripts/tilebig.py` measured crops+tiles against crops at **1,150,180** rows
and found +3.30 pp [+3.0, +3.6] at 25 km, median 54.8 -> 38.4 km. The series it
extended had not flattened: +1.80 pp at 25k, +2.60 at 400k, +3.30 at 1.15M. So
the number that decides whether tiles ship is the one at the size the bank
actually is, and until `chain_tiles.sh` finished on 2026-09-07 that number could
not be computed -- `bank_ext2`, `bank_ext3` and `bank_ext4` had no tiles at all.

They do now, 750,000 rows each, done-masks verified. This script does the part
that chain deliberately stopped short of: pool, stack, project, index, compare,
at **3,400,180** bank rows.

Nothing here is new work; it is `tilebig`'s second half with three extensions
where it had one. What is new is only the arithmetic of the stack:

    pyr_l0_b115 (1.25M) ++ bank_ext2_* -> _b190 ++ bank_ext3_* -> _b265
                                                ++ bank_ext4_* -> _b340

`_bNNN` counts *bank* rows -- the release's 400,180 train images plus the
extensions -- which is what `b115` meant and what the gain curve is plotted
against. The files themselves hold 500,000 more, the release's val and test
rows, which `build_knn` excludes from the bank and uses as queries.

**The extensions are fed from `_bal`, not `_dual`, and that is not a seam.**
Only `bank_ext` has a `_dual` cache; 2, 3 and 4 have `_bal`. `pool_pyramid`
normalises per 768-d token before it pools, so any positive per-encoder
rescaling cancels: its docstring records `L0(dual_c3)` and `L0(dual_bal)`
agreeing at cosine 0.99999976 while their raw SigLIP norms are 20.53 and 82.73.
Picking wrong would otherwise be exactly the kind of silent seam that reads as
a weaker bank rather than as an error.

**Order is checked, not assumed.** `bank_ext70_meta.npz` was verified to be
`bank_ext ++ bank_ext2 ++ bank_ext3 ++ bank_ext4` id for id before this was
written, so the incrementally stacked file's row space digests the same as the
merged one and `build_knn --bank-ext bank_ext70` validates it. Two extensions
swapped produce a file of exactly the right length, so length is not evidence.

The basis is reused, not refitted -- `pyr768_l0_pca.npz` and
`pyr768_mix_pca.npz`, the same ones the 1.15M point used. One basis for every
row is the whole point: refitting per bank size would put the two densities in
different spaces and the difference between them would read as a result.

Order of the plan: both arms are pooled, stacked, projected and indexed, then
`knn_gap` answers the question in minutes. The `cell8` pair is planned last
because it is the honest number and the expensive one -- §8c found the 25 km
gain 5.6x larger on `sequence` than on the geographic holdout, and at 1.15M the
holdout kept only +0.41 pp. If the window closes before it runs, the sequence
number is still a complete paired result; the holdout is not.

    OSV_RELEASE=s10 py scripts/tilefull.py 10
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

REL = "s10"
if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = REL

import config                                            # noqa: E402
import overnight as O                                    # noqa: E402
from overnight import Stage, log, load_state, run_stage  # noqa: E402

# The extensions this adds, in the order `bank_ext70_meta.npz` records them.
# Stacking them in any other order builds a file of the right length whose
# rows name the wrong photographs.
EXTS = [("bank_ext2", "tile6_ext2", "b190"),
        ("bank_ext3", "tile6_ext3", "b265"),
        ("bank_ext4", "tile6_ext4", "b340")]
MERGED = "bank_ext70"          # == bank_ext ++ bank_ext2 ++ bank_ext3 ++ ext4
FINAL = "b340"                 # 400,180 + 750,000 already stacked + 3 x 750,000

# arm -> (stem the 1.15M stack already has, suffix for the pooled extensions,
#         whether the arm carries tiles, PCA basis, projected-file infix)
ARMS = {
    "pyrL0":   ("pyr_l0_b115",   "pyrl0",   False, "pyr768_l0_pca.npz",  "l0"),
    "pyrL0L1": ("pyr_l0l1_b115", "pyrl0l1", True,  "pyr768_mix_pca.npz", "l0l1"),
}


def pooled(ext, arm):
    """`bank_ext2_pyrl0` -- the extension stem must come first.

    `provenance.ext_stem_for` strips suffixes from the RIGHT to find the
    metadata that names an extension's images, so `pyr_l0_ext2` resolves to
    None and `stack_bank` refuses the arm with "no metadata found for
    extension". `bank_ext2_pyrl0` resolves to `bank_ext2`, which is the file
    that actually names these 750,000 ids.
    """
    return "{}_{}".format(ext, ARMS[arm][1])


def stack_at(arm, size):
    return "pyr_{}_{}".format(ARMS[arm][4], size)


def projected(arm):
    return "pyr768_{}_{}.f16.npy".format(ARMS[arm][4], FINAL)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return 0
    hours = float(arg) if arg else 10.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("tilefull: the paired tiles experiment at 3,400,180 bank rows")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    plan = []
    for arm, (base0, _sfx, tiles, basis, _ifx) in ARMS.items():
        # Pool each extension into the arm's 1536-d space. The tiles arm
        # refuses any extension whose tile cache is not complete, which is the
        # guard that makes an all-or-nothing cache safe to consume.
        for ext, tstem, _size in EXTS:
            argv = ["scripts/pool_pyramid.py",
                    "--src", ext + "_bal.f16.npy",
                    "--out", pooled(ext, arm) + ".f16.npy"]
            if tiles:
                argv += ["--tiles", tstem]
            plan.append(Stage("tilefull-pool-{}-{}".format(arm, ext), argv,
                              release=REL, est=6 * 60, retries=2,
                              critical=True))
        # Grow the stack one extension at a time, which is the documented
        # path: --base may itself be a stack, and each step records the
        # extension it appended so the next one can reconstruct the layout.
        base = base0
        for ext, _tstem, size in EXTS:
            out = stack_at(arm, size)
            plan.append(Stage(
                "tilefull-stack-{}-{}".format(arm, size),
                ["scripts/stack_bank.py", "--base", base,
                 "--ext", pooled(ext, arm), "--out", out],
                release=REL, est=12 * 60, retries=2, critical=True))
            base = out
        plan.append(Stage(
            "tilefull-proj-" + arm,
            ["scripts/project_street.py",
             "--src", stack_at(arm, FINAL) + ".f16.npy",
             "--out", projected(arm), "--dim", "768", "--basis", basis],
            release=REL, est=20 * 60, retries=2, critical=True))
        plan.append(Stage(
            "tilefull-knn-" + arm,
            ["scripts/build_knn.py", "--street-file", projected(arm),
             "--split-mode", "sequence", "--k", "32", "--bank-ext", MERGED],
            release=REL, est=60 * 60, retries=2, critical=True))

    # The question. Neighbour tables only -- minutes, no GPU.
    plan.append(Stage(
        "tilefull-knngap",
        ["scripts/knn_gap.py",
         "--a", config.knn_name(projected("pyrL0"), "sequence", ext=MERGED),
         "--b", config.knn_name(projected("pyrL0L1"), "sequence", ext=MERGED),
         "--ranks", "1,16,32", "--flows"],
        release=REL, est=8 * 60, retries=1))

    # The geographic holdout, planned last: it is two more full index builds
    # over the same 3.5M rows, and the sequence pair above is already a
    # complete result without it.
    for arm in ARMS:
        # critical for the same reason the sequence builds are: the cell8
        # comparison below reads these paths, and a previous run's caches are
        # usually already there, so a failure here does not stop the
        # comparison -- it changes which experiment the comparison is of.
        plan.append(Stage(
            "tilefull-knn-cell8-" + arm,
            ["scripts/build_knn.py", "--street-file", projected(arm),
             "--split-mode", "cell8", "--k", "32", "--bank-ext", MERGED],
            release=REL, est=60 * 60, retries=2, critical=True))
    plan.append(Stage(
        "tilefull-knngap-cell8",
        ["scripts/knn_gap.py",
         "--a", config.knn_name(projected("pyrL0"), "cell8", ext=MERGED),
         "--b", config.knn_name(projected("pyrL0L1"), "cell8", ext=MERGED),
         "--split-mode", "cell8", "--ranks", "1,16", "--flows"],
        release=REL, est=8 * 60, retries=1))

    # `Stage.critical` says "if it fails, dependents are dropped" and NOTHING
    # in this repository reads it -- not overnight.py, which sets it on six
    # stages, and not tilebig.py. It reads as protection and provides none: a
    # failed pool would have let its stack, projection and index run on
    # whatever was previously at those paths, and the comparison at the end
    # would have been published from a stale arm. Honour it here.
    failed = []
    for st in plan:
        if O.now() > deadline:
            log("deadline reached with {} stages unrun".format(
                len(plan) - plan.index(st)))
            break
        if not run_stage(st, state, deadline):
            failed.append(st.name)
            if st.critical:
                log("critical stage {} failed; everything after it reads what "
                    "it was supposed to write, and the arms cannot be "
                    "compared, so the remaining {} stages are dropped"
                    .format(st.name, len(plan) - plan.index(st) - 1))
                break
    # Signal completion on every path, not just the happy one: tilebig died on
    # a missing method AFTER every stage had finished, the completion line
    # never appeared, and the monitor watching for it never fired.
    O.SAMPLER.stop_flag.set()
    log("{} done. {} stages recorded, {} failed this run{}".format(
        "tilefull", len(state["done"]), len(failed),
        (": " + ", ".join(failed)) if failed else ""))
    # A runner that exits 0 after a stage failed tells a caller, a chain
    # script and a monitor that the measurement is ready. It is not.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
