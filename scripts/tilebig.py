"""Tile the first bank extension, then repeat the paired experiment at 1.15M.

`scripts/tiledrisk.py` asks whether crops+tiles helps a *trained* agent on a
400,180-row bank. This carries the same pair to **1,150,180 rows** -- the
release's 400,180 train rows plus `bank_ext.parquet`'s 750,000 -- which is the
largest bank that can be tiled inside one window.

**Completeness is why it stops there.** A partly-tiled bank is not a smaller
tiled bank, it is a broken one: tiled rows are `l2((L0+L1)/2)`, untiled rows are
`L0`, and one cosine ranks both, so an untiled row has the query's `L1` half
scored against its `L0` -- cross-space, systematically lower, silently demoted.
So the extension is taken one whole parquet at a time.

Two things this settles that 400k cannot.

  * **Does the gain grow with density?** `runs/XBANK.md` §3 measured it growing
    (+1.80 pp at 25k rows to +2.60 at 400k) and that was the entire argument for
    paying for a bigger tiled corpus. **That argument is now in doubt**:
    `runs/GAIN_DENSITY.md` tested the mechanism behind it -- coverage giving way
    to discrimination -- by varying *local* density inside one fixed bank, and
    the gain is flat across four orders of magnitude of it. Local and global
    density are different manipulations, so that is not a refutation, but it is
    the only independent check and it comes back flat. 1.15M is therefore the
    first point that **measures** the extrapolation rather than assuming it, and
    `knn_gap` answers it within minutes of the caches existing, long before the
    ladders finish.
  * **Does it still transfer to the agent when the bank is 2.9x denser?** A
    denser bank is limited by discrimination rather than coverage, which is
    where finer features are supposed to help most.

The extension is embedded with `autocast`, matching `tile6`. `runs/TILEBENCH.md`
measured every faster regime moving the nearest neighbour for 0.8-1.0% of
queries, against a ceiling of 1.218x -- so extending a cache in a different
numeric regime buys about 45 minutes here and costs a permanent, undetectable
seam through the middle of the bank.

Alignment checked before any of this was written: `bank_ext_meta.npz` and
`bank_ext.parquet` hold the same 750,000 `image_id` values in the same order
(`np.array_equal`: True), so the tile cache and `bank_ext_dual` index the same
photographs row for row. Getting that wrong is not an error, it is a bank that
pairs each image's crops with another image's tiles.

    OSV_RELEASE=s10 py scripts/tilebig.py 9.0
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
from marathon import ARCH, tiles_up                      # noqa: E402

EXT_PQ = "bank_ext.parquet"
EXT_TILES = "tile6_ext"
EXT_ROWS = 750000

# arm -> (release 1536-d cache, extension 1536-d cache, tile stem or None)
ARMS = {
    "pyrL0": ("pyr_l0", "pyr_l0_ext", None),
    "pyrMIX": ("pyr_mix", "pyr_mix_ext", EXT_TILES),
}
LADDER = {n: ["{}115-e{}".format(n, e) for e in (2, 4, 6)] for n in ARMS}


def stacked(arm):
    return "{}_b115".format(ARMS[arm][0])


def projected(arm):
    return "{}768_b115.f16.npy".format(arm.lower())


def rung(tag, street, init):
    """Identical to tiledrisk's rung, so the two densities are comparable."""
    a = ["src/train.py", "--tag", tag, "--epochs", "2",
         "--batch", "64", "--limit", "400000",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", "sequence", "--street-file", street,
         "--knn-file", config.knn_name(street, "sequence", ext="bank_ext"),
         "--seed", "0", "--neg-random"] + ARCH + ["--retr-drop", "0.7"]
    return a + (["--init", init] if init else [])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return
    hours = float(arg) if arg else 9.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("tilebig: the paired tiles experiment at 1,150,180 bank rows")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    plan = []
    # First, and cheap: does the gain survive a geographic holdout? Everything
    # measured so far is on `sequence`, which is the benchmark split, but §8c
    # found the 25 km gain 5.6x larger there than on `cell8` -- most of it is
    # same-region matching that a geographic split strips out. These reuse the
    # 400k caches already on disk, so it is two kNN builds and a lookup, and it
    # runs before the 4 h tile pass rather than after it.
    #
    # Read the ABSOLUTE cell8 numbers with care: the PCA basis these caches
    # were projected with was fitted on the *sequence* train split, part of
    # which is cell8's test side, so it is mildly transductive here. Both arms
    # share one basis, so the paired contrast -- the only thing being claimed
    # -- is unaffected; the levels are slightly optimistic.
    for arm, street in (("pyrL0", "pyr768_l0.f16.npy"),
                        ("pyrMIX", "pyr768_mix.f16.npy")):
        plan.append(Stage(
            "tilebig-knn-cell8-" + arm,
            ["scripts/build_knn.py", "--street-file", street,
             "--split-mode", "cell8", "--k", "32"],
            release=REL, est=8 * 60, retries=2))
    plan.append(Stage(
        "tilebig-knngap-cell8",
        ["scripts/knn_gap.py",
         "--a", config.knn_name("pyr768_l0.f16.npy", "cell8"),
         "--b", config.knn_name("pyr768_mix.f16.npy", "cell8"),
         "--split-mode", "cell8", "--ranks", "1,16", "--flows"],
        release=REL, est=5 * 60, retries=1))
    plan += [
        # ~4 h. autocast, matching tile6 -- see the module docstring.
        Stage("tilebig-tiles",
              ["scripts/tile_cache.py", "--parquet", EXT_PQ,
               "--out", EXT_TILES, "--n", str(EXT_ROWS), "--grid", "3x2"],
              release=REL, est=4 * 3600, retries=2, critical=True),
    ]
    for arm, (rel_c, ext_c, tstem) in ARMS.items():
        pool = ["scripts/pool_pyramid.py", "--src", "bank_ext_dual.f16.npy",
                "--out", ext_c + ".f16.npy"]
        if tstem:
            pool += ["--tiles", tstem]
        plan.append(Stage("tilebig-pool-" + arm, pool,
                          release=REL, est=5 * 60, retries=2, critical=True))
        plan.append(Stage(
            "tilebig-stack-" + arm,
            ["scripts/stack_bank.py", "--base", rel_c, "--ext", ext_c,
             "--out", stacked(arm)],
            release=REL, est=10 * 60, retries=2, critical=True))
        # Reuse the basis fitted on the release's training rows for this arm,
        # rather than refitting over the stack. One basis for every row is the
        # whole point: fitting per file would put the two corpora in different
        # spaces, and the failure would look like a weaker bank, not an error.
        plan.append(Stage(
            "tilebig-proj-" + arm,
            ["scripts/project_street.py", "--src", stacked(arm) + ".f16.npy",
             "--out", projected(arm), "--dim", "768",
             "--basis", "pyr768_{}_pca.npz".format(
                 "l0" if arm == "pyrL0" else "mix")],
            release=REL, est=10 * 60, retries=2, critical=True))
        plan.append(Stage(
            "tilebig-knn-" + arm,
            ["scripts/build_knn.py", "--street-file", projected(arm),
             "--split-mode", "sequence", "--k", "32",
             "--bank-ext", "bank_ext"],
            release=REL, est=20 * 60, retries=2, critical=True))
    # Cheap and decisive: does the gain grow with density, as the 25k -> 400k
    # series predicts? Runs off the neighbour tables alone, minutes, no GPU.
    plan.append(Stage(
        "tilebig-knngap",
        ["scripts/knn_gap.py",
         "--a", config.knn_name(projected("pyrL0"), "sequence", ext="bank_ext"),
         "--b", config.knn_name(projected("pyrMIX"), "sequence", ext="bank_ext"),
         "--ranks", "1,16,32"],
        release=REL, est=5 * 60, retries=1))
    # Interleaved, so a window that closes early leaves a comparable pair.
    for i in range(3):
        for arm in ARMS:
            plan.append(Stage(
                "train-{}115-e{}".format(arm, 2 * (i + 1)),
                rung(LADDER[arm][i], projected(arm),
                     LADDER[arm][i - 1] if i else None),
                release=REL, est=30 * 60, retries=3))
    tags = [LADDER[n][-1] for n in ARMS] + [LADDER[n][-2] for n in ARMS]
    plan.append(Stage(
        "tilebig-boot",
        ["scripts/bootstrap.py", "--tags", ",".join(tags),
         "--split", "test", "--n", "5000", "--beam", "2", "--score-steps", "3",
         "--out", str(O.RUNS / "BOOTSTRAP_tilebig.md")],
        release=REL, est=20 * 60, retries=2))

    for st in plan:
        if O.now() > deadline:
            log("deadline reached with {} stages unrun".format(
                len(plan) - plan.index(st)))
            break
        run_stage(st, state, deadline)
    O.SAMPLER.stop()
    log("tilebig done")


if __name__ == "__main__":
    main()
