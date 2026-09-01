"""What to run once `tonight.py` is done, without waiting to be asked.

Two things the overnight plan left open, in order of what the evidence says is
worth the compute.

**1. The corpus, which is the whole point of block 3.**  Bank size is the axis
with the largest measured effect in this project: +20.4 pp for 400k -> 1.15M.
Block 3 embeds and pools 750k more images but stops there, so on its own it
changes no metric.  This runs the chain that turns those bytes into a number --
stack, merge the two extension metadata files, rebuild the neighbour index at
2.0M, then the same 2+2+2 ladder `poolbank.py` used, so the only thing that
differs from `s10_pool_c6` is how many images the bank holds.

Matched on schedule, not just on epochs: `s10_pool_c6` was 2 epochs, then +2
from a restart, then +2, and one long cosine is a different trajectory.

**2. Two controls the fusion-head block earned.**

`fh_pos1` is confounded and should not be read as it stands.  Tightening
positives from 5 km to 1 km cut the pair count 377,302 -> 25,532, so the arm
trained for 1,188 steps against 17,676 -- a 15x smaller budget.  It came out
worse, which is exactly what undertraining looks like.  `--epochs 179` restores
the step count at 99 steps/epoch, so radius and budget stop being the same
knob.

`fh_seed1` re-runs the best config under a different seed.  Every fusion-head
number so far comes from a single seed, and on this project seeds have differed
by 18 km on the agent's own metric.  `fh_combo` and `fh_tau` are separated from
the mean-pool baseline by a wide margin, but they are *not* separated from each
other, and a second seed is the cheapest thing that says whether that gap is
real or noise.

Deliberately **not** here: wiring the fusion head into the agent.  It needs tile
embeddings for the whole corpus, and at the measured 19 ms/image that is over
nine hours for 1.75M images -- a night of its own, not a tail-end stage.

Gated: the corpus block is skipped, with a line saying so, if block 3 did not
finish.  Nothing here touches dataset.parquet or the split hash.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

REL = "s10"

# BEFORE importing config, which reads this at import time and defaults to
# "s01".  Every Stage carries release=REL into its own subprocess, so the stages
# were always right; what was wrong was this process, whose STREET_CACHE
# decides the `have_ext2` gate below.  Launched without the variable it looked
# in cache/street/s01, found no bank_ext2_pool, and skipped the corpus block --
# the largest piece of work of the night -- with a message saying block 3 had
# not finished, when it had.  Mixing releases is silent, not loud, so the
# default is pinned here rather than left to the caller.
if not os.environ.get("OSV_RELEASE"):   # unset *or* empty
    os.environ["OSV_RELEASE"] = REL

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, boot, tiles_up

SRC_BANK = "pool_bal_bank25"          # 500k release + 750k bank_ext, pooled
EXT2_POOL = "bank_ext2_pool"          # 750k more, pooled, from block 3
BANK40 = "pool_bal_bank40"            # the two stacked: 2.0M rows at 1536-d
META40 = "bank_ext40"
KNN40 = config.knn_name(BANK40 + ".f16.npy", "sequence", ext=META40)


def train(tag, init=None, epochs=2):
    cmd = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
           "--batch", "64", "--limit", "400000",
           "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
           "--split-mode", "sequence", "--street-file", BANK40 + ".f16.npy",
           "--knn-file", KNN40] + ARCH
    return cmd + (["--init", init] if init else [])


def fuse(extra):
    return ["scripts/fuse_head.py", "--batch", "256",
            "--queries", "3000"] + list(extra)


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("after: corpus to 2.0M, then the two fusion-head controls")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    have_ext2 = (config.STREET_CACHE / (EXT2_POOL + ".f16.npy")).exists()
    log("{} {}".format(EXT2_POOL,
                       "present -- corpus block runs" if have_ext2 else
                       "MISSING -- block 3 did not finish, corpus block skipped"))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    corpus = [
        (Stage("b40_stack",
               ["scripts/stack_bank.py", "--base", SRC_BANK,
                "--ext", EXT2_POOL, "--out", BANK40],
               release=REL, est=15 * 60, retries=1), False),
        (Stage("b40_meta",
               ["scripts/merge_bank_meta.py", "--parts", "bank_ext,bank_ext2",
                "--out", META40, "--emb", BANK40 + ".f16.npy"],
               release=REL, est=3 * 60, retries=1), False),
        (Stage("b40_knn",
               ["scripts/build_knn.py", "--street-file", BANK40 + ".f16.npy",
                "--split-mode", "sequence", "--k", "32",
                "--bank-ext", META40], release=REL, est=30 * 60, retries=1),
         False),
        (Stage("b40_2", train("s10_b40"), release=REL, est=30 * 60,
               retries=1), True),
        (Stage("b40_4", train("s10_b40_c4", "s10_b40"), release=REL,
               est=30 * 60, retries=1), True),
        (Stage("b40_6", train("s10_b40_c6", "s10_b40_c4"), release=REL,
               est=30 * 60, retries=1), True),
        (Stage("b40_eval",
               boot("s10_bal_bank25_c6,s10_pool_c6,s10_b40,s10_b40_c4,"
                    "s10_b40_c6", "BOOTSTRAP_bank40.md"),
               release=REL, est=15 * 60, retries=1), True),
    ]

    controls = [
        # step-matched against fh_combo's 17,676: 25,532 pairs / 256 = 99/epoch
        (Stage("fh_pos1m", fuse(["--pos-km", "1.0", "--epochs", "179"]),
               release=REL, est=15 * 60, retries=1), False),
        (Stage("fh_seed1", fuse(["--tau", "0.02", "--epochs", "12",
                                 "--seed", "1"]),
               release=REL, est=15 * 60, retries=1), False),
    ]

    plan = (corpus if have_ext2 else []) + controls
    for st, needs_tiles in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        if needs_tiles and not tiles_up():
            log("skip   {}  (tile server down)".format(st.name))
            continue
        run_stage(st, state, deadline)

    log("-" * 72)
    log("done; logs in {}".format(O.LOGS))
    if O.SAMPLER:
        log("system, whole run: " + O.SAMPLER.summary())


if __name__ == "__main__":
    main()
