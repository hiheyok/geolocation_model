"""Eight hours, unattended, ordered so that a failure late costs nothing early.

Three blocks, deliberately in this order:

1. **Fusion head** -- self-contained, runs on cached tokens, no tile server, no
   network. Finishes the experiment that was mid-flight: the combination arm
   with an equal-bytes control, then three swings at the objective, which is
   where it actually failed. Global in-batch negatives taught it "same
   continent?" (+3.93 pp at 2500 km, -4.07 at 25 km); one-bucket negatives made
   the task so hard it never left chance (loss 5.08 against ln(256)=5.55). The
   sweeps look for the middle.

2. **Pooled representation on cell8** -- the one gap in a validated result. The
   1536-d pooled vector matched the 4608-d one on `sequence` at a third of the
   bank bytes, but was never checked on the geographic holdout, where the bank
   contributes very differently (+2.0 pp against +20.4 pp for corpus scaling).
   Uses only established scripts.

3. **Corpus extension** -- the largest prize by far, and the riskiest chain, so
   it goes last. Bank size is the axis with the biggest measured effect in this
   project (+20.4 pp for 400k -> 1.15M) and pooling just tripled the ceiling to
   3.75M at the same RAM. Five steps, each its own stage so partial progress is
   kept: metadata, two encoder passes, the equal-norm join, then pooling.
   Nothing here touches dataset.parquet or the split hash, so the benchmark
   stays intact whatever happens.

Every training stage needs the tile server, because `--select hit` decodes 2,000
validation images an epoch. The fusion-head block does not, so it survives the
server going away.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, boot, tiles_up

REL = "s10"
POOL_BANK = "pool_bal_bank25.f16.npy"
KNN_C8 = config.knn_name(POOL_BANK, "cell8", ext="bank_ext")
EXT2 = "bank_ext2"
SHARDS2 = ",".join("{:02d}".format(i) for i in range(25, 40))


def fuse(tag, extra=()):
    return ["scripts/fuse_head.py", "--epochs", "12", "--batch", "256",
            "--queries", "3000"] + list(extra)


def train_c8(tag, init=None, epochs=2):
    cmd = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
           "--batch", "64", "--limit", "400000", "--lr", "1e-4",
           "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
           "--split-mode", "cell8", "--street-file", POOL_BANK,
           "--knn-file", KNN_C8] + ARCH
    return cmd + (["--init", init] if init else [])


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("overnight: fusion head, pooled cell8, then corpus extension")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    up = tiles_up()
    log("tile server {}".format("up" if up else "DOWN -- training stages will "
                                "be skipped, the fusion head block will not"))

    plan = [
        # ---- block 1: fusion head, no external dependencies -------------
        (Stage("fh_combo", fuse("combo"), release=REL, est=12 * 60, retries=1),
         False),
        (Stage("fh_pos1", fuse("pos1", ["--pos-km", "1.0"]), release=REL,
               est=12 * 60, retries=1), False),
        (Stage("fh_tau", fuse("tau", ["--tau", "0.02"]), release=REL,
               est=12 * 60, retries=1), False),
        (Stage("fh_big", fuse("big", ["--d", "384", "--epochs", "24"]),
               release=REL, est=25 * 60, retries=1), False),

        # ---- block 2: the pooled vector on the geographic holdout --------
        (Stage("pc8_knn",
               ["scripts/build_knn.py", "--street-file", POOL_BANK,
                "--split-mode", "cell8", "--k", "32", "--bank-ext", "bank_ext"],
               release=REL, est=10 * 60, retries=1), False),
        (Stage("pc8_a", train_c8("d1536-b115-e2-cell8"), release=REL, est=30 * 60,
               retries=1), True),
        (Stage("pc8_b", train_c8("d1536-b115-e4-cell8", "d1536-b115-e2-cell8"), release=REL,
               est=30 * 60, retries=1), True),
        (Stage("pc8_c", train_c8("d1536-b115-e6-cell8", "d1536-b115-e4-cell8"),
               release=REL, est=30 * 60, retries=1), True),
        (Stage("pc8_eval",
               boot("d4608-b115-e4-cell8,d1536-b115-e2-cell8,d1536-b115-e4-cell8,"
                    "d1536-b115-e6-cell8", "BOOTSTRAP_pool_cell8.md"),
               release=REL, est=15 * 60, retries=1), True),

        # ---- block 3: more corpus, the biggest prize and the longest chain
        (Stage("ext2_meta",
               ["scripts/build_bank_ext.py", "--shards", SHARDS2,
                "--out", EXT2], release=REL, est=20 * 60, retries=1), False),
        (Stage("ext2_dino",
               ["scripts/embed_street.py", "--crops", "3",
                "--model", "vit_base_patch14_dinov2.lvd142m",
                "--parquet", str(config.PROCESSED / (EXT2 + ".parquet")),
                "--out", EXT2 + "_dino"], release=REL, est=75 * 60, retries=1),
         False),
        (Stage("ext2_siglip",
               ["scripts/embed_street.py", "--crops", "3",
                "--model", "vit_base_patch16_siglip_224.v2_webli",
                "--parquet", str(config.PROCESSED / (EXT2 + ".parquet")),
                "--out", EXT2 + "_siglip"], release=REL, est=75 * 60,
               retries=1), False),
        (Stage("ext2_join",
               ["scripts/concat_street.py", "--a", EXT2 + "_dino",
                "--b", EXT2 + "_siglip", "--scale-b", "4.03",
                "--out", EXT2 + "_bal"], release=REL, est=15 * 60,
               retries=1), False),
        (Stage("ext2_pool",
               ["scripts/pool_street.py", "--src", EXT2 + "_bal.f16.npy",
                "--out", EXT2 + "_pool.f16.npy"], release=REL, est=10 * 60,
               retries=1), False),
    ]

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
