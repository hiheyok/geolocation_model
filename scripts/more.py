"""The next point on the corpus curve: 15 more shards, 1.90M -> 2.65M bank.

Bank size just produced the largest single gain this project has measured --
1.15M -> 1.90M took the `sequence` test split from 7.7 km / 64.1% to 3.6 km /
70.6%, `<25 km` +6.50 pp [+5.50, +7.64].  It did **not** flatten: a
log-extrapolation predicted ~+9 pp and the observed +6.5 sits below that line
but well above the +2 to +4 that flattening would have looked like.  So the
cheapest next experiment is another block of shards, and this repeats the last
step exactly -- 15 shards, 750k images -- so the runtime is known and the two
points on the curve are directly comparable.

    step 1   1.15M -> 1.90M   x1.65   +6.50 pp
    step 2   1.90M -> 2.65M   x1.39   this run

Expect less than +6.5: the ratio is smaller and returns are sub-log.  If this
comes in under about +2 pp the corpus axis has finally flattened and the next
night belongs to architecture instead -- most likely the s1 policy failure in
`docs/STATE.md`, which is a defect rather than a scaling knob.

Budget: two encoder passes at the measured 153 and 181 img/s dominate, ~2.5 h,
then a 5 min index build and the same 2+2+2 ladder.  Disk cost about 28 GB
against 290 GB free; host RAM for the neighbour table goes 6.14 -> 8.45 GB.

Nothing here touches dataset.parquet or the split hash.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

REL = "s10"
if not os.environ.get("OSV_RELEASE"):   # config reads this at import time
    os.environ["OSV_RELEASE"] = REL

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, boot, tiles_up

EXT3 = "bank_ext3"
SHARDS3 = ",".join("{:02d}".format(i) for i in range(40, 55))
SRC_BANK = "pool_bal_bank40"          # 2.0M rows: 500k release + 1.5M ext
BANK55 = "pool_bal_bank55"            # 2.75M rows
META55 = "bank_ext55"
KNN55 = config.knn_name(BANK55 + ".f16.npy", "sequence", ext=META55)


def train(tag, init=None, epochs=2):
    cmd = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
           "--batch", "64", "--limit", "400000",
           "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
           "--split-mode", "sequence", "--street-file", BANK55 + ".f16.npy",
           "--knn-file", KNN55] + ARCH
    return cmd + (["--init", init] if init else [])


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 7.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("more: corpus 1.90M -> 2.65M, shards {}".format(SHARDS3))
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    pq = str(config.PROCESSED / (EXT3 + ".parquet"))
    plan = [
        (Stage("e3_meta", ["scripts/build_bank_ext.py", "--shards", SHARDS3,
                           "--out", EXT3], release=REL, est=5 * 60,
               retries=1), False),
        (Stage("e3_dino", ["scripts/embed_street.py", "--crops", "3",
                           "--model", "vit_base_patch14_dinov2.lvd142m",
                           "--parquet", pq, "--out", EXT3 + "_dino"],
               release=REL, est=90 * 60, retries=1), False),
        (Stage("e3_siglip", ["scripts/embed_street.py", "--crops", "3",
                             "--model", "vit_base_patch16_siglip_224.v2_webli",
                             "--parquet", pq, "--out", EXT3 + "_siglip"],
               release=REL, est=80 * 60, retries=1), False),
        (Stage("e3_join", ["scripts/concat_street.py", "--a", EXT3 + "_dino",
                           "--b", EXT3 + "_siglip", "--scale-b", "4.03",
                           "--out", EXT3 + "_bal"], release=REL, est=5 * 60,
               retries=1), False),
        (Stage("e3_pool", ["scripts/pool_street.py", "--src", EXT3 + "_bal.f16.npy",
                           "--out", EXT3 + "_pool.f16.npy"], release=REL,
               est=5 * 60, retries=1), False),
        (Stage("b55_stack", ["scripts/stack_bank.py", "--base", SRC_BANK,
                             "--ext", EXT3 + "_pool", "--out", BANK55],
               release=REL, est=10 * 60, retries=1), False),
        (Stage("b55_meta", ["scripts/merge_bank_meta.py", "--parts",
                            "bank_ext,bank_ext2,bank_ext3", "--out", META55,
                            "--emb", BANK55 + ".f16.npy"], release=REL,
               est=3 * 60, retries=1), False),
        (Stage("b55_knn", ["scripts/build_knn.py", "--street-file",
                           BANK55 + ".f16.npy", "--split-mode", "sequence",
                           "--k", "32", "--bank-ext", META55], release=REL,
               est=15 * 60, retries=1), False),
        (Stage("b55_2", train("d1536-b265-e2"), release=REL, est=30 * 60,
               retries=1), True),
        (Stage("b55_4", train("d1536-b265-e4", "d1536-b265-e2"), release=REL,
               est=30 * 60, retries=1), True),
        (Stage("b55_6", train("d1536-b265-e6", "d1536-b265-e4"), release=REL,
               est=30 * 60, retries=1), True),
        (Stage("b55_eval", boot("d1536-b115-e6,d1536-b190-e6,d1536-b265-e2,d1536-b265-e4,"
                                "d1536-b265-e6", "BOOTSTRAP_bank55.md"),
               release=REL, est=15 * 60, retries=1), True),
    ]

    for st, needs_tiles in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        if needs_tiles and not tiles_up():
            log("skip   {}  (tile server down)".format(st.name))
            continue
        if not run_stage(st, state, deadline) and st.name.startswith(
                ("e3_", "b55_stack", "b55_meta", "b55_knn")):
            # every later stage reads what this one writes
            log("stopping: {} is a prerequisite for everything after it"
                .format(st.name))
            break

    log("-" * 72)
    log("done; logs in {}".format(O.LOGS))
    if O.SAMPLER:
        log("system, whole run: " + O.SAMPLER.summary())


if __name__ == "__main__":
    main()
