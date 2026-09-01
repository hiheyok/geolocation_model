"""Everything that does not need the tile server, for the window without one.

Every training stage needs the map server, because `--select hit` decodes 2,000
validation images an epoch, and so does every beam evaluation. The server goes
away at 13:30. That does not mean the machine idles: the expensive part of this
project has never been the training, it is the encoder passes, and those touch
nothing but the GPU and the shard zips.

So the order inverts. Do the staging now, and leave the window when the server
comes back for training alone.

  1. **768-d projection and its index.** `width_probe.py` put the compression
     knee at 768 (-0.17 pp [-1.00, +0.63] against 1536, while 384 is separated
     at -0.87). Projecting and indexing are tile-free; only the confirmation
     ladder is not, and `half.py` will resume it.

  2. **The next shard block, 55-69.** Corpus is still the axis with the largest
     measured effect -- 1.15M -> 1.90M was worth +6.50 pp -- and the retrieval
     curve is log-linear with no knee in sight, so a fourth block is the
     highest-value thing the GPU can do unattended. Two encoder passes at the
     measured 153 and 181 img/s dominate at about 2.5 hours.

  3. **Screening the high-resolution harvest.** 1.45% of OSV-5M frames have
     their true GPS burned in as a dashcam overlay, worth about +1.3 pp if
     exploited. It is harmless today only because everything is resized to 224.
     Any high-resolution direction walks into it, so the harvest gets screened
     before its numbers are used, not after.

Nothing here touches dataset.parquet or the split hash. Every stage is
resumable through `runs/marks/`, so this can be re-run safely.
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

EXT4 = "bank_ext4"
SHARDS4 = ",".join("{:02d}".format(i) for i in range(55, 70))
BANK55 = "pool_bal_bank55.f16.npy"     # 2.75M x 1536, from more.py
W768 = "pca768_bank55.f16.npy"
BANK70 = "pool_bal_bank70"             # 3.50M x 1536
META70 = "bank_ext70"
HARVEST = "E:/data/kartaview_hr"


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    # optional second argument: run only these stages, so the tile-free 768
    # staging can go first, concurrently with a ladder that still has tiles
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("offline: 768 projection, shards 55-69, and the leak screen")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("no stage here needs the tile server")

    pq4 = str(config.PROCESSED / (EXT4 + ".parquet"))
    plan = []

    if (config.STREET_CACHE / BANK55).exists():
        plan += [
            Stage("w768_proj", ["scripts/project_street.py", "--src", BANK55,
                                "--out", W768, "--dim", "768"],
                  release=REL, est=20 * 60, retries=1),
            Stage("w768_knn", ["scripts/build_knn.py", "--street-file", W768,
                               "--split-mode", "sequence", "--k", "32",
                               "--bank-ext", "bank_ext55"],
                  release=REL, est=15 * 60, retries=1),
        ]
    else:
        log("{} missing -- the corpus run has not written it yet; the 768 "
            "stages are skipped and half.py will do them".format(BANK55))

    plan += [
        Stage("e4_meta", ["scripts/build_bank_ext.py", "--shards", SHARDS4,
                          "--out", EXT4], release=REL, est=5 * 60, retries=1),
        Stage("e4_dino", ["scripts/embed_street.py", "--crops", "3",
                          "--model", "vit_base_patch14_dinov2.lvd142m",
                          "--parquet", pq4, "--out", EXT4 + "_dino"],
              release=REL, est=90 * 60, retries=1),
        Stage("e4_siglip", ["scripts/embed_street.py", "--crops", "3",
                            "--model", "vit_base_patch16_siglip_224.v2_webli",
                            "--parquet", pq4, "--out", EXT4 + "_siglip"],
              release=REL, est=80 * 60, retries=1),
        Stage("e4_join", ["scripts/concat_street.py", "--a", EXT4 + "_dino",
                          "--b", EXT4 + "_siglip", "--scale-b", "4.03",
                          "--out", EXT4 + "_bal"], release=REL, est=5 * 60,
              retries=1),
        Stage("e4_pool", ["scripts/pool_street.py", "--src", EXT4 + "_bal.f16.npy",
                          "--out", EXT4 + "_pool.f16.npy"], release=REL,
              est=5 * 60, retries=1),
        Stage("b70_stack", ["scripts/stack_bank.py", "--base",
                            BANK55.replace(".f16.npy", ""),
                            "--ext", EXT4 + "_pool", "--out", BANK70],
              release=REL, est=15 * 60, retries=1),
        Stage("b70_meta", ["scripts/merge_bank_meta.py", "--parts",
                           "bank_ext,bank_ext2,bank_ext3,bank_ext4",
                           "--out", META70, "--emb", BANK70 + ".f16.npy"],
              release=REL, est=5 * 60, retries=1),
        Stage("b70_knn", ["scripts/build_knn.py", "--street-file",
                          BANK70 + ".f16.npy", "--split-mode", "sequence",
                          "--k", "32", "--bank-ext", META70],
              release=REL, est=20 * 60, retries=1),
        Stage("kv_screen", ["scripts/screen_leak.py", "--data", HARVEST,
                            "--n", "2000"], release=REL, est=20 * 60,
              retries=1),
    ]

    stop_after = ("e4_meta", "e4_dino", "e4_siglip", "e4_join", "e4_pool",
                  "b70_stack", "b70_meta")
    if only:
        plan = [st for st in plan if st.name in only]
        log("running only: {}".format(", ".join(st.name for st in plan)))
    for st in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        if not run_stage(st, state, deadline) and st.name in stop_after:
            log("stopping: {} is a prerequisite for everything after it"
                .format(st.name))
            break

    log("-" * 72)
    log("done; logs in {}".format(O.LOGS))
    log("when the tile server is back: scripts/half.py for the 768 ladder, "
        "then a b70 ladder for the 3.50M corpus")
    if O.SAMPLER:
        log("system, whole run: " + O.SAMPLER.summary())


if __name__ == "__main__":
    main()
