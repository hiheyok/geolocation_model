"""Does 768-d survive into the agent's own metric?

`scripts/width_probe.py` says a fitted PCA to 768 costs -0.17 pp [-1.00, +0.63]
on any-of-32 `<25 km` -- inside noise -- while 384 is separated at -0.87. So the
knee is at 768, and at 2.65M images that is 4.07 GB instead of 8.14, meaning the
same host RAM would hold about 5.3M images.

That is a retrieval-level result, and this project's rule is that those do not
count until a training run agrees. Pooling 4608 -> 1536 also measured free at
retrieval level and still had to be confirmed twice, on `sequence` and again on
`cell8`, before anything was built on it. The failure mode is specific: the
agent does not use the street vector the way a cosine does -- it goes through
`StreetProj` and the retrieval keys, both of which are shaped by the input
width -- so a projection can preserve neighbour ranking and still cost the
policy something.

Matched on schedule as well as on epochs, the same 2+2+2 ladder every arm in
this comparison has used, against the same 2.65M corpus. The only difference
from `s10_b55_c6` is the width of the street vector.

Read it the way the pooling runs were read: 768 does not need to win. It needs
to not lose by more than the noise floor, about 1 pp of hit rate at n=5,000,
because "the same, at half the memory" is the result worth having.
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

SRC = "pool_bal_bank55.f16.npy"        # 2.75M x 1536
DST = "pca768_bank55.f16.npy"          # 2.75M x 768
META = "bank_ext55"
KNN = config.knn_name(DST, "sequence", ext=META)


def train(tag, init=None, epochs=2):
    cmd = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
           "--batch", "64", "--limit", "400000",
           "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
           "--split-mode", "sequence", "--street-file", DST,
           "--knn-file", KNN] + ARCH
    return cmd + (["--init", init] if init else [])


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("half: street vector 1536 -> 768 on the 2.65M bank")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    if not (config.STREET_CACHE / SRC).exists():
        log("{} missing -- the corpus run did not finish; nothing to do"
            .format(SRC))
        return

    plan = [
        (Stage("w768_proj",
               ["scripts/project_street.py", "--src", SRC, "--out", DST,
                "--dim", "768"], release=REL, est=15 * 60, retries=1), False),
        (Stage("w768_knn",
               ["scripts/build_knn.py", "--street-file", DST,
                "--split-mode", "sequence", "--k", "32",
                "--bank-ext", META], release=REL, est=15 * 60, retries=1),
         False),
        (Stage("w768_2", train("s10_w768"), release=REL, est=20 * 60,
               retries=1), True),
        (Stage("w768_4", train("s10_w768_c4", "s10_w768"), release=REL,
               est=20 * 60, retries=1), True),
        (Stage("w768_6", train("s10_w768_c6", "s10_w768_c4"), release=REL,
               est=20 * 60, retries=1), True),
        # boot_existing, not boot: if the last rung is dropped for time its
        # checkpoint will not exist, and bootstrap.py would fail on the missing
        # tag -- losing the arms that did finish, inside the last minutes the
        # tile server is up.
        (Stage("w768_eval",
               ["scripts/boot_existing.py", "--tags",
                "s10_b40_c6,s10_b55_c6,s10_w768,s10_w768_c4,s10_w768_c6",
                "--out", str(O.RUNS / "BOOTSTRAP_w768.md")],
               release=REL, est=2 * 60, retries=1), True),
    ]

    for st, needs_tiles in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        if needs_tiles and not tiles_up():
            log("skip   {}  (tile server down)".format(st.name))
            continue
        if not run_stage(st, state, deadline) and st.name in ("w768_proj",
                                                              "w768_knn"):
            log("stopping: {} is a prerequisite for everything after it"
                .format(st.name))
            break

    log("-" * 72)
    log("done; logs in {}".format(O.LOGS))
    if O.SAMPLER:
        log("system, whole run: " + O.SAMPLER.summary())


if __name__ == "__main__":
    main()
