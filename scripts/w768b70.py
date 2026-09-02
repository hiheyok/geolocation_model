"""768-d on the 3.50M bank: the width reduction where it actually matters.

At 2.65M the projection to 768 cost +1.05 pp [+0.28, +1.82] -- separated, but
small. The reason to want it is not accuracy, it is that at 1536-d a 3.50M table
is 10.75 GB and a 4.90M one is 15.05 GB, neither of which can ever be resident
on a 31.7 GB machine: the tier needs `0.4 x free` to exceed the table, which is
unreachable. `d1536-b350-e6` trained tonight on the memmap tier for exactly that
reason. At 768-d the same corpus is 5.38 GB and stays in RAM, and the whole of
OSV-5M would be 7.53 GB.

**Prediction, recorded before the run.** The width cost should be roughly
constant in the bank, since it is a property of the projection rather than of
the corpus: **~1 pp, so about 75.2% and a ~2.0 km median** against
`d1536-b350-e6` at 1.8 km / 76.2%. If it comes in much worse, the cost grows
with corpus and 768-d stops being the way to reach 4.90M; if much better, the
projection is cheaper on a denser bank and the 2.65M measurement was pessimistic.

**One basis, not two.** The projection reuses `pca768_bank55_pca.npz` rather
than fitting fresh. It was fitted on the release rows, which are the same 500k
rows either way, so a refit would be equivalent -- but reusing keeps
`d768-b265-*` and `d768-b350-*` in one space, so the two are directly
comparable rather than each being right only against its own bank.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

REL = "s10"
if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = REL

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, tiles_up

SRC = "pool_bal_bank70.f16.npy"         # 3.50M x 1536
DST = "pca768_bank70.f16.npy"           # 3.50M x 768, 5.38 GB
BASIS = "pca768_bank55_pca.npz"
META = "bank_ext70"
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
    log("w768b70: 768-d on the 3.50M bank, predicted ~1 pp under d1536-b350-e6")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))
    if not (config.STREET_CACHE / SRC).exists():
        log("{} missing".format(SRC))
        return

    plan = [
        (Stage("w70_proj",
               ["scripts/project_street.py", "--src", SRC, "--out", DST,
                "--dim", "768", "--basis", BASIS],
               release=REL, est=10 * 60, retries=1), False),
        (Stage("w70_knn",
               ["scripts/build_knn.py", "--street-file", DST,
                "--split-mode", "sequence", "--k", "32", "--bank-ext", META],
               release=REL, est=15 * 60, retries=1), False),
        (Stage("w70_2", train("d768-b350-e2"), release=REL, est=25 * 60,
               retries=1), True),
        (Stage("w70_4", train("d768-b350-e4", "d768-b350-e2"), release=REL,
               est=25 * 60, retries=1), True),
        (Stage("w70_6", train("d768-b350-e6", "d768-b350-e4"), release=REL,
               est=25 * 60, retries=1), True),
        (Stage("w70_eval",
               ["scripts/boot_existing.py", "--tags",
                "d1536-b350-e6,d768-b265-e6,d768-b350-e2,d768-b350-e4,"
                "d768-b350-e6",
                "--out", str(O.RUNS / "BOOTSTRAP_w768_b70.md")],
               release=REL, est=5 * 60, retries=1), True),
    ]

    for st, needs_tiles in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        if needs_tiles and not tiles_up():
            log("skip   {}  (tile server down)".format(st.name))
            continue
        if not run_stage(st, state, deadline) and st.name in ("w70_proj",
                                                              "w70_knn"):
            log("stopping: {} is a prerequisite for everything after it"
                .format(st.name))
            break

    log("-" * 72)
    log("done; logs in {}".format(O.LOGS))
    if O.SAMPLER:
        log("system, whole run: " + O.SAMPLER.summary())


if __name__ == "__main__":
    main()
