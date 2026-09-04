"""Review item 29, measured: does decaying the learned retrieval keys matter?

The substring rule `"pos" in name` exempted `retr.q_pos.weight` and
`retr.k_pos.weight` from weight decay -- 196,608 parameters, 3.7% of the model,
and the learned retrieval keys, worth +4.1 pp on record.  So the branch
`--retr-drop` exists to regularise was the branch training with no decay.

**Both arms are trained fresh, under one seed.**  Not the shipping checkpoint
against a new one: all 122 checkpoints on file predate `--seed`, so an old arm
differs from a new one in the seed as well as the rule, and the seed is not
small here -- two seeds of the same configuration differ by 18 km on the median
(`select-on-hit-rate-not-median`).  Pairing a seed difference with a treatment
difference is the mistake the corpus caveat made, where the error turned out to
be correlated with the treatment.  So: same seed, same data, same ladder, one
thing different.

The ladder is reproduced exactly as the shipping arm was built -- three rungs
of two epochs with `--init` between them, not one six-epoch run -- because that
is what `d768-b350-e6-drop70` is, and a re-baseline that changes the schedule
measures the schedule too.

Roughly 28 minutes a rung, so about three hours for both ladders, then the
paired bootstrap on the sequence benchmark and the external KartaView set.

    python scripts/wd29.py 6          # hours of deadline
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
from overnight import Stage, log, load_state, save_state, run_stage  # noqa: E402
from marathon import ARCH, tiles_up                      # noqa: E402

DST = "pca768_bank70.f16.npy"
META = "bank_ext70"
KNN = config.knn_name(DST, "sequence", ext=META)
SEED = "0"


def rung(tag, init, legacy):
    """One two-epoch rung of the d768/3.50M/p=0.7 ladder."""
    a = ["src/train.py", "--tag", tag, "--epochs", "2",
         "--batch", "64", "--limit", "400000",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", "sequence", "--street-file", DST,
         "--knn-file", KNN, "--seed", SEED] + ARCH + ["--retr-drop", "0.7"]
    if init:
        a += ["--init", init]
    if legacy:
        a += ["--wd-legacy"]
    return a


def ladder(prefix, legacy):
    """e2 -> e4 -> e6, each rung initialised from the one before."""
    out, prev = [], None
    for e in (2, 4, 6):
        tag = "{}-e{}".format(prefix, e)
        out.append(Stage("train-" + tag, rung(tag, prev, legacy),
                         release=REL, est=28 * 60, retries=3))
        prev = tag
    return out


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("wd29: does decaying the learned retrieval keys change anything?")
    log("both ladders fresh at seed {}, so the seed is not confounded with "
        "the rule".format(SEED))
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))
    if not (config.STREET_CACHE / DST).exists():
        log("{} missing -- run scripts/w768b70.py first".format(DST))
        return

    # Fixed rule first. If only one ladder finishes, the one worth having is
    # the one that tells us what the code now does.
    plan = ladder("wd29-fix", legacy=False) + ladder("wd29-legacy", legacy=True)

    plan.append(Stage(
        "wd29-boot",
        ["scripts/bootstrap.py", "--tags",
         "wd29-fix-e6,wd29-legacy-e6,d768-b350-e6-drop70",
         "--split", "test", "--n", "5000",
         "--out", str(O.RUNS / "BOOTSTRAP_wd29.md")],
        release=REL, est=20 * 60, retries=2))

    for tag in ("wd29-fix-e6", "wd29-legacy-e6"):
        plan.append(Stage(
            "wd29-hr-" + tag.split("-")[1],
            ["scripts/eval_highres.py", "--tag", tag, "--n", "5000",
             "--export", str(O.RUNS / ("wd29_" + tag.split("-")[1] + ".npz"))],
            release=REL, est=25 * 60, retries=2))

    for st in plan:
        if O.now() > deadline:
            log("deadline reached with {} stages unrun".format(
                len(plan) - plan.index(st)))
            break
        run_stage(st, state, deadline)

    log("done. {} stages recorded, {} failed".format(
        len(state["done"]), len(state["failed"])))


if __name__ == "__main__":
    main()
