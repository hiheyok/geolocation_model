"""Two questions that fit one window: is the schedule a confound, and do tiles read geography?

A window, not an experiment -- the GPU is serial and these are independent, so
they share the night. Ordered by value, because a deadline truncates the tail.

**1. Has every ladder here been mis-scheduled?**

Every ladder in this project is separate `--epochs 2` runs chained with
`--init`, and `--init` restores WEIGHTS ONLY: Adam's moments restart and the
LR schedule is a fresh cosine. `train.py` compensates -- with `--init` the
default lr drops to 1e-4 from 3e-4 and warmup to 100 steps, "because the
previous run annealed to zero" -- but that is still a re-heat of a just-
annealed model, three times over.

Five ladders have now peaked before their last rung, and it has always been
read as `epochs-do-not-substitute-for-data`. That reading has never been
separated from the schedule. `chain_lr.sh` tests the peak height of the last
rung; this tests the chaining itself: **one `--epochs 6` cosine from scratch**,
both arms, everything else identical.

    c6 beats the chained e6 and e4  -> the sawtooth was costing, and five
                                       results need re-reading
    c6 ties e4                      -> the model is done at 4 epochs and the
                                       schedule was never the issue

**2. Do tiles read geography, for the agent?**

`runs/TILEFULL.md` put the tile gain at +3.72 pp on the benchmark split and
**+0.49 pp on the cell8 geographic holdout** -- but both are RETRIEVAL. No
agent has ever been trained on cell8 with tiles, so the transfer question that
`benchmark-measures-bank-coverage` keeps raising has never been answered where
it matters. The caches exist (`knn_pyr768_*_b340_cell8_k32_bank_ext70.npz`),
so this is four rungs and a bootstrap.

Read it against the sequence result, not on its own: the interesting number is
whether the agent's cell8 gain is a seventh of its sequence gain, as retrieval
says it should be, or whether training changes that ratio.

**3. Finish the clean baseline.**

`shipclean-e2/e4` are running now -- the shipping street file trained against
the rebuilt cache, because every shipping-family checkpoint predates the
2026-09-04 04:05 rebuild. e6 completes that ladder so it can be compared rung
for rung with the pyramid arms.

    OSV_RELEASE=s10 py scripts/night0908.py 8
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
from marathon import ARCH                                # noqa: E402

MERGED = "bank_ext70"
STREET = {"pyrL0": "pyr768_l0_b340.f16.npy",
          "pyrL0L1": "pyr768_l0l1_b340.f16.npy"}


def rung(tag, street, mode, epochs, init=None, lr=None):
    """The ladder rung, with the schedule and split as parameters.

    `tilefull_ladder`'s rung with two knobs opened up. `--init` is omitted for
    the single-cosine arms, which is the whole point: without it `train.py`
    keeps lr 3e-4 and warmup 500 and anneals once over `--epochs`.
    """
    a = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
         "--batch", "64", "--limit", "400000",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", mode, "--street-file", street,
         "--knn-file", config.knn_name(street, mode, ext=MERGED),
         "--seed", "0", "--neg-random"] + ARCH + ["--retr-drop", "0.7"]
    if init:
        a += ["--init", init]
    if lr:
        a += ["--lr", str(lr)]
    return a


def boot(name, tags, out, needs=(), est=25 * 60):
    """A comparison, and the stages it is a comparison OF.

    `needs` is not decoration. A bootstrap reads checkpoints by tag, and a
    tag that failed to train usually still has a file -- the previous rung's,
    or an older run's. So a failed training stage does not stop its own
    comparison from producing a clean-looking report of the wrong thing.
    `critical` does not cover it: a non-critical failure is one the window can
    survive, not one its dependents can ignore.
    """
    st = Stage(name, ["scripts/bootstrap.py", "--tags", ",".join(tags),
                      "--split", "test", "--n", "5000", "--beam", "2",
                      "--score-steps", "3", "--out", str(O.RUNS / out)],
               release=REL, est=est, retries=2)
    st.prereqs = tuple(needs)
    return st


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return 0
    hours = float(arg) if arg else 8.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("night0908: schedule control, cell8 agent ladder, clean baseline e6")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    plan = []

    # 1. One cosine over six epochs, from scratch, both arms. ~90 min each.
    #    Critical: the comparison needs the pair, and the bootstrap after it
    #    reads both.
    for arm in ("pyrL0L1", "pyrL0"):
        plan.append(Stage(
            "night-c6-" + arm,
            rung(arm + "-b340-c6", STREET[arm], "sequence", 6),
            release=REL, est=95 * 60, retries=2, critical=True))
    plan.append(boot("night-boot-schedule",
                     ["pyrL0L1-b340-e4", "pyrL0L1-b340-e6", "pyrL0L1-b340-c6",
                      "pyrL0-b340-e4", "pyrL0-b340-e6", "pyrL0-b340-c6"],
                     "BOOTSTRAP_schedule.md",
                     needs=["night-c6-pyrL0L1", "night-c6-pyrL0"],
                     est=30 * 60))

    # 2. cell8 agent ladder, arms interleaved so an early close still leaves a
    #    comparable pair. Two rungs each: TILEBIG and this project's other
    #    ladders all peak at e4.
    for i, ep in enumerate((2, 4)):
        for arm in ("pyrL0", "pyrL0L1"):
            plan.append(Stage(
                "night-cell8-{}-e{}".format(arm, ep),
                rung("{}-c8-e{}".format(arm, ep), STREET[arm], "cell8", 2,
                     init=("{}-c8-e2".format(arm) if i else None)),
                release=REL, est=32 * 60, retries=2, critical=True))
    plan.append(boot("night-boot-cell8",
                     ["pyrL0-c8-e4", "pyrL0L1-c8-e4",
                      "pyrL0-c8-e2", "pyrL0L1-c8-e2"],
                     "BOOTSTRAP_cell8_tiles.md",
                     needs=["night-cell8-pyrL0-e2", "night-cell8-pyrL0L1-e2",
                            "night-cell8-pyrL0-e4", "night-cell8-pyrL0L1-e4"]))

    # 3. Cheapest and least decisive, so it goes last.
    plan.append(Stage(
        "night-shipclean-e6",
        rung("shipclean-e6", "pca768_bank70.f16.npy", "sequence", 2,
             init="shipclean-e4"),
        release=REL, est=32 * 60, retries=2))
    plan.append(boot("night-boot-shipclean6",
                     ["shipclean-e4", "shipclean-e6",
                      "pyrL0-b340-e4", "pyrL0L1-b340-e4"],
                     "BOOTSTRAP_shipclean6.md",
                     needs=["night-shipclean-e6"]))

    failed, unrun, skipped = [], [], []
    for i, st in enumerate(plan):
        gone = [d for d in getattr(st, "prereqs", ()) if d in failed]
        if gone:
            log("skip   {}  ({} failed, so this would compare whatever "
                "checkpoint happens to be on disk)".format(
                    st.name, ", ".join(gone)))
            skipped.append(st.name)
            continue
        if O.now() > deadline:
            unrun = [s.name for s in plan[i:]]
            log("deadline reached with {} stages unrun: {}".format(
                len(unrun), ", ".join(unrun)))
            break
        if not run_stage(st, state, deadline):
            failed.append(st.name)
            if st.critical:
                log("critical stage {} failed; its pair cannot be compared, "
                    "so the remaining {} stages are dropped".format(
                        st.name, len(plan) - i - 1))
                unrun = [s.name for s in plan[i + 1:]]
                break
    O.SAMPLER.stop_flag.set()
    log("night0908 done. {} stages recorded, {} failed this run{}".format(
        len(state["done"]), len(failed),
        (": " + ", ".join(failed)) if failed else ""))
    if skipped:
        log("{} stages skipped for a failed prerequisite: {}"
            .format(len(skipped), ", ".join(skipped)))
    if unrun:
        log("{} stages did not run; this is not a completed window"
            .format(len(unrun)))
    return 1 if (failed or unrun or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
