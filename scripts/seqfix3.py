"""Re-run the clean-bank ladder with one variable instead of two.

`scripts/seqfix2.py` asked the question that turns 57.5% from a lower bound
into a number: every arm on record was *trained* against a bank that served
same-drive frames as neighbours, so what does an arm trained against a clean
one score? It answered -2.5 to -4.8 pp, separated, and that answer is void.

**Why it is void.** I committed a change to `train.py` at 12:32:53 that seeds
sink negatives from `--seed` instead of OS entropy. `train-clean-e4` launched
at 12:33:38. The rungs therefore disagree with each other:

    clean-drop70-e2   written 12:33:37   neg_random=None    (OS entropy)
    clean-drop70-e4   written 12:47:25   neg_random=False   (seeded)
    clean-drop70-e6   written 13:28:15   neg_random=False   (seeded)

Seeded negatives are a pure function of (split, seed, index), so an image sees
the same four off-path tiles every epoch rather than fresh ones each time --
the tradeoff named in `GeoStepDataset.neg_rng`. Across e4 and e6 the sink class
saw 1.6M distinct negatives repeated four times where the reference arms saw
6.4M distinct. That is a plausible cause of both the decline from 54.65% at e2
to 54.10% at e6 and of the deficit itself, and it is a fact about my edit
rather than about leaky versus clean banks.

So this ladder passes `--neg-random`, which is what every arm on record used.
The bank it trained against is then the only thing separating `clean2-drop70-e6`
from `wd29-fix-e6`: same seed, same schedule, same width, same corpus, same
weight-decay rule, same negative draw.

All three rungs are bootstrapped, not just the last. The first ladder peaked at
e2 and declined by e6, so reporting only e6 would understate a clean-trained
arm even once the confound is gone -- and if that shape survives here it is
itself the result, since every reference arm is an e6.

    OSV_RELEASE=s10 python scripts/seqfix3.py 4
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

DST = "pca768_bank70.f16.npy"
KNN = config.knn_name(DST, "sequence", ext="bank_ext70")

LADDER = ["clean2-drop70-e2", "clean2-drop70-e4", "clean2-drop70-e6"]
REFS = ["wd29-fix-e6", "d768-b350-e6-drop70"]


def rung(tag, init):
    """One rung, against the CLEAN cache, with negatives drawn as on record."""
    a = ["src/train.py", "--tag", tag, "--epochs", "2",
         "--batch", "64", "--limit", "400000",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", "sequence", "--street-file", DST,
         "--knn-file", KNN, "--seed", "0",
         # the point of this run: match the reference arms' negative draw
         "--neg-random"] + ARCH + ["--retr-drop", "0.7"]
    return a + (["--init", init] if init else [])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return
    hours = float(arg) if arg else 4.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("seqfix3: the clean-bank ladder, with the negative draw held fixed")
    log("seqfix2's answer was confounded by a train.py edit mid-ladder")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    plan = [
        Stage("train-clean2-e2", rung(LADDER[0], None),
              release=REL, est=28 * 60, retries=3),
        Stage("train-clean2-e4", rung(LADDER[1], LADDER[0]),
              release=REL, est=28 * 60, retries=3),
        Stage("train-clean2-e6", rung(LADDER[2], LADDER[1]),
              release=REL, est=28 * 60, retries=3),
        # The whole ladder against both references, so the epoch shape is
        # visible rather than assumed from the val selection sample.
        Stage("seqfix3-boot",
              ["scripts/bootstrap.py", "--tags", ",".join(LADDER + REFS),
               "--split", "test", "--n", "5000",
               "--out", str(O.RUNS / "BOOTSTRAP_cleantrain2.md")],
              release=REL, est=20 * 60, retries=2),
        # KartaView is the selection benchmark: it has no same-drive frames in
        # the bank, so it is the one measurement the leak never touched.
        Stage("seqfix3-hr",
              ["scripts/eval_highres.py", "--tag", LADDER[2],
               "--n", "5000",
               "--export", str(O.RUNS / "hr_clean2drop70.npz")],
              release=REL, est=25 * 60, retries=2),
    ]

    for st in plan:
        if O.now() > deadline:
            log("deadline reached with {} stages unrun".format(
                len(plan) - plan.index(st)))
            break
        ok = run_stage(st, state, deadline)
        if not ok and getattr(st, "critical", False):
            log("critical stage {} failed; stopping".format(st.name))
            break

    log("done. {} recorded, {} failed".format(
        len(state["done"]), len(state["failed"])))


if __name__ == "__main__":
    main()
