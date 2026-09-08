"""Does the 3.4M tile gain survive training? The one question TILEFULL cannot answer.

`runs/TILEFULL.md` measured crops+tiles against crops on the neighbour tables
at 3,400,180 bank rows: **+3.72 pp [+3.4, +4.0]** at 25 km. That is retrieval
only. The agent consumes neighbours through a *trained* retrieval prior, so
only a retrain converts it into an agent number -- and this project has twice
had inference-level reasoning predict the wrong sign for a trained agent.

At 1,150,180 rows `runs/TILEBIG.md` found the gain not merely surviving but
growing: +3.30 pp on the table became **+2.76 to +4.88 pp** for the agent. It
also found **six epochs worse than four in both arms**, [-1.34,-0.14] and
[-1.28,-0.04] pp -- the third time a ladder here peaked before its last rung.
So this runs 2/4/6 and expects to read e4, with e6 as the check on that.

**The rung is byte-identical to tilebig's**, except for the street file, the
neighbour table and the extension stem. That is deliberate: the interesting
comparison is 1.15M against 3.4M, and it is only a comparison if the training
recipe is the same one. Same seed, same limit, same architecture, same
`--retr-drop 0.7`, same selection on hit rate rather than median
(the median swings 300 km between epochs).

**Arms are interleaved rung by rung**, so a window that closes early leaves a
comparable pair rather than one finished arm and one absent.

What this cannot settle: `cell8`. TILEFULL put the geographic gain at +0.49 pp
against +3.72 on the benchmark, and no agent ladder is run on the holdout here
-- same as TILEBIG, so the two are comparable, but do not read an agent number
below as evidence about unseen geography.

    OSV_RELEASE=s10 py scripts/tilefull_ladder.py 6
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
EPOCHS = (2, 4, 6)
LADDER = {n: ["{}-b340-e{}".format(n, e) for e in EPOCHS] for n in STREET}


def rung(tag, street, init):
    """Identical to tilebig's rung, so the two densities are comparable.

    The only differences are the street file, its neighbour table, and the
    extension stem -- the three things that ARE the manipulation. Changing
    anything else here would make the 1.15M number stop being a baseline.
    """
    a = ["src/train.py", "--tag", tag, "--epochs", "2",
         "--batch", "64", "--limit", "400000",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", "sequence", "--street-file", street,
         "--knn-file", config.knn_name(street, "sequence", ext=MERGED),
         "--seed", "0", "--neg-random"] + ARCH + ["--retr-drop", "0.7"]
    return a + (["--init", init] if init else [])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return 0
    hours = float(arg) if arg else 6.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("tilefull_ladder: does the 3.4M tile gain survive training?")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    plan = []
    for i, ep in enumerate(EPOCHS):
        for arm in STREET:
            plan.append(Stage(
                "train-{}-b340-e{}".format(arm, ep),
                rung(LADDER[arm][i], STREET[arm],
                     LADDER[arm][i - 1] if i else None),
                release=REL, est=30 * 60, retries=3, critical=True))
    # e4 and e6 for both arms: the pair TILEBIG reported, so the numbers line
    # up against its +2.76 to +4.88 pp.
    tags = [LADDER[n][1] for n in STREET] + [LADDER[n][2] for n in STREET]
    plan.append(Stage(
        "tilefull-ladder-boot",
        ["scripts/bootstrap.py", "--tags", ",".join(tags),
         "--split", "test", "--n", "5000", "--beam", "2", "--score-steps", "3",
         "--out", str(O.RUNS / "BOOTSTRAP_tilefull.md")],
        release=REL, est=25 * 60, retries=2))

    # `Stage.critical` is honoured here for the reason #57 established: each
    # rung is `--init` the previous one, so a failed e2 does not stop e4 -- it
    # silently trains e4 from scratch, or from an older checkpoint of the same
    # tag, and the ladder that gets published is not the ladder that ran.
    failed = []
    unrun = []
    for i, st in enumerate(plan):
        if O.now() > deadline:
            # Record what did not run, and exit non-zero for it. Breaking out
            # silently made a window that closed early indistinguishable from
            # a complete run: six rungs could succeed, the comparison never
            # happen, and the process still exit 0 -- telling a caller, a
            # chain script and a monitor that the measurement is ready.
            unrun = [s.name for s in plan[i:]]
            log("deadline reached with {} stages unrun: {}".format(
                len(unrun), ", ".join(unrun)))
            break
        if not run_stage(st, state, deadline):
            failed.append(st.name)
            if st.critical:
                log("critical stage {} failed; every later rung inits from it, "
                    "so the remaining {} stages are dropped".format(
                        st.name, len(plan) - plan.index(st) - 1))
                unrun = [s.name for s in plan[i + 1:]]
                break
    O.SAMPLER.stop_flag.set()
    log("{} done. {} stages recorded, {} failed this run{}".format(
        "tilefull_ladder", len(state["done"]), len(failed),
        (": " + ", ".join(failed)) if failed else ""))
    if unrun:
        log("{} stages did not run; this is not a completed measurement"
            .format(len(unrun)))
    return 1 if (failed or unrun) else 0


if __name__ == "__main__":
    sys.exit(main())
