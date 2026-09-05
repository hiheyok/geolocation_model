"""Does the tiles gain survive into the agent? Two ladders, one flag apart.

`runs/RESMATCH.md` decided the rebuild is tiles at 224 and `runs/XBANK.md`
measured the matched gain at **+2.63 pp [+1.3, +4.0]** on `<25 km`. Both are
**retrieval probes**: encode, cosine, nearest neighbour, great-circle error. No
gradient step anywhere. The agent consumes 16 neighbours through a *trained*
retrieval prior, and this project has twice had inference-level reasoning
predict the wrong sign -- most recently the clean-bank experiment, where I
argued 57.5% was a floor and it turned out to be a ceiling.

So before spending the extension pass on tiles, the cheap version of the
question gets asked: **on a bank that is already fully tiled, does a trained
agent do better?**

    bank      400,180 release train rows -- every one of them tiled in `tile6`
    arms      pyrL0   crops only
              pyrMIX  l2((L0 + L1)/2), the exact vector xbank measured
    ladder    e2 -> e4 -> e6, --init chaining, seqfix3's recipe unchanged

The two arms differ in **one** thing: whether `pool_pyramid.py` was given
`--tiles tile6`. Same crops cache, same pooling code, same PCA width fitted the
same way on the same training split, same kNN build, same `--seed 0`, same
`--neg-random`, same architecture, same schedule. `build_knn` excluded the same
789,748 same-sequence pairs in both, which is the check that matters: same
corpus and same sequences, so the exclusion must not depend on the embedding.

**This bank is 400,180 rows, not the shipping 3.4M**, so neither arm's absolute
number is comparable to anything on record -- corpus is the axis that has moved
this metric most. Only the paired contrast means anything here. And note it is
measured where the gain is *smallest*: `XBANK.md` §3 has the matched gain
growing with density, +1.80 pp at 25k rows to +2.60 at 400k.

    OSV_RELEASE=s10 py scripts/tiledrisk.py 4.0
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

ARMS = {
    "pyrL0": "pyr768_l0.f16.npy",
    "pyrMIX": "pyr768_mix.f16.npy",
}
LADDER = {name: ["{}-e{}".format(name, e) for e in (2, 4, 6)] for name in ARMS}


def rung(tag, street, init):
    """One rung. Identical to seqfix3's recipe except the street cache.

    `--neg-random` and `--seed 0` are both carried over deliberately: seqfix2's
    answer was voided by a `train.py` edit that changed the negative draw
    partway through a ladder, so the draw is pinned here rather than left to
    whatever the default is on the day.
    """
    a = ["src/train.py", "--tag", tag, "--epochs", "2",
         "--batch", "64", "--limit", "400000",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", "sequence", "--street-file", street,
         "--knn-file", config.knn_name(street, "sequence"),
         "--seed", "0", "--neg-random"] + ARCH + ["--retr-drop", "0.7"]
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
    log("tiledrisk: does +2.63 pp of retrieval become an agent gain?")
    log("400,180-row bank, fully tiled; one flag between the arms")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    plan = []
    # Interleaved, not one ladder then the other. If the window closes early
    # the surviving rungs are still a pair at some epoch count rather than one
    # complete arm and nothing to compare it against.
    for i in range(3):
        for name, street in ARMS.items():
            plan.append(Stage(
                "train-{}-e{}".format(name, 2 * (i + 1)),
                rung(LADDER[name][i], street,
                     LADDER[name][i - 1] if i else None),
                release=REL, est=28 * 60, retries=3))
    tags = [LADDER[n][-1] for n in ARMS] + [LADDER[n][-2] for n in ARMS]
    plan.append(Stage(
        "tiledrisk-boot",
        ["scripts/bootstrap.py", "--tags", ",".join(tags),
         "--split", "test", "--n", "5000", "--beam", "2", "--score-steps", "3",
         "--out", str(O.RUNS / "BOOTSTRAP_tiledrisk.md")],
        release=REL, est=20 * 60, retries=2))

    for st in plan:
        if O.now() > deadline:
            log("deadline reached with {} stages unrun".format(
                len(plan) - plan.index(st)))
            break
        run_stage(st, state, deadline)
    # `Sampler` exposes a `stop_flag` Event, not a `stop()` method. Calling the
    # method that does not exist raised AFTER every stage had finished, so no
    # result was lost -- but the script died on a traceback instead of logging
    # its completion line, and the monitor watching for that line never fired.
    # Nobody read the result for six hours and the GPU sat idle. A completion
    # signal that only appears on the happy path is not a completion signal.
    O.SAMPLER.stop_flag.set()
    log("{} done. {} stages recorded, {} failed".format(
        "tiledrisk", len(state["done"]), len(state["failed"])))


if __name__ == "__main__":
    main()
