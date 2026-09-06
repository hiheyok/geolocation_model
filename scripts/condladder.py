"""Does native-resolution detail help when it rides BESIDE retrieval?

`runs/PYR_LEVELS.md` closed the obvious route. Pushed *into* the retrieval
vector against a fixed `L0L1` bank, an extra level at weight 0.10 replaces 8.2%
of the top-16 and flips the top-1 for 12.7% of queries -- and the hit rate does
not move at all, 59.6% -> 59.6%. It reorders neighbours without being about
location, and retrieval is offline, so no head downstream can un-retrieve what
the cosine chose.

So the conditioning rides beside the retrieval vector rather than inside it:

    [ retrieval 768 | conditioning 1536 ] = 2304-d, StreetProj splits it

`cond_native.f16.npy` is 500,000 release rows at each encoder's own native size
-- DINOv2 at 518, its exact training resolution, and SigLIP at 512, the frame's
own pixels with no resampling. It is conditioning only and never reaches a
cosine.

**The two arms differ in one argument, `--street-file`**, and not even in a
flag: `d_cond` is read from the cache's own sidecar, because a flag that
disagreed with the file would split the tensor in the wrong place and take part
of the retrieval vector as conditioning.

    control   pyr768_mix.f16.npy        d_cond 0, no adapter
    treated   pyr768_mix_cond.f16.npy   d_cond 1536, zero-gated adapter

**The control is not optional.** Both arms are +2 epochs from `pyrMIX-e6`, so
without it the comparison confounds conditioning with two more epochs of
training -- and `tiledrisk` measured e6-vs-e4 inside noise within both of its
arms, which is suggestive but is not the same claim.

**Prior: small.** With retrieval off the system collapses to 2.7%, and top-1
retrieval alone scores 56.4% where the full agent scores 57.5%. The
conditioning path has historically carried very little. That is what makes it
a real experiment rather than a confirmation.

    OSV_RELEASE=s10 py scripts/condladder.py 3.0
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

import overnight as O                                    # noqa: E402
from overnight import Stage, log, load_state, run_stage  # noqa: E402
from marathon import ARCH, tiles_up                      # noqa: E402

BASE = "pyrMIX-e6"
RETR = "pyr768_mix.f16.npy"
COND = "pyr768_mix_cond.f16.npy"

# Both arms retrieve in the SAME space, so both name the same cache. Derived
# from the retrieval file, never from the street file: `config.knn_name(COND)`
# would ask for `knn_pyr768_mix_cond_sequence_k32.npz`, a cache that should not
# exist, because the k-NN was deliberately built on the retrieval prefix. That
# is REVIEW6 #5, fixed in `train.py`'s default derivation -- named explicitly
# here anyway, since this run costs an hour of GPU and an explicit argument
# cannot be wrong in a new way.
KNN = "knn_pyr768_mix_sequence_k32.npz"

ARMS = [("pyrMIX-e8", RETR), ("pyrMIX-cond-e8", COND)]


def rung(tag, street):
    """Two more epochs from BASE. `tiledrisk`'s recipe, unchanged.

    `--seed 0` and `--neg-random` are pinned rather than left to the day's
    default: seqfix2's answer was voided when a `train.py` edit changed the
    negative draw partway through a ladder.
    """
    return ["src/train.py", "--tag", tag, "--epochs", "2",
            "--batch", "64", "--limit", "400000",
            "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
            "--split-mode", "sequence", "--street-file", street,
            "--knn-file", KNN,
            "--seed", "0", "--neg-random", "--init", BASE] + ARCH \
        + ["--retr-drop", "0.7"]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return
    hours = float(arg) if arg else 3.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("condladder: does native-resolution detail help beside retrieval?")
    log("both arms +2 epochs from {}; one argument apart".format(BASE))
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    plan = [Stage("train-" + tag, rung(tag, street),
                  release=REL, est=28 * 60, retries=3)
            for tag, street in ARMS]
    # BASE is in the bootstrap because it is the point both arms start from:
    # without it, "+2 epochs with the adapter beats +2 epochs without" cannot
    # be separated from "+2 epochs is worse than stopping at 6".
    plan.append(Stage(
        "condladder-boot",
        ["scripts/bootstrap.py", "--tags",
         ",".join([BASE] + [t for t, _ in ARMS]),
         "--split", "test", "--n", "5000", "--beam", "2", "--score-steps", "3",
         "--out", str(O.RUNS / "BOOTSTRAP_condladder.md")],
        release=REL, est=20 * 60, retries=2))

    for st in plan:
        if O.now() > deadline:
            log("deadline reached with {} stages unrun".format(
                len(plan) - plan.index(st)))
            break
        run_stage(st, state, deadline)
    # `Sampler` exposes a `stop_flag` Event, not a `stop()` method. The method
    # that does not exist raised AFTER every stage finished, so no result was
    # lost -- but the script died on a traceback instead of logging its
    # completion line, the monitor watching for that line never fired, and the
    # GPU sat idle for six hours.
    O.SAMPLER.stop_flag.set()
    log("condladder done. {} stages recorded, {} failed".format(
        len(state["done"]), len(state["failed"])))


if __name__ == "__main__":
    main()
