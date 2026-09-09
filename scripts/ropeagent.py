"""Can the agent read positional phase that the cosine cannot?

`runs/BOOTSTRAP_lrlong.md`'s companion, `ropeladder`, put the fixed rotation
at **net +0.06 to +0.14 pp on retrieval, spanning zero at every threshold**.
That is a null for `knn_gap` and it is *not* a null for the agent, which is a
distinction this project has already been burned by twice --
`pool_pyramid`'s own docstring says so: "the agent consumes neighbours through
a *trained* retrieval prior, so only a retrain converts that into an agent
number, and this project has twice had inference-level reasoning predict the
wrong sign."

The measurement that makes this worth doing is the pair of facts either side
of the null:

* the projected banks are **mostly different** -- mean cosine to arm 0 of
  0.326, 0.384 and 0.238 -- so the rotation is not being washed out by the
  PCA before anything can use it;
* the neighbour **ordering** is unchanged.

Same neighbours, different vectors. The plain sum destroys which view a
feature came from; the rotated sum keeps it in the phase relations. The cosine
is a fixed, unlearned reader of that vector and finds nothing. The agent's
first operation on the street embedding is a learned `Linear(768 -> 512)`, and
there is a learned key pathway over `nbr_emb` besides. A learned reader can
extract structure a fixed dot product cannot, and nothing measured so far
speaks to whether it does.

**One flag between the arms.** Every arm is the `pyrL0L1-b340-c6` recipe -- one
6-epoch cosine, lr 3e-4, seed 0, `--retr-drop 0.7` -- imported from
`night0908.rung` so it cannot drift. Only `--street-file` and its derived
`--knn-file` change.

`rope0` is trained even though it should equal the shipping arm, because
"should" is the word that has cost this project the most. Its bank is
bit-identical to `pyr768_l0l1_b340` on 2.65M of 3.4M rows and differs by fp16
noise (2.44e-04) on the 750k that `tilebig` pooled from `bank_ext_dual` rather
than `bank_ext_bal`. That moved 0 to 4 queries in 500,000, so `rope0` and the
shipping arm should land on top of each other -- and if they do not, the gap
is what a chain rebuild costs, which every arm above inherits.

`pyrL0L1-b340-c6` goes in the bootstrap as a fifth arm for that reason, and
because `tiles-gain-grows-with-corpus` is the memory that says always to.

    OSV_RELEASE=s10 py scripts/ropeagent.py 7
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

import config                                   # noqa: E402
import overnight as O                           # noqa: E402
from overnight import Stage, log, load_state    # noqa: E402

# The recipe is imported, not restated, so every arm is comparable with
# `pyrL0L1-b340-c6` rung for rung by construction.
from night0908 import rung                      # noqa: E402

ARMS = ["rope0", "ropeD", "ropeDR", "ropeDRC"]
SHIP = "pyrL0L1-b340-c6"       # the arm already trained on the same recipe
EPOCHS = 6                     # one cosine; `night0908` showed the sawtooth costs


def street(arm):
    return "pyr768_{}_b340.f16.npy".format(arm)


def tag(arm):
    return "{}-c6".format(arm)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return 0
    hours = float(arg) if arg else 7.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    # A bank that is not there is a stage that fails four times over 20
    # minutes and buries the reason. Checked once, up front, by name.
    missing = [a for a in ARMS
               if not (config.STREET_CACHE / street(a)).exists()
               or not (config.STREET_CACHE / config.knn_name(
                   street(a), "sequence", ext="bank_ext70")).exists()]
    if missing:
        return log("no bank or neighbour table for {}; run scripts/"
                   "ropeladder.py first".format(", ".join(missing))) or 1

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("ropeagent: can the agent read phase the cosine cannot?")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    # 775 s/epoch measured on `pyrL0L1-b340-c6`, plus a load-and-index
    # allowance. Not critical: each arm is independent, and three arms plus
    # the shipping reference is still a result.
    plan = [Stage(tag(a), rung(tag(a), street(a), "sequence", EPOCHS),
                  release=REL, est=EPOCHS * 800 + 600, retries=2)
            for a in ARMS]
    res = O.run_plan(plan, state, deadline)

    tags = [tag(a) for a in ARMS
            if tag(a) in res["done"]
            and (config.CHECKPOINTS / (tag(a) + ".pt")).exists()]
    if len(tags) > 1:
        boot = Stage(
            "ropeagent-boot",
            ["scripts/bootstrap.py", "--tags", ",".join(tags + [SHIP]),
             "--split", "test", "--n", "5000", "--beam", "2",
             "--score-steps", "3",
             "--out", str(O.RUNS / "BOOTSTRAP_ropeagent.md")],
            release=REL, est=8 * 60 * (len(tags) + 1), retries=2)
        for k, v in O.run_plan([boot], state, deadline).items():
            res[k] += v
    else:
        # One arm and the shipping reference is a two-arm bootstrap with no
        # control for the chain rebuild, which is exactly the shape
        # `runs/TILESHIP.md` got wrong.
        log("{} arm(s) trained; not comparing, because one rotated arm against "
            "the shipping one cannot separate the rotation from the rebuild"
            .format(len(tags)))
        res["skipped"].append("ropeagent-boot")

    O.SAMPLER.stop_flag.set()
    log("ropeagent done. {} trained: {}".format(len(tags), ", ".join(tags)))
    for k in ("failed", "skipped", "unrun"):
        if res[k]:
            log("  {}: {}".format(k, ", ".join(res[k])))
    return 1 if (res["failed"] or res["skipped"] or res["unrun"]) else 0


if __name__ == "__main__":
    sys.exit(main())
