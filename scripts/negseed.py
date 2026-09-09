"""What does a run-to-run difference cost, and what does removing it cost?

`runs/BOOTSTRAP_ropeagent20k.md` separated `rope0-c6` from
`pyrL0L1-b340-c6` at [+0.12, +1.02] pp. Those two arms are the same
experiment: same recipe, same seed, same epochs, and neighbour tables whose
top-1 differs on **84 of 500,000** queries. They disagree on the `<25 km`
outcome for **10.1% of test images** -- 1,070 flips one way, 959 the other --
and the separation is the +111 residual of that churn, p = 0.015 by an exact
sign test, one of ten contrasts in the table.

The paired bootstrap resamples test images and contains no training-run
variance at all, so at 20,000 images it will confidently separate two
identical experiments. That is a property of the instrument, and this project
reads 1-4 pp gaps off it routinely.

**The cause is already known and already fixed, and every recipe opts out.**
`--seed` covers weight init, the shuffling loader and dropout;
`--neg-random` overrides only the sink negatives, drawing them from OS
entropy inside each `__getitem__` so no two runs see the same off-path tiles.
It is an opt-in flag restoring old behaviour, and twelve call sites pass it.

Dropping it is not free, which is why this measures rather than assumes:
seeded negatives are a pure function of (split, seed, index), so an image sees
the *same* off-path tiles every epoch. Less negative diversity, and `--neg 4`
is the shipping configuration.

Two runs of one command, identical but for the tag:

    A vs B          -> the run-to-run floor with negatives seeded. If they
                       land on top of each other, the floor collapses and a
                       0.5 pp separation starts meaning something again.
    A vs rope0-c6   -> what seeding the negatives costs, against the arm whose
                       bank and recipe it shares exactly.

The argv is `rope0-c6`'s with one token removed, asserted in the tests, so
neither contrast carries anything else.

    OSV_RELEASE=s10 py scripts/negseed.py 5
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
from night0908 import rung                      # noqa: E402

BANK = "pyr768_rope0_b340.f16.npy"    # the arm this is paired against
REF = "rope0-c6"
ARMS = ["negseed-a-c6", "negseed-b-c6"]
EPOCHS = 6


def argv_for(tag):
    """`rope0-c6`'s command line, minus `--neg-random`.

    Removed rather than rebuilt, so the two arms cannot differ in anything
    else. `--neg-random` is a store_true flag, so dropping the token is the
    whole edit -- `neg_random` then takes its default of False and the
    negatives become a pure function of (split, seed, index).
    """
    a = rung(tag, BANK, "sequence", EPOCHS)
    return [t for t in a if t != "--neg-random"]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return 0
    hours = float(arg) if arg else 5.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    if not (config.CHECKPOINTS / (REF + ".pt")).exists():
        return log("no {}.pt, and both contrasts are against it".format(REF)) or 1

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("negseed: the run-to-run floor, and what removing it costs")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    plan = [Stage(t, argv_for(t), release=REL, est=EPOCHS * 800 + 600,
                  retries=2)
            for t in ARMS]
    res = O.run_plan(plan, state, deadline)

    tags = [t for t in ARMS
            if t in res["done"] and (config.CHECKPOINTS / (t + ".pt")).exists()]
    # Both arms or neither: A alone against `rope0-c6` measures the seeding
    # change confounded with the very run-to-run noise this exists to bound,
    # which is the question, not an answer to it.
    if len(tags) == len(ARMS):
        boot = Stage(
            "negseed-boot",
            ["scripts/bootstrap.py", "--tags", ",".join(tags + [REF]),
             "--split", "test", "--n", "20000", "--beam", "2",
             "--score-steps", "3",
             "--out", str(O.RUNS / "BOOTSTRAP_negseed.md")],
            release=REL, est=30 * 60, retries=2)
        for k, v in O.run_plan([boot], state, deadline).items():
            res[k] += v
    else:
        log("{} of {} arms trained; not comparing, because one seeded arm "
            "against {} cannot separate the seeding change from the "
            "run-to-run spread it is measuring"
            .format(len(tags), len(ARMS), REF))
        res["skipped"].append("negseed-boot")

    O.SAMPLER.stop_flag.set()
    log("negseed done. {} trained: {}".format(len(tags), ", ".join(tags)))
    for k in ("failed", "skipped", "unrun"):
        if res[k]:
            log("  {}: {}".format(k, ", ".join(res[k])))
    return 1 if (res["failed"] or res["skipped"] or res["unrun"]) else 0


if __name__ == "__main__":
    sys.exit(main())
