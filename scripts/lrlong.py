"""Does more training help, and does the learning rate change the answer?

Both questions at once, because they are the same question. `night0908`
settled that the *sawtooth* was costing: one 6-epoch cosine (`c6`) beats the
chained `e6` by +0.38 to +2.18 pp and ties `e4`. What it did not settle is
whether six epochs is where this arm stops improving, or only where this
schedule stops improving.

**The c6 log says the ceiling is not the schedule.** Selection hit rate by
epoch was 57.4, 58.1, 58.1, 57.5, 56.8 -- it peaks at epoch 3 and falls for
three straight epochs while the train loss keeps dropping (10.6 -> 9.7) and
the val loss flattens at 9.56. That is overfitting on a fixed 400,180-image
train split, not a mis-annealed schedule, and `--limit 400000` is already the
whole split, so there is no more data to reach for
(`epochs-do-not-substitute-for-data`, `data-is-the-binding-constraint`).

So the prior here is a null, and the arms are chosen to say *which* null:

1. **`c6x2-lr5e-5`** -- continue `c6` for two more epochs at a modulated rate.
   The literal question. Its `e4`-based analogue (`e6-lr5e-5`) landed inside
   noise against `e4`: the low rate removed the chaining regression without
   adding anything. If that repeats from a schedule that was not mis-annealed
   to begin with, "continue training" is answered.
2. **`c6-lr15` / `c6-lr6`** -- the same 6-epoch cosine at half and double the
   peak rate. This is the knob that is actually untested: every arm in this
   project ran a 3e-4 peak, and `runs/BOOTSTRAP_lr.md` only ever varied the
   rate of a *continuation*, never of the schedule itself. A lower peak is the
   plausible one -- it is the cheapest thing that could slow the overfit.
3. **`c10`** -- ten epochs in one cosine. Last because it costs the most and
   is the least likely to win: a longer cosine holds a higher rate through the
   epochs where c6 already turned over.

Read it as a set. If every arm is inside noise against `c6`, the arm is
data-limited at ~58%, the epoch budget is settled at 4-6, and the next lever
is the corpus rather than the schedule.

    OSV_RELEASE=s10 py scripts/lrlong.py 7
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

# The recipe is imported, not restated. Every arm here has to be comparable
# rung-for-rung with `pyrL0L1-b340-c6`, and the only way to guarantee that is
# to build the argv with the same function that built c6's -- a copied
# `rung()` drifts silently, and a drifted flag looks exactly like a result.
from night0908 import rung, STREET                # noqa: E402

ARM = "pyrL0L1"
BASE = "pyrL0L1-b340-c6"           # the arm every new one is measured against


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

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("lrlong: is 6 epochs the ceiling, and does the peak rate move it?")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    st = STREET[ARM]
    # (tag, epochs, lr, init) -- cheapest first, so a short window still
    # answers the question that was actually asked.
    arms = [
        ("pyrL0L1-b340-c6x2-lr5e-5", 2, 5e-5, BASE),
        ("pyrL0L1-b340-c6-lr15", 6, 1.5e-4, None),
        ("pyrL0L1-b340-c6-lr6", 6, 6e-4, None),
        ("pyrL0L1-b340-c10", 10, None, None),
    ]
    # 775 s/epoch measured on the c6 run, plus a load-and-index allowance.
    plan = [Stage(tag, rung(tag, st, "sequence", ep, init=init, lr=lr),
                  release=REL, est=ep * 800 + 600, retries=2)
            for tag, ep, lr, init in arms]

    res = O.run_plan(plan, state, deadline)

    # The comparison is assembled from what actually trained, not from the
    # list above. These tags are all new, so a missing checkpoint means the
    # stage really did not produce one -- but requiring both the recorded
    # success and the file is what keeps a re-run of this script from
    # comparing a half-written checkpoint from the attempt before it.
    tags = [BASE, "pyrL0L1-b340-e4"]
    for tag, _, _, _ in arms:
        if tag in res["done"] and (config.CHECKPOINTS / (tag + ".pt")).exists():
            tags.append(tag)
    if len(tags) > 2:
        O.run_stage(Stage(
            "lrlong-boot",
            ["scripts/bootstrap.py", "--tags", ",".join(tags),
             "--split", "test", "--n", "5000", "--beam", "2",
             "--score-steps", "3", "--out", str(O.RUNS / "BOOTSTRAP_lrlong.md")],
            release=REL, est=8 * 60 * len(tags), retries=2), state, deadline)
    else:
        log("no arm trained, so there is nothing to compare; skipping the "
            "bootstrap rather than re-reporting c6 against e4")

    O.SAMPLER.stop_flag.set()
    log("lrlong done. {} trained, {} failed, {} skipped, {} unrun".format(
        len(res["done"]), len(res["failed"]), len(res["skipped"]),
        len(res["unrun"])))
    for k in ("failed", "skipped", "unrun"):
        if res[k]:
            log("  {}: {}".format(k, ", ".join(res[k])))
    return 1 if (res["failed"] or res["skipped"] or res["unrun"]) else 0


if __name__ == "__main__":
    sys.exit(main())
