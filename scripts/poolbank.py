"""Does mean-pooling the crop tokens survive into the agent's own metric?

At retrieval level, pooling the three crop tokens into one 1536-d vector instead
of concatenating them into 4608 measured at +0.20 pp [-0.30, +0.70] on any-of-32
`<25 km` -- indistinguishable, at a third of the bytes.  Two reasons that is
worth a training run rather than a footnote:

  * the neighbour table is 11.52 GB for 1.25M images at 4608-d and 3.53 GB at
    1536-d, and corpus size is the axis that has moved this metric most
    (+20.4 pp for 400k -> 1.15M), so a third of the bytes is three times the
    corpus at the same RAM;
  * `StreetProj` and the retrieval keys are both shaped by the 4608 input and
    are together 4.72M of the 9.22M trainable parameters.

The comparison is against `s10_bal_bank25_c6`, and it is matched on schedule
rather than only on epoch count.  That arm was not trained as one six-epoch
cosine -- it was 2 epochs, then +2 from a restart, then +2 -- and a single long
cosine is a different optimiser trajectory, so this runs the same 2+2+2 ladder.
Same batch, same 400k train limit, same architecture flags, same selection rule.
The only difference is the width of the street vector.

Reading it: the pooled arm does not need to win.  It needs to not lose by more
than the noise floor measured today, which is about 1 pp of hit rate on this
recipe at n=5,000 -- because "the same, at a third of the memory" is the result
that matters here.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, boot

REL = "s10"
BANK = "pool_bal_bank25.f16.npy"
KNN = config.knn_name(BANK, "sequence", ext="bank_ext")


def train(tag, init=None, epochs=2):
    cmd = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
           "--batch", "64", "--limit", "400000",
           "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
           "--split-mode", "sequence", "--street-file", BANK,
           "--knn-file", KNN] + ARCH
    return cmd + (["--init", init] if init else [])


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("pooled street vector: 4608 -> 1536, same 2+2+2 ladder as the baseline")
    log("bank {}   knn {}".format(BANK, KNN))
    log("deadline {}".format(O.hhmm(deadline)))

    plan = [
        Stage("pb_2", train("s10_pool"), release=REL, est=27 * 60, retries=1),
        Stage("pb_4", train("s10_pool_c4", "s10_pool"), release=REL,
              est=27 * 60, retries=1),
        Stage("pb_6", train("s10_pool_c6", "s10_pool_c4"), release=REL,
              est=27 * 60, retries=1),
        Stage("pb_eval",
              boot("s10_bal_bank25_c6,s10_pool,s10_pool_c4,s10_pool_c6",
                   "BOOTSTRAP_pool.md"), release=REL, est=10 * 60, retries=1),
    ]
    for st in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        run_stage(st, state, deadline)
    log("done")


if __name__ == "__main__":
    main()
