"""Continue the arm that turned out to be the good one.

d4608-b115-e2 -- equal-norm encoders over the 1.15M bank -- reached 8.4 km and
63.0% in TWO epochs, matching what the unbalanced arm needed six to reach. The
marathon had no stage for it because the result that justifies it did not exist
when the plan was written.

Two epochs at a time, evaluated between, so the ladder is measured rather than
assumed: the unbalanced arm's gains halved at each step (+2.6 pp then +0.8) and
there is no reason to expect this one to behave differently. Every stage needs
the tile server, so the whole script has to be done before it goes.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, BAL_BANK, KNN_BALBANK_SEQ, boot, cutoff_at, tiles_up

REL = "s10"


def cont(tag, init, epochs=2):
    return ["src/train.py", "--tag", tag, "--epochs", str(epochs),
            "--batch", "64", "--limit", "400000", "--init", init,
            "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
            "--split-mode", "sequence", "--street-file", BAL_BANK,
            "--knn-file", KNN_BALBANK_SEQ] + ARCH


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600
    tiles_until = cutoff_at(9, 30)

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("followup: continue d4608-b115-e2, the equal-norm full-bank arm")
    log("deadline {}   tiles until {}".format(O.hhmm(deadline),
                                              O.hhmm(tiles_until)))

    plan = [
        Stage("fu_bal4", cont("d4608-b115-e4", "d4608-b115-e2"),
              release=REL, est=35 * 60, retries=1),
        Stage("fu_bal4_eval",
              boot("d4608-b115-e2,d4608-b115-e4,s10_n400k_bank25_c4",
                   "BOOTSTRAP_bal4.md"), release=REL, est=15 * 60, retries=1),
        Stage("fu_bal6", cont("d4608-b115-e6", "d4608-b115-e4"),
              release=REL, est=35 * 60, retries=1),
        Stage("fu_bal6_eval",
              boot("d4608-b115-e4,d4608-b115-e6",
                   "BOOTSTRAP_bal6.md"), release=REL, est=10 * 60, retries=1),
    ]

    alive = True
    for st in plan:
        if O.now() + st.est > tiles_until:
            log("skip   {}  -- needs tiles, would run past {}".format(
                st.name, O.hhmm(tiles_until)))
            continue
        if not alive or not tiles_up():
            alive = False
            log("skip   {}  -- tile server gone".format(st.name))
            continue
        run_stage(st, state, deadline)

    try:
        import subprocess
        subprocess.call([O.PY, "scripts/digest.py"], cwd=str(ROOT))
    except Exception as e:
        log("digest failed: {}".format(e))
    O.finish(state, deadline)


if __name__ == "__main__":
    main()
