"""The 3.50M corpus ladder, and the first run on locked large pages.

Two questions at once, which is fine because they are independent.

**Does the corpus curve still track the model?** Three routes agreed on a
prediction before this ran, and it is recorded here so it can fail:

    metric-side declining-rate model          +2.01 pp
    retrieval delta x average conversion      +2.64 pp
    retrieval delta x most recent conversion  +2.38 pp

so **+2.0 to +2.6 pp, centre about +2.3**, on `<25 km` against `s10_b55_c6`
at 2.5 km / 73.8%. The earlier +2 pp "flattening threshold" was the wrong test:
the curve is decelerating on schedule, not flattening, and what matters is
whether the observed gain lands in the band, not whether it clears a line.

The retrieval curve it comes from, same 500,000 queries throughout:

    1.15M 0.8926 -> 1.90M 0.9000 -> 2.65M 0.9046 -> 3.50M 0.9080
    conversion measured at 8.8 then 7.0 pp per 0.01 of similarity

**Does the large-page tier hold up under a real run?** At 1536-d this table is
10.75 GB, which the host-RAM heuristic can never select on a 31.7 GB machine, so
before today it would have paged. `s10_b55` on an 8.45 GB resident table ran
788 s on its first epoch and 720 s warm. If the locked table is in that
neighbourhood despite being 27% larger, the tier is working; if it is far worse,
something about locking 10.75 GB is costing more than it saves and
`NO_LARGE_PAGES=1` turns it off without touching code.

Same 2+2+2 ladder, same 400k train limit, same architecture. The only
difference from `s10_b55_c6` is 750,000 more bank images.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

REL = "s10"
if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = REL

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, tiles_up

BANK = "pool_bal_bank70.f16.npy"        # 3.50M x 1536
META = "bank_ext70"
KNN = config.knn_name(BANK, "sequence", ext=META)


def train(tag, init=None, epochs=2):
    cmd = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
           "--batch", "64", "--limit", "400000",
           "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
           "--split-mode", "sequence", "--street-file", BANK,
           "--knn-file", KNN] + ARCH
    return cmd + (["--init", init] if init else [])


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("b70: 3.50M corpus, predicted +2.0 to +2.6 pp over s10_b55_c6")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))
    if not (config.STREET_CACHE / KNN).exists():
        log("{} missing -- build_knn has not run for this bank".format(KNN))
        return

    plan = [
        Stage("b70_2", train("s10_b70"), release=REL, est=30 * 60, retries=1),
        Stage("b70_4", train("s10_b70_c4", "s10_b70"), release=REL,
              est=30 * 60, retries=1),
        Stage("b70_6", train("s10_b70_c6", "s10_b70_c4"), release=REL,
              est=30 * 60, retries=1),
        Stage("b70_eval",
              ["scripts/boot_existing.py", "--tags",
               "s10_b55_c6,s10_b70,s10_b70_c4,s10_b70_c6",
               "--out", str(O.RUNS / "BOOTSTRAP_bank70.md")],
              release=REL, est=5 * 60, retries=1),
    ]
    for st in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        if not tiles_up():
            log("skip   {}  (tile server down)".format(st.name))
            continue
        run_stage(st, state, deadline)

    log("-" * 72)
    log("done; logs in {}".format(O.LOGS))
    if O.SAMPLER:
        log("system, whole run: " + O.SAMPLER.summary())


if __name__ == "__main__":
    main()
