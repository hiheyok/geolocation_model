"""Re-export the KartaView reference arms so paired intervals become possible.

Every `runs/hr_*.npz` on disk was written on 2026-09-02, when `eval_highres`
searched all 3,500,000 bank rows. The `hrfix-*` runs on 09-03 corrected that to
3,400,180 -- the release's val and test images excluded from the corpus, which
is the bank definition the OSV-5M evaluation uses, so the system under test is
the same one on both benchmarks. Those corrected runs are the numbers quoted in
STATE.md 2, and they exported nothing.

The consequence is easy to miss and I walked into it: the exports look like
per-image errors for the arms in that table, and they are not. Pairing one
against a current run compares two protocols, and the difference between the
protocols is larger than the differences being measured -- drop70 alone moves
13.4% -> 14.1% and 435.7 -> 409.8 km. A paired bootstrap over that mixture
reported `<1 km` as separated, which was an artifact of the bank, not a fact
about the arms.

So this re-runs the two reference arms with `--export`, at the corrected
protocol, on the frozen cohort. After it, `clean2-drop70-e6` from seqfix3 can
be compared to either with a paired interval instead of two point estimates
0.1 pp apart.

Run this AFTER seqfix3 finishes -- never stack GPU jobs (STATE.md 10).

    OSV_RELEASE=s10 py scripts/hr_pairs.py 2
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
from marathon import tiles_up                            # noqa: E402

# The arm the ladder is matched to, and the shipping arm. Both already have a
# corrected-protocol number (12.3% and 13.4%); this adds the per-image errors
# behind them.
ARMS = ["wd29-fix-e6", "d768-b350-e6-drop70"]


def stage(tag):
    return Stage(
        "hrpair-" + tag,
        ["scripts/eval_highres.py", "--tag", tag, "--n", "5000",
         "--export", str(O.RUNS / "hr_{}_fixed.npz".format(tag.replace("-", "_")))],
        release=REL, est=17 * 60, retries=2)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return
    hours = float(arg) if arg else 2.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("hr_pairs: per-image KartaView errors at the corrected bank protocol")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    for tag in ARMS:
        if O.now() > deadline:
            log("deadline reached")
            break
        run_stage(stage(tag), state, deadline)

    log("done. {} recorded, {} failed".format(
        len(state["done"]), len(state["failed"])))


if __name__ == "__main__":
    main()
