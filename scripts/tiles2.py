"""The tiling probe again, with enough bank to actually resolve the question.

The first run embedded 30,000 images, and at that size the retrieval hit rate
is ~0.7% -- about ten successful queries out of 1,500. crop3 and tile6 differed
by roughly seven queries, which is Poisson noise wearing a result's clothing.
Worse, the same cached embeddings at 28,500 and 40,000 bank rows score 0.67%
and 5.47%, so the metric's sampling variance at that scale dwarfs any effect
being looked for.

Two changes buy the bank this needs. crop3 comes from the cached dual_c3 rather
than being recomputed -- it is the shipping pipeline rather than a
re-implementation, and it drops the per-image cost from nine encoder passes to
six. And the run is sized to the whole remaining window rather than to a round
number.

Entirely offline: no tile fetches anywhere in it, so the map server going at
09:30 is irrelevant to it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import overnight as O
from overnight import Stage, load_state, log, run_stage


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("tiles2: 3x2 tiling vs 3 crops, at a bank size that can resolve it")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    run_stage(Stage("tiles2_probe",
                    ["scripts/tile_probe.py", "--n", "120000",
                     "--queries", "3000", "--crop3-cached",
                     "--dims", "256,512,1024,2048"],
                    release="s10", est=150 * 60, retries=1), state, deadline)

    try:
        import subprocess
        subprocess.call([O.PY, "scripts/digest.py"], cwd=str(ROOT))
    except Exception as e:
        log("digest failed: {}".format(e))
    O.finish(state, deadline)


if __name__ == "__main__":
    main()
