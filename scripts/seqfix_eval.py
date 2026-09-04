"""What the OSV-5M benchmark says once the retrieval bank stops leaking.

`build_knn` served same-sequence frames as neighbours across the corpus
boundary (review item 61): 41.6% of test queries had a same-drive top-1, at a
median 0.31 km.  The cache has been rebuilt with exclusion applied across both
corpora; this re-measures the arms against it.

**What this asks, and what it does not.**  These arms were *trained* against
the leaky cache.  Evaluating them against a clean one answers "how much of the
published number was the leak" -- the immediate question, and cheap.  It does
not answer "what would an arm trained on clean retrieval score", which needs a
retrain and is a separate, larger run.  An arm that leaned on near-duplicate
neighbours during training may be worse at using honest ones, so this is a
lower bound on the clean-trained number, not an estimate of it.

The old cache is kept beside the new one as
`*.LEAKY-preseqfix.npz`, so every published number stays reproducible.

    OSV_RELEASE=s10 python scripts/seqfix_eval.py 3
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
from marathon import tiles_up                            # noqa: E402

# The shipping arm, the arm it is measured against, and the two wd29 arms --
# which are the only ones trained at a known seed, so they are the pair whose
# difference is attributable to anything at all.
ARMS = ["d768-b350-e6-drop70", "d768-b350-e6",
        "wd29-fix-e6", "wd29-legacy-e6"]


def main():
    # These stage runners take a bare hours argument rather than argparse, so
    # `--help` used to crash on float("--help"). The whole family does it; my
    # own --help sweep flagged them and this one at least should not.
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
    log("seqfix: the benchmark, re-measured on a bank that no longer leaks")
    log("arms trained against the LEAKY cache, scored against the clean one --")
    log("this bounds how much of the published number was the leak")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    knn = config.STREET_CACHE / config.knn_name(
        "pca768_bank70.f16.npy", "sequence", ext="bank_ext70")
    if not knn.exists():
        log("{} missing -- run build_knn first".format(knn.name))
        return

    plan = [
        # Prove the rebuild before measuring anything with it. A cache that
        # still leaks would produce a smaller, believable drop and be read as
        # the answer.
        Stage("seqfix-verify",
              ["scripts/seqleak.py", "--verify"],
              release=REL, est=2 * 60, retries=1, critical=True),
        Stage("seqfix-boot",
              ["scripts/bootstrap.py", "--tags", ",".join(ARMS),
               "--split", "test", "--n", "5000",
               "--out", str(O.RUNS / "BOOTSTRAP_seqfix.md")],
              release=REL, est=30 * 60, retries=2),
    ]

    for st in plan:
        if O.now() > deadline:
            log("deadline reached")
            break
        if not run_stage(st, state, deadline) and st.critical:
            log("critical stage {} failed; stopping".format(st.name))
            break

    log("done")


if __name__ == "__main__":
    main()
