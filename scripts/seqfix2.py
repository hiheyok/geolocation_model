"""Finish the leak fix: rebuild the last two caches, then train one arm clean.

Part one re-measures the arms whose retrieval caches are still leaky. Three
caches serve every arm on record and only `pca768_bank70` has been rebuilt, so
the d1536 family (`pool_bal_bank70`) and the 2.65M-bank arm (`pca768_bank55`)
are still reporting numbers inflated by roughly 18 pp.

Part two is the one that turns a bound into an estimate. Every arm on record
was *trained* against a leaking bank, so scoring them clean says how much of
the published number was the leak -- a lower bound on a cleanly-trained arm,
because a model that learned to lean on near-duplicate neighbours may be worse
at using honest ones. Retraining the shipping ladder against the clean cache is
what answers "what is this system actually worth".

`--seed 0`, matching the wd29 ladders, so the new arm is comparable to those
without a seed confound. It differs from `wd29-fix-e6` in exactly one thing:
the retrieval cache it trained against.

Each rebuild is followed by its own verify, and the verify is critical. A cache
that still leaked would produce a smaller, entirely believable drop and be read
as the answer.

    OSV_RELEASE=s10 python scripts/seqfix2.py 5
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
from marathon import ARCH, tiles_up                      # noqa: E402

DST = "pca768_bank70.f16.npy"
KNN = config.knn_name(DST, "sequence", ext="bank_ext70")

# every arm whose OSV-5M column is stale, plus the two already re-measured so
# the table is one paired run rather than two that cannot be compared
ARMS = ["d768-b350-e6-drop70", "d768-b350-e6", "d768-b265-e6",
        "d768-b350-e6-drop30", "d768-b350-e6-drop90",
        "d1536-b350-e6", "d1536-b350-e6-drop30", "d1536-b350-e6-drop70"]


def rung(tag, init):
    """One rung of the shipping ladder, trained against the CLEAN cache."""
    a = ["src/train.py", "--tag", tag, "--epochs", "2",
         "--batch", "64", "--limit", "400000",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", "sequence", "--street-file", DST,
         "--knn-file", KNN, "--seed", "0"] + ARCH + ["--retr-drop", "0.7"]
    return a + (["--init", init] if init else [])


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("-h", "--help"):
        print(__doc__)
        return
    hours = float(arg) if arg else 5.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("seqfix2: the last two caches, then one arm trained on a clean bank")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server {}".format("up" if tiles_up() else "DOWN"))

    plan = [
        # ---- part one: the two caches still leaking ----------------------
        Stage("seqfix2-knn-d1536",
              ["scripts/build_knn.py", "--street-file", "pool_bal_bank70.f16.npy",
               "--split-mode", "sequence", "--k", "32",
               "--bank-ext", "bank_ext70"],
              release=REL, est=10 * 60, retries=1, critical=True),
        Stage("seqfix2-verify-d1536",
              ["scripts/seqleak.py", "--verify", "--ext", "bank_ext70",
               "--knn", "knn_pool_bal_bank70_sequence_k32_bank_ext70.npz"],
              release=REL, est=2 * 60, retries=1, critical=True),
        Stage("seqfix2-knn-b55",
              ["scripts/build_knn.py", "--street-file", "pca768_bank55.f16.npy",
               "--split-mode", "sequence", "--k", "32",
               "--bank-ext", "bank_ext55"],
              release=REL, est=10 * 60, retries=1, critical=True),
        Stage("seqfix2-verify-b55",
              ["scripts/seqleak.py", "--verify", "--ext", "bank_ext55",
               "--knn", "knn_pca768_bank55_sequence_k32_bank_ext55.npz"],
              release=REL, est=2 * 60, retries=1, critical=True),
        Stage("seqfix2-boot-all",
              ["scripts/bootstrap.py", "--tags", ",".join(ARMS),
               "--split", "test", "--n", "5000",
               "--out", str(O.RUNS / "BOOTSTRAP_seqfix_all.md")],
              release=REL, est=30 * 60, retries=2),

        # ---- part two: one arm trained against a bank that does not leak --
        Stage("train-clean-e2", rung("clean-drop70-e2", None),
              release=REL, est=28 * 60, retries=3),
        Stage("train-clean-e4", rung("clean-drop70-e4", "clean-drop70-e2"),
              release=REL, est=28 * 60, retries=3),
        Stage("train-clean-e6", rung("clean-drop70-e6", "clean-drop70-e4"),
              release=REL, est=28 * 60, retries=3),

        # The comparison that matters: same seed, same data, same schedule,
        # differing only in whether the bank it trained against leaked.
        Stage("seqfix2-boot-clean",
              ["scripts/bootstrap.py", "--tags",
               "clean-drop70-e6,wd29-fix-e6,d768-b350-e6-drop70",
               "--split", "test", "--n", "5000",
               "--out", str(O.RUNS / "BOOTSTRAP_cleantrain.md")],
              release=REL, est=20 * 60, retries=2),
        # KartaView is the selection benchmark now, so the new arm has to be
        # measured there before anything is claimed about it.
        Stage("seqfix2-hr-clean",
              ["scripts/eval_highres.py", "--tag", "clean-drop70-e6",
               "--n", "5000",
               "--export", str(O.RUNS / "hr_cleandrop70.npz")],
              release=REL, est=25 * 60, retries=2),
    ]

    for st in plan:
        if O.now() > deadline:
            log("deadline reached with {} stages unrun".format(
                len(plan) - plan.index(st)))
            break
        ok = run_stage(st, state, deadline)
        if not ok and getattr(st, "critical", False):
            log("critical stage {} failed; stopping".format(st.name))
            break

    log("done. {} recorded, {} failed".format(
        len(state["done"]), len(state["failed"])))


if __name__ == "__main__":
    main()
