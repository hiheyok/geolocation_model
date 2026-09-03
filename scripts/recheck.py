"""Re-measure what the keys bug invalidated, and finish the arms it left open.

Three jobs, in descending order of what they change:

  1. The compute curve. Accuracy versus test-time compute is the figure the
     whole thesis rests on, and both of its axes -- beam width and ranking
     depth -- were tuned against a decode that discarded the learned retrieval
     keys. The keys change the fine-step likelihood, which is exactly the
     quantity `--score-steps` trades away, so `score_steps=3` was chosen for a
     model that no longer exists. Cheap: evaluation only, no training.

  2. The shipping arm's schedule. `s10_n400k_bank25` trained a full two-epoch
     cosine and scored 53.8% at both epochs; `--select hit` keeps the earlier
     one on an exact tie, which is why the checkpoint reads epoch 1. It is not
     undertrained -- but a tie on a 2,000-image selector is about 1 pp of
     resolution, so "plateaued" is a weak claim. The cell8 arms improved
     monotonically under exactly this continuation, so it is worth one test.

  3. cell8's learning rate. Those arms declined under the 3e-4 cosine and then
     improved under a 1e-4 continuation, which says mistuned rather than
     saturated. cell8 is the number quoted for external comparison, so leaving
     a known-mistuned arm as the headline is the weakest part of the record.
     A fresh 4-epoch run at each LR settles it; continuing does not, because a
     continuation confounds the rate with the extra epochs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage

REL = "s10"
EXT = "bank_ext"
STACKED = "dual_c3_bank25"
SHIP = "s10_n400k_bank25"
KNN_SEQ = config.knn_name(STACKED + ".f16.npy", "sequence", ext=EXT)
KNN_C8 = config.knn_name(STACKED + ".f16.npy", "cell8", ext=EXT)

ARCH = ["--pool", "attn", "--pool-q", "4", "--pos", "both",
        "--map-layers", "1", "--neg", "4",
        "--retr", "--retr-k", "16", "--retr-mode", "dual", "--d-key", "128"]

EVAL = ["--split", "test", "--n", "5000"]


def train_argv(tag, knn, mode, epochs, lr=None, init=None):
    a = ["src/train.py", "--tag", tag, "--epochs", str(epochs), "--batch", "64",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", mode, "--street-file", STACKED + ".f16.npy",
         "--knn-file", knn]
    if lr is not None:
        a += ["--lr", str(lr)]
    if init is not None:
        a += ["--init", init]
    return a + ARCH


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("recheck: compute curve, shipping-arm continuation, cell8 learning rate")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    stages = [
        # ---- 1. the compute curve, keys live ----------------------------
        Stage("rc_width",
              ["src/evaluate.py", "--tag", SHIP] + EVAL +
              ["--ks", "1,2,4,8,16", "--score-steps", "3"],
              release=REL, est=25 * 60, retries=1),
    ]
    # ranking depth at a fixed width; separate stages so one failure is not all
    for d in (1, 2, 3, 4):
        stages.append(
            Stage("rc_depth{}".format(d),
                  ["src/evaluate.py", "--tag", SHIP] + EVAL +
                  ["--beam", "4", "--score-steps", str(d)],
                  release=REL, est=8 * 60, retries=1))

    stages += [
        # ---- 2. does the shipping arm have anything left? ---------------
        Stage("rc_cont",
              train_argv(SHIP + "_cont", KNN_SEQ, "sequence", 2, init=SHIP),
              release=REL, est=40 * 60, retries=1),
        Stage("rc_cont_eval",
              ["scripts/bootstrap.py", "--tags", SHIP + "," + SHIP + "_cont",
               "--split", "test", "--n", "5000", "--beam", "2",
               "--score-steps", "3",
               "--out", str(O.RUNS / "BOOTSTRAP_cont.md")],
              release=REL, est=10 * 60, retries=1),

        # ---- 3. cell8 learning rate, fresh runs at matched epochs -------
        # 3e-4 is the current default and the arm that declined; 1e-4 is what
        # the continuation ran at. Same epochs, same data, one knob.
        Stage("rc_c8_lr3",
              train_argv("s10_cell8_bank25_lr3e4", KNN_C8, "cell8", 4,
                         lr=3e-4),
              release=REL, est=60 * 60, retries=1),
        Stage("rc_c8_lr1",
              train_argv("d4608-b115-e2-cell8", KNN_C8, "cell8", 4,
                         lr=1e-4),
              release=REL, est=60 * 60, retries=1),
        Stage("rc_c8_eval",
              ["scripts/bootstrap.py", "--tags",
               "s10_cell8_bank25_cont,s10_cell8_bank25_lr3e4,"
               "d4608-b115-e2-cell8",
               "--split", "test", "--n", "5000", "--beam", "2",
               "--score-steps", "3",
               "--out", str(O.RUNS / "BOOTSTRAP_cell8_lr.md")],
              release=REL, est=15 * 60, retries=1),
    ]

    for st in stages:
        run_stage(st, state, deadline)

    O.finish(state, deadline)


if __name__ == "__main__":
    main()
