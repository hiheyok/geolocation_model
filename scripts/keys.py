"""Do the learned retrieval keys earn their place?  Re-run, honestly this time.

`RetrievalPrior` offers four ways to turn 16 retrieved neighbours into a bias
over the 256 child cells, each a strict superset of the one before:

    scalar  one gate, shared by every step and by the sink bucket
    cond    per-step gates for cells and sink, modulated by retrieval quality
    pos     + a learned positive key: re-weight the *same* neighbours in a
            learned space instead of trusting raw cosine
    dual    + a separately keyed negative branch, whose job is the neighbour
            that is visually close and geographically wrong

The first comparison of these was void.  `beam.search` built the retrieval bias
itself and never passed the neighbour embeddings, so `weights()` fell through to
the branch that drops the `w_pos * cos(q_pos, k_pos)` term and returns no
negative weights at all.  At inference `pos` and `dual` therefore *were* `cond`,
exactly, and the ablation compared three copies of the same decode rule.  It is
also why the learned key looked unidentified: a parameter that settles at +0.85
in one run and -0.85 in another, with no effect on the metric, is a parameter
the metric never reads.

Training was always correct, so this asks two separable questions:

  * s01 arms -- re-evaluate the checkpoints trained back then, with the keys
    now live.  Weights unchanged; only the decode is fixed.  Note their epoch
    was *selected* by the broken rollout, so this is a lower bound.
  * s10 arms -- retrain all four at the shipping scale, where selection sees
    the same decode that evaluation will.  A no-retrieval control anchors the
    ladder, so "what is the prior worth" and "what is the key worth" get
    separate answers.

The dual arm here repeats s10_n400k_e2's configuration under a new tag, which
makes the pair a free seed-to-seed variance estimate -- the thing every
conclusion in this project has been missing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import overnight as O
from overnight import Stage, load_state, log, run_stage

BASE = "dual_c3.f16.npy"
KNN = "knn_dual_c3_sequence_k32.npz"
LIMIT = "400000"
EPOCHS = "2"

# everything except the retrieval head is held at the shipping configuration
ARCH = ["--pool", "attn", "--pool-q", "4", "--pos", "both",
        "--map-layers", "1", "--neg", "4"]

# (tag suffix, extra args).  "none" omits --retr entirely: the control.
MODES = [
    ("none", []),
    ("scalar", ["--retr", "--retr-k", "16", "--retr-mode", "scalar"]),
    ("cond", ["--retr", "--retr-k", "16", "--retr-mode", "cond"]),
    ("pos", ["--retr", "--retr-k", "16", "--retr-mode", "pos", "--d-key", "128"]),
    ("dual", ["--retr", "--retr-k", "16", "--retr-mode", "dual", "--d-key", "128"]),
]

# the s01 checkpoints whose evaluation the bug invalidated
S01 = ["retr_seq", "retr_gate", "retr_lkey", "retr_dual", "retr_dual32"]


def train_argv(mode, extra):
    return (["src/train.py", "--tag", "s10_key_" + mode,
             "--epochs", EPOCHS, "--batch", "64", "--limit", LIMIT,
             "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
             "--split-mode", "sequence", "--street-file", BASE,
             "--knn-file", KNN] + ARCH + extra)


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("retrieval key ablation, keys live at inference")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    stages = [
        # cheap and first: the old arms, re-decoded.  Answers "what did the bug
        # cost" before spending two hours on the retrain.
        Stage("key_s01_eval",
              ["scripts/bootstrap.py", "--tags", ",".join(S01),
               "--split", "test", "--n", "5000", "--beam", "2",
               "--score-steps", "3",
               "--out", str(O.RUNS / "BOOTSTRAP_keys_s01.md")],
              release="s01", est=20 * 60, retries=1),
    ]
    for mode, extra in MODES:
        stages.append(Stage("key_train_" + mode, train_argv(mode, extra),
                            release="s10", est=32 * 60, retries=1,
                            critical=False))

    for st in stages:
        run_stage(st, state, deadline)

    tags = ",".join("s10_key_" + m for m, _ in MODES)
    run_stage(Stage("key_eval",
                    ["scripts/bootstrap.py", "--tags", tags,
                     "--split", "test", "--n", "5000", "--beam", "2",
                     "--score-steps", "3",
                     "--out", str(O.RUNS / "BOOTSTRAP_keys.md")],
                    release="s10", est=45 * 60, retries=1), state, deadline)

    # the dual arm is a reseed of s10_n400k_e2; that contrast is the variance
    run_stage(Stage("key_seed",
                    ["scripts/bootstrap.py", "--tags",
                     "s10_key_dual,s10_n400k_e2",
                     "--split", "test", "--n", "5000", "--beam", "2",
                     "--score-steps", "3",
                     "--out", str(O.RUNS / "BOOTSTRAP_seed.md")],
                    release="s10", est=5 * 60, retries=1), state, deadline)

    O.finish(state, deadline)


if __name__ == "__main__":
    main()
