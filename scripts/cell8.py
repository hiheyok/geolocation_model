"""The geographic holdout: does any of the retrieval win survive unseen regions?

The sequence split only guarantees that a capture sequence never spans
train/test, so a test image can sit metres from a training image on the same
street. Scaling the bank to 1.15M images took the median from 81.6 to 14.2 km
under that split -- but a denser corpus is also more likely to *contain* the
test location, and nothing in that experiment separates "learned to read
geography" from "has seen this place before".

The cell8 split removes whole z8 cells (156 km across) from training. Here the
model cannot have seen the region at all, and neither can the bank: the
extension is filtered to the same held-out cells, which drops 16.5% of it and
leaves 1,044,404 images against the sequence bank's 1,150,180 -- within 10%, so
a collapse cannot be blamed on having less to retrieve from.

Completes a 2x2 of {sequence, cell8} x {release bank, extended bank}. The
interesting cell is the bottom right: if the extended bank still helps on unseen
regions, retrieval is doing something better than remembering.
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
BASE = "dual_c3.f16.npy"
STACKED = "dual_c3_bank25.f16.npy"

KNN_BASE = config.knn_name(BASE, "cell8")
KNN_EXT = config.knn_name(STACKED, "cell8", ext=EXT)

ARCH = ["--pool", "attn", "--pool-q", "4", "--pos", "both",
        "--map-layers", "1", "--neg", "4",
        "--retr", "--retr-k", "16", "--retr-mode", "dual", "--d-key", "128"]


def train_argv(tag, street, knn, epochs, init=None):
    a = [] if init is None else ["--init", init]
    return (["src/train.py", "--tag", tag, "--epochs", str(epochs), "--batch", "64"]
            + a + [
             "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
             "--split-mode", "cell8", "--street-file", street,
             "--knn-file", knn] + ARCH)


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    # epochs is part of the arm identity: a longer run is a different cosine
    # schedule, not a continuation, so it gets its own tag and its own marker
    # "cont" continues the 2-epoch arms for 2 more epochs from their saved
    # weights; an integer instead trains that many epochs from scratch
    arg = sys.argv[2] if len(sys.argv) > 2 else "2"
    cont = arg == "cont"
    ep = 2 if cont else int(arg)
    sfx = "_cont" if cont else ("" if ep == 2 else "_e{}".format(ep))
    init = (lambda t: t if cont else None)
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("cell8: geographic holdout, bank held to the same cells, "
        "{} epochs".format(ep))
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    stages = [
        Stage("c8_knn_base",
              ["scripts/build_knn.py", "--street-file", BASE,
               "--split-mode", "cell8", "--chunk", "512",
               "--bank-block", "200000"],
              release=REL, est=20 * 60, critical=True),

        Stage("c8_knn_ext",
              ["scripts/build_knn.py", "--street-file", STACKED,
               "--split-mode", "cell8", "--chunk", "512",
               "--bank-ext", EXT, "--bank-block", "150000"],
              release=REL, est=40 * 60, critical=True),

        Stage("c8_train_base" + sfx,
              train_argv("s10_cell8_base" + sfx, BASE, KNN_BASE, ep,
                         init("s10_cell8_base")),
              release=REL, est=(30 * ep + 10) * 60, retries=1),

        Stage("c8_train_ext" + sfx,
              train_argv("s10_cell8_bank25" + sfx, STACKED, KNN_EXT, ep,
                         init("s10_cell8_bank25")),
              release=REL, est=(32 * ep + 10) * 60, retries=1),
    ]

    ok = True
    for st in stages:
        if not run_stage(st, state, deadline) and st.critical:
            ok = False
            break

    if ok:
        # both arms see the same cell8 test images, so this is paired
        run_stage(Stage("c8_eval" + sfx,
                        ["scripts/bootstrap.py", "--tags",
                         "s10_cell8_base{0},s10_cell8_bank25{0}".format(sfx),
                         "--split", "test", "--n", "5000", "--beam", "2",
                         "--score-steps", "3",
                         "--out", str(O.RUNS / ("BOOTSTRAP_cell8%s.md" % sfx))],
                        release=REL, est=25 * 60, retries=1), state, deadline)

    O.finish(state, deadline)


if __name__ == "__main__":
    main()
