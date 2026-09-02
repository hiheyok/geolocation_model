"""Run bootstrap.py over whichever of the named checkpoints actually exist.

A ladder that runs out of time leaves its last rung untrained. The evaluation
stage still names it, so `bootstrap.py` fails on a missing tag and the whole
comparison is lost -- including the arms that *did* finish, and including the
one-minute window in which the tile server was still up.

That is the wrong failure. A partial ladder is worth reading: the pooling work
was decided on its c4 and c6 rungs and the c4 number alone would have been
informative. So filter to the tags that have a checkpoint, say plainly which
were dropped, and evaluate the rest.

Refuses to run if fewer than two tags survive, because a bootstrap of one arm
against nothing is not a comparison.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", default="5000")
    ap.add_argument("--beam", default="2")
    ap.add_argument("--score-steps", default="3")
    a = ap.parse_args()

    want = [t.strip() for t in a.tags.split(",") if t.strip()]
    have, missing = [], []
    for t in want:
        (have if (config.CHECKPOINTS / (t + ".pt")).exists() else missing).append(t)

    if missing:
        print("no checkpoint, dropped from the comparison: {}"
              .format(", ".join(missing)), flush=True)
    if len(have) < 2:
        raise SystemExit("only {} of {} tags have checkpoints; a bootstrap "
                         "needs at least two arms".format(len(have), len(want)))
    print("evaluating: {}\n".format(", ".join(have)), flush=True)

    cmd = [sys.executable, "scripts/bootstrap.py", "--tags", ",".join(have),
           "--split", a.split, "--n", a.n, "--beam", a.beam,
           "--score-steps", a.score_steps, "--out", a.out]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
