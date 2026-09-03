"""Does the encoder blend want to change with the zoom step?

Measured, at retrieval level: SigLIP alone beats DINOv2 alone by 5 pp at a
2500 km threshold and loses by 2.2 pp at 25 km, with the benefit of the second
encoder growing monotonically with the threshold (+0.6, +1.8, +3.4, +4.8 pp at
25/200/750/2500 km).  This agent's cells are 2504 km wide at step 0, 156 km at
step 1, 9.8 km at step 2 and 611 m at step 3, so the encoder that deserves the
weight is a different one at each step -- and today a single `StreetProj` shared
across steps can express exactly one ratio for all of them.

`--enc-gate` adds one scalar per (step, encoder block): ten parameters, zero-
init, applied as exp(g), so it starts as the identity and `--init` from an
ungated checkpoint is exact.

Two arms, both continuing from the shipping best for the same number of epochs,
because "continue training and it improved" is not evidence about the gate:

    gate    d4608-b115-e6 + 2 epochs, --enc-gate
    ctrl    d4608-b115-e6 + 2 epochs, unchanged

The metric comparison is paired.  But the *mechanistic* readout is worth more
than the metric here and cannot be faked: if the learned gates come out flat,
the hypothesis is dead whatever the hit rate does, and if they come out ordered
-- more SigLIP at step 0, less at step 3 -- that is the measured scale split
reproducing itself inside the trained model.  Ten parameters cannot overfit
their way to that shape by accident.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage
from marathon import ARCH, BAL_BANK, KNN_BALBANK_SEQ, boot, tiles_up

REL = "s10"
FROM = "d4608-b115-e6"


def cont(tag, extra=(), epochs=2):
    return ["src/train.py", "--tag", tag, "--epochs", str(epochs),
            "--batch", "64", "--limit", "400000", "--init", FROM,
            "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
            "--split-mode", "sequence", "--street-file", BAL_BANK,
            "--knn-file", KNN_BALBANK_SEQ] + ARCH + list(extra)


def report():
    """Print the learned gates as a SigLIP-relative log ratio per step."""
    import numpy as np
    import torch
    ck = torch.load(config.CHECKPOINTS / "s10_bal_gate.pt",
                    map_location="cpu", weights_only=False)
    g = ck["model"].get("street.gate")
    if g is None:
        log("no street.gate in the checkpoint -- did --enc-gate reach train.py?")
        return
    g = g.float().numpy()
    width = [2504, 156, 9.8, 0.611, 0.611]
    log("")
    log("learned encoder gates, exp(g), by step")
    log("  %-6s %10s %9s %9s %9s" % ("step", "cell km", "dino", "siglip",
                                     "siglip/dino"))
    for t in range(g.shape[0]):
        d, s = float(np.exp(g[t, 0])), float(np.exp(g[t, 1]))
        log("  %-6d %10.3f %9.3f %9.3f %9.3f" % (t, width[t], d, s, s / d))
    r = np.exp(g[:, 1] - g[:, 0])
    log("  ratio at step 0 / ratio at step 3: %.3f" % (r[0] / r[3]))
    log("  (>1 means the model leans on SigLIP for the coarse decision, which "
        "is what the retrieval measurement predicted)")


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("per-step encoder gate against a matched-epoch control")
    log("deadline {}".format(O.hhmm(deadline)))
    if not tiles_up():
        log("tile server is down; --select hit decodes 2,000 val images an "
            "epoch, so there is nothing to run")
        return

    plan = [
        Stage("eg_gate", cont("s10_bal_gate", ["--enc-gate"]),
              release=REL, est=35 * 60, retries=1),
        Stage("eg_ctrl", cont("s10_bal_c8"), release=REL, est=35 * 60,
              retries=1),
        Stage("eg_eval",
              boot("d4608-b115-e6,s10_bal_gate,s10_bal_c8",
                   "BOOTSTRAP_encgate.md"), release=REL, est=15 * 60,
              retries=1),
    ]
    for st in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        run_stage(st, state, deadline)
    report()
    log("done")


if __name__ == "__main__":
    main()
