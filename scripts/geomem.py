"""A learned key per map tile, in the policy readout rather than the fusion input.

The earlier rejection of tile memory is stale in three separate ways, all
checked before this was built:

  * **Mechanism.** What was measured (`--mem z4`) fed *one* vector for the
    current tile's z4 ancestor into the fusion input, identical across all 256
    actions, so it could not say "child 37 rather than child 38". This is a
    per-action key.
  * **Density.** Occupancy was measured at 50k images. At 500k, z8 went from
    14.6 images per populated cell to 106.3, and the share of test images
    landing in a z8 cell no training image occupies is 0.2%.
  * **Leakage.** The fact that killed it was that the count table
    `p(child | parent)` scored 6.58% at step 2 where the model scored 5.9%.
    Re-measured at s10: the count table is 8.0% and the model is 42.2%. The
    model is now 5.3x above co-location, so a tile table can no longer be
    dismissed as measuring the split.

That last number also sets the expectation. Under teacher forcing a per-tile
*scalar* converges to exactly that count table, so `--geo bias` should add
close to nothing -- it is the control, not a candidate. Everything a
`--geo key` arm earns above it is the query-dependent part, which is the only
part that could be new.

Both arms continue from the shipping checkpoint for two epochs, and the control
is `s10_bal_c8` -- the same checkpoint trained the same two epochs with nothing
added, which already exists. Zero-init table and gate, so the arms start as
exactly the baseline.

The gate and the learned key norms are the readout that matters. A flat gate
kills it whatever the hit rate does, exactly as it did for the per-step encoder
gate; norms that rise with cell occupancy would say the table is fitting where
it has support.
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


def cont(tag, extra, epochs=2):
    return ["src/train.py", "--tag", tag, "--epochs", str(epochs),
            "--batch", "64", "--limit", "400000", "--init", FROM,
            "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
            "--split-mode", "sequence", "--street-file", BAL_BANK,
            "--knn-file", KNN_BALBANK_SEQ] + ARCH + list(extra)


def report():
    """Gate values, and whether the learned keys track cell occupancy."""
    import numpy as np
    import torch
    import pyarrow.parquet as pq
    import splits as sp

    ds = pq.read_table(config.DATASET_PARQUET)
    lat = np.clip(np.asarray(ds["lat"], np.float64), -85.0511, 85.0511)
    lon = np.asarray(ds["lon"], np.float64)
    n = 1 << 16
    s = np.sin(np.radians(lat))
    x16 = np.clip(((lon + 180.0) / 360.0 * n), 0, n - 1).astype(np.int64)
    y16 = np.clip(((0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n),
                  0, n - 1).astype(np.int64)
    tr = sp.read(ds, "sequence")[0] == "train"

    for tag in ("s10_geo_bias", "s10_geo_key"):
        p = config.CHECKPOINTS / (tag + ".pt")
        if not p.exists():
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        w = ck["model"]["geo.emb.weight"].float().numpy()
        gate = float(ck["model"]["geo.gate"])
        log("")
        log("{}  gate {:+.4f}  table {} rows x {}".format(
            tag, gate, w.shape[0], w.shape[1]))
        for z, off in ((4, 0), (8, 256)):
            side = 1 << z
            key = (y16 >> (16 - z)) * side + (x16 >> (16 - z))
            u, c = np.unique(key[tr], return_counts=True)
            rows = off + u
            nrm = np.linalg.norm(w[rows], axis=1)
            dead = np.linalg.norm(
                w[off:off + side * side], axis=1)
            empty = np.setdiff1d(np.arange(side * side), u)
            log("  z{}  populated {:,}  |key| mean {:.4f}  max {:.4f}   "
                "unpopulated |key| mean {:.4f}".format(
                    z, len(u), nrm.mean(), nrm.max(),
                    dead[empty].mean() if len(empty) else float("nan")))
            if len(u) > 20 and nrm.std() > 0:
                r = np.corrcoef(np.log(c), nrm)[0, 1]
                log("       corr(log images in cell, |key|) = {:+.3f}".format(r))
        log("  UNK |key| {:.4f}".format(float(np.linalg.norm(w[-1]))))


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("per-tile keys in the policy readout, dense z4+z8")
    log("deadline {}".format(O.hhmm(deadline)))
    if not tiles_up():
        log("tile server is down; --select hit needs it every epoch")
        return

    plan = [
        Stage("gm_bias", cont("s10_geo_bias", ["--geo", "bias"]),
              release=REL, est=27 * 60, retries=1),
        Stage("gm_key", cont("s10_geo_key", ["--geo", "key", "--d-geo", "128"]),
              release=REL, est=27 * 60, retries=1),
        Stage("gm_eval",
              boot("d4608-b115-e6,s10_bal_c8,s10_geo_bias,s10_geo_key",
                   "BOOTSTRAP_geomem.md"), release=REL, est=10 * 60,
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
