"""How much does the retrieval prior actually move a logit, per row?

`docs/ARCHITECTURE_NEXT.md` records the static gates read straight from five
checkpoints: `g_cell` sits between -0.11 and +0.17 and `log_eps` floors at about
-5.1, so the worst-case static contribution to a cell logit is under a point.
That bounded the review's "product of experts can annihilate a good visual
prediction" argument -- but only on the statics, and the note says so:

    `cond` adds a per-row `delta` to both gates, so the effective gate is
    input-dependent and these statics are a floor, not the whole story.
    Measuring the delta distribution needs a forward pass over real data.
    Do that before acting on items 1 or 10.

This is that forward pass. It hooks the `cond` MLP, runs real teacher-forced
batches, and reports the distribution of the *effective* gates -- static plus
delta -- per step, together with the logit contribution they imply at the eps
floor. Teacher-forced, so it reads the cached map tokens and touches no tiles.

    OSV_RELEASE=s10 python scripts/cond_probe.py --tags d768-b350-e6,d768-b350-e6-drop70
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import splits as sp  # noqa: E402
import tile_math as tm  # noqa: E402
from dataset import (GeoStepDataset, gather_nbr, street_table,  # noqa: E402
                     to_device)
from evaluate import check_split, load_model, street_file_for  # noqa: E402


@torch.no_grad()
def probe(tag, dev, n_batches, batch):
    model, ck, d_street = load_model(tag, dev)
    prior = getattr(model, "retr", None)
    if prior is None or getattr(prior, "cond", None) is None:
        return None, ck

    mode = ck.get("split_mode", sp.PRIMARY)
    check_split(ck, mode, "val")
    sf = street_file_for(ck, d_street)
    ds = GeoStepDataset("val", street_file=sf, split_mode=mode,
                        knn_file=ck.get("knn_file"),
                        knn_k=ck.get("retr_k", 0) if ck.get("retr") else 0,
                        cache=ck.get("map_cache"))
    tbl = (street_table(config.STREET_CACHE / sf, dev)
           if ck.get("retr_mode") in ("pos", "dual") else None)

    grabbed = []
    h = prior.cond.register_forward_hook(
        lambda m, i, o: grabbed.append(o.detach().float().cpu()))
    steps = []
    try:
        for bi, b in enumerate(DataLoader(ds, batch_size=batch)):
            if bi >= n_batches:
                break
            b = to_device(b, dev)
            if tbl is not None and "nbr_row" in b:
                b["nbr_emb"] = gather_nbr(tbl, b["nbr_row"], dev)
            model(b)
            steps.append(b["step"][:, :tm.STEPS].reshape(-1).cpu())
    finally:
        h.remove()

    delta = torch.cat(grabbed).numpy()          # (rows, 2)
    st = torch.cat(steps).numpy()
    out = {"eps": float(prior.log_eps.exp()), "log_eps": float(prior.log_eps),
           "steps": {}}
    for t in range(tm.STEPS):
        m = st == t
        if not m.any():
            continue
        out["steps"][t] = {
            "cell": float(prior.g_cell[t]) + delta[m, 0],
            "sink": float(prior.g_sink[t]) + delta[m, 1],
        }
    del model, tbl
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="d768-b350-e6,d768-b350-e6-drop70")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        out, ck = probe(tag, dev, a.batches, a.batch)
        if out is None:
            print("\n{}: no conditional gate (retr_mode={!r})"
                  .format(tag, ck.get("retr_mode")))
            continue
        n = sum(len(v["cell"]) for v in out["steps"].values())
        print("\n{}   retr_mode={}   eps={:.4f}  log eps={:.2f}   {:,} rows"
              .format(tag, ck.get("retr_mode"), out["eps"], out["log_eps"], n))
        print("  {:<5} {:<6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}".format(
            "step", "gate", "p1", "p25", "median", "p75", "p99", "neg %"))
        for t, v in sorted(out["steps"].items()):
            for name in ("cell", "sink"):
                x = v[name]
                q = np.percentile(x, [1, 25, 50, 75, 99])
                print("  {:<5} {:<6} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} "
                      "{:>8.3f} {:>8.1f}%".format(
                          t, name, *q, 100 * (x < 0).mean()))
        # what that gate is worth on a logit at the eps floor
        worst = max(abs(np.percentile(v["cell"], [1, 99])).max()
                    for v in out["steps"].values())
        print("  widest effective |g_cell| (p1/p99) = {:.3f};  times log(eps) "
              "= {:.2f} logits".format(worst, worst * out["log_eps"]))

    print("\nThe review's argument needs the prior able to erase a confident "
          "visual\nprediction. Compare the last line against a policy logit "
          "spread, which is\norder 10 before the softmax.")


if __name__ == "__main__":
    main()
