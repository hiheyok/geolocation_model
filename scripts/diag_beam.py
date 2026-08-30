"""Why does a better encoder bring back the beam's degradation?

Two candidate mechanisms, and they are distinguishable:

  headroom     Beam search only pays when the correct action is findable but
               not rank 1.  A stronger policy shrinks that band -- more of the
               truth moves into rank 1, where greedy already had it -- while
               the cost of expanding confidently-wrong siblings is unchanged.
               Signature: recall@k minus top-1 shrinks.

  confidence   A stronger encoder sharpens the step-0 distribution, so one
               confident wrong first move dominates the cumulative path score
               and no later step can outvote it.
               Signature: lower entropy, higher max probability at step 0.

Teacher-forced, so this measures the policy's distribution directly with no
rollout and no tile fetching.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import tile_math as tm
from dataset import GeoStepDataset
from evaluate import load_model, street_file_for


@torch.no_grad()
def stats(tag, dev, ks=(1, 2, 4, 8, 16), batch=64):
    model, ck, d_street = load_model(tag, dev)
    ds = GeoStepDataset("val", street_file=street_file_for(d_street))
    dl = DataLoader(ds, batch_size=batch)
    steps = tm.STEPS
    rec = {k: np.zeros(steps) for k in ks}
    ent = np.zeros(steps)
    top = np.zeros(steps)
    n = 0
    for b in dl:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits, _ = model(b)
        lg = logits.float()[..., :tm.actions()]      # drop the sink column
        p = torch.softmax(lg, -1)
        a = b["action"]
        order = lg.argsort(-1, descending=True)
        hit = (order == a.unsqueeze(-1))
        for k in ks:
            rec[k] += hit[..., :k].any(-1).float().sum(0).cpu().numpy()
        ent += (-(p * p.clamp_min(1e-12).log()).sum(-1)).sum(0).cpu().numpy()
        top += p.max(-1).values.sum(0).cpu().numpy()
        n += a.shape[0]
    return {"rec": {k: rec[k] / n for k in ks}, "ent": ent / n, "top": top / n,
            "n": n, "ks": ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    o = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {t: stats(t, dev) for t in o.tags}

    ks = out[o.tags[0]]["ks"]
    print("\nRecall@k of the true action, teacher-forced val ({:,} images)\n"
          .format(out[o.tags[0]]["n"]))
    for step in (0, 1):
        print("step {}".format(step))
        print("  {:<12}".format("model") + "".join("{:>9}".format("@" + str(k)) for k in ks)
              + "{:>12}".format("headroom"))
        for t in o.tags:
            r = out[t]["rec"]
            head = r[max(ks)][step] - r[1][step]
            print("  {:<12}".format(t)
                  + "".join("{:>8.1%} ".format(r[k][step]) for k in ks)
                  + "{:>11.1%}".format(head))
        print()

    print("Step-0 sharpness (256-way; uniform entropy = {:.2f} nats)\n"
          .format(float(np.log(tm.actions()))))
    print("  {:<12} {:>10} {:>12}".format("model", "entropy", "max prob"))
    for t in o.tags:
        print("  {:<12} {:>10.3f} {:>11.1%}".format(t, out[t]["ent"][0], out[t]["top"][0]))

    if len(o.tags) == 2:
        a, b = o.tags
        print("\nverdict")
        for step in (0, 1):
            ha = out[a]["rec"][max(ks)][step] - out[a]["rec"][1][step]
            hb = out[b]["rec"][max(ks)][step] - out[b]["rec"][1][step]
            print("  step {} headroom  {}: {:.1%}   {}: {:.1%}   change {:+.1f} pp"
                  .format(step, a, ha, b, hb, 100 * (hb - ha)))
        print("  step 0 entropy   {}: {:.3f}   {}: {:.3f}   change {:+.3f}"
              .format(a, out[a]["ent"][0], b, out[b]["ent"][0],
                      out[b]["ent"][0] - out[a]["ent"][0]))


if __name__ == "__main__":
    main()
