"""One table: per-step accuracy and loss on train and test, plus rollout error.

Two different things get called "step accuracy" in this project and they must not
be mixed:

  * **Teacher-forced** -- the input tile at step t is the true prefix, so every
    step is scored independently.  This is what train.py logs, and it is the only
    version that can be compared between train and test, because a train/test gap
    is only meaningful when both sides are measured the same way.
  * **Rollout** -- the input tile at step t is whatever the beam chose at t-1, so
    errors compound and s3 is conditioned on s0-s2 being right.  This is what
    evaluate.py reports and what the median km actually comes from.

The train side is sampled from the arm's *own* training subset, reproducing
train.py --limit's draw, so "train accuracy" means the images that arm really
saw.  Everything runs in eval mode, so the train/test gap is a generalisation
gap and not a dropout artefact.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import splits as sp
import tile_math as tm
from beam import TokenSource, source_for
from dataset import GeoStepDataset, street_table
from evaluate import evaluate, load_model, street_file_for
from train import run_epoch

CAP = 5000


def subset_like_train(ds, limit, cap, seed):
    """The arm's own training images, then a fixed sample of them."""
    idx = np.arange(len(ds))
    if limit and limit < len(ds):
        idx = np.sort(np.random.default_rng(config.SPLIT_SEED).choice(
            len(ds), limit, replace=False))
    if cap and cap < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return Subset(ds, idx.tolist())


def teacher_forced(model, ds, dev, tbl, batch=64):
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        return run_epoch(model, dl, dev, None, None, tm.STEPS, street_gpu=tbl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--score-steps", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "SUMMARY.md"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        model, ck, d_street = load_model(tag, dev)
        # per arm, not once: arms in one table may be tokenised at different
        # `sub`, and a shared source scores one against the other's map
        source = source_for(ck)
        sf = street_file_for(ck, d_street)
        tbl = (street_table(config.STREET_CACHE / sf, dev)
               if ck.get("retr_mode") in ("pos", "dual") else None)
        kn = dict(knn_file=ck.get("knn_file"),
                  knn_k=ck.get("retr_k", 0) if ck.get("retr") else 0,
                  cache=ck.get("map_cache"))
        mode = ck.get("split_mode", sp.PRIMARY)
        neg = dict(n_neg=ck.get("neg", 0), neg_random=False, neg_seed=11)

        # limit is encoded in the tag, e.g. s10_n100k_e10
        limit = 0
        for part in tag.split("_"):
            if part.startswith("n") and part.endswith("k") and part[1:-1].isdigit():
                limit = int(part[1:-1]) * 1000

        tr = GeoStepDataset("train", street_file=sf, split_mode=mode, **kn, **neg)
        te = GeoStepDataset("test", street_file=sf, split_mode=mode, **kn, **neg)
        n_train_total = len(tr)
        tr_s = subset_like_train(tr, limit, a.cap, 101)
        # the same images the rollout scores, in the same order -- otherwise
        # teacher-forced and rollout accuracy are two different samples and
        # the difference between them reads as an effect that is not there
        te_idx = (np.arange(len(te)) if a.cap >= len(te) else
                  np.sort(np.random.default_rng(1234).choice(
                      len(te), a.cap, replace=False)))
        te_s = Subset(te, te_idx.tolist())

        mtr = teacher_forced(model, tr_s, dev, tbl)
        mte = teacher_forced(model, te_s, dev, tbl)
        roll = evaluate(model, te, source, dev, a.cap, beam_k=a.beam,
                        top_m=max(4, a.beam), greedy=(a.beam == 1),
                        score_steps=a.score_steps, street_gpu=tbl)

        rows.append({
            "tag": tag, "release": ck.get("release", "s01"),
            "n_train": limit or n_train_total, "epoch": ck.get("epoch"),
            "tf_train": mtr["acc"], "tf_test": mte["acc"],
            "l_train": mtr["loss"], "l_test": mte["loss"],
            "roll": roll["step_acc"],
            "median": float(np.median(roll["err"])),
            "mean": float(roll["err"].mean()),
            "hit": float((roll["err"] < 25).mean()),
        })
        print("done {:<18} train loss {:6.3f}  test loss {:6.3f}  median {:6.1f}"
              .format(tag, mtr["loss"], mte["loss"], rows[-1]["median"]),
              flush=True)
        del model, tbl
        torch.cuda.empty_cache()

    L = []
    L.append("# Per-step accuracy and loss, train vs test\n")
    L.append("Teacher-forced accuracy scores each step against the true prefix, so "
             "train and test are measured the same way. Rollout accuracy feeds the "
             "beam's own choice forward, so errors compound -- it is lower at every "
             "step and it is where the km numbers come from.\n")
    L.append("{:,} images per side; test rollout at k={}, ranked s0-s{}.\n".format(
        a.cap, a.beam, a.score_steps - 1))

    head = ("| arm | train imgs | ep | "
            "trn s0 | trn s1 | trn s2 | trn s3 | "
            "tst s0 | tst s1 | tst s2 | tst s3 | "
            "gap s0 | gap s1 | gap s2 | gap s3 | "
            "roll s0 | roll s1 | roll s2 | roll s3 | "
            "loss trn | loss tst | Δloss | median km | mean km | <25km |")
    L.append(head)
    L.append("|" + "---|" * (head.count("|") - 1))
    for r in rows:
        cells = [r["tag"], "{:,}".format(r["n_train"]), str(r["epoch"])]
        cells += ["{:.1f}%".format(100 * x) for x in r["tf_train"]]
        cells += ["{:.1f}%".format(100 * x) for x in r["tf_test"]]
        cells += ["{:+.1f}".format(100 * (a_ - b_))
                  for a_, b_ in zip(r["tf_train"], r["tf_test"])]
        cells += ["{:.1f}%".format(100 * x) for x in r["roll"]]
        cells += ["{:.3f}".format(r["l_train"]), "{:.3f}".format(r["l_test"]),
                  "{:+.3f}".format(r["l_test"] - r["l_train"]),
                  "{:.1f}".format(r["median"]), "{:.0f}".format(r["mean"]),
                  "{:.1f}%".format(100 * r["hit"])]
        L.append("| " + " | ".join(cells) + " |")

    txt = "\n".join(L) + "\n"
    Path(a.out).write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
