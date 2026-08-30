"""Teacher-forced training.  No rollout: every row is independent.

Watch per-step tile accuracy above all else -- it localises failure.  Step 0
flat means the street encoder or the region prior; step 3 flat means the map
tokens are not discriminative at z12.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import splits as sp
import tile_math as tm
from dataset import GeoStepDataset, gather_nbr, street_table
from model import GeoAgent, param_report


def param_groups(model, wd):
    """No decay on norms, biases or embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "pos" in name or "step" in name) else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def soft_targets(cell_xy, g, temp):
    """Sub-cell position -> a distance-weighted distribution over the g*g cells.

    Hard cross-entropy scores the neighbouring cell exactly as wrong as the
    opposite hemisphere.  At step 0 a cell is ~2500 km across, so that is the
    difference between a near miss and a catastrophe, and the loss cannot see
    it.  Weighting by squared distance to each cell centre, in cell units, makes
    near misses cheap.  exp(-d^2/T) normalised is just softmax(-d^2/T).

    T is per step, and it has to be: a cell is ~2500 km wide at step 0 and 611 m
    at step 3, so one temperature in cell units means "a neighbour is nearly
    right" about a different continent and about the next street.  Measured, a
    single T=0.25 across all steps costs 11% median km and triples the beam's
    degradation with width.
    """
    idx = torch.arange(g * g, device=cell_xy.device)
    cx = (idx % g).float() + 0.5
    cy = torch.div(idx, g, rounding_mode="floor").float() + 0.5
    d2 = (cell_xy[..., :1] - cx) ** 2 + (cell_xy[..., 1:2] - cy) ** 2
    return torch.softmax(-d2 / temp, dim=-1)


def run_epoch(model, loader, dev, opt=None, sched=None, steps=tm.STEPS, clip=1.0,
              noise=0.0, drop=0.0, smooth=0.0, soft=None, g=tm.G, sink_w=1.0,
              street_gpu=None):
    """Street embeddings are frozen and unique per image, which is exactly what
    the model memorises.  Images cannot be re-augmented without re-embedding, so
    noise and dropout on the embedding stand in for image augmentation."""
    soft = [0.0] * steps if soft is None else soft
    train = opt is not None
    n_sink = 0.0
    sink_hit = 0.0
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=smooth)
    sl1 = nn.SmoothL1Loss()

    tot = {"loss": 0.0, "click": 0.0, "n": 0}
    hit = np.zeros(steps)
    per_step_loss = np.zeros(steps)
    uv_err = 0.0

    for batch in loader:
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
        if street_gpu is not None and "nbr_row" in b:
            b["nbr_emb"] = gather_nbr(street_gpu, b["nbr_row"], dev)
        if train and (noise > 0 or drop > 0):
            st = b["street"]
            if noise > 0:
                st = st + torch.randn_like(st) * (noise * st.std())
            if drop > 0:
                st = torch.nn.functional.dropout(st, p=drop, training=True)
            b = dict(b, street=st)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logits, uv = model(b)
            lp = torch.log_softmax(logits.float(), dim=-1) if any(soft) else None
            losses = []
            for t in range(steps):
                if soft[t] > 0:
                    w = soft_targets(b["cell_xy"][:, t], g, soft[t])
                    losses.append(-(w * lp[:, t]).sum(-1).mean())
                else:
                    losses.append(ce(logits[:, t].float(), b["action"][:, t]))
            click = sl1(uv.float(), b["uv"])
            loss = sum(losses) + click          # uniform weights, lambda = 1
            if "neg_tokens" in b:
                # Off-path views: the answer is not in this tile, so the target
                # is the sink.  Without these the sink class has no positives --
                # every teacher-forced row is on-path by construction.
                nbrs = ((b["nbr_x"], b["nbr_y"], b["nbr_sim"])
                        if "nbr_x" in b else None)
                nl = model.policy_from(b["street"], b["neg_tokens"],
                                       b["neg_x0"], b["neg_y0"], b["neg_step"],
                                       nbrs)
                tgt = torch.full((nl.shape[0],), nl.shape[-1] - 1,
                                 dtype=torch.long, device=nl.device)
                sink_loss = nn.functional.cross_entropy(nl.float(), tgt)
                loss = loss + sink_w * sink_loss

        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched:
                sched.step()

        n = b["action"].shape[0]
        tot["loss"] += float(loss) * n
        tot["click"] += float(click) * n
        tot["n"] += n
        with torch.no_grad():
            pred = logits.argmax(-1)
            hit += (pred == b["action"]).float().sum(0).cpu().numpy()
            for t in range(steps):
                per_step_loss[t] += float(losses[t]) * n
            uv_err += float((uv.float() - b["uv"]).abs().mean()) * n
            if "neg_tokens" in b:
                # can the model tell an off-path tile from an on-path one at all?
                sink_hit += float((nl.argmax(-1) == nl.shape[-1] - 1).float().sum())
                n_sink += nl.shape[0]

    n = max(tot["n"], 1)
    return {"loss": tot["loss"] / n, "click": tot["click"] / n,
            "uv_mae": uv_err / n, "acc": hit / n, "step_loss": per_step_loss / n,
            "sink_acc": (sink_hit / n_sink) if n_sink else float("nan")}


def fmt(tag, m, steps):
    acc = "  ".join("s{} {:5.1%}".format(t, m["acc"][t]) for t in range(steps))
    sk = ("" if m["sink_acc"] != m["sink_acc"]
          else "  sink {:5.1%}".format(m["sink_acc"]))
    return ("{:<5} loss {:6.3f}  click {:6.4f}  uv_mae {:6.4f}{}  | {}"
            .format(tag, m["loss"], m["click"], m["uv_mae"], sk, acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4,
                    help="loader processes; measured 77.8 -> 68.4 ms/iter "
                         "against 0, and numerically neutral")
    ap.add_argument("--limit", type=int, default=0,
                    help="train on a random N-image subset; for learning curves. "
                         "Scale --epochs inversely to hold optimizer steps fixed.")
    ap.add_argument("--overfit", type=int, default=0,
                    help="train and eval on the same N images; loss must reach ~0")
    ap.add_argument("--tag", default="g16")
    ap.add_argument("--emb-noise", type=float, default=0.0,
                    help="gaussian noise on street embeddings, in units of their std")
    ap.add_argument("--emb-drop", type=float, default=0.0)
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--soft", default="0",
                    help="soft-label temperature in squared cell units: one "
                         "value for all steps, or one per step (0,0,0.25,0.5). "
                         "0 = hard cross-entropy at that step")
    ap.add_argument("--street-file", default="embeddings.f16.npy")
    ap.add_argument("--map-layers", type=int, default=0,
                    help="self-attention layers over map tokens; 0 = linear readout")
    ap.add_argument("--retr", action="store_true",
                    help="content-keyed retrieval prior over child cells")
    ap.add_argument("--retr-k", type=int, default=16,
                    help="neighbours used; must not exceed the cached k")
    ap.add_argument("--retr-tau", type=float, default=0.07)
    ap.add_argument("--retr-mode", choices=["scalar", "cond", "pos", "dual"],
                    default="scalar",
                    help="scalar: one gate. cond: per-step gates modulated by "
                         "retrieval quality. dual: adds a separately keyed "
                         "negative branch. Every gate starts at zero, so all "
                         "three begin identical to a model with no retrieval.")
    ap.add_argument("--d-key", type=int, default=128)
    ap.add_argument("--knn-file", default=None,
                    help="defaults to the cache matching --street-file and "
                         "--split-mode")
    ap.add_argument("--mem", choices=["none", "z4", "rand"], default="none",
                    help="persistent z4 region prior; rand freezes it at init")
    ap.add_argument("--d-mem", type=int, default=64)
    ap.add_argument("--mem-drop", type=float, default=0.0,
                    help="probability of zeroing the region vector per row")
    ap.add_argument("--split-mode", default=sp.PRIMARY, choices=sorted(sp.MODES),
                    help="benchmark to train against; see src/splits.py")
    ap.add_argument("--pool", choices=["mean", "attn"], default="mean",
                    help="map readout into fusion; mean is a class histogram")
    ap.add_argument("--pool-q", type=int, default=4,
                    help="learned queries when --pool attn")
    ap.add_argument("--map-loop", action="store_true",
                    help="reuse one map block --map-layers times (weight-tied) "
                         "instead of that many distinct blocks")
    ap.add_argument("--neg", type=int, default=0,
                    help="off-path negatives per image; >0 enables the sink class")
    ap.add_argument("--sink-w", type=float, default=1.0,
                    help="weight on the sink loss")
    ap.add_argument("--pos", choices=["learned", "rope", "both"], default="learned",
                    help="map token positions: additive embedding, 2D rotary, or both")
    ap.add_argument("--select", choices=["hit", "km", "loss"], default="hit",
                    help="what decides which epoch is kept, all measured on a "
                         "greedy decode over --sel-n val images. hit: the "
                         "<25 km rate, the lowest-variance statistic here and "
                         "the default. km: median km, which is the headline "
                         "number but swings by hundreds of km between adjacent "
                         "epochs at this sample size. loss: val loss, a proxy, "
                         "and the only mode that keeps training offline.")
    ap.add_argument("--sel-n", type=int, default=1000,
                    help="val images decoded per epoch for --select km")
    ap.add_argument("--val-n", type=int, default=5000,
                    help="images in the per-epoch val LOSS pass. The loss "
                         "is a diagnostic -- selection is --select km over "
                         "--sel-n images -- so walking all 50k of a large "
                         "release costs more than the training step it "
                         "reports on. 0 = the whole split.")
    ap.add_argument("--sel-k", type=int, default=1,
                    help="beam width for selection; 1 = greedy, which is what "
                         "the plan specifies -- full beam is for evaluation")
    ap.add_argument("--sel-score-steps", type=int, default=3,
                    help="ranking depth when --sel-k > 1; ignored at k=1")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    steps = tm.STEPS
    soft = [float(v) for v in str(a.soft).split(",")]
    soft = soft * steps if len(soft) == 1 else soft
    if len(soft) != steps:
        raise SystemExit("--soft needs 1 or {} values".format(steps))

    knn_file = a.knn_file
    if a.retr and knn_file is None:
        knn_file = config.knn_name(a.street_file, a.split_mode)
    kn = dict(knn_file=knn_file, knn_k=a.retr_k if a.retr else 0)
    tr = GeoStepDataset("train", street_file=a.street_file, n_neg=a.neg,
                        split_mode=a.split_mode, **kn)
    # val negatives are seeded, so sink accuracy is measured on the same tiles
    # every epoch and across arms
    va = GeoStepDataset("val", street_file=a.street_file, n_neg=a.neg,
                        split_mode=a.split_mode, **kn,
                        neg_random=False, neg_seed=11)
    # capture before any Subset wrapping -- the stamp belongs to the benchmark,
    # not to whatever slice of it this run happened to use
    split_mode, split_hash = tr.split_mode, tr.split_hash
    if a.limit and a.limit < len(tr):
        keep = np.random.default_rng(config.SPLIT_SEED).choice(
            len(tr), a.limit, replace=False)
        tr = Subset(tr, sorted(keep.tolist()))
    va_ds = va          # rollout needs the dataset, not a Subset view
    if a.val_n and a.val_n < len(va):
        # evenly spread rather than random, so the sample is the same one
        # every epoch and every arm and the losses stay comparable
        pick = np.linspace(0, len(va) - 1, a.val_n).astype(np.int64)
        va = Subset(va, np.unique(pick).tolist())
    if a.overfit:
        tr = Subset(tr, list(range(a.overfit)))
        va = tr
        va_ds = None
    print("train {:,} images   val {:,} images   grid g={} steps={}"
          .format(len(tr), len(va), tm.G, steps))

    # persistent_workers matters on Windows, where processes are spawned rather
    # than forked and would otherwise be rebuilt every epoch
    dl_kw = dict(num_workers=a.workers, pin_memory=(dev == "cuda"))
    if a.workers:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)
    ltr = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=False, **dl_kw)
    lva = DataLoader(va, batch_size=a.batch, shuffle=False, **dl_kw)

    street_gpu = None
    if a.retr and a.retr_mode in ("pos", "dual"):
        street_gpu = street_table(config.STREET_CACHE / a.street_file, dev)

    # Selecting on val loss was a regression against the written plan, which
    # says "early-stop on val median km error, not on loss -- the losses are
    # proxies and can improve while the metric that matters stalls".  Measured
    # here, they do worse than stall: they move the wrong way.  A greedy decode
    # is cheap because steps 0-2 read z0/z4/z8, all of which are cached
    # exhaustively; only the step-3 z12 view and the z16 click can miss, and
    # TokenSource keeps live fetches for the rest of the run.
    src = rollout = None
    if a.select in ("km", "hit"):
        from beam import TokenSource
        from evaluate import evaluate as rollout
        src = TokenSource(tm.G)

    model = GeoAgent(d_street=tr.dataset.dim_street if hasattr(tr, "dataset")
                     else tr.dim_street,
                     n_actions=tm.actions(), n_steps=steps + 1,
                     map_layers=a.map_layers, pool=a.pool,
                     n_pool_q=a.pool_q, pos=a.pos, sink=(a.neg > 0),
                     map_loop=a.map_loop,
                     mem=a.mem, d_mem=a.d_mem, mem_drop=a.mem_drop,
                     retr=a.retr, retr_tau=a.retr_tau,
                     retr_mode=a.retr_mode, d_key=a.d_key).to(dev)
    rep, total = param_report(model)
    print("\n" + rep + "\n")

    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.95))
    total_steps = max(1, a.epochs * math.ceil(len(tr) / a.batch))
    warm = min(a.warmup, max(1, total_steps // 10))

    def lr_at(s):
        if s < warm:
            return s / warm
        p = (s - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = float("inf")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        mtr = run_epoch(model, ltr, dev, opt, sched, steps,
                        street_gpu=street_gpu,
                        noise=a.emb_noise, drop=a.emb_drop, smooth=a.smooth,
                        soft=soft, sink_w=a.sink_w)
        with torch.no_grad():
            mva = run_epoch(model, lva, dev, None, None, steps,
                            street_gpu=street_gpu, soft=soft,
                            sink_w=a.sink_w)
        print("ep {:>3}  {:5.1f}s  lr {:.2e}".format(ep, time.time() - t0,
                                                     sched.get_last_lr()[0]), flush=True)
        print("   " + fmt("train", mtr, steps), flush=True)
        print("   " + fmt("val", mva, steps), flush=True)

        km = hit = float("nan")
        if rollout is not None and va_ds is not None:
            model.eval()
            r = rollout(model, va_ds, src, dev, a.sel_n, beam_k=a.sel_k,
                        top_m=max(4, a.sel_k), greedy=(a.sel_k == 1),
                        score_steps=(None if a.sel_k == 1 else a.sel_score_steps),
                        street_gpu=street_gpu)
            km = float(np.median(r["err"]))
            hit = float((r["err"] < 25).mean())
            print("   {:<5} median {:7.1f} km   <25km {:5.1%}   {:,} images  "
                  "{:4.1f}s".format("sel", km, hit, r["n"], r["secs"]),
                  flush=True)

        # minimised, so the hit rate enters negated; the median breaks ties
        # at the 0.1 pp granularity of a 1-2k sample
        crit = {"hit": -hit + km * 1e-9,
                "km": km,
                "loss": mva["loss"]}[a.select]
        if crit == crit and crit < best:
            best = crit
            torch.save({"model": model.state_dict(), "g": tm.G, "steps": steps,
                        "map_layers": a.map_layers, "street_file": a.street_file,
                        "pool": a.pool, "pool_q": a.pool_q, "pos": a.pos,
                        "soft": a.soft, "neg": a.neg, "map_loop": a.map_loop,
                        "mem": a.mem, "d_mem": a.d_mem,
                        "retr": a.retr, "retr_k": a.retr_k,
                        "retr_mode": a.retr_mode, "d_key": a.d_key,
                        "retr_tau": a.retr_tau, "knn_file": knn_file,
                        "split_mode": split_mode,
                        "split_hash": split_hash,
                        "release": config.RELEASE,
                        "epoch": ep, "val_loss": mva["loss"],
                        "val_km": km, "val_hit": hit, "select": a.select,
                        "sel_n": a.sel_n, "sel_k": a.sel_k},
                       config.CHECKPOINTS / (a.tag + ".pt"))
    if best == float("inf"):
        raise SystemExit(
            "no checkpoint was written: the {!r} criterion never produced a "
            "finite value. A training run that saves nothing is a failure, "
            "not a result.".format(a.select))
    print("\nbest val {} {:.4f} -> {}".format(
        {"hit": "-<25km", "km": "median km", "loss": "loss"}[a.select],
        best, a.tag + ".pt"))


if __name__ == "__main__":
    main()
