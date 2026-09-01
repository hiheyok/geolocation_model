"""Rollout metrics: greedy and beam, against the baseline ladder."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import splits as sp
import tile_math as tm
from baselines import great_circle_km, print_table, report
from beam import TokenSource, search
from dataset import GeoStepDataset, gather_nbr, street_table
from model import GeoAgent


def load_model(tag, dev):
    ck = torch.load(config.CHECKPOINTS / (tag + ".pt"), map_location=dev,
                    weights_only=False)
    # Infer the street width from the checkpoint so older files stay loadable.
    d_street = ck["model"]["street.proj.weight"].shape[1]
    m = GeoAgent(d_street=d_street, n_actions=tm.actions(), n_steps=tm.STEPS + 1,
                 map_layers=ck.get("map_layers", 0),
                 pool=ck.get("pool", "mean"), n_pool_q=ck.get("pool_q", 4),
                 pos=ck.get("pos", "learned"),
                 sink=ck.get("neg", 0) > 0,
                 map_loop=ck.get("map_loop", False),
                 mem=ck.get("mem", "none"),
                 d_mem=ck.get("d_mem", 64),
                 retr=ck.get("retr", False),
                 retr_tau=ck.get("retr_tau", 0.07),
                 retr_mode=ck.get("retr_mode", "scalar"),
                 d_key=ck.get("d_key", 128),
                 enc_gate=ck.get("enc_gate", False),
                 geo=ck.get("geo", "none"),
                 d_geo=ck.get("d_geo", 128)).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck, d_street


def street_file_for(ck, dim, override=None):
    """The embedding cache this checkpoint was trained against.

    Order of trust: an explicit override, then the name the checkpoint recorded,
    then -- only for checkpoints predating that field -- a scan for a cache of
    the right width.

    The scan cannot be the primary answer any more. Several caches now share
    width 4608: the release's own, a bank extension's, and the two stacked into
    one index space. They are not interchangeable -- the extension holds no
    release rows at all -- and a scan would silently return whichever sorts
    first.
    """
    if override:
        return override
    named = ck.get("street_file")
    if named and (config.STREET_CACHE / named).exists():
        got = np.load(config.STREET_CACHE / named, mmap_mode="r").shape[1]
        if got != dim:
            raise SystemExit(
                "checkpoint names {} (width {}) but the model wants {}".format(
                    named, got, dim))
        return named
    for f in sorted(config.STREET_CACHE.glob("*.f16.npy")):
        if np.load(f, mmap_mode="r").shape[1] == dim:
            print("street     {}  (checkpoint records none; matched on width)"
                  .format(f.name))
            return f.name
    raise SystemExit("no cached embeddings with dim {}".format(dim))


def check_split(ck, mode, split, allow_dirty=False):
    """A number is only comparable to numbers from the same benchmark.

    The dangerous case is not a mismatched name, it is a mismatched *set*.
    The stress split reshuffles which images are held out, so evaluating a
    sequence-trained model on cell8-val scores it largely on images it was
    trained on and produces a spectacular, meaningless number.  So this
    measures the contamination directly rather than trusting the labels.
    """
    import pyarrow.parquet as pq

    # Release first: the split hash only proves two splits agree over the same
    # rows, and every cache here is indexed by row order, so a checkpoint read
    # against a different release silently pairs each image with another
    # image's embedding rather than failing.
    rel = ck.get("release", "s01")
    if rel != config.RELEASE:
        raise SystemExit(
            "release mismatch: checkpoint trained on {!r}, OSV_RELEASE is {!r}. "
            "Every cache is indexed by row order in that release's "
            "dataset.parquet, so this comparison would be meaningless. "
            "Re-run with OSV_RELEASE={}.".format(rel, config.RELEASE, rel))

    tbl = pq.read_table(config.DATASET_PARQUET)
    live_lab, live = sp.read(tbl, mode)
    was, want = ck.get("split_mode"), ck.get("split_hash")
    trained_on = was or sp.PRIMARY

    if was is not None and was == mode and want != live:
        raise SystemExit(
            "split hash mismatch on {!r}: checkpoint {}, data on disk {}. "
            "dataset.parquet changed since training; re-run "
            "scripts/resplit.py or retrain.".format(mode, want, live))

    stamp = "" if was is not None else "  (checkpoint predates the split stamp)"
    if trained_on == mode:
        print("split      {} {}  eval on {!r}{}".format(mode, live, split, stamp))
        return

    train_lab, _ = sp.read(tbl, trained_on)
    ev = live_lab == split
    dirty = float((train_lab[ev] == "train").mean())
    print("split      evaluating {!r} a model trained on {!r} -- transfer test"
          .format(mode, trained_on))
    print("           {:.1%} of this {} set was in the training set of {!r}"
          .format(dirty, split, trained_on))
    if dirty > 0.005 and not allow_dirty:
        raise SystemExit(
            "REFUSING: {:.1%} of the evaluation set is training data for this "
            "checkpoint, so any metric here is scored partly on memorised "
            "images. To run the {!r} stress test properly, train a model with "
            "--split-mode {}. Pass --allow-dirty only for a deliberate "
            "train-set readout.".format(dirty, mode, mode))


def evaluate(model, ds, source, dev, n=None, beam_k=16, top_m=16,
             greedy=False, batch=32, sink_prune=1.0, score_steps=None,
             street_gpu=None, sample_seed=1234):
    """n < len(ds) draws a *seeded random* subset, not the first n rows.

    Split order is the DuckDB join order over the shards, so the head of the
    file is not a random draw over geography: the first 5,000 test rows score a
    test loss 0.87 lower than a random 5,000 of the same split.  Seeded, so the
    sample is identical across arms and paired tests stay valid.
    """
    if n is None or n >= len(ds):
        idx = np.arange(len(ds))
    else:
        idx = np.sort(np.random.default_rng(sample_seed).choice(
            len(ds), n, replace=False))
    errs, radii, step_hit = [], [], np.zeros(tm.STEPS)
    t0 = time.time()
    for lo in range(0, len(idx), batch):
        sel = idx[lo:lo + batch]
        street = torch.from_numpy(
            np.asarray(ds.street[ds.rows[sel]], dtype=np.float32)).to(dev)
        nbrs = None
        if getattr(ds, "knn_k", 0):
            r = ds.rows[sel]
            nbrs = [torch.from_numpy(ds.all_x16[ds.knn_idx[r]].astype(np.int64)).to(dev),
                    torch.from_numpy(ds.all_y16[ds.knn_idx[r]].astype(np.int64)).to(dev),
                    torch.from_numpy(ds.knn_sim[r]).to(dev)]
            if street_gpu is not None:
                nbrs.append(gather_nbr(
                    street_gpu,
                    torch.from_numpy(ds.knn_idx[r].astype(np.int64)), dev))
        res = search(model, street, source, dev, beam_k, top_m, greedy=greedy,
                     sink_prune=sink_prune, score_steps=score_steps, nbrs=nbrs)
        for i, r in zip(sel, res):
            b = r["best"]
            errs.append(great_circle_km(np.array(b["lat"]), np.array(b["lon"]),
                                        np.array(ds.lat[i]), np.array(ds.lon[i])))
            radii.append(r["confidence_radius_km"])
            step_hit += (np.array(b["path"]) == ds.action[i]).astype(float)
    el = time.time() - t0
    errs = np.array(errs)
    return {"err": errs, "radius": np.array(radii),
            "step_acc": step_hit / len(idx), "secs": el, "n": len(idx)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="g16")
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="permit evaluating on images the checkpoint trained on")
    ap.add_argument("--split-mode", default=None, choices=sorted(sp.MODES),
                    help="benchmark to evaluate on; defaults to the one the\n"
                         "checkpoint was trained against")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--street-file", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="accuracy vs test-time compute: beam_k in 1,2,4,8,16,32")
    ap.add_argument("--sink-prune", type=float, default=1.0,
                    help="drop a beam once p(sink) exceeds this; 1.0 = never, "
                         "the sink still discounts the score either way")
    ap.add_argument("--score-steps", type=int, default=None,
                    help="rank beams on the first N steps only; later steps "
                         "expand greedily and do not enter the path score")
    ap.add_argument("--ks", default=None,
                    help="explicit beam widths, comma separated; overrides --sweep")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, d_street = load_model(a.tag, dev)
    mode = a.split_mode or ck.get("split_mode", sp.PRIMARY)
    check_split(ck, mode, a.split, a.allow_dirty)
    sf = street_file_for(ck, d_street, a.street_file)
    street_gpu = None
    if ck.get("retr_mode") in ("pos", "dual"):
        street_gpu = street_table(config.STREET_CACHE / sf, dev)
    ds = GeoStepDataset(a.split, street_file=sf, split_mode=mode,
                        knn_file=ck.get("knn_file"),
                        knn_k=ck.get("retr_k", 0) if ck.get("retr") else 0)
    src = TokenSource(tm.G)
    km = ck.get("val_km")
    sel = ("" if km is None or km != km else
           "  val_km {:.1f} (selected on {})".format(km, ck.get("select", "loss")))
    print("checkpoint {}  release {}  epoch {}  val_loss {:.4f}{}".format(
        a.tag, ck.get("release", "s01"), ck["epoch"], ck["val_loss"], sel))
    print("{} split, {:,} images evaluated\n".format(a.split, min(a.n, len(ds))))

    ks = ([int(k) for k in a.ks.split(",")] if a.ks else
          [1, 2, 4, 8, 16, 32] if a.sweep else [a.beam])
    rows = []
    for k in ks:
        m = evaluate(model, ds, src, dev, a.n, beam_k=k, top_m=max(4, k),
                     greedy=(k == 1), sink_prune=a.sink_prune,
                     score_steps=a.score_steps, street_gpu=street_gpu)
        r = report("agent beam_k={}".format(k), m["err"])
        r["secs"] = m["secs"]
        r["step_acc"] = m["step_acc"]
        rows.append(r)
        print("beam_k {:<3} {:5.1f}s   step acc "
              .format(k, m["secs"]) +
              "  ".join("s{} {:5.1%}".format(t, m["step_acc"][t])
                        for t in range(tm.STEPS)))
    print()
    print_table(rows)
    print("\ntile cache: {:,} hits, {:,} live fetches".format(src.n_hit, src.n_miss))


if __name__ == "__main__":
    main()
