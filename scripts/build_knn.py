"""Precompute visual neighbours: for every image, its nearest TRAIN images.

This is the key half of content-keyed memory.  A tile-ID table cannot help in a
region never seen in training because the key does not exist there; retrieval by
street appearance always has a key, so the memory transfers to unseen geography.

Two things here are not optional.

Same-sequence exclusion.  OSV-5M captures run consecutively along a road, and
the benchmark splits on sequence -- so a *training* query sitting in the bank
alongside its own near-duplicate frames would retrieve itself and read off the
answer exactly, while a val query never can.  The model would then learn to
trust a signal far stronger at train time than at eval time, and the whole
experiment would measure that mismatch.  Neighbours from the query's own
sequence are dropped before the top-k.

Bank follows the split.  The bank is the train side of one split mode, so the
cache is written per (street file, split mode) and never shared between them.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import splits as sp


def cache_path(street_file, mode, k):
    return config.STREET_CACHE / "knn_{}_{}_k{}.npz".format(
        street_file.replace(".f16.npy", ""), mode, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street-file", default="dual_c3.f16.npy")
    ap.add_argument("--split-mode", default=sp.PRIMARY, choices=sorted(sp.MODES))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    labels, shash = sp.read(ds, a.split_mode)
    seq = np.asarray(ds["sequence"].to_pylist(), dtype=object)
    _, seq_id = np.unique(seq, return_inverse=True)

    emb = np.load(config.STREET_CACHE / a.street_file, mmap_mode="r")
    n, d = emb.shape
    bank_rows = np.flatnonzero(labels == "train").astype(np.int64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    B = torch.from_numpy(np.asarray(emb[bank_rows], dtype=np.float32)).to(dev)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-6)
    B = B.half()
    bank_seq = torch.from_numpy(seq_id[bank_rows]).to(dev)

    print("bank       {:,} train images, dim {}".format(len(bank_rows), d))
    print("split      {} {}".format(a.split_mode, shash))

    idx_out = np.zeros((n, a.k), dtype=np.int64)
    sim_out = np.zeros((n, a.k), dtype=np.float32)
    dropped = 0
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        Q = torch.from_numpy(np.asarray(emb[lo:hi], dtype=np.float32)).to(dev)
        Q = (Q / Q.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()
        sim = (Q @ B.T).float()
        same = bank_seq.unsqueeze(0) == torch.from_numpy(seq_id[lo:hi]).to(dev).unsqueeze(1)
        dropped += int(same.sum())
        sim = sim.masked_fill(same, -2.0)
        s, j = sim.topk(a.k, dim=1)
        # store rows in dataset.parquet order, not bank order, so the dataset
        # can index neighbours the same way it indexes everything else
        idx_out[lo:hi] = bank_rows[j.cpu().numpy()]
        sim_out[lo:hi] = s.cpu().numpy()

    out = cache_path(a.street_file, a.split_mode, a.k)
    np.savez(out, idx=idx_out, sim=sim_out.astype(np.float16),
             split_mode=a.split_mode, split_hash=shash,
             street_file=a.street_file)
    print("excluded   {:,} same-sequence pairs ({:.1f} per query)".format(
        dropped, dropped / n))
    print("top-1 sim  mean {:.4f}   top-{} sim mean {:.4f}".format(
        sim_out[:, 0].mean(), a.k, sim_out[:, -1].mean()))
    print("wrote      {}  ({:.1f} MB)".format(out.name, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
