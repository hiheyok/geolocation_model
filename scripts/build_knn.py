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


def cache_path(street_file, mode, k, bank_limit=0, ext=None):
    tail = "" if not bank_limit else "_bank{}k".format(bank_limit // 1000)
    tail += "" if not ext else "_" + ext
    return config.STREET_CACHE / "knn_{}_{}_k{}{}.npz".format(
        street_file.replace(".f16.npy", ""), mode, k, tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street-file", default="dual_c3.f16.npy")
    ap.add_argument("--split-mode", default=sp.PRIMARY, choices=sorted(sp.MODES))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--bank-ext", default=None,
                    help="stem of a bank extension (build_bank_ext.py): its "
                         "embeddings are appended to the bank and its rows are "
                         "addressed as n_release + i. Queries stay the release.")
    ap.add_argument("--bank-block", type=int, default=200000,
                    help="bank rows held on the card at once")
    ap.add_argument("--bank-limit", type=int, default=0,
                    help="restrict the bank to the same N train images that "
                         "train.py --limit N uses, so an arm's retrieval "
                         "memory matches its training set instead of always "
                         "being the full release. 0 = every train image.")
    a = ap.parse_args()

    ds = pq.read_table(config.DATASET_PARQUET)
    labels, shash = sp.read(ds, a.split_mode)
    seq = np.asarray(ds["sequence"].to_pylist(), dtype=object)
    _, seq_id = np.unique(seq, return_inverse=True)

    emb = np.load(config.STREET_CACHE / a.street_file, mmap_mode="r")
    n, d = emb.shape
    bank_rows = np.flatnonzero(labels == "train").astype(np.int64)
    if a.bank_limit and a.bank_limit < len(bank_rows):
        # exactly train.py's subset: same rng, same seed, same draw over the
        # train rows in parquet order
        keep = np.random.default_rng(config.SPLIT_SEED).choice(
            len(bank_rows), a.bank_limit, replace=False)
        bank_rows = bank_rows[np.sort(keep)]
        print("bank limit {:,} images (matching train.py --limit)"
              .format(a.bank_limit))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # A bank extension is a corpus that never joined the release: bank-only
    # shards with an embedding and a z16 address but no split, no tiles and no
    # training rows.  Its embeddings live in the same file, appended after the
    # release rows, so a neighbour index addresses both spaces uniformly.
    ext_n = 0
    if a.bank_ext:
        m = np.load(config.STREET_CACHE / (a.bank_ext + "_meta.npz"),
                    allow_pickle=True)
        ext_n = len(m["x16"])
        if emb.shape[0] < n + ext_n:
            raise SystemExit(
                "{} has {:,} rows but the release has {:,} and the extension "
                "{:,}; run scripts/stack_bank.py first".format(
                    a.street_file, emb.shape[0], n, ext_n))
        ext_rows = np.arange(n, n + ext_n, dtype=np.int64)
        bank_rows = np.concatenate([bank_rows, ext_rows])
        # sequence ids must not collide across the two corpora
        _, ext_seq = np.unique(m["sequence"], return_inverse=True)
        seq_id = np.concatenate([seq_id, ext_seq + seq_id.max() + 1])
        print("bank ext   {:,} images from {}".format(
            ext_n, ", ".join(str(x) for x in m["shards"])))

    # The bank no longer fits on the card: at 1.15M x 4608 it is 10.6 GB in
    # fp16.  Similarity runs against one block at a time and a running top-k is
    # merged across blocks -- identical result, bounded memory.
    nb = len(bank_rows)
    blocks = [(lo, min(lo + a.bank_block, nb))
              for lo in range(0, nb, a.bank_block)]
    bank_seq_all = torch.from_numpy(seq_id[bank_rows])

    print("bank       {:,} images, dim {}  ({:.2f} GB fp16, {} block(s) of {:,})"
          .format(nb, d, nb * d * 2 / 1e9, len(blocks), a.bank_block))
    print("split      {} {}".format(a.split_mode, shash))

    idx_out = np.zeros((n, a.k), dtype=np.int64)
    sim_out = np.zeros((n, a.k), dtype=np.float32)
    dropped = 0
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        Q = torch.from_numpy(np.asarray(emb[lo:hi], dtype=np.float32)).to(dev)
        Q = (Q / Q.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()
        qseq = torch.from_numpy(seq_id[lo:hi]).to(dev).unsqueeze(1)

        best_s, best_j = None, None
        for blo, bhi in blocks:
            rows = bank_rows[blo:bhi]
            B = torch.from_numpy(np.asarray(emb[rows], dtype=np.float32)).to(dev)
            B = (B / B.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()
            sim = (Q @ B.T).float()
            del B
            same = bank_seq_all[blo:bhi].to(dev).unsqueeze(0) == qseq
            dropped += int(same.sum())
            sim = sim.masked_fill(same, -2.0)
            kk = min(a.k, sim.shape[1])
            s, j = sim.topk(kk, dim=1)
            j = j + blo                       # into bank_rows, not the block
            del sim, same
            if best_s is None:
                best_s, best_j = s, j
            else:
                cs = torch.cat([best_s, s], 1)
                cj = torch.cat([best_j, j], 1)
                best_s, order = cs.topk(a.k, dim=1)
                best_j = torch.gather(cj, 1, order)

        # store rows in dataset.parquet order, not bank order, so the dataset
        # can index neighbours the same way it indexes everything else
        idx_out[lo:hi] = bank_rows[best_j.cpu().numpy()]
        sim_out[lo:hi] = best_s.cpu().numpy()
        if (lo // a.chunk) % 200 == 0:
            print("  {:>7,}/{:,}".format(hi, n), flush=True)

    out = cache_path(a.street_file, a.split_mode, a.k, a.bank_limit,
                     a.bank_ext)
    np.savez(out, idx=idx_out, sim=sim_out.astype(np.float16),
             split_mode=a.split_mode, split_hash=shash,
             street_file=a.street_file,
             bank_ext=(a.bank_ext or ""), bank_n=len(bank_rows))
    print("excluded   {:,} same-sequence pairs ({:.1f} per query)".format(
        dropped, dropped / n))
    print("top-1 sim  mean {:.4f}   top-{} sim mean {:.4f}".format(
        sim_out[:, 0].mean(), a.k, sim_out[:, -1].mean()))
    print("wrote      {}  ({:.1f} MB)".format(out.name, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
