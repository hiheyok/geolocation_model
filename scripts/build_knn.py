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
import provenance as prov
import splits as sp
import tile_math as tm


def cache_path(street_file, mode, k, bank_limit=0, ext=None):
    """config.knn_name owns the convention; train.py derives the same name."""
    return config.STREET_CACHE / config.knn_name(
        street_file, mode, k, bank_limit, ext)



def held_cells(ds, labels, zc):
    """The cells this split holds out, at the split's OWN zoom.

    `cell_z8` was read whatever the mode said, so a cell12 or cell16 split
    compared z8 cell ids against z12/z16 ids -- different numbering, so almost
    nothing matched, so the extension served neighbours from precisely the
    regions being held out and the transfer question went unasked while the
    run printed a plausible "N dropped" line (item 60).

    Both sides are derived the same way here, from the z16 address, so they
    cannot disagree about what a cell id means. Latent until now: only
    `sequence` (no held cells at all) and `cell8` are in use, and at z8 the
    column happened to be right.
    """
    # tile_math.project is scalar (math.sin, min/max), so the array form is
    # written out here exactly as build_dataset and dataset.py write it.
    lat = np.clip(np.asarray(ds["lat"], dtype=np.float64),
                  -tm.MAX_LAT, tm.MAX_LAT)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    sin = np.sin(np.radians(lat))
    px = (lon + 180.0) / 360.0
    py = 0.5 - np.log((1.0 + sin) / (1.0 - sin)) / (4.0 * np.pi)
    n16 = 1 << (4 * tm.STEPS)
    x16 = np.clip((px * n16).astype(np.int64), 0, n16 - 1)
    y16 = np.clip((py * n16).astype(np.int64), 0, n16 - 1)
    return cell_ids(x16, y16, zc)[labels != "train"]


def cell_ids(x16, y16, zc):
    """z16 addresses -> cell ids at zoom zc. One definition, used both sides."""
    sh = 4 * tm.STEPS - zc
    return ((x16.astype(np.int64) >> sh) * (1 << zc)
            + (y16.astype(np.int64) >> sh))


def bank_rows_for(ds, labels, mode, ext_stem=None, bank_limit=0):
    """The rows that make up the bank, in the order they are addressed.

    Release train rows first, then a bank extension appended after them, minus
    any extension image sitting in a cell this split holds out. Exported because
    the demo server has to reconstruct exactly the same bank the cache was built
    from -- it used to approximate it by the extension rows that happened to be
    someone's neighbour, which quietly lost 29,469 images.
    """
    import config
    rows = np.flatnonzero(labels == "train").astype(np.int64)
    if bank_limit and bank_limit < len(rows):
        keep = np.random.default_rng(config.SPLIT_SEED).choice(
            len(rows), bank_limit, replace=False)
        rows = rows[np.sort(keep)]
    if not ext_stem:
        return rows, None
    m = prov.bank_ext(ext_stem, config.RELEASE)
    n_rel = len(labels)
    keep = np.ones(len(m["x16"]), dtype=bool)
    zc = sp.MODES[mode]
    if zc is not None:
        held = np.unique(held_cells(ds, labels, zc))
        keep = ~np.isin(cell_ids(m["x16"], m["y16"], zc), held)
    return np.concatenate([rows, np.arange(n_rel, n_rel + len(keep))[keep]]), m


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
    # two different counts, equal until a bank extension exists: n_rel is
    # how many images the release has and therefore how many queries there
    # are; emb may hold more, because bank-only rows are appended after it
    n_rel, d = len(labels), emb.shape[1]
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
        m = prov.bank_ext(a.bank_ext, config.RELEASE)
        ext_n = len(m["x16"])
        if emb.shape[0] < n_rel + ext_n:
            raise SystemExit(
                "{} has {:,} rows but the release has {:,} and the extension "
                "{:,}; run scripts/stack_bank.py first".format(
                    a.street_file, emb.shape[0], n_rel, ext_n))
        # Length was the only thing ever checked, and every wrong order
        # satisfies it too. The stacked file records release-then-extension;
        # this is where that record is read (item 66).
        prov.check_stack(config.STREET_CACHE / a.street_file,
                         [np.asarray(ds["image_id"]), m["image_id"]],
                         a.street_file)
        ext_rows = np.arange(n_rel, n_rel + ext_n, dtype=np.int64)
        ext_keep = np.ones(ext_n, dtype=bool)

        zc = sp.MODES[a.split_mode]
        if zc is not None:
            # A cell split holds whole cells out of training to ask whether the
            # model transfers to unseen regions. The extension has no split
            # label, so without this it would serve neighbours from exactly the
            # held-out cells and the question would go unasked.
            held = np.unique(held_cells(ds, labels, zc))
            ext_keep = ~np.isin(cell_ids(m["x16"], m["y16"], zc), held)
            print("bank ext   {:,} of {:,} dropped: they sit in z{} cells this "
                  "split holds out".format(int((~ext_keep).sum()), ext_n, zc))

        ext_rows = ext_rows[ext_keep]
        bank_rows = np.concatenate([bank_rows, ext_rows])
        # Factorise the two corpora TOGETHER, so a sequence appearing in both
        # gets one id. Offsetting the extension's ids made the corpora disjoint
        # by construction, which is not a fact about the data: one real drive
        # crossing the boundary then had two ids, and same-sequence exclusion
        # -- the entire reason this column exists -- silently stopped applying
        # to it (item 61).
        both = np.concatenate([seq.astype("U40"),
                               np.asarray(m["sequence"]).astype("U40")])
        _, seq_id = np.unique(both, return_inverse=True)
        shared = len(np.intersect1d(seq.astype("U40"),
                                    np.asarray(m["sequence"]).astype("U40")))
        print("bank ext   {:,} sequence ids shared with the release{}".format(
            shared, " -- exclusion now covers them" if shared else ""))
        print("bank ext   {:,} images kept from {}".format(
            len(ext_rows), ", ".join(str(x) for x in m["shards"])))

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

    # Queries normalised once into host memory: the inner loop below runs
    # len(blocks) times over them, and re-reading 4.6 GB from the memmap each
    # pass would dominate the matmul it feeds.
    Qh = torch.empty((n_rel, d), dtype=torch.float16)
    for lo in range(0, n_rel, 8192):
        hi = min(lo + 8192, n_rel)
        blk = torch.from_numpy(np.asarray(emb[lo:hi], dtype=np.float32))
        Qh[lo:hi] = (blk / blk.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()
    qseq_h = torch.from_numpy(seq_id[:n_rel])

    # running top-k over the whole query set, because the bank is now the outer
    # loop; -3 is below any cosine and below the -2 a masked pair scores
    best_s = torch.full((n_rel, a.k), -3.0, dtype=torch.float32)
    best_j = torch.zeros((n_rel, a.k), dtype=torch.int64)
    dropped = 0
    for bi, (blo, bhi) in enumerate(blocks):
        # one read of this slice of the bank, reused by every query
        B = torch.empty((bhi - blo, d), dtype=torch.float16, device=dev)
        rows = bank_rows[blo:bhi]
        for lo in range(0, len(rows), 8192):
            hi = min(lo + 8192, len(rows))
            blk = torch.from_numpy(
                np.asarray(emb[rows[lo:hi]], dtype=np.float32)).to(dev)
            B[lo:hi] = (blk / blk.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()
            del blk
        bseq = bank_seq_all[blo:bhi].to(dev).unsqueeze(0)

        for lo in range(0, n_rel, a.chunk):
            hi = min(lo + a.chunk, n_rel)
            Q = Qh[lo:hi].to(dev, non_blocking=True)
            sim = (Q @ B.T).float()
            same = bseq == qseq_h[lo:hi].to(dev).unsqueeze(1)
            dropped += int(same.sum())
            sim = sim.masked_fill(same, -2.0)
            kk = min(a.k, sim.shape[1])
            s, j = sim.topk(kk, dim=1)
            j = j + blo                       # into bank_rows, not the block
            del sim, same
            # merge this block's best into the running best for these queries
            cs = torch.cat([best_s[lo:hi].to(dev), s], 1)
            cj = torch.cat([best_j[lo:hi].to(dev), j], 1)
            ms, order = cs.topk(a.k, dim=1)
            best_s[lo:hi] = ms.cpu()
            best_j[lo:hi] = torch.gather(cj, 1, order).cpu()
        del B, bseq
        torch.cuda.empty_cache()
        print("  bank block {}/{}  ({:,} of {:,} bank rows)".format(
            bi + 1, len(blocks), bhi, nb), flush=True)

    # store rows in dataset.parquet order, not bank order, so the dataset can
    # index neighbours the same way it indexes everything else
    idx_out = bank_rows[best_j.numpy()]
    sim_out = best_s.numpy()

    # Excluded rows are masked to -2.0 rather than removed, so a query with
    # fewer than k legal neighbours in the bank keeps the placeholders and
    # they reach the cache as ordinary neighbours with an ordinary index
    # (item 64). Their weight is near zero -- `score = sim/tau + ...` with
    # tau ~ 0.07 makes -2.0 worth exp(-29) against a real 0.9 -- but "near
    # zero" is not "absent", and the index still points at an excluded row.
    #
    # Fail rather than emit them. With a 3.4M-row bank this cannot happen
    # short of a pathological query, and if it ever does the operator should
    # lower --k rather than train on rows the exclusion meant to remove. The
    # caches on disk hold none: min similarity is 0.340.
    short = sim_out <= -1.99
    if short.any():
        rows_hit = int(short.any(1).sum())
        raise SystemExit(
            "{:,} of {:,} queries have fewer than k={} legal neighbours, so "
            "{:,} slots would carry an excluded row at similarity -2.0. Lower "
            "--k, or widen the bank. (This became reachable when same-sequence "
            "exclusion started applying across the corpus boundary.)"
            .format(rows_hit, len(sim_out), a.k, int(short.sum())))

    out = cache_path(a.street_file, a.split_mode, a.k, a.bank_limit,
                     a.bank_ext)
    np.savez(out, idx=idx_out, sim=sim_out.astype(np.float16),
             split_mode=a.split_mode, split_hash=shash,
             street_file=a.street_file,
             bank_ext=(a.bank_ext or ""), bank_n=len(bank_rows),
             bank_rows=bank_rows)
    print("excluded   {:,} same-sequence pairs ({:.1f} per query)".format(
        dropped, dropped / n_rel))
    print("top-1 sim  mean {:.4f}   top-{} sim mean {:.4f}".format(
        sim_out[:, 0].mean(), a.k, sim_out[:, -1].mean()))
    print("wrote      {}  ({:.1f} MB)".format(out.name, out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
