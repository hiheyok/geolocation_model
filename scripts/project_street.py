"""Project a street cache down to a narrower width with a fitted PCA.

`pool_street.py` halved 4608 -> 1536 by averaging crop tokens, which is free
because the three crops live in one space. Halving again cannot work that way:
[DINOv2 | SigLIP] are unrelated bases and averaging them is arbitrary. A fitted
projection is the honest version, and at 768 it measured inside noise on
any-of-32 `<25 km` (-0.17 pp [-1.00, +0.63], `scripts/width_probe.py`) while
384 did not, so 768 is where the knee is.

Two things this has to get right.

**One basis for every row.** Queries and bank must land in the same space, so
the basis is fitted once on a subsample of the *training* rows and then applied
to everything, extensions included. Fitting per-file would silently put the two
corpora in different spaces -- the failure would look like a weaker bank rather
than an error.

**Fitted on training rows only.** `--fit-from` bounds the sample to the release
rather than the bank, which looks like a split and is not one: the release is
the whole 80/10/10 split, so the basis used to be fitted with about a fifth of
its sample drawn from val and test. `--fit-all-rows` restores that behaviour
for anyone who wants it, and says in the saved basis that it was used. The
`pca768_*_pca.npz` bases already on disk were fitted the old way; refitting
them changes the coordinate system, so every 768-d arm would need re-measuring
against a rebuilt bank before its number could be compared to the others.

**The basis is saved.** Without it no future extension can be projected to
match, and the cache becomes a dead end.

Writes `<out>.f16.npy` plus `<out>_pca.npz` holding the mean and the basis.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config
import provenance as prov
import safeio
import splits as sp
from tile_pool import pca_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--fit-rows", type=int, default=200000,
                    help="rows sampled to fit the basis")
    ap.add_argument("--fit-from", type=int, default=500000,
                    help="fit only within the first N rows, i.e. the release; "
                         "0 to sample the whole file")
    ap.add_argument("--split-mode", default=sp.PRIMARY,
                    help="which split defines the training rows the basis may "
                         "be fitted on")
    ap.add_argument("--fit-all-rows", action="store_true",
                    help="fit on every row in range, held-out rows included. "
                         "Transductive: the basis then depends on the "
                         "evaluation data it will be measured against.")
    ap.add_argument("--basis", default="",
                    help="reuse a saved <stem>_pca.npz instead of fitting")
    ap.add_argument("--block", type=int, default=100000)
    a = ap.parse_args()

    src = config.STREET_CACHE / a.src
    out = config.STREET_CACHE / a.out
    if src == out:
        sys.exit("refusing to write over the source")
    X = np.load(src, mmap_mode="r")
    n, d = X.shape
    print("{}  {:,} x {}  ->  {:,} x {}   {:.2f} -> {:.2f} GB".format(
        a.src, n, d, n, a.dim, X.nbytes / 1e9, n * a.dim * 2 / 1e9), flush=True)

    if a.basis:
        z = np.load(config.STREET_CACHE / a.basis, allow_pickle=True)
        mu, P = z["mu"], z["P"]
        if P.shape != (d, a.dim):
            sys.exit("basis is {} but this needs {}".format(P.shape, (d, a.dim)))
        # Shape was the only check, and shape is not the space. Several caches
        # here share a width and are different encoders: reusing a basis
        # fitted on one to project another puts the two in coordinate systems
        # that agree on their dimensions and on nothing else, and every
        # similarity computed across them is meaningless and finite. Reusing a
        # basis across *rows* is the intended use and stays allowed; reusing
        # it across encoders is what this refuses (item 47).
        want = json.loads(str(z["encoder"])) if "encoder" in z.files else None
        got = prov.encoder_of(src)
        if want and got and want != got:
            sys.exit(
                "basis {} was fitted on embeddings from {} but {} was built "
                "by {}. Same width, different space -- the projection would "
                "succeed and mean nothing.".format(a.basis, want, a.src, got))
        if not want:
            print("note: {} predates the encoder stamp, so nothing proves it "
                  "was fitted in this cache's space".format(a.basis),
                  flush=True)
        print("reusing basis {}".format(a.basis), flush=True)
        basis_rec = None                # nothing new to publish
    else:
        hi = min(a.fit_from or n, n)
        rng = np.random.default_rng(0)
        # The basis is fitted on the TRAINING rows, not on the first `hi` of
        # them. `--fit-from 500000` bounds the sample to the release rather
        # than the bank, which reads like a split but is not one: on s10 the
        # release *is* the whole 80/10/10 split, so roughly a fifth of the fit
        # sample was val and test. PCA is unsupervised, so this is milder than
        # label leakage, but held-out queries still shaped the coordinate
        # system their similarity to the bank is measured in -- and the
        # external corpora are then compared against a basis fitted partly on
        # the internal evaluation set. Round one removed exactly this from
        # `fuse_head.py` and `width_probe.py`; the pipeline that actually
        # built `pca768_*` kept it.
        pool, note = np.arange(hi), "every row"
        if not a.fit_all_rows:
            labels, _ = sp.read(pq.read_table(config.DATASET_PARQUET),
                                a.split_mode)
            pool = np.flatnonzero(labels[:hi] == "train") \
                if len(labels) >= hi else np.arange(hi)
            if len(labels) < hi:
                sys.exit(
                    "--fit-from {:,} reaches past the {:,} rows the split "
                    "covers, so the rows beyond it cannot be shown to be "
                    "training rows. Lower --fit-from to the release length, "
                    "or pass --fit-all-rows and accept a transductive basis."
                    .format(hi, len(labels)))
            note = "the {!r} training split".format(a.split_mode)
        pick = np.sort(rng.choice(pool, min(a.fit_rows, len(pool)),
                                  replace=False))
        print("fitting on {:,} rows drawn from {} within the first {:,}"
              .format(len(pick), note, hi), flush=True)
        F = np.asarray(X[pick], dtype=np.float32)
        mu, P = pca_fit(F, a.dim, np.random.default_rng(0), fit_rows=len(pick))
        del F
        # Held back until the projection it describes has been written and
        # verified. Saving it here meant a run that stopped after this line
        # left the OLD projected bank and its old, still-valid sidecar on disk
        # while the basis path they name held NEW components. Serving and
        # external evaluation load that path to project fresh queries, so the
        # queries land in the new space and the bank sits in the old one --
        # dimensions agree, every similarity is finite, and the row-provenance
        # check cannot see it because no row changed.
        basis_rec = dict(
            mu=mu, P=P, src=a.src, dim=a.dim,
            fit_split=("all rows" if a.fit_all_rows else a.split_mode),
            fit_split_hash=("" if a.fit_all_rows
                            else sp.split_hash(a.split_mode, labels)),
            encoder=json.dumps(prov.encoder_of(src), sort_keys=True))

    Y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(n, a.dim))
    t0 = time.time()
    for s in range(0, n, a.block):
        e = min(s + a.block, n)
        Y[s:e] = ((np.asarray(X[s:e], dtype=np.float32) - mu) @ P
                  ).astype(np.float16)
        el = time.time() - t0
        print("  {:>9,}/{:,}  {:.0f}s  eta {:.0f}s".format(
            e, n, el, el * (n - e) / max(e, 1)), flush=True)
    Y.flush()
    # One output row per input row, in order, so the output describes exactly
    # the rows the input did.
    prov.carry(src, out, n, dim=a.dim,
               projection=a.basis or (out.name.replace(".f16.npy", "")
                                      + "_pca.npz"))

    # the projection must be reproducible from the saved basis, not just from
    # this process; recompute a handful of rows the long way round
    rng = np.random.default_rng(1)
    for i in rng.choice(n, 5, replace=False):
        want = (np.asarray(X[i], dtype=np.float32) - mu[0]) @ P
        got = np.asarray(Y[i], dtype=np.float32)
        assert np.allclose(want, got, atol=3e-2), "row {} disagrees".format(i)
    zeros = 0
    for s in range(0, n, a.block):
        blk = np.asarray(Y[s:min(s + a.block, n)], dtype=np.float32)
        zeros += int((np.abs(blk).sum(1) == 0).sum())
    if zeros:
        raise SystemExit("{:,} all-zero rows".format(zeros))
    # One generation: the basis is published only now, after the data it
    # describes is complete and has been checked against it. Written to a
    # temporary name and re-read before it is published, because the check
    # above compares against the in-memory mu/P -- which proves this process
    # was consistent with itself, not that the file anyone else will load
    # reproduces the projection. That is the claim the comment above makes and
    # it was not being tested.
    if basis_rec is not None:
        final = Path(str(out).replace(".f16.npy", "") + "_pca.npz")
        tmp = final.with_name(final.name + ".tmp.npz")
        np.savez(tmp, **basis_rec)
        z = np.load(tmp, allow_pickle=True)
        for i in rng.choice(n, 3, replace=False):
            want = (np.asarray(X[i], dtype=np.float32) - z["mu"][0]) @ z["P"]
            got = np.asarray(Y[i], dtype=np.float32)
            if not np.allclose(want, got, atol=3e-2):
                tmp.unlink()
                raise SystemExit(
                    "the saved basis does not reproduce row {} of the output "
                    "it was fitted for; refusing to publish it".format(i))
        del z
        safeio.replace_from(tmp, final)
        print("saved basis alongside the output, verified by re-reading it",
              flush=True)
    print("\nwrote {} in {:.0f}s; 5 rows verified, no zero rows".format(
        out.name, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
