"""Scale the retrieval bank to 25 shards, holding the training set fixed.

The s10 control measured the bank at roughly six times the training set per
image, so this scales the axis that was winning: 15 more shards, 750,000 more
images, into the bank only.  The training set stays at the same 400,180 images
and the test set stays byte-identical, so the arm differs from s10_n400k_e2 in
exactly one thing.

Bank-only images cost an embedding and nothing else -- their z16 address is
arithmetic on lat/lon, so no tiles are fetched and dataset.parquet, the split
hash and every existing checkpoint are untouched.

Resumable and deadline-aware, reusing the machinery from overnight.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage

REL = "s10"
EXT = "bank_ext"
DINO = O.DINO
SIGLIP = O.SIGLIP
STACKED = "dual_c3_bank25"
# derived, not written out: build_knn generates the same name from the
# same function, so the two cannot disagree
KNN = config.knn_name(STACKED + ".f16.npy", "sequence", ext=EXT)

# same arm as s10_n400k_e2 in every respect but the bank it reads
SHIP_BANK = ["--street-file", STACKED + ".f16.npy",
             "--pool", "attn", "--pool-q", "4", "--pos", "both",
             "--map-layers", "1", "--neg", "4",
             "--retr", "--retr-k", "16", "--retr-mode", "dual", "--d-key", "128",
             "--knn-file", KNN]


def emb_rows(name, dim):
    p = ROOT / "cache" / "street" / REL / (name + ".f16.npy")
    if not p.exists():
        return 0
    try:
        a = np.load(p, mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != dim:
            return 0
        probe = np.linspace(0, a.shape[0] - 1, 96).astype(np.int64)
        r = np.asarray(a[probe], dtype=np.float32)
        return a.shape[0] if bool((np.abs(r).sum(1) > 0).all()) else 0
    except Exception:
        return 0


def ext_n():
    p = ROOT / "data" / "processed" / REL / (EXT + ".parquet")
    if not p.exists():
        return 0
    import pyarrow.parquet as pq
    return pq.ParquetFile(p).metadata.num_rows


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    start = O.now()
    deadline = start + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("bank25: 15 shards into the bank, training set held at 400,180")
    log("deadline {}  ({:.1f} h)".format(O.hhmm(deadline), hours))

    n_rel = 500000
    stages = [
        Stage("bx_meta", ["scripts/build_bank_ext.py", "--shards",
                          ",".join("{:02d}".format(i) for i in range(10, 25)),
                          "--out", EXT],
              release=REL, est=15 * 60, critical=True,
              check=lambda: ext_n() >= 700000),

        Stage("bx_dino", ["scripts/embed_street.py", "--parquet", EXT + ".parquet",
                          "--model", DINO, "--crops", "3", "--out", EXT + "_dino"],
              release=REL, est=100 * 60, critical=True,
              check=lambda: emb_rows(EXT + "_dino", 2304) >= 700000),

        Stage("bx_siglip", ["scripts/embed_street.py", "--parquet", EXT + ".parquet",
                            "--model", SIGLIP, "--crops", "3",
                            "--out", EXT + "_siglip"],
              release=REL, est=90 * 60, critical=True,
              check=lambda: emb_rows(EXT + "_siglip", 2304) >= 700000),

        Stage("bx_concat", ["scripts/concat_street.py", "--a", EXT + "_dino",
                            "--b", EXT + "_siglip", "--out", EXT + "_dual"],
              release=REL, est=8 * 60, critical=True,
              check=lambda: emb_rows(EXT + "_dual", 4608) >= 700000),

        Stage("bx_stack", ["scripts/stack_bank.py", "--base", "dual_c3",
                           "--ext", EXT + "_dual", "--out", STACKED],
              release=REL, est=12 * 60, critical=True,
              check=lambda: emb_rows(STACKED, 4608) >= n_rel + 700000),

        Stage("bx_knn", ["scripts/build_knn.py", "--street-file",
                         STACKED + ".f16.npy", "--split-mode", "sequence",
                         "--chunk", "256", "--bank-ext", EXT,
                         "--bank-block", "150000"],
              release=REL, est=60 * 60, critical=True),
    ]

    ok = True
    for st in stages:
        if not run_stage(st, state, deadline):
            ok = False
            break

    if ok:
        run_stage(Stage("bank25_train",
                        ["src/train.py", "--tag", "s10_n400k_bank25",
                         "--epochs", "2", "--batch", "64", "--select", "hit",
                         "--sel-n", "2000", "--val-n", "5000"] + SHIP_BANK,
                        release=REL, est=45 * 60, retries=1), state, deadline)
        run_stage(Stage("bank25_eval",
                        ["scripts/bootstrap.py", "--tags",
                         "s10_n400k_e2,s10_n400k_bank25", "--split", "test",
                         "--n", "5000", "--beam", "2", "--score-steps", "3",
                         "--out", str(O.RUNS / "BOOTSTRAP_bank25.md")],
                        release=REL, est=20 * 60, retries=1), state, deadline)
    else:
        log("prep incomplete -- not training")

    O.finish(state, deadline)


if __name__ == "__main__":
    main()
