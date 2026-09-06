"""Which side has to change: the query, the bank, or both?

Adding a tile level to the crops is worth about +3 pp when *both* sides are
rebuilt. The shipping constraint is that the bank is 3.4M rows and days of GPU,
so the useful question is how much of that survives changing only one side.

The 2x2, measured rather than argued:

    query       bank        what it tests
    crops       crops       the incumbent
    crops+tiles crops       change only the uploaded image
    crops       crops+tiles change only the corpus
    crops+tiles crops+tiles the ceiling

The asymmetry is the interesting part. Retrieval is a cosine, so with a blended
query `(L0+L1)/2` against an `L0` bank the score is

    cos(q, b) ~ 1/2 cos(L0_q, L0_b) + 1/2 cos(L1_q, L0_b)

and the second term compares tile features against crop features -- half the
query spent on a dimension the bank never encoded. The mirrored case has the
same algebra but not the same consequence, because *which* side carries the
orphaned half changes what it does to the ranking: a constant-ish term added to
every bank row shifts scores together, while a term that varies per bank row
reorders them.

Run on the whole release now that `tile_cache` has covered it: a ~400k bank
instead of the 96k subset the first pass used, and the release's own sequence
split. Still a retrieval probe, not the agent.

    OSV_RELEASE=s10 py scripts/xbank.py --queries 3000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import splits as sp                              # noqa: E402
from tile_pool import paired                     # noqa: E402
from osv_pyramid import complete_rows            # noqa: E402
from query_only import great_circle, top1        # noqa: E402

THRESH = (1, 25, 200, 750, 2500)
D_ENC = 768


def build(rel_rows, tile_pos, dev, block=4096):
    """(L0, MIX) for every row, without holding the token array.

    The token form is (n, 9, 2, 768) -- 13.8 GB at 500,000 rows as float16,
    which is more than this machine wants to hold beside two output banks. The
    two 1536-d representations are all anything downstream needs, so they are
    accumulated block by block and the tokens never exist all at once.
    """
    C_all = np.load(config.STREET_CACHE / "dual_c3.f16.npy", mmap_mode="r")
    T_all = np.load(config.STREET_CACHE / "tile6.f16.npy", mmap_mode="r")
    h = C_all.shape[1] // 2
    nc = h // D_ENC
    n = len(rel_rows)
    L0 = np.empty((n, 2 * D_ENC), np.float32)
    MIX = np.empty((n, 2 * D_ENC), np.float32)
    nrm = lambda t: t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        for s in range(0, n, block):
            e = min(s + block, n)
            C = torch.from_numpy(
                np.asarray(C_all[rel_rows[s:e]], np.float32)).to(dev)
            C = torch.stack([C[:, :h].reshape(-1, nc, D_ENC),
                             C[:, h:].reshape(-1, nc, D_ENC)], dim=2)
            T = torch.from_numpy(
                np.asarray(T_all[tile_pos[s:e]], np.float32)).to(dev)
            T = torch.stack([T[:, :, :D_ENC], T[:, :, D_ENC:]], dim=2)
            C, T = nrm(C), nrm(T)          # per token, as the bank rows are
            l0 = nrm(torch.cat([nrm(C[:, :, 0].mean(1)),
                                nrm(C[:, :, 1].mean(1))], dim=1))
            l1 = nrm(torch.cat([nrm(T[:, :, 0].mean(1)),
                                nrm(T[:, :, 1].mean(1))], dim=1))
            L0[s:e] = l0.cpu().numpy()
            MIX[s:e] = nrm((l0 + l1) / 2.0).cpu().numpy()
            del C, T
    return L0, MIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--split-mode", default="sequence")
    ap.add_argument("--bank-sizes", default="",
                    help="also score the matched arm against subsampled banks, "
                         "to see whether the gain decays with density")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    rel, pos = complete_rows("tile6")
    ds = pq.read_table(config.DATASET_PARQUET)
    lat = np.asarray(ds["lat"], np.float64)[rel]
    lon = np.asarray(ds["lon"], np.float64)[rel]
    seq = np.asarray(ds["sequence"]).astype("U40")[rel]
    spl = sp.read(ds, a.split_mode)[0][rel]
    tr = np.flatnonzero(spl == "train")
    # A seeded RANDOM sample, not the first n. `dataset.parquet` is written in
    # shard order, so the leading test rows are a different continent mix and a
    # denser one: the first 3,000 are BR/AR/ZA with a median nearest legal bank
    # row of 0.212 km, against US/DE/RU and 0.337 km for a random 3,000. Rank-1
    # retrieval scores 56.4% on that head and 49.4% on a random draw -- a 7 pp
    # gap that was published as a fact about the system. Pairing does not
    # repair it: a paired interval on a biased cohort estimates the effect on
    # that cohort. Seeded so arms stay comparable across runs.
    _te = np.flatnonzero(spl == "test")
    te = _te[np.sort(np.random.default_rng(config.SPLIT_SEED).choice(
        len(_te), min(a.queries, len(_te)), replace=False))]
    print("{:,} tiled release rows: {:,} bank, {:,} queries\n".format(
        len(rel), len(tr), len(te)), flush=True)

    L0, MIX = build(rel, pos, dev)
    print("both representations built in {:.0f}s".format(time.time() - t0),
          flush=True)

    hdr = "%-34s %9s %s" % ("query / bank", "median km",
                            " ".join("%8s" % ("<%gkm" % t) for t in THRESH))
    print(hdr); print("-" * len(hdr))
    err = {}
    combos = [("crops", "crops", L0, L0, "   <- the incumbent"),
              ("crops+tiles", "crops", MIX, L0, "   <- query only"),
              ("crops", "crops+tiles", L0, MIX, "   <- bank only"),
              ("crops+tiles", "crops+tiles", MIX, MIX, "   <- both")]
    for qn, bn, Q, B, note in combos:
        who, _ = top1(Q[te], B[tr], seq[tr], seq[te], dev)
        e = great_circle(lat[te], lon[te], lat[tr][who], lon[tr][who])
        err[(qn, bn)] = e
        print("%-34s %9.1f %s" % (
            "%s / %s%s" % (qn, bn, note), np.median(e),
            " ".join("%7.1f%%" % (100 * (e < t).mean()) for t in THRESH)),
            flush=True)

    rng = np.random.default_rng(0)
    base = err[("crops", "crops")]
    print("\n--- against the incumbent (crops / crops) ---")
    for k in [("crops+tiles", "crops"), ("crops", "crops+tiles"),
              ("crops+tiles", "crops+tiles")]:
        cells = []
        for t in THRESH:
            lo, hi = paired(base < t, err[k] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[k] < t).mean() - (base < t).mean()),
                lo, hi, " " if lo * hi > 0 else "~"))
        print("%-26s %s" % ("%s / %s" % k, " ".join(cells)))

    # Does the matched gain survive a denser bank? Every pyramid number so far
    # was measured on 96k-400k rows and the shipping bank is 3.4M. A gain that
    # decays with density is a gain measured where the bank was weakest, and
    # the decay rate is what decides whether a rebuild is worth 16 hours.
    sizes = [int(x) for x in a.bank_sizes.split(",") if x.strip()]
    if sizes:
        print("\n--- matched gain against bank density ---")
        print("%-12s %10s %12s %10s" % ("bank rows", "crops", "crops+tiles",
                                        "gain pp"))
        print("-" * 48)
        rs = np.random.default_rng(0)
        for n in sizes:
            if n > len(tr):
                continue
            sub = tr[np.sort(rs.choice(len(tr), n, replace=False))]
            got = []
            for V in (L0, MIX):
                who, _ = top1(V[te], V[sub], seq[sub], seq[te], dev)
                e = great_circle(lat[te], lon[te], lat[sub][who], lon[sub][who])
                got.append(100 * (e < 25).mean())
            print("%-12s %9.1f%% %11.1f%% %+10.2f" % (
                "{:,}".format(n), got[0], got[1], got[1] - got[0]), flush=True)

    print("\n~ spans zero. {:.0f}s total".format(time.time() - t0))


if __name__ == "__main__":
    main()
