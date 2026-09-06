"""Does an extra query level help or hurt when the bank does not have it?

Two claims of mine need testing rather than extrapolating.

**The mismatch claim.** The 2x2 in `runs/XBANK.md` measured query `L0L1` against
a bank of `L0` and found it separated *harmful* (-1.43 pp), and I generalised
that to all mismatches -- including query `L0L1L2` against an `L0L1` bank, which
is one level up and was never measured. The reasoning was that the query's
extra component scores against a dimension the bank never encoded, so a third
of the query's norm is spent on a term that cannot discriminate. That argument
is level-agnostic in form, but its *magnitude* is not, and L1 and L2 are far
more alike (both are means of 224-crop embeddings of sub-regions) than L1 and
L0 are. So it is measured here.

**The L2 claim.** The three-level test on record used 5,426 images and 1,200
queries, and weighted levels 1:1:1 or by token count -- neither chosen. This
project has twice had an unchosen constant hide a real effect (the fusion
head's 0.5 was ~10x too much; DINOv2 took 81% of the cosine on raw norms), and
its own three-level result swung 4 pp between the two weightings. So the L2
null is re-run at 47,646 images with the L2 weight **swept**.

`pyr47.f16.npy` is 47,646 KartaView images x 33 tokens x 2 encoders x 768,
token-major as 3 crops, 6 tiles, 24 tiles. **This is the only corpus where L2
exists at all**: a 6x4 grid of 224 tiles needs 1344x896 pixels and 0% of
OSV-5M frames have them.

A retrieval probe, so it bounds what any trained head could exploit -- the
top-k set is chosen by cosine before the model sees anything.

    OSV_RELEASE=s10 py scripts/pyr_levels.py --queries 5000
"""

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
from tile_pool import paired                     # noqa: E402
from query_only import great_circle              # noqa: E402

THRESH = (1, 25, 200, 750, 2500)
D_ENC = 768
# token-major: 3 crops, then 6 tiles, then 24 tiles
SPAN = {"L0": (0, 3), "L1": (3, 9), "L2": (9, 33)}


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def levels(stem, dev, block=4096):
    """(n, 1536) per level, pooled exactly as `pool_pyramid.levels` does."""
    A = np.load(config.STREET_CACHE / (stem + ".f16.npy"), mmap_mode="r")
    n = A.shape[0]
    out = {k: np.empty((n, 2 * D_ENC), np.float32) for k in SPAN}
    with torch.no_grad():
        for s in range(0, n, block):
            e = min(s + block, n)
            T = torch.from_numpy(np.asarray(A[s:e], np.float32)).to(dev)
            T = nrm(T)                       # per token, as the bank rows are
            for k, (lo, hi) in SPAN.items():
                B = T[:, lo:hi]
                out[k][s:e] = nrm(torch.cat(
                    [nrm(B[:, :, 0].mean(1)), nrm(B[:, :, 1].mean(1))], 1
                )).cpu().numpy()
            del T
    return out, n


def blend(L, spec, w2=None):
    """`spec` is a level list; equal weights unless `w2` re-weights L2.

    With `w2`, the shared levels share `1 - w2` and L2 takes `w2`, so the sweep
    varies only how much of the vector the third level owns. Equal weighting of
    three levels is w2 = 1/3, which is what the test on record used without
    ever choosing it.
    """
    if w2 is None or "L2" not in spec:
        V = sum(L[k] for k in spec) / len(spec)
    else:
        base = [k for k in spec if k != "L2"]
        V = (1.0 - w2) * sum(L[k] for k in base) / len(base) + w2 * L["L2"]
    return nrm(torch.from_numpy(V)).numpy()


def topk(Q, B, seq_q, seq_b, dev, k, block=8192):
    """Best of k neighbours, same-sequence excluded. KartaView drives repeat."""
    Qt = torch.from_numpy(Q).to(dev)
    best = torch.full((len(Q), k), -1, dtype=torch.long, device=dev)
    bestv = torch.full((len(Q), k), -2.0, device=dev)
    sq = torch.from_numpy(seq_q).to(dev)
    for s in range(0, len(B), block):
        e = min(s + block, len(B))
        S = Qt @ torch.from_numpy(B[s:e]).to(dev).T
        S[sq[:, None] == torch.from_numpy(seq_b[s:e]).to(dev)[None, :]] = -2.0
        v, i = torch.cat([bestv, S], 1).topk(k, dim=1)
        idx = torch.cat([best, torch.arange(s, e, device=dev)
                         .expand(len(Q), -1)], 1)
        best, bestv = idx.gather(1, i), v
    return best.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="pyr47")
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--k", type=int, default=16,
                    help="the agent's retrieval window; results also at top-1")
    ap.add_argument("--w2", default="",
                    help="comma-separated L2 weights to sweep, e.g. .05,.1,.2")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = np.load(config.STREET_CACHE / (a.stem + "_meta.npz"), allow_pickle=True)
    lat = np.asarray(m["lat"], np.float64)
    lon = np.asarray(m["lon"], np.float64)
    seq = np.asarray(m["sequence"]).astype("U40")
    L, n = levels(a.stem, dev)
    print("{}  {:,} images, levels {}".format(
        a.stem, n, " ".join("{}={}t".format(k, SPAN[k][1] - SPAN[k][0])
                            for k in ("L0", "L1", "L2"))), flush=True)

    # Split whole sequences, not rows: KartaView captures run along a road, so
    # a row split puts near-duplicate frames of one drive on both sides.
    uq = np.unique(seq)
    h = np.array([zlib.crc32(s.encode()) % 100 for s in uq])
    qseq = set(uq[h < 12].tolist())
    is_q = np.array([s in qseq for s in seq])
    # A seeded RANDOM sample, not the first n. `dataset.parquet` is written in
    # shard order, so the leading test rows are a different continent mix and a
    # denser one: the first 3,000 are BR/AR/ZA with a median nearest legal bank
    # row of 0.212 km, against US/DE/RU and 0.337 km for a random 3,000. Rank-1
    # retrieval scores 56.4% on that head and 49.4% on a random draw -- a 7 pp
    # gap that was published as a fact about the system. Pairing does not
    # repair it: a paired interval on a biased cohort estimates the effect on
    # that cohort. Seeded so arms stay comparable across runs.
    _te = np.flatnonzero(is_q)
    te = _te[np.sort(np.random.default_rng(config.SPLIT_SEED).choice(
        len(_te), min(a.queries, len(_te)), replace=False))]
    tr = np.flatnonzero(~is_q)
    sid = {s: i for i, s in enumerate(uq)}
    seq_i = np.array([sid[s] for s in seq], np.int64)
    print("{:,} bank, {:,} queries, split on {:,} whole sequences, "
          "{:.0f}s".format(len(tr), len(te), len(uq), time.time() - t0),
          flush=True)

    arms = [("L0", ["L0"], None), ("L0L1", ["L0", "L1"], None),
            ("L0L1L2", ["L0", "L1", "L2"], None)]
    for w in (float(x) for x in a.w2.split(",") if x.strip()):
        arms.append(("L0L1+L2@{:g}".format(w), ["L0", "L1", "L2"], w))
    V = {name: blend(L, spec, w2) for name, spec, w2 in arms}

    # query x bank. The cell that matters is L0L1L2 query against an L0L1 bank:
    # the configuration a high-res upload would hit against an OSV-5M corpus
    # that cannot carry L2.
    banks = ["L0", "L0L1", "L0L1L2"]
    cells = [(q, b) for b in banks for q, _, _ in arms
             if not (q == b == "L0")] + [("L0", "L0")]
    err = {}
    for depth in (1, a.k):
        hdr = "%-26s %9s %s" % (
            "query / bank  (any-of-%d)" % depth, "median km",
            " ".join("%8s" % ("<%gkm" % t) for t in THRESH))
        print("\n" + hdr); print("-" * len(hdr))
        for q, b in cells:
            who = topk(V[q][te], V[b][tr], seq_i[te], seq_i[tr], dev, depth)
            src = tr[who]
            e = great_circle(np.repeat(lat[te], depth),
                             np.repeat(lon[te], depth),
                             lat[src].ravel(), lon[src].ravel()
                             ).reshape(len(te), depth).min(1)
            err[(q, b, depth)] = e
            print("%-26s %9.1f %s" % (
                "%s / %s" % (q, b), np.median(e),
                " ".join("%7.1f%%" % (100 * (e < t).mean()) for t in THRESH)),
                flush=True)

        rng = np.random.default_rng(0)
        base = err[("L0L1", "L0L1", depth)]
        print("\n--- against the matched L0L1 / L0L1 arm ---")
        for q, b in cells:
            if (q, b) == ("L0L1", "L0L1"):
                continue
            cs = []
            for t in THRESH:
                lo, hi = paired(base < t, err[(q, b, depth)] < t, rng)
                cs.append("%+5.2f[%+.1f,%+.1f]%s" % (
                    100 * ((err[(q, b, depth)] < t).mean()
                           - (base < t).mean()),
                    lo, hi, " " if lo * hi > 0 else "~"))
            print("%-26s %s" % ("%s / %s" % (q, b), " ".join(cs)))

    print("\n~ spans zero. {:.0f}s total".format(time.time() - t0))


if __name__ == "__main__":
    main()
