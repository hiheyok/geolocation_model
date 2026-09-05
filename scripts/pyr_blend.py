"""Where the pyramid's levels and the fusion head each earn their keep.

The fusion head is a negative result on its own (-11.30 pp at 25 km against
plain mean pooling) and the concatenation of head and mean is the one arm that
beat *both* at 750 km. That is not an averaging artifact -- it is the signature
of two scorers making different mistakes at coarse scale. This measures the
family that observation sits in, which nobody swept.

**What the current concat actually is.** `fuse_head.py` builds it as

    combo = concat([l2(base_un), l2(Z)])

Each block is unit-normalised before concatenation, so for two such vectors

    cos_combo = (cos_mean + cos_head) / 2

exactly. The "fusion" is an equal-weight average of two similarities, and the
0.5 was never chosen -- it fell out of concatenating two unit blocks. Write the
blend with explicit weights and the whole family opens up:

    concat([sqrt(1-w) * l2(mean), sqrt(w) * l2(head)])
      ->  cos = (1-w) * cos_mean + w * cos_head

which is the same lesson as `--scale-b 4.03` for [DINOv2 | SigLIP]: in a
concatenated vector the blend weight is decided by block norms, and not
choosing it is still choosing one.

**Why this is the resolution question.** The pyramid exists because a
higher-resolution source supports more levels -- 3 crops, then +6 tiles, then
+24. `baseline()` weights the levels equally, which is as arbitrary as the 0.5.
If deeper levels want *more* weight as they get sharper, then more resolution
buys more than one extra averaged token set; if they want less, the extra
levels are dilution and the ceiling is near. Same machinery answers both, since
every component here is a unit block and every score is a convex combination:

    cos = sum_c w_c * cos_c ,  w on the simplex

so four similarity matrices are computed once and every weighting is free.

Also measured: **cascade** instead of blend. Blending makes one vector serve
both regimes. Retrieving a wide candidate set with the coarse-strong scorer and
re-ranking it with the fine-strong one uses each only where it wins.

    OSV_RELEASE=s10 py scripts/pyr_blend.py --pyr-stem pyr47 --queries 3000
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
import maskio                                    # noqa: E402
from tile_pool import l2, paired                 # noqa: E402
from tile_match import dense_sim, topk_stats     # noqa: E402
from fuse_head import FuseHead                   # noqa: E402

THRESH = (1, 25, 200, 750, 2500)
D_ENC = 768


def per_level(pyr, level_of, block=4096):
    """One unit vector per pyramid level, built the way `base_un` builds one.

    For level l and encoder e: mean the level's tokens, L2 them; concatenate
    the two encoders; L2 the pair. Every component this script blends is a
    unit block, which is what makes the cosine of a weighted concatenation an
    exactly weighted sum of the components' cosines.

    Blocked over rows because the raw cache is (n, 33, 2, 768) -- 9.6 GB as
    float32, against 293 MB for one level's output.
    """
    lo = np.asarray(level_of)
    n_lvl = int(lo.max()) + 1
    n = len(pyr)
    out = [np.empty((n, 2 * D_ENC), np.float32) for _ in range(n_lvl)]
    for s in range(0, n, block):
        e = min(s + block, n)
        B = np.asarray(pyr[s:e], np.float32)
        B /= np.linalg.norm(B, axis=-1, keepdims=True).clip(1e-6)
        for l in range(n_lvl):
            m = B[:, lo == l]                       # (b, tokens_l, 2, 768)
            out[l][s:e] = l2(np.concatenate(
                [l2(m[:, :, 0].mean(1)), l2(m[:, :, 1].mean(1))], axis=1))
    return out


def level_mean(levels):
    """The published baseline: equal weight per level, inside each encoder."""
    per_enc = []
    for i in range(2):
        per_enc.append(l2(np.stack([V[:, i * D_ENC:(i + 1) * D_ENC]
                                    for V in levels]).mean(0)))
    return l2(np.concatenate(per_enc, axis=1))


def head_vectors(pyr, level_of, ckpt, dev, block=512):
    """Run the trained fusion head over the pyramid cache."""
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st["state"]
    n_reg = int(sd["reg.weight"].shape[0])
    out_d = int(sd["out.1.weight"].shape[0])
    model = FuseHead(d=st.get("d", 256), out=out_d, n_reg=n_reg,
                     level_of=list(level_of)).to(dev).eval()
    model.load_state_dict(sd)
    Z = np.empty((len(pyr), out_d), np.float32)
    with torch.no_grad():
        for s in range(0, len(pyr), block):
            e = min(s + block, len(pyr))
            B = np.asarray(pyr[s:e], np.float32)
            B /= np.linalg.norm(B, axis=-1, keepdims=True).clip(1e-6)
            Z[s:e] = model(torch.from_numpy(B).to(dev)).float().cpu().numpy()
    return l2(Z)


def score(S, lat, lon, qi, bi, same):
    e, _ = topk_stats(S, lat, lon, qi, bi, same)
    return e


def row(name, e, width=30):
    return "%-*s %9.1f %s" % (width, name, np.median(e),
                              " ".join("%7.1f%%" % (100 * (e < t).mean())
                                       for t in THRESH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyr-stem", default="pyr47")
    ap.add_argument("--head", default="pyr47_fuse_p05_head.pt")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--cascade-k", type=int, default=200,
                    help="candidates the coarse scorer hands to the re-ranker")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = np.load(config.STREET_CACHE / (a.pyr_stem + "_meta.npz"),
                allow_pickle=True)
    lat, lon = m["lat"], m["lon"]
    seq = m["sequence"].astype("U40")
    level_of = m["level_of"].tolist()

    pyr_p = config.STREET_CACHE / (a.pyr_stem + ".f16.npy")
    pyr = np.load(pyr_p, mmap_mode="r")
    done_p = config.STREET_CACHE / (a.pyr_stem + "_done.u8.npy")
    if done_p.exists():
        d = maskio.load_mask(done_p, len(pyr), a.pyr_stem)
        if not maskio.is_complete(d, len(pyr)):
            raise SystemExit("{} is incomplete: {:,} of {:,} rows written"
                             .format(done_p.name, maskio.complete(d), len(pyr)))

    # Whole sequences, as the main benchmark splits: frames from one drive are
    # near-duplicates, so a row-wise split puts a near-copy of every test image
    # in the bank. crc32 rather than hash(), which Python salts per process.
    h = np.array([zlib.crc32(x.encode()) % 10 for x in seq.tolist()])
    tr, te = np.flatnonzero(h < 8), np.flatnonzero(h >= 8)
    nq = min(a.queries, len(te))
    qi, bi = te[:nq], tr
    same = seq[qi][:, None] == seq[bi][None, :]
    # The sweep tables below are exploratory and print over every query. The
    # paired intervals at the end are not: an arm picked as the best of a
    # sweep and then scored on the same queries carries whatever noise
    # favoured it. Those are reported on a held-out half, split by sequence
    # so adjacent frames of one drive cannot straddle it. Measured cost of
    # ignoring this: the 25 km gain went from +2.10 pp "separated" to
    # +1.18 pp spanning zero.
    hq = h[qi]
    rep = np.flatnonzero((hq % 2) == 1)

    print("{}  {:,} rows, levels {}".format(
        a.pyr_stem, len(pyr), np.bincount(level_of)), flush=True)
    print("{:,} test queries  {:,} bank  {:,} same-sequence masked\n".format(
        nq, len(bi), int(same.sum())), flush=True)

    levels = per_level(pyr, level_of)
    base = level_mean(levels)
    Z = head_vectors(pyr, level_of, config.STREET_CACHE / a.head, dev)
    print("components built in {:.0f}s".format(time.time() - t0), flush=True)

    comp = {"L0": levels[0], "L1": levels[1], "L2": levels[2],
            "mean": base, "head": Z}
    S = {k: dense_sim(V[qi], V[bi], dev) for k, V in comp.items()}
    print("similarities in {:.0f}s\n".format(time.time() - t0), flush=True)

    hdr = "%-30s %9s %s" % ("arm", "median km",
                            " ".join("%8s" % ("<%gkm" % t) for t in THRESH))
    print(hdr); print("-" * len(hdr))
    err = {}
    for k in ("L0", "L1", "L2", "mean", "head"):
        err[k] = score(S[k], lat, lon, qi, bi, same)
        print(row(k if k in ("mean", "head") else k + " alone", err[k]))

    # ---- 1. the knob the concat fixed at 0.5 without saying so -------------
    print("\n--- mean/head blend:  cos = (1-w) * cos_mean + w * cos_head ---")
    print(hdr); print("-" * len(hdr))
    blend = {}
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        e = score((1 - w) * S["mean"] + w * S["head"], lat, lon, qi, bi, same)
        blend[w] = e
        print(row("w = %.2f%s" % (w, "   <- the concat" if w == 0.5 else ""), e))

    # ---- 2. do deeper levels want more weight, or less? --------------------
    print("\n--- level weights (head excluded); equal is the published mean ---")
    print(hdr); print("-" * len(hdr))
    grid = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (2, 1, 1), (1, 2, 1),
            (1, 1, 2), (3, 2, 1), (1, 2, 3), (2, 2, 1), (1, 2, 2), (0, 1, 1),
            (1, 1, 0), (1, 0, 1)]
    lvl_err = {}
    for g in grid:
        w = np.asarray(g, np.float64) / sum(g)
        e = score(sum(w[i] * S["L%d" % i] for i in range(3)),
                  lat, lon, qi, bi, same)
        lvl_err[g] = e
        print(row("L0:L1:L2 = %d:%d:%d%s" % (
            g + ("   <- equal" if g == (1, 1, 1) else "",)), e))

    # ---- 3. best level mix, then add the head -----------------------------
    best_g = max(grid, key=lambda g: (lvl_err[g] < 25).mean())
    wbest = np.asarray(best_g, np.float64) / sum(best_g)
    Sl = sum(wbest[i] * S["L%d" % i] for i in range(3))
    print("\n--- best level mix %s + head ---" % (best_g,))
    print(hdr); print("-" * len(hdr))
    both = {}
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
        e = score((1 - w) * Sl + w * S["head"], lat, lon, qi, bi, same)
        both[w] = e
        print(row("levels %d:%d:%d, head w = %.2f" % (best_g + (w,)), e))
    best_w = max(both, key=lambda w: (both[w] < 25).mean())

    # ---- 4. cascade: each scorer only where it wins ------------------------
    print("\n--- cascade: top-%d by A, re-ranked by B ---" % a.cascade_k)
    print(hdr); print("-" * len(hdr))
    for aname, bname in (("head", "mean"), ("mean", "head"),
                         ("L2", "mean"), ("mean", "L0")):
        SA, SB = S[aname], S[bname]
        k = min(a.cascade_k, SA.shape[1])
        keep = np.argpartition(-np.where(same, -2.0, SA), k - 1, axis=1)[:, :k]
        M = np.full_like(SB, -2.0)
        np.put_along_axis(M, keep, np.take_along_axis(SB, keep, 1), 1)
        e = score(M, lat, lon, qi, bi, same)
        print(row("%s retrieves -> %s ranks" % (aname, bname), e))

    # ---- paired intervals against the published baseline ------------------
    rng = np.random.default_rng(0)
    print("\n--- paired against `mean` (the published level-weighted pool) ---")
    cands = {"blend w=0.05": blend[0.05], "blend w=0.1": blend[0.1],
             "blend w=0.5 (concat)": blend[0.5],
             "levels %d:%d:%d" % best_g: lvl_err[best_g],
             "levels %d:%d:%d + head %.2f" % (best_g + (best_w,)):
                 both[best_w]}
    print("(reported on {:,} held-out queries; the arms were chosen on the "
          "other {:,})".format(len(rep), nq - len(rep)))
    for name, e in cands.items():
        cells = []
        for t in THRESH:
            lo_, hi_ = paired(err["mean"][rep] < t, e[rep] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((e[rep] < t).mean() - (err["mean"][rep] < t).mean()),
                lo_, hi_, " " if lo_ * hi_ > 0 else "~"))
        print("%-24s %s" % (name, " ".join(cells)))
    print("\n~ marks an interval spanning zero. {:.0f}s total"
          .format(time.time() - t0))


if __name__ == "__main__":
    main()
