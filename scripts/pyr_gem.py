"""Within a pyramid level, 24 tiles are averaged equally. Should they be?

Two unswept constants have already turned up in this pipeline, both of the same
shape -- a weight that looked like the absence of a choice. The head/mean blend
was fixed at 0.5 by concatenating two unit blocks; the levels were fixed at
1:1:1 by `stack(per).mean(0)`. Sweeping the second found the deepest level
wants roughly double.

This is the same constant one level further down. `FuseHead.baseline` pools a
level with `x[:, level_of == l].mean(1)` -- **24 tiles, equal weight**. A
featureless highway frame has 24 tiles of which perhaps two carry signal, and a
mean dilutes those by twelve.

Attention was already tried as the alternative and lost badly: 2.5M parameters
against 38,009 images overfit, and every trained configuration scored below the
head at initialisation. So the thing to test is not *more* capacity but a
**fixed non-uniform pooling** with no parameters at all.

GeM is that. With `p = 1` it is the mean; as `p` grows it approaches a max,
up-weighting strongly-responding tiles without learning anything:

    gem_p(X) = sign(m) * |m|^(1/p),   m = mean_i( sign(x_i) * |x_i|^p )

The sign-preserving form matters because these are encoder outputs, not
post-ReLU activations, so components are signed. `tile_pool.pool` already
implements this (and once shipped it without the p-th root, which made it a
signed third moment rather than a mean -- review item 57). No comparison of the
schemes was ever recorded.

Tokens are loaded once and normalised, then every `p` is a cheap re-pool, so
the sweep costs one pass over the cache.

    OSV_RELEASE=s10 py scripts/pyr_gem.py --pyr-stem pyr47 --queries 3000
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
from pyr_blend import head_vectors, THRESH       # noqa: E402

D_ENC = 768


def gem(X, p, axis=1):
    """Generalised mean over `axis`, sign-preserving. Reference implementation.

    p=1 is the arithmetic mean and p->inf approaches a max. The p-th root is
    what makes it a *mean*: without it the result is a moment, which is not
    scale-equivalent to its inputs.

    Kept as the definition the GPU path is checked against, not as the path
    the sweep runs on -- see `gem_t`.
    """
    if p == 1.0:
        return X.mean(axis)
    m = (np.sign(X) * np.abs(X) ** p).mean(axis)
    return np.sign(m) * np.abs(m) ** (1.0 / p)


def gem_t(X, p, dim=1):
    """The same thing in torch, so it runs on the GPU.

    `|x|**p` over 2.4 billion elements per `p` is a single-threaded numpy
    ufunc -- not a BLAS call, so 16 cores sit idle while one works and the
    card idles at 21%. On the GPU the same sweep is bounded by the PCIe copy
    instead, which is the right thing to be bounded by.
    """
    if p == 1.0:
        return X.mean(dim)
    m = (torch.sign(X) * X.abs().pow(p)).mean(dim)
    return torch.sign(m) * m.abs().pow(1.0 / p)


def load_tokens(pyr, block=4096):
    """The whole cache, per-token L2-normalised, in RAM as float16.

    Held once so each `p` is a re-pool rather than another pass over 4.8 GB.
    """
    out = torch.empty(tuple(pyr.shape), dtype=torch.float16)
    for s in range(0, len(pyr), block):
        e = min(s + block, len(pyr))
        B = torch.from_numpy(np.asarray(pyr[s:e], np.float32))
        B /= B.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        out[s:e] = B.half()
    return out


def level_vectors(tok, level_of, p, dev, block=2048):
    """Per-level unit vectors, pooling each level's tokens with GeM(p).

    Blocked so the fp32 working copy stays a few hundred MB: the whole cache
    upcast at once is 9.7 GB, which is what made even p=1 slow.
    """
    lo = torch.as_tensor(np.asarray(level_of))
    n_lvl = int(lo.max()) + 1
    idx = [torch.nonzero(lo == lv, as_tuple=True)[0].to(dev)
           for lv in range(n_lvl)]
    out = [np.empty((len(tok), 2 * D_ENC), np.float32) for _ in range(n_lvl)]
    nrm = lambda t: t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        for s in range(0, len(tok), block):
            e = min(s + block, len(tok))
            B = tok[s:e].to(dev, non_blocking=True).float()
            for lv in range(n_lvl):
                m = B.index_select(1, idx[lv])
                v = torch.cat([nrm(gem_t(m[:, :, 0], p)),
                               nrm(gem_t(m[:, :, 1], p))], dim=1)
                out[lv][s:e] = nrm(v).cpu().numpy()
            del B
    return out


def combine(levels):
    """Equal weight across levels, per encoder -- the published baseline."""
    per = [l2(np.stack([V[:, i * D_ENC:(i + 1) * D_ENC] for V in levels]
                       ).mean(0)) for i in range(2)]
    return l2(np.concatenate(per, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyr-stem", default="pyr47")
    ap.add_argument("--head", default="pyr47_fuse_p05_head.pt")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--head-w", type=float, default=0.04,
                    help="head contribution, from the flat plateau in pyr_blend")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    m = np.load(config.STREET_CACHE / (a.pyr_stem + "_meta.npz"),
                allow_pickle=True)
    lat, lon = m["lat"], m["lon"]
    seq = m["sequence"].astype("U40")
    level_of = m["level_of"].tolist()

    pyr = np.load(config.STREET_CACHE / (a.pyr_stem + ".f16.npy"), mmap_mode="r")
    done_p = config.STREET_CACHE / (a.pyr_stem + "_done.u8.npy")
    if done_p.exists():
        d = maskio.load_mask(done_p, len(pyr), a.pyr_stem)
        if not maskio.is_complete(d, len(pyr)):
            raise SystemExit("{} is incomplete".format(done_p.name))

    h = np.array([zlib.crc32(x.encode()) % 10 for x in seq.tolist()])
    tr, te = np.flatnonzero(h < 8), np.flatnonzero(h >= 8)
    nq = min(a.queries, len(te))
    qi, bi = te[:nq], tr
    same = seq[qi][:, None] == seq[bi][None, :]
    hq = h[qi]
    sel, rep = np.flatnonzero(hq % 2 == 0), np.flatnonzero(hq % 2 == 1)
    print("{}  {:,} rows, levels {}".format(
        a.pyr_stem, len(pyr), np.bincount(level_of)))
    print("{:,} queries  {:,} bank   {:,} choose / {:,} report\n".format(
        nq, len(bi), len(sel), len(rep)), flush=True)

    tok = load_tokens(pyr)
    Z = head_vectors(pyr, level_of, config.STREET_CACHE / a.head, dev)
    SH = dense_sim(Z[qi], Z[bi], dev)
    print("tokens + head in {:.0f}s".format(time.time() - t0), flush=True)

    ps = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
    hdr = "%-22s %9s %s" % ("pooling", "median km",
                            " ".join("%8s" % ("<%gkm" % t) for t in THRESH))
    print(hdr); print("-" * len(hdr))
    err, rerr = {}, {}
    for p in ps:
        lv = level_vectors(tok, level_of, p, dev)
        V = combine(lv)
        S = dense_sim(V[qi], V[bi], dev)
        for tag, Su in (("", S), ("+head", (1 - a.head_w) * S + a.head_w * SH)):
            e, _ = topk_stats(Su, lat, lon, qi, bi, same)
            key = (p, tag)
            err[key] = e
            rerr[key] = tuple((e[rep] < t).mean() for t in THRESH)
            name = "p = %-5g%s%s" % (p, tag, "   <- the mean" if p == 1.0
                                     and not tag else "")
            print("%-22s %9.1f %s" % (name, np.median(e[rep]), " ".join(
                "%7.1f%%" % (100 * x) for x in rerr[key])))
        del lv, V, S

    # chosen on the half that chooses, reported on the other
    best = max(ps, key=lambda p: (err[(p, "")][sel] < 25).mean())
    print("\nbest p on <25 km: selection half {:g}, reporting half {:g}".format(
        best, max(ps, key=lambda p: (err[(p, "")][rep] < 25).mean())))

    rng = np.random.default_rng(0)
    base = err[(1.0, "")]
    print("\n--- against the mean (p=1), on the {:,} held-out queries ---"
          .format(len(rep)))
    for key in [(p, "") for p in ps if p != 1.0] + [(best, "+head")]:
        cells = []
        for t in THRESH:
            lo_, hi_ = paired(base[rep] < t, err[key][rep] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[key][rep] < t).mean() - (base[rep] < t).mean()),
                lo_, hi_, " " if lo_ * hi_ > 0 else "~"))
        print("%-14s %s" % ("p = %g%s" % key, " ".join(cells)))
    print("\n~ marks an interval spanning zero. {:.0f}s total"
          .format(time.time() - t0))


if __name__ == "__main__":
    main()
