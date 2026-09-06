"""Resolution or tiles? Four matched banks, to decide what a rebuild should be.

Everything is now pointing at a bank rebuild -- query-side changes are a
dose-response failure and only the matched arm gains. So the question is no
longer *whether* to rebuild but *what to build*, and there are two candidates
that have never been compared:

    tiles      add detail by fragmenting the scene -- 9 forwards, 51.5 GB of
               tile cache, ~16 h to cover the bank extension
    resolution add detail by sampling the scene properly -- 3 forwards, no new
               cache, ~3-4 h for the extension

They may well be alternatives rather than complements, and picking wrong costs
either 13 hours or the whole gain.

**Each encoder at its own native size.** An earlier sweep held both to one
shared input, which forces multiples of lcm(14, 16) = 112 and lands on 448 --
a compromise where DINOv2 sits 14% below its native grid and SigLIP is
upsampled. Nothing requires a shared size: the encoders run independently and
only their 768-d outputs are concatenated. Frames are 512 tall, so

    SigLIP  /16 at 512 = 32x16   the frame's own pixels, no resampling at all
    DINOv2  /14 at 518 = 37x14   its exact training resolution, +1.2% resize

Every arm is **matched** -- query and bank built the same way -- because the
2x2 established that nothing else is informative.

    OSV_RELEASE=s10 py scripts/resmatch.py --rows 50000
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
import shards                                    # noqa: E402
import splits as sp                              # noqa: E402
from tile_pool import paired                     # noqa: E402
from osv_pyramid import complete_rows            # noqa: E402
from query_only import great_circle, top1        # noqa: E402

THRESH = (1, 25, 200, 750, 2500)
D_ENC = 768

# (spec, patch, size). Sizes chosen per encoder, not shared -- see the module
# docstring. DINOv2's default img_size really is 518; the pipeline has been
# overriding it to 224 since the beginning.
#
# The SigLIP entry was `vit_base_patch16_siglip_224.v2_webli` run at
# img_size=512: a 224 checkpoint with interpolated position embeddings, which
# is out of distribution rather than native (REVIEW5 #10). That made the
# resolution arms unreadable -- "more pixels do not help" and "this checkpoint
# cannot use them" look identical. `timm` ships a real 512 checkpoint and
# `embed_native` already uses it, so the native arms now get an encoder at
# home rather than one extrapolating.
ENCODERS = [("vit_base_patch14_dinov2.lvd142m", 14, 518),
            ("vit_base_patch16_siglip_512.v2_webli", 16, 512)]


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def cached_levels(rel_rows, tile_pos, dev, block=4096):
    """(L0, L1) at 224 from the caches already on disk -- no encoding."""
    C_all = np.load(config.STREET_CACHE / "dual_c3.f16.npy", mmap_mode="r")
    T_all = np.load(config.STREET_CACHE / "tile6.f16.npy", mmap_mode="r")
    h = C_all.shape[1] // 2
    nc = h // D_ENC
    n = len(rel_rows)
    L0 = np.empty((n, 2 * D_ENC), np.float32)
    L1 = np.empty((n, 2 * D_ENC), np.float32)
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
            C, T = nrm(C), nrm(T)
            L0[s:e] = nrm(torch.cat([nrm(C[:, :, 0].mean(1)),
                                     nrm(C[:, :, 1].mean(1))], 1)).cpu().numpy()
            L1[s:e] = nrm(torch.cat([nrm(T[:, :, 0].mean(1)),
                                     nrm(T[:, :, 1].mean(1))], 1)).cpu().numpy()
            del C, T
    return L0, L1


def encode_native(blobs, dev, crops=3, batch=16):
    """(n, 1536) crops at each encoder's own native input size."""
    import timm
    from embed_street import preprocess

    per = []
    for spec, patch, size in ENCODERS:
        assert size % patch == 0, (spec, size, patch)
        # bf16 WEIGHTS, not merely bf16 autocast. Benchmarked on this card:
        # 31.2 -> 35.7 img/s and 2.30 -> 1.48 GB, because autocast re-casts
        # fp32 weights on every op while this pays once. Batch size is already
        # saturated at 16 (48 is no faster and 2.5x the VRAM) and
        # channels_last does nothing to a ViT, so this is the only free win
        # left: fused attention is already active, and the card sits at 220 W
        # of 300 W because ViT attention at 1,369 tokens is bandwidth-bound,
        # not FMA-bound.
        net = timm.create_model(spec, pretrained=True, num_classes=0,
                                img_size=size).eval().to(dev).to(torch.bfloat16)
        cfg = timm.data.resolve_data_config({}, model=net)
        mu = np.array(cfg["mean"], np.float32).reshape(3, 1, 1)
        sd = np.array(cfg["std"], np.float32).reshape(3, 1, 1)
        out = np.empty((len(blobs), D_ENC), np.float32)
        t0 = time.time()

        # Preprocess on worker threads, one batch ahead of the GPU.
        # Serially, the card sawtooths 100% -> 0%: it idles through every
        # JPEG decode, 921x518 resize and float32 normalise, then saturates
        # for one forward. Measured 18.2 img/s with 25% of 16 cores busy --
        # one core doing all of it. PIL and numpy release the GIL, so threads
        # genuinely overlap with the forward rather than fighting it.
        def prep(lo, hi):
            return np.concatenate(list(pool.map(
                lambda b: preprocess(b, size=size, crops=crops,
                                     mean=mu, std=sd), blobs[lo:hi])))

        with torch.no_grad(), ThreadPoolExecutor(max_workers=8) as pool:
            nxt = pool.submit(prep, 0, min(batch, len(blobs)))
            for s in range(0, len(blobs), batch):
                e = min(s + batch, len(blobs))
                x = nxt.result()
                if e < len(blobs):
                    nxt = pool.submit(prep, e, min(e + batch, len(blobs)))
                x = torch.from_numpy(x).to(dev, torch.bfloat16,
                                             non_blocking=True)
                f = net(x).float()          # weights are already bf16
                # mean over crops, then L2 -- exactly how the 224 arm pools
                out[s:e] = nrm(f.reshape(e - s, crops, D_ENC).mean(1)
                               ).cpu().numpy()
                if s % (batch * 200) == 0:
                    el = time.time() - t0
                    print("   {:<8} {:>7,}/{:,}  {:.0f}s  eta {:.0f}s".format(
                        spec.split("_")[2][:6], e, len(blobs), el,
                        el * (len(blobs) - e) / max(e, 1)), flush=True)
        per.append(out)
        del net
        torch.cuda.empty_cache()
    return nrm(torch.from_numpy(np.concatenate(per, axis=1))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50000,
                    help="release rows to use; the bank is this minus the "
                         "held-out queries")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--split-mode", default="sequence")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    rel, pos = complete_rows("tile6")
    rs = np.random.default_rng(0)
    take = np.sort(rs.choice(len(rel), min(a.rows, len(rel)), replace=False))
    rel, pos = rel[take], pos[take]

    ds = pq.read_table(config.DATASET_PARQUET)
    lat = np.asarray(ds["lat"], np.float64)[rel]
    lon = np.asarray(ds["lon"], np.float64)[rel]
    seq = np.asarray(ds["sequence"]).astype("U40")[rel]
    zname = np.asarray(ds["zip_name"].to_pylist())[rel]
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
    print("{:,} rows: {:,} bank, {:,} queries".format(len(rel), len(tr), len(te)))
    print("encoders: " + ", ".join("{} @{}".format(s.split('.')[0], z)
                                   for s, _, z in ENCODERS) + "\n", flush=True)

    L0, L1 = cached_levels(rel, pos, dev)
    print("224 caches read in {:.0f}s".format(time.time() - t0), flush=True)

    # `shards.blobs`, not a per-member `ZipFile.read`. The loop that was here
    # walked the members in parquet order, which is a random seek per image
    # across ~98 archives on a 5900 RPM disk. It is invisible on a warm run --
    # the same read took 7s with the pages cached -- and it cost 999s cold,
    # 143x, on the run that produced the numbers in runs/RESMATCH.md. The
    # reader groups by shard and reads each one sequentially into RAM.
    blobs = shards.blobs(zname)
    print("{:,} JPEGs read in {:.0f}s; encoding at native size"
          .format(len(blobs), time.time() - t0), flush=True)
    NAT = encode_native(blobs, dev)
    print("native encoding done at {:.0f}s".format(time.time() - t0), flush=True)

    mix = lambda a_, b_: nrm(torch.from_numpy((a_ + b_) / 2.0)).numpy()
    arms = [("1  crops @224 (incumbent)", L0),
            ("2  crops+tiles @224", mix(L0, L1)),
            ("3  crops @518/512 native", NAT),
            ("4  native crops + 224 tiles", mix(NAT, L1))]

    hdr = "%-30s %9s %s" % ("arm", "median km",
                            " ".join("%8s" % ("<%gkm" % t) for t in THRESH))
    print("\n" + hdr); print("-" * len(hdr))
    err = {}
    for name, V in arms:
        who, _ = top1(V[te], V[tr], seq[tr], seq[te], dev)
        e = great_circle(lat[te], lon[te], lat[tr][who], lon[tr][who])
        err[name] = e
        print("%-30s %9.1f %s" % (name, np.median(e), " ".join(
            "%7.1f%%" % (100 * (e < t).mean()) for t in THRESH)), flush=True)

    rng = np.random.default_rng(0)
    base = err[arms[0][0]]
    print("\n--- against crops @224, the incumbent ---")
    for name, _ in arms[1:]:
        cells = []
        for t in THRESH:
            lo, hi = paired(base < t, err[name] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[name] < t).mean() - (base < t).mean()),
                lo, hi, " " if lo * hi > 0 else "~"))
        print("%-30s %s" % (name, " ".join(cells)))
    print("\n~ spans zero. {:.0f}s total".format(time.time() - t0))


if __name__ == "__main__":
    main()
