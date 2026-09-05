"""How fast can the 19 h tile pass run, and does going faster move the vectors?

`tile_cache.py` is about to embed 3,000,000 extension rows at roughly 19 hours,
so its per-image cost is worth an hour of measurement first. Two levers are
untested on *this* workload:

    bf16 weights    `.to(torch.bfloat16)` on the model instead of
                    `torch.autocast`. Measured 1.36x elsewhere this session
                    (resmatch's native pass), because autocast re-casts fp32
                    weights on every op while this pays the cast once.
    torch.compile   fuses the memory-bound elementwise chains between the
                    matmuls -- layernorm, GELU, softmax, residuals -- which is
                    what leaves the card at 220 W of 300 W with the SMs
                    resident but stalling.

**Speed is only half the question, and the smaller half.** `tile6`'s 500,000
release rows were embedded under autocast. If the extension is embedded any
other way, one bank holds two numerically different halves, and the retrieval
that consumes it is a cosine across the boundary. This project's standing rule
is to never compare two numbers from different embedding spaces, so the
agreement between the arms is measured here rather than assumed:

    cosine          per-row, arm against the autocast incumbent
    top-1 crossing  embed the bank one way and the query the other, then ask
                    how often the nearest neighbour changes identity

The second is the one that matters. A cosine of 0.9999 is reassuring and not
sufficient: retrieval only cares whether the *ranking* survives, and a bank
whose two halves sit at slightly different norms can reorder near-ties without
either half being wrong.

    OSV_RELEASE=s10 py scripts/tilebench.py --n 2000
"""

import argparse
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                       # noqa: E402
from embed_street import slurp                      # noqa: E402
from tile_cache import DINO, SIGLIP, S, D_ENC, tile_uint8   # noqa: E402


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def load_tiles(n, gc, gr, seed=0, max_shards=0):
    """(n, n_tile, 224, 224, 3) uint8 from as few shards as possible."""
    zn = np.asarray(pq.read_table(config.DATASET_PARQUET,
                                  columns=["zip_name"])["zip_name"]).astype("U40")
    keep = np.arange(len(zn))
    if max_shards:
        allow = sorted({m.split("/")[0] for m in zn})[:max_shards]
        keep = np.flatnonzero(np.isin([m.split("/")[0] for m in zn], allow))
    sel = keep[np.sort(np.random.default_rng(seed).permutation(len(keep))[:n])]
    by_zip = defaultdict(list)
    for i in sel:
        by_zip[zn[i].split("/")[0]].append(zn[i])
    out, got = np.empty((n, gc * gr, S, S, 3), np.uint8), 0
    for shard, members in sorted(by_zip.items()):
        zp = Path(config.OSV_ROOT) / "images" / "train" / (shard + ".zip")
        with zipfile.ZipFile(slurp(zp)) as zf:
            for m in members:
                out[got] = tile_uint8(zf.read(m), gc, gr)
                got += 1
        print("  loaded {:,}/{:,}".format(got, n), flush=True)
    return out[:got]


def build(mode, dev):
    """Two encoders under one numeric regime.

    A mode is a `+`-joined set of independent switches, because they compose
    and the interesting cell is the composition: `autocast+compile` is the only
    compiled variant that could actually be used for the extension, since it is
    the one whose arithmetic stays closest to the autocast `tile6` already on
    disk. Testing `compile` only on top of bf16 weights would confound the two.
    """
    import timm
    from timm.data import resolve_model_data_config
    encs = []
    for name in (DINO, SIGLIP):
        m = timm.create_model(name, pretrained=True, num_classes=0,
                              img_size=S).eval().to(dev)
        cfg = resolve_model_data_config(m)
        # Statistics stay fp32 and the normalise happens in fp32; only the
        # weights change regime. Casting the constants too would confound a
        # weight-dtype result with an input-dtype one.
        mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
        opt = set(mode.split("+"))
        if "bf16" in opt:
            m = m.to(torch.bfloat16)
        if "compile" in opt:
            m = torch.compile(m)
        encs.append((m, mean, std))
    return encs


def encode(encs, tiles, mode, dev, batch, n_tile):
    """(n, n_tile, 1536) float32, in the layout tile_cache writes."""
    n = len(tiles)
    out = np.empty((n, n_tile, 2 * D_ENC), np.float32)
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            x = torch.from_numpy(tiles[s:e].reshape(-1, S, S, 3)).to(dev)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            got = []
            for m, mean, std in encs:
                z = (x - mean) / std
                if "bf16" in mode.split("+"):
                    f = m(z.to(torch.bfloat16))
                else:
                    with torch.autocast(dev, dtype=torch.bfloat16,
                                        enabled=(dev == "cuda")):
                        f = m(z)
                got.append(f.float())
            out[s:e] = torch.cat(got, dim=1).reshape(
                e - s, n_tile, 2 * D_ENC).cpu().numpy()
    return out


def level(V, dev):
    """(n, n_tile, 1536) -> the 1536-d L1 vector, exactly as the probes pool it."""
    T = torch.from_numpy(V).to(dev)
    T = torch.stack([T[:, :, :D_ENC], T[:, :, D_ENC:]], dim=2)
    T = nrm(T)
    return nrm(torch.cat([nrm(T[:, :, 0].mean(1)),
                          nrm(T[:, :, 1].mean(1))], dim=1))


def top1(Q, B):
    """Nearest-neighbour index of each query row in the bank."""
    return (Q @ B.T).argmax(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--grid", default="3x2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--modes", default="autocast,bf16,autocast+compile",
                    help="comma-separated; each is a +-joined set of switches "
                         "from {autocast, bf16, compile}")
    ap.add_argument("--repeat", type=int, default=2,
                    help="timed passes after a warm-up pass, best kept")
    ap.add_argument("--max-shards", type=int, default=0,
                    help="draw only from the first N shards. Each shard is "
                         "preloaded whole (2.5 GB, ~30 s), and a uniform "
                         "sample of 2,000 release rows touches all ten, so "
                         "the load dominates a short benchmark. Shards are "
                         "themselves uniform samples of OSV-5M, so bounding "
                         "them costs diversity that is not there to lose.")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gc, gr = (int(v) for v in a.grid.lower().split("x"))
    n_tile = gc * gr
    modes = [m for m in a.modes.split(",") if m]

    print("loading {:,} images as {}x{} tiles".format(a.n, gc, gr), flush=True)
    tiles = load_tiles(a.n, gc, gr, max_shards=a.max_shards)
    print("{:,} images, {:,} tile forwards per pass\n".format(
        len(tiles), len(tiles) * n_tile), flush=True)

    res = {}
    for mode in list(modes):
        try:
            encs = build(mode, dev)
        except Exception as exc:                      # noqa: BLE001
            print("{:<18} unavailable: {}".format(
                mode, str(exc).strip().split(chr(10))[0]), flush=True)
            modes.remove(mode)
            continue
        # Warm-up is not optional for compile -- the first pass pays the
        # compilation -- and it is honest for the others too, since it takes
        # the cuDNN autotuner and the allocator out of the timed region.
        try:
            encode(encs, tiles[:a.batch * 2], mode, dev, a.batch, n_tile)
        except Exception as exc:                      # noqa: BLE001
            # torch.compile defers everything to the first real call, so this
            # is where it actually fails. Losing one arm must not lose the
            # agreement measurement, which is the half that decides anything.
            print("{:<18} unavailable: {}".format(
                mode, str(exc).strip().split(chr(10))[0]), flush=True)
            modes.remove(mode)
            del encs
            if dev == "cuda":
                torch.cuda.empty_cache()
            continue
        torch.cuda.synchronize() if dev == "cuda" else None
        best = None
        for _ in range(a.repeat):
            t0 = time.time()
            V = encode(encs, tiles, mode, dev, a.batch, n_tile)
            if dev == "cuda":
                torch.cuda.synchronize()
            el = time.time() - t0
            best = el if best is None else min(best, el)
        res[mode] = (V, best)
        print("{:<18} {:6.1f}s  {:6.1f} img/s  {:7.1f} tile/s  {:.2f} GB".format(
            mode, best, len(tiles) / best, len(tiles) * n_tile / best,
            torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0),
            flush=True)
        del encs
        if dev == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    base = modes[0]
    print("\n--- speed, against {} ---".format(base))
    for m in modes[1:]:
        print("{:<18} {:.3f}x".format(m, res[base][1] / res[m][1]))

    # Agreement. The cosine is the reassuring number; the crossing rate is the
    # one that decides whether a bank may hold both regimes.
    print("\n--- agreement with {}, on the pooled 1536-d level ---".format(base))
    L = {m: level(res[m][0], dev) for m in modes}
    half = len(tiles) // 2
    ref_who = top1(L[base][half:], L[base][:half])
    print("{:<18} {:>12} {:>12} {:>14}".format(
        "arm", "min cos", "mean cos", "top-1 changed"))
    for m in modes:
        cos = (L[base] * L[m]).sum(1)
        # Query embedded with `m`, bank still the incumbent: exactly the mixed
        # bank the extension would create.
        who = top1(L[m][half:], L[base][:half])
        print("{:<18} {:12.6f} {:12.6f} {:11.2f}% {}".format(
            m, float(cos.min()), float(cos.mean()),
            100.0 * (who != ref_who).mean(),
            "<- baseline" if m == base else ""), flush=True)

    print("\nA cosine near 1 is not sufficient; the crossing rate is the test. "
          "Anything above ~0.5% means the extension must be embedded the same "
          "way tile6 was, whatever the speed says.")


if __name__ == "__main__":
    main()
