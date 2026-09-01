"""Does higher source resolution actually help, and which scheme spends it?

The trap this is built around: **source resolution on its own changes nothing.**
`preprocess` scales the short side to 224 before anything else, so an 8-megapixel
KartaView frame and a 682x512 OSV-5M frame become almost the same tensor --
resizing 3264 -> 398 in one step or via 682 differs only by resampling error.
Any gain has to come from a scheme that *consumes* the extra pixels.

So the arms separate those two things:

    osv_sim     downsample to 682x512 first, then the shipping pipeline.
                Simulates today's data.
    crop3_224   full-res source, then the shipping pipeline.
                **Should equal osv_sim.** If it does not, the difference is
                resampling artefacts, not information -- and the whole premise
                needs rechecking before reading anything else here.
    crop3_448   short side to 448, three 448 crops, encoder at img_size=448.
                Twice the linear resolution, four times the compute, no
                architecture change. The cheapest way to actually spend pixels.
    tile6       resize to 672x448, six 224 tiles. This is the scheme that lost
                on OSV-5M, where the source was 682x512 and the tiles were
                barely sharper than crops. Here the source is 8 MP, so for the
                first time the tiles are genuinely sharper rather than merely
                more numerous.
    tile24      resize to 1344x896, twenty-four 224 tiles. Four times tile6's
                pixels; impossible on OSV-5M source, which is the point.

Every arm is reduced the same way -- per-token L2 then mean-pool to 768-d -- so
they are compared at identical width and identical bytes, which is what the
earlier tiling probe could not do.

Same-sequence neighbours are masked, as everywhere else in this project:
consecutive frames of one drive are near-duplicates and would make every scheme
look excellent.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from tile_pool import great_circle, l2, paired
from tile_match import dense_sim, topk_stats

DINO = "vit_base_patch14_dinov2.lvd142m"
SIGLIP = "vit_base_patch16_siglip_224.v2_webli"
# DINOv2 is self-supervised on structure and texture, which is largely
# scale-robust; SigLIP is language-supervised and encodes nameable content --
# signage, storefronts, script -- which is what extra resolution surfaces. They
# may not answer the resolution question the same way.
MODELS = {"dinov2": DINO, "siglip": SIGLIP}
THRESH = (1, 25, 200, 750, 2500)

# name -> (encoder input size, grid cols, grid rows, pre-downsample short side)
ARMS = {
    "osv_sim":   (224, 3, 1, 512),
    "crop3_224": (224, 3, 1, None),
    "crop3_448": (448, 3, 1, None),
    "tile6":     (224, 3, 2, None),
    "tile24":    (224, 6, 4, None),
}


def variants(img, arms):
    """One decoded PIL image -> {arm: (n_parts, H, W, 3) uint8}."""
    from PIL import Image
    out = {}
    for name in arms:
        size, gc, gr, pre = ARMS[name]
        im = img
        if pre:                      # simulate a lower-resolution source first
            w, h = im.size
            s = pre / min(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.BILINEAR)
        if gr == 1:
            # the shipping scheme: short side to `size`, then crops across width
            w, h = im.size
            s = size / min(w, h)
            if s != 1.0:
                im2 = im.resize((max(size, round(w * s)), max(size, round(h * s))),
                                Image.BILINEAR)
            else:
                im2 = im
            w, h = im2.size
            t = (h - size) // 2
            lefts = [round(i * (w - size) / (gc - 1)) for i in range(gc)]
            parts = [im2.crop((l, t, l + size, t + size)) for l in lefts]
        else:
            im2 = im.resize((gc * size, gr * size), Image.BILINEAR)
            parts = [im2.crop((c * size, r * size, (c + 1) * size, (r + 1) * size))
                     for r in range(gr) for c in range(gc)]
        out[name] = np.stack([np.asarray(p.convert("RGB"), np.uint8)
                              for p in parts])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/data/kartaview")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n", type=int, default=0, help="cap on images used")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=24, help="images per forward")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="dinov2", choices=sorted(MODELS))
    ap.add_argument("--save-emb", default=None,
                    help="write per-arm embeddings here, so mixtures of "
                         "encoders can be scored without re-embedding")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import timm
    from timm.data import resolve_model_data_config
    from PIL import Image

    root = Path(a.data)
    recs = [json.loads(l) for l in (root / "manifest.jsonl").open(encoding="utf-8")]
    seen, uniq = set(), []
    for r in recs:                                   # the manifest is append-only
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    recs = uniq
    rng = np.random.default_rng(a.seed)
    rng.shuffle(recs)
    if a.n:
        recs = recs[:a.n]
    arms = [x for x in a.arms.split(",") if x in ARMS]
    mp = np.array([r["w"] * r["h"] for r in recs]) / 1e6
    print("{:,} images  median {:.1f} MP  (OSV-5M is 0.35)  encoder {}  "
          "arms: {}".format(len(recs), np.median(mp), a.model,
                            ", ".join(arms)), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, stats = {}, {}
    name = MODELS[a.model]
    for size in sorted({ARMS[x][0] for x in arms}):
        m = timm.create_model(name, pretrained=True, num_classes=0,
                              img_size=size).eval().to(dev)
        cfg = resolve_model_data_config(m)
        models[size] = m
        stats[size] = (torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1),
                       torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1))
        patch = int(name.split("patch")[1].split("_")[0])
        print("  {} at {}px  ({} patches)".format(
            a.model, size, (size // patch) ** 2), flush=True)

    emb = {x: np.zeros((len(recs), 768), np.float32) for x in arms}
    pool = ThreadPoolExecutor(a.workers)

    def load(i):
        try:
            return i, variants(Image.open(root / "img" / recs[i]["file"]), arms)
        except Exception:
            return i, None

    t0, done, bad = time.time(), 0, 0
    buf = {x: [] for x in arms}
    idx = []

    def flush():
        nonlocal buf, idx
        if not idx:
            return
        for x in arms:
            size = ARMS[x][0]
            mean, std = stats[size]
            v = np.concatenate(buf[x])
            t = torch.from_numpy(v).to(dev).permute(0, 3, 1, 2).float().div_(255)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16,
                                                 enabled=(dev == "cuda")):
                f = models[size]((t - mean) / std).float()
            f = f.view(len(idx), -1, 768)
            f = torch.nn.functional.normalize(f, dim=-1).mean(1)
            emb[x][idx] = f.cpu().numpy()
        buf = {x: [] for x in arms}
        idx = []

    for i, v in pool.map(load, range(len(recs))):
        if v is None:
            bad += 1
            continue
        for x in arms:
            buf[x].append(v[x])
        idx.append(i)
        if len(idx) >= a.batch:
            flush()
            done += a.batch
            if done % 480 == 0:
                el = time.time() - t0
                print("  {:,}/{:,}  {:.0f}s  eta {:.0f}s".format(
                    done, len(recs), el, el * (len(recs) - done) / max(done, 1)),
                    flush=True)
    flush()
    pool.shutdown()
    print("embedded {:,} images in {:.0f}s ({} unreadable)".format(
        len(recs) - bad, time.time() - t0, bad), flush=True)

    if a.save_emb:
        d = Path(a.save_emb)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / ("emb_%s.npz" % a.model),
                 ids=np.array([r["id"] for r in recs]),
                 lat=np.array([r["lat"] for r in recs]),
                 lon=np.array([r["lon"] for r in recs]),
                 seq=np.array([r["sequence_id"] for r in recs]),
                 **{x: emb[x] for x in arms})
        print("saved embeddings -> {}".format(d / ("emb_%s.npz" % a.model)),
              flush=True)

    lat = np.array([r["lat"] for r in recs])
    lon = np.array([r["lon"] for r in recs])
    seq = np.array([r["sequence_id"] for r in recs])
    nq = min(a.queries, len(recs) // 4)
    qi, bi = np.arange(nq), np.arange(nq, len(recs))
    same = seq[qi][:, None] == seq[bi][None, :]
    print("{:,} queries  {:,} bank  {:,} same-sequence pairs masked\n".format(
        nq, len(bi), int(same.sum())), flush=True)

    err = {}
    print("%-12s %11s %s" % ("arm", "median km",
                            " ".join("%9s" % ("<%gkm" % t) for t in THRESH)))
    print("-" * 74)
    for x in arms:
        V = emb[x]
        d1, _ = topk_stats(dense_sim(V[qi], V[bi], dev), lat, lon, qi, bi, same)
        err[x] = d1
        print("%-12s %11.1f %s" % (x, np.median(d1), " ".join(
            "%8.1f%%" % (100 * (d1 < t).mean()) for t in THRESH)), flush=True)

    rng2 = np.random.default_rng(0)
    base = "osv_sim" if "osv_sim" in err else arms[0]
    print("\npaired against {}, percentage points, ~ spans zero".format(base))
    print("%-12s %s" % ("arm", " ".join("%18s" % ("<%gkm" % t) for t in THRESH)))
    print("-" * 108)
    for x in arms:
        if x == base:
            continue
        cells = []
        for t in THRESH:
            lo, hi = paired(err[base] < t, err[x] < t, rng2)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[x] < t).mean() - (err[base] < t).mean()), lo, hi,
                " " if lo * hi > 0 else "~"))
        print("%-12s %s" % (x, " ".join(cells)))
    print("\nRead crop3_224 against osv_sim first: they should agree. A gap "
          "there is resampling, not information, and invalidates the rest.")


if __name__ == "__main__":
    main()
