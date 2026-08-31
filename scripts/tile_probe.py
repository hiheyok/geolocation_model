"""Do more tiles beat more dimensions, at equal bank bytes?

`preprocess` scales the short side to 224 and then takes three 224x224 crops
across the width. OSV-5M frames are 910x512 or 682x512, so that first scale
throws away 2.3x of linear resolution before the encoder ever sees the image --
over 80% of the pixels. The hypothesis is that a fixed-width vector is a
capacity bottleneck, and that giving the encoder a subsection at a time
extracts more of what is there.

Two schemes, both fed to the same frozen encoders:

  crop3   today: short side -> 224, three 224x224 crops across the width.
          3 x 224^2 = 150,528 pixels, from a 2.3x downsample.
  tile6   resize to 672x448 -- three columns by two rows of 224 -- and take all
          six. 301,056 pixels, at roughly native resolution for a 682x512
          frame. Six is not arbitrary: 512/224 = 2.3, so a third row would be
          upsampled interpolation and could add nothing.

The comparison that matters is **equal bank bytes**, not equal dimensions.
tile6 is 9,216-d against crop3's 4,608-d, and the bank is the memory-bound part
of the system, so tile6 must justify twice the storage or be compressed to pay
for itself. Both are therefore also compared after PCA to a common width: same
bytes per image, different amounts of the photograph behind them.

Retrieval quality is the metric because that is what the bank is for. Same
-sequence neighbours are dropped exactly as build_knn does, or a query
retrieves its own near-duplicates and every scheme looks perfect.
"""

import argparse
import io
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config

DINO = "vit_base_patch14_dinov2.lvd142m"
SIGLIP = "vit_base_patch16_siglip_224.v2_webli"
S = 224
GRID = (3, 2)                      # cols, rows -> 672 x 448
K = 32


def crops_today(img, mean, std):
    """Exactly what scripts/embed_street.py does, for the control arm."""
    w, h = img.size
    sc = S / min(w, h)
    if sc != 1.0:
        img = img.resize((max(S, round(w * sc)), max(S, round(h * sc))),
                         Image.BILINEAR)
    w, h = img.size
    t, n = (h - S) // 2, 3
    lefts = [round(i * (w - S) / (n - 1)) for i in range(n)]
    return [img.crop((l, t, l + S, t + S)) for l in lefts]


def tiles(img, mean, std):
    """A gc x gr grid at 224, resized so the grid is as close to native as it
    can be without upsampling the common frame."""
    gc, gr = GRID
    img = img.resize((gc * S, gr * S), Image.BILINEAR)
    return [img.crop((c * S, r * S, (c + 1) * S, (r + 1) * S))
            for r in range(gr) for c in range(gc)]


def to_batch(parts, mean, std):
    out = np.empty((len(parts), 3, S, S), dtype=np.float32)
    for i, c in enumerate(parts):
        x = np.array(c, np.uint8).transpose(2, 0, 1).astype(np.float32) / 255.0
        out[i] = (x - mean) / std
    return out


def great_circle(a_lat, a_lon, b_lat, b_lon):
    p = np.pi / 180.0
    dlat, dlon = (b_lat - a_lat) * p, (b_lon - a_lon) * p
    h = (np.sin(dlat / 2) ** 2 +
         np.cos(a_lat * p) * np.cos(b_lat * p) * np.sin(dlon / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def pca_fit(X, d, rng):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    Om = rng.standard_normal((Xc.shape[1], d + 64)).astype(np.float32)
    Y = Xc @ Om
    for _ in range(2):
        Y = Xc @ (Xc.T @ Y)
    Q, _ = np.linalg.qr(Y)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return mu, Vt[:d].T


def score(Eq, Eb, lat, lon, qi, bi, same):
    k = min(K, Eb.shape[0] - 1)
    Eq = Eq / np.linalg.norm(Eq, axis=1, keepdims=True).clip(1e-6)
    Eb = Eb / np.linalg.norm(Eb, axis=1, keepdims=True).clip(1e-6)
    Sm = Eq @ Eb.T
    Sm[same] = -2.0
    j = np.argpartition(-Sm, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(Sm, j, 1), axis=1)
    got = np.take_along_axis(j, o, 1)
    d1 = great_circle(lat[qi], lon[qi], lat[bi[got[:, 0]]], lon[bi[got[:, 0]]])
    dk = great_circle(np.repeat(lat[qi], k), np.repeat(lon[qi], k),
                      lat[bi[got.ravel()]], lon[bi[got.ravel()]]).reshape(-1, k)
    return np.median(d1), (d1 < 25).mean(), (dk < 25).any(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="images embedded")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--dims", default="256,512,1024")
    ap.add_argument("--batch", type=int, default=48, help="tiles per forward")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import timm
    from timm.data import resolve_model_data_config

    ds = pq.read_table(config.DATASET_PARQUET,
                       columns=["zip_name", "lat", "lon", "sequence"])
    zn = np.asarray(ds["zip_name"]).astype("U40")
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    seq = np.asarray(ds["sequence"]).astype("U40")

    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(zn), a.n, replace=False))
    print("release {}  {:,} images  grid {}x{}  batch {}".format(
        config.RELEASE, a.n, GRID[0], GRID[1], a.batch), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    encs = []
    for name in (DINO, SIGLIP):
        m = timm.create_model(name, pretrained=True, num_classes=0,
                              img_size=S).eval().to(dev)
        cfg = resolve_model_data_config(m)
        encs.append((m,
                     np.array(cfg["mean"], np.float32).reshape(3, 1, 1),
                     np.array(cfg["std"], np.float32).reshape(3, 1, 1)))
        print("encoder    {}".format(name), flush=True)

    n_tile = GRID[0] * GRID[1]
    E3 = np.zeros((a.n, 3 * 768 * 2), dtype=np.float32)
    E6 = np.zeros((a.n, n_tile * 768 * 2), dtype=np.float32)

    by_zip = defaultdict(list)
    for i, r in enumerate(sel):
        by_zip[zn[r].split("/")[0]].append((i, zn[r]))

    t0, done = time.time(), 0
    with torch.no_grad():
        for shard, items in sorted(by_zip.items()):
            zp = Path(config.OSV_ROOT) / "images" / "train" / (shard + ".zip")
            with zipfile.ZipFile(zp) as zf:
                for i, name in items:
                    try:
                        img = Image.open(io.BytesIO(zf.read(name)))
                        img = img.convert("RGB")
                    except Exception:
                        continue
                    for scheme, store in ((crops_today, E3), (tiles, E6)):
                        parts = scheme(img, None, None)
                        vecs = []
                        for m, mean, std in encs:
                            x = torch.from_numpy(to_batch(parts, mean, std)).to(dev)
                            with torch.autocast(dev, dtype=torch.bfloat16,
                                                enabled=(dev == "cuda")):
                                f = m(x)
                            vecs.append(f.float().reshape(-1).cpu().numpy())
                        store[i] = np.concatenate(vecs)
                    done += 1
                    if done % 2000 == 0:
                        el = time.time() - t0
                        print("  {:>6,}/{:,}  {:.0f}s  eta {:.0f}s".format(
                            done, a.n, el, el * (a.n - done) / max(done, 1)),
                            flush=True)

    ok = np.abs(E3).sum(1) > 0
    print("\nembedded {:,} of {:,} in {:.0f}s".format(
        int(ok.sum()), a.n, time.time() - t0), flush=True)
    sel, E3, E6 = sel[ok], E3[ok], E6[ok]

    nq = a.queries
    qi, bi = sel[:nq], sel[nq:]
    same = seq[qi][:, None] == seq[bi][None, :]
    print("same-sequence pairs masked: {:,}".format(int(same.sum())))

    def row(tag, dims, m, h, a32):
        print("%-10s %7d %11.1f %12.1f%% %13.1f%%" % (tag, dims, m, 100 * h,
                                                      100 * a32))

    print("\n%-10s %7s %11s %12s %13s" %
          ("scheme", "dims", "top1 km", "top1 <25km", "any32 <25km"))
    print("-" * 58)
    for tag, E in (("crop3", E3), ("tile6", E6)):
        row(tag + " full", E.shape[1], *score(E[:nq], E[nq:], lat, lon, qi, bi, same))

    for d in [int(x) for x in a.dims.split(",")]:
        for tag, E in (("crop3", E3), ("tile6", E6)):
            if d >= E.shape[1]:
                continue
            mu, P = pca_fit(E[nq:], d, rng)
            row("%s pca%d" % (tag, d), d,
                *score((E[:nq] - mu) @ P, (E[nq:] - mu) @ P, lat, lon, qi, bi, same))

    print("\nequal bytes is the comparison: tile6 stores {:.1f}x crop3 raw, so "
          "read the pca rows.".format(E6.shape[1] / E3.shape[1]))


if __name__ == "__main__":
    main()
