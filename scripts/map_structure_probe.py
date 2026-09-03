"""How much structure does the 12-d class histogram throw away?

Each map token is a 32x32 patch -- 1024 pixels -- reduced to 12 class
fractions. That is aggregation, which is the right call for class ids (they must
never be interpolated), but it is orientation-blind by construction: a patch
holding a north-south road and one holding an east-west road have *identical*
tokens, and so do a T-junction and a straight road carrying the same number of
road pixels. Street layout is what separates one place from another at z12-z16,
so this is worth measuring rather than assuming.

The probe asks one question with a clean answer: **can the orientation of a
patch's linear structure be recovered from what the model is actually given?**

  * label each patch by the dominant orientation of its road pixels, from the
    structure tensor of the full-resolution binary mask -- horizontal, vertical,
    or neither
  * try to predict that label from the 12-d histogram the model sees
  * try again from a 4x4 sub-histogram (192-d), which is the cheapest
    representation that *can* express orientation

If the 12-d probe sits at the majority-class rate while the 192-d probe is well
above it, the information is present in the tile and destroyed by the
tokenizer, and enriching the token is worth a training run. If both are high,
the histogram already carries it indirectly and there is nothing to recover.

No GPU: masks come from the tile server and everything else is numpy.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config
import tile_math as tm
import tiles as T


def sub_tokens(mask, grid=16, sub=4):
    """(512,512) mask -> (grid*grid, 12*sub*sub) sub-patch class fractions.

    The same aggregation as to_tokens, applied to sub x sub cells inside each
    patch, so within-patch layout survives. sub=1 reproduces to_tokens exactly.
    """
    px = mask.shape[0]
    p = px // grid
    c = p // sub
    ids = (mask // T.CLASS_STEP).astype(np.uint8)
    # (grid, sub, c, grid, sub, c) -> one row per patch, cells in row-major
    a = ids.reshape(grid, sub, c, grid, sub, c)
    a = a.transpose(0, 3, 1, 4, 2, 5).reshape(grid * grid, sub * sub, c * c)
    out = np.zeros((grid * grid, sub * sub, T.N_CLASSES), np.float32)
    for k in range(T.N_CLASSES):
        out[:, :, k] = (a == k).sum(-1)
    out /= float(c * c)
    return out.reshape(grid * grid, sub * sub * T.N_CLASSES)


def orientation(mask, grid=16, road_ids=(1, 2, 3), min_px=40):
    """Per patch: 0 horizontal, 1 vertical, 2 neither/ambiguous.

    Structure tensor of the binary road mask. A patch whose road pixels form a
    coherent line gets that line's direction; anything sparse or isotropic is
    dropped rather than guessed, so the label is only defined where there is
    real structure to find.
    """
    px = mask.shape[0]
    p = px // grid
    ids = (mask // T.CLASS_STEP).astype(np.uint8)
    road = np.isin(ids, list(road_ids)).astype(np.float32)
    gy, gx = np.gradient(road)
    lab = np.full(grid * grid, 2, np.int8)
    for i in range(grid):
        for j in range(grid):
            sy = slice(i * p, (i + 1) * p)
            sx = slice(j * p, (j + 1) * p)
            if road[sy, sx].sum() < min_px:
                continue
            ex, ey = gx[sy, sx].ravel(), gy[sy, sx].ravel()
            jxx, jyy, jxy = (ex * ex).sum(), (ey * ey).sum(), (ex * ey).sum()
            tr = jxx + jyy
            if tr < 1e-6:
                continue
            # coherence: how line-like the gradient distribution is
            det = jxx * jyy - jxy * jxy
            coh = np.sqrt(max(tr * tr - 4 * det, 0.0)) / tr
            if coh < 0.5:
                continue
            # dominant gradient direction; the edge runs perpendicular to it
            th = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
            lab[i * grid + j] = 0 if abs(np.cos(th)) > abs(np.sin(th)) else 1
    return lab


def mlp_acc(Xtr, ytr, Xte, yte, hidden=96, epochs=400, seed=0):
    """One hidden layer, CPU torch.

    A linear probe is the wrong instrument here and will understate what the
    sub-histogram holds: a road at an arbitrary offset makes orientation a
    disjunction -- *some* column is filled across all rows -- and no linear
    function of cell occupancies expresses that. If a representation carries the
    structure at all, a single hidden layer should find it.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(Xte); yv = torch.from_numpy(yte)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], hidden),
                              torch.nn.GELU(),
                              torch.nn.Linear(hidden, 2))
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.05, generator=g)
            torch.nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((net(xv).argmax(1) == yv).float().mean())


def cnn_acc(Ptr, ytr, Pte, yte, ch=16, epochs=250, seed=0):
    """A small conv over the raw one-hot patch, as the ceiling.

    The sub-histograms are a hand-designed compromise; a conv sees all 1024
    pixels and can learn oriented filters directly. What matters is not whether
    it beats them -- it must -- but by how much, because that gap is the whole
    payoff for caching masks and paying convolution on every training step,
    against a wider Linear for sub=2.
    """
    import torch
    g = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(12, ch, 3, stride=2, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), torch.nn.GELU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
        torch.nn.Linear(ch * 2, 2))
    for m in net:
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.normal_(m.weight, 0, 0.08, generator=g)
            torch.nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    xt = torch.from_numpy(Ptr); yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(Pte); yv = torch.from_numpy(yte)
    n, bs = len(xt), 512
    for _ in range(epochs):
        i = torch.randint(0, n, (bs,), generator=g)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(xt[i]), yt[i]).backward()
        opt.step()
    with torch.no_grad():
        pred = torch.cat([net(xv[i:i + 1024]).argmax(1)
                          for i in range(0, len(xv), 1024)])
        return float((pred == yv).float().mean())


def raw_patches(mask, grid=16):
    """(grid*grid, 12, p, p) one-hot patches -- never interpolated."""
    px = mask.shape[0]
    p = px // grid
    ids = (mask // T.CLASS_STEP).astype(np.uint8)
    a = ids.reshape(grid, p, grid, p).transpose(0, 2, 1, 3)
    a = a.reshape(grid * grid, p, p)
    out = np.zeros((grid * grid, T.N_CLASSES, p, p), np.float32)
    for k in range(T.N_CLASSES):
        out[:, k] = (a == k)
    return out


def logreg(X, y, n_class, epochs=300, lr=0.5, seed=0):
    """Multinomial logistic regression, full batch. Small enough for numpy."""
    rng = np.random.default_rng(seed)
    X = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    W = np.zeros((X.shape[1], n_class), np.float32)
    Y = np.eye(n_class, dtype=np.float32)[y]
    for _ in range(epochs):
        z = X @ W
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        W -= lr * (X.T @ (p - Y)) / len(X)
    return W


def acc(W, X, y):
    X = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1)
    return float((np.argmax(X @ W, 1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="tiles to sample")
    ap.add_argument("--zoom", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    ds = pq.read_table(config.DATASET_PARQUET, columns=["lat", "lon"])
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    rng = np.random.default_rng(a.seed)
    pick = rng.permutation(len(lat))[:a.n]

    client = T.TileClient(config.TILE_SERVER)
    X12, X48, X192, RAW, Y = [], [], [], [], []
    kept = 0
    for n, i in enumerate(pick):
        _, x, y = tm.tile_for(float(lat[i]), float(lon[i]), a.zoom)
        try:
            m = client.mask(a.zoom, x, y)
        except Exception:
            continue
        lab = orientation(m)
        sel = lab < 2
        if not sel.any():
            continue
        X12.append(T.to_tokens(m)[sel])
        X48.append(sub_tokens(m, sub=2)[sel])
        X192.append(sub_tokens(m)[sel])
        if len(RAW) < 40:
            RAW.append(raw_patches(m)[sel])
        Y.append(lab[sel])
        kept += 1
        if (n + 1) % 100 == 0:
            print("  {}/{} tiles, {:,} oriented patches"
                  .format(n + 1, a.n, sum(len(v) for v in Y)), flush=True)

    X12 = np.concatenate(X12).astype(np.float32)
    X48 = np.concatenate(X48).astype(np.float32)
    X192 = np.concatenate(X192).astype(np.float32)
    RAW = np.concatenate(RAW).astype(np.float32)
    Y = np.concatenate(Y).astype(np.int64)
    n = len(Y)
    print("\n{:,} patches from {:,} tiles at z{} with a coherent road direction"
          .format(n, kept, a.zoom))
    print("  horizontal {:.1%}   vertical {:.1%}"
          .format((Y == 0).mean(), (Y == 1).mean()))

    idx = np.random.default_rng(1).permutation(n)
    cut = int(n * 0.7)
    tr, te = idx[:cut], idx[cut:]
    base = max((Y[te] == 0).mean(), (Y[te] == 1).mean())

    print("\npredicting road orientation from the token the model is given")
    print("  {:<28} {:>7}".format("majority class", "{:.1%}".format(base)))
    for name, X in (("12-d histogram (current)", X12),
                    ("48-d 2x2 sub-histogram", X48),
                    ("192-d 4x4 sub-histogram", X192)):
        W = logreg(X[tr], Y[tr], 2)
        lin = acc(W, X[te], Y[te])
        nl = mlp_acc(X[tr], Y[tr], X[te], Y[te])
        print("  {:<28} {:>7} linear   {:>7} MLP".format(
            name, "{:.1%}".format(lin), "{:.1%}".format(nl)))

    nr = len(RAW)
    Yr = Y[:nr]
    ir = np.random.default_rng(2).permutation(nr)
    ct = int(nr * 0.7)
    rtr, rte = ir[:ct], ir[ct:]
    rbase = max((Yr[rte] == 0).mean(), (Yr[rte] == 1).mean())
    print("")
    print("ceiling: a small conv over the raw 32x32 one-hot patch")
    print("  {:<28} {:>7}  (n={:,} patches)".format(
        "majority class", "{:.1%}".format(rbase), nr))
    for name, X in (("   12-d, same subset", X12[:nr]),
                    ("   48-d, same subset", X48[:nr]),
                    ("  192-d, same subset", X192[:nr])):
        print("  {:<28} {:>7}".format(name, "{:.1%}".format(
            mlp_acc(X[rtr], Yr[rtr], X[rte], Yr[rte]))))
    print("  {:<28} {:>7}".format("  conv on raw pixels",
          "{:.1%}".format(cnn_acc(RAW[rtr], Yr[rtr],
                                  RAW[rte], Yr[rte]))))

    print("\nA 12-d score at the majority rate means the orientation is in the\n"
          "tile and destroyed by the tokenizer, not absent from the map.")


if __name__ == "__main__":
    main()
