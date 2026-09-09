"""Pretrain the fusion trunk by reconstruction, on 90x the images.

`pyramid-fusion-head-is-negative` recorded the failure and named the cause:

    | epochs | train loss | head <25 km | median   |
    |--------|-----------|-------------|----------|
    | init   | --        | 33.1%       | (= mean) |
    | 1      | --        | 32.2%       | 207.4 km |
    | 12     | 0.063     | 20.5%       | 384.1 km |
    | 12, 0.5 km pos | 0.0145 | 12.9%  | 670.5 km |

Loss falls, retrieval falls with it, monotonically -- 2.5M parameters against
**38,009** images. Its own post-mortem says what to do about it: "if it is
ever reopened, the thing to attack is capacity and data volume, not the
objective's details."

This attacks data volume. The contrastive objective needs an anchor with a
positive within `--pos-km` from a *different sequence*, and only 38,009 images
have one. Reconstruction needs no pairs, so it trains on every row the corpus
has -- 3,400,180, from vectors already on disk.

**What is pretrained, and what deliberately is not.** Only the trunk: the
input projections, the region/level/encoder embeddings, the cross- and
self-attention, the FFN and the attention pool. `out` is left untouched at its
zero initialisation, so the contrastive phase still emits exactly the
mean-pooled baseline at step 0 and can only depart if it pays. That property
is what makes the head's numbers comparable at all, and pretraining the output
layer would spend it.

**What this cannot fix.** Reconstruction preserves variance; retrieval needs
neighbourhood structure, and nothing connects the two. A linear autoencoder at
this bottleneck would be exactly PCA (Eckart-Young) and the pipeline already
ends in PCA-768, so any difference here comes from the nonlinearity and from
the extra data, not from the idea of compression. The honest read of a null is
therefore "reconstruction is the wrong pretext", not "pooling cannot be
learned".

    OSV_RELEASE=s10 py scripts/fuse_pretrain.py --epochs 3 --out fuse_trunk.pt
"""

import argparse
import sys
import time
from pathlib import Path

import pyarrow.dataset          # noqa: F401  (see pool_pyramid: DLL load order)
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config                                        # noqa: E402
import safeio                                        # noqa: E402
from fuse_head import FuseHead, D_ENC                # noqa: E402
from pool_pyramid import tile_positions              # noqa: E402

# (crop cache, tile cache). The release first, then the extensions in the
# order `bank_ext70_meta.npz` records -- not because reconstruction cares
# about order, but because a mismatched pair is one image's crops beside
# another image's tiles, which trains fine and means nothing.
CORPORA = [("dual_c3.f16.npy", "tile6"),
           ("bank_ext_bal.f16.npy", "tile6_ext"),
           ("bank_ext2_bal.f16.npy", "tile6_ext2"),
           ("bank_ext3_bal.f16.npy", "tile6_ext3"),
           ("bank_ext4_bal.f16.npy", "tile6_ext4")]

# Everything the contrastive phase must NOT inherit. `out` is the residual on
# the mean-pooled baseline and is zero-initialised on purpose.
NOT_PRETRAINED = ("out.",)


def drop_views(x, rng, keep_min=3):
    """Blank a random subset of the nine views, per row.

    The free half of geometric invariance. What we actually want is
    "stretch the frame and the vector should not move", and that cannot be
    synthesised: every cached vector was encoded from a 910x512 frame, so a
    stretched view needs a new DINOv2 and SigLIP pass. The corruption here has
    the same SHAPE -- degrade the input, target the clean embedding -- and
    costs nothing, but it teaches robustness to *which views arrive* rather
    than to the frame they were cut from.

    That distinction matters because aspect ratio changes what the views ARE.
    `embed_street.preprocess` scales the short side to 224 and slides a 224
    window across the width, so the three crops sit at columns 0.34/1.00/1.66
    on a 910x512 frame, 0.62/1.01/1.38 at 4:3, and at 1.00/1.00/1.00 on
    anything square or taller -- three copies of one crop. A square photograph
    is a degenerate pyramid this corpus contains no examples of.
    """
    b, r = x.shape[0], x.shape[1]
    keep = torch.ones(b, r, 1, 1, device=x.device)
    n_drop = int(rng.integers(0, r - keep_min + 1))
    if n_drop:
        idx = torch.argsort(torch.rand(b, r, device=x.device), dim=1)[:, :n_drop]
        keep.scatter_(1, idx.view(b, n_drop, 1, 1), 0.0)
    return x * keep


class Decoder(nn.Module):
    """Pooled trunk output -> the 18 tokens it came from.

    Deliberately small and deliberately thrown away. It exists only to give
    the trunk a gradient; keeping it would invite someone to reuse a decoder
    that was never evaluated on anything.
    """

    def __init__(self, d, n_reg, hidden=2048):
        super().__init__()
        self.n_reg = n_reg
        self.net = nn.Sequential(
            nn.LayerNorm(4 * d), nn.Linear(4 * d, hidden), nn.GELU(),
            nn.Linear(hidden, n_reg * 2 * D_ENC))

    def forward(self, p):
        return self.net(p).view(-1, self.n_reg, 2, D_ENC)


def trunk_forward(model, x):
    """`FuseHead.forward` up to the pool, without the residual or the norm.

    Copied structure rather than refactored: `forward` is the shipping path
    for three recorded results, and splitting it would put this experiment's
    changes inside them.
    """
    B, R, E, _ = x.shape
    reg = torch.arange(R, device=x.device)
    h = torch.stack([model.proj[e](x[:, :, e]) for e in range(E)], dim=2)
    h = h + (model.reg(reg) + model.lvl(model.level_of)).unsqueeze(0).unsqueeze(2)
    h = h + model.enc(torch.arange(E, device=x.device)).view(1, 1, E, -1)
    a, b = h[:, :, 0], h[:, :, 1]
    an, bn = model.n_cross(a), model.n_cross(b)
    a = a + model.cross(an, bn, bn, need_weights=False)[0]
    b = b + model.cross(bn, an, an, need_weights=False)[0]
    t = torch.cat([a, b], dim=1)
    tn = model.n_self(t)
    t = t + model.self_attn(tn, tn, tn, need_weights=False)[0]
    t = t + model.ffn(model.n_ffn(t))
    q = model.q.unsqueeze(0).expand(B, -1, -1)
    return model.pool(q, t, t, need_weights=False)[0].reshape(B, -1)


def corpus_blocks(block):
    """Yield aligned (crops, tiles) blocks across every corpus.

    Streamed rather than loaded: the five pairs are about 96 GB together.
    """
    for src, tstem in CORPORA:
        cp = config.STREET_CACHE / src
        tp = config.STREET_CACHE / (tstem + ".f16.npy")
        if not cp.exists() or not tp.exists():
            print("skip {} + {} (absent)".format(src, tstem), flush=True)
            continue
        C = np.load(cp, mmap_mode="r")
        T = np.load(tp, mmap_mode="r")
        n = len(C)
        # The tile cache stores a SELECTION; this is the guard that refuses a
        # partly-written one rather than training on zero rows that
        # L2-normalise to unit-length nothing.
        pos = tile_positions(tstem, np.arange(n))
        half = C.shape[1] // 2
        crops = half // D_ENC
        for s in range(0, n, block):
            e = min(s + block, n)
            c = np.asarray(C[s:e], np.float32)
            c = np.stack([c[:, :half].reshape(-1, crops, D_ENC),
                          c[:, half:].reshape(-1, crops, D_ENC)], axis=2)
            t = np.asarray(T[pos[s:e]], np.float32)
            t = np.stack([t[:, :, :D_ENC], t[:, :, D_ENC:]], axis=2)
            x = np.concatenate([c, t], axis=1)
            x /= np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-6)
            yield src, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--block", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many rows per epoch; 0 = all")
    ap.add_argument("--inv-weight", type=float, default=1.0,
                    help="weight on the view-dropout invariance term. 0 makes "
                         "this a plain autoencoder, which at a linear "
                         "bottleneck would be PCA and the pipeline already "
                         "ends in one.")
    ap.add_argument("--out", default="fuse_trunk.pt")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = FuseHead(d=a.d).to(dev)
    dec = Decoder(a.d, int(model.level_of.numel())).to(dev)
    # `out` never receives a gradient here: it is the residual on the
    # mean-pooled baseline and must still be zero when the contrastive phase
    # starts, or that phase no longer begins at the baseline.
    for p in model.out.parameters():
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params + list(dec.parameters()), lr=a.lr)

    print("trunk {:,} params, decoder {:,}".format(
        sum(p.numel() for p in params),
        sum(p.numel() for p in dec.parameters())), flush=True)

    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        seen, tot, nb, inv_tot = 0, 0.0, 0, 0.0
        for src, block in corpus_blocks(a.block):
            X = torch.from_numpy(block)
            for s in range(0, len(X), a.batch):
                x = X[s:s + a.batch].to(dev, non_blocking=True)
                p = trunk_forward(model, x)
                # Cosine rather than MSE: every token is already unit length,
                # so MSE would be a monotone function of this anyway, and the
                # cosine keeps the scale of the loss readable across widths.
                r = torch.nn.functional.normalize(dec(p), dim=-1)
                loss = (1.0 - (r * x).sum(-1)).mean()
                if a.inv_weight:
                    # Predict the CLEAN pooled vector from a degraded input.
                    # The target is detached: without that, the cheapest way
                    # to agree is for both to collapse, which is the standard
                    # failure of a two-view objective with no negatives.
                    q = trunk_forward(model, drop_views(x, rng))
                    inv = 1.0 - torch.nn.functional.cosine_similarity(
                        q, p.detach(), dim=-1).mean()
                    loss = loss + a.inv_weight * inv
                    inv_tot += float(inv)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    params + list(dec.parameters()), 1.0)
                opt.step()
                tot += float(loss)
                nb += 1
                seen += len(x)
            if a.limit and seen >= a.limit:
                break
        print("ep {:2d}  {:>10,} rows  recon {:.4f}  inv {:.4f}  {:.0f}s"
              .format(ep, seen, tot / max(nb, 1), inv_tot / max(nb, 1),
                      time.time() - t0), flush=True)

    state = {k: v for k, v in model.state_dict().items()
             if not any(k.startswith(p) for p in NOT_PRETRAINED)}
    out = config.CHECKPOINTS / a.out
    safeio.save_torch({"trunk": state, "d": a.d, "epochs": a.epochs,
                       "rows_per_epoch": seen, "recon": tot / max(nb, 1),
                       "corpora": [c for c, _ in CORPORA]}, out)
    print("wrote {}  ({} tensors, `out` deliberately excluded)"
          .format(out.name, len(state)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
