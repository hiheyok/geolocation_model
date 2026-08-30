"""The three encoders.  All small: the heavy lifting is frozen and precomputed."""

import math

import torch
import torch.nn as nn


def sinusoidal(v, n_freq=8):
    """Scalar in [0,1] -> multi-scale features.

    A bare float is hard for an MLP to use across scales; log-spaced sin/cos
    gives it the same handle RoPE provides.  v: (...,) -> (..., 2*n_freq)
    """
    freqs = (2.0 ** torch.arange(n_freq, device=v.device, dtype=v.dtype)) * math.pi
    a = v.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def rope2d_tables(n_actions=256, d_tok=256):
    """Per-position rotation angles for an axial 2D rotary encoding.

    Half the channel pairs carry the column, half the row.  Wavelengths are
    spread geometrically from 2 to 2*g patches rather than using the usual
    base-10000 schedule: that schedule is built for sequences of thousands, and
    across a 16-wide axis its high-index channels turn by less than 0.01 rad end
    to end, spending dimensions on nothing.
    """
    g = int(round(n_actions ** 0.5))
    assert g * g == n_actions, "grid must be square"
    assert d_tok % 4 == 0, "need an even number of rotation pairs per axis"
    n_pair = d_tok // 4
    k = torch.arange(n_pair, dtype=torch.float32)
    theta = 2 * math.pi / (2.0 * g ** (k / max(1, n_pair - 1)))
    idx = torch.arange(n_actions)
    col = (idx % g).float().unsqueeze(1)
    row = torch.div(idx, g, rounding_mode="floor").float().unsqueeze(1)
    ang = torch.cat([col * theta, row * theta], dim=1)   # (A, d_tok//2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate each adjacent channel pair of x by its position's angle.

    x (..., A, d); cos/sin (A, d/2).  Norm-preserving, so it composes with a
    LayerNorm applied beforehand instead of undoing it.
    """
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


class StreetProj(nn.Module):
    """Frozen DINOv2 vector -> model width."""

    def __init__(self, d_in=768, d=512):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.proj(x))


class AttnPool(nn.Module):
    """A few learned queries read the map tokens, instead of averaging them.

    Positions are added before the mean in `MapTokenizer`, so `mean(pos)` is a
    constant and the mean hands fusion a class histogram -- "30% water" -- with
    the arrangement gone.  Measured on a trained checkpoint: permuting all 256
    patches within a tile moves the pooled vector 0.3% as far as swapping in a
    different tile does, and 83% of its variance is linear in the bare 12-d
    histogram.

    Attention weights depend on content, so the position components of the
    values no longer cancel: each query's output says *where* its feature is,
    not just how much of it there is.  Separate queries can specialise.
    """

    def __init__(self, d_tok=256, d=512, n_q=4, heads=4, dropout=0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_q, d_tok) * d_tok ** -0.5)
        self.attn = nn.MultiheadAttention(d_tok, heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_tok)
        self.out = nn.Sequential(
            nn.Linear(n_q * d_tok, d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, k):
        q = self.q.unsqueeze(0).expand(k.shape[0], -1, -1)
        h = self.norm(self.attn(q, k, k, need_weights=False)[0])
        return self.out(h.reshape(k.shape[0], -1))


class MapTokenizer(nn.Module):
    """Cached per-patch class histograms -> attention keys.

    The keys double as the action space: token i *is* action i, so the policy
    head scores the map grid directly instead of learning the correspondence.

    `pool="mean"` reproduces the original readout exactly, so the attention
    pool is a controlled ablation rather than a replacement.
    """

    def __init__(self, n_classes=12, n_actions=256, d_tok=256, d=512,
                 pool="mean", n_q=4, dropout=0.1, pos="learned"):
        super().__init__()
        self.proj = nn.Linear(n_classes, d_tok)
        self.pos_kind = pos
        self.pos = (nn.Embedding(n_actions, d_tok)
                    if pos in ("learned", "both") else None)
        if pos in ("rope", "both"):
            cos, sin = rope2d_tables(n_actions, d_tok)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        self.norm = nn.LayerNorm(d_tok)
        self.pool_kind = pool
        if pool == "attn":
            self.pool = AttnPool(d_tok, d, n_q, dropout=dropout)
        else:
            self.pool = nn.Sequential(
                nn.Linear(d_tok, d), nn.GELU(), nn.LayerNorm(d))
        self.n_actions = n_actions

    def forward(self, tokens):
        # tokens: (N, A, C) -> keys (N, A, d_tok), pooled (N, d)
        h = self.proj(tokens)
        if self.pos is not None:
            h = h + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        k = self.norm(h)
        if self.pos_kind in ("rope", "both"):
            # after the norm: rotation preserves length, so a per-dim
            # renormalisation afterwards would partly undo it.
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        return k, self.pool(k if self.pool_kind == "attn" else k.mean(dim=1))


class MapBlock(nn.Module):
    """Pre-norm self-attention + FFN over map tokens.

    Without this the map tokens never interact: each one knows only its own
    patch histogram and its position, so the encoder cannot represent a
    coastline, a junction or a peninsula -- contrast against neighbours is
    exactly what those are.  Those features are what street-to-map matching
    runs on at fine zoom, which is where the model is weakest.
    """

    def __init__(self, d=256, heads=4, dropout=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop(self.ffn(self.n2(x)))


class MapTransformer(nn.Module):
    """N blocks over [cond; K0..K255], conditioned on the fused vector.

    No circularity: K -> pooled -> fused -> here -> policy readout.
    """

    def __init__(self, n_layers=2, d_tok=256, d=512, heads=4, dropout=0.1,
                 loop=False):
        super().__init__()
        self.cond = nn.Linear(d, d_tok)
        self.n_iter = n_layers
        self.loop = loop
        # Weight tying gives depth without capacity.  That is the interesting
        # property here rather than the parameter saving as such: the learning
        # curve says this model is data-limited, and a tied loop is a
        # regulariser where untied layers are the opposite.
        self.blocks = nn.ModuleList(
            [MapBlock(d_tok, heads, dropout) for _ in range(1 if loop else n_layers)])
        # Without a per-iteration signal the tied block cannot tell its passes
        # apart and the extra ones are close to wasted -- the same reason the
        # policy needs a step embedding.
        self.iter_emb = nn.Embedding(n_layers, d_tok) if loop else None
        self.out = nn.LayerNorm(d_tok)

    def forward(self, keys, fused):
        x = torch.cat([self.cond(fused).unsqueeze(1), keys], dim=1)
        for i in range(self.n_iter):
            if self.iter_emb is not None:
                x = x + self.iter_emb.weight[i]
            x = self.blocks[0 if self.loop else i](x)
        return self.out(x[:, 1:])


class StateEncoder(nn.Module):
    """Current tile only -- no trajectory.

    Position is what makes the policy Markov: without it two similar-looking
    tiles at different world positions alias into one state.
    """

    def __init__(self, n_steps=5, n_freq=8, d_step=128, d=512):
        super().__init__()
        self.n_freq = n_freq
        self.step = nn.Embedding(n_steps, d_step)
        d_in = 4 * n_freq + d_step
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, x0, y0, step):
        f = torch.cat([sinusoidal(x0, self.n_freq),
                       sinusoidal(y0, self.n_freq),
                       self.step(step)], dim=-1)
        return self.mlp(f)
