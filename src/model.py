"""GeoAgent: a Markov policy over map tokens, plus a click head.

One refinement against the written plan.  The plan bypassed the map encoder at
step 0, on the grounds that the world view is byte-identical for every sample so
pushing it through an encoder is a no-op.  That held when the map encoder was a
separate CNN producing only a pooled vector.  It no longer does: the policy head
needs the world tile's *keys* to score the 256 world cells, so the tokens must be
computed at step 0 regardless, and mean-pooling them is then nearly free.  Step 0
is treated uniformly; the step embedding already tells the model where it is.
"""

import math
import sys
from pathlib import Path

from typing import NamedTuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import (GeoMem, MapTokenizer, MapTransformer, StateEncoder,
                      StreetProj)
from retrieval import RetrievalPrior


class NeighborBatch(NamedTuple):
    """The retrieved neighbours of a batch of images, by name rather than index.

    This was four positional slots whose meaning was assigned by `nbrs[0]`
    through `nbrs[3]`, whose optional last element was detected with
    `len(nbrs) > 3`, and which three call sites built independently
    (`GeoAgent.forward`, `train.run_epoch`, `evaluate`). That shape has already
    cost twice: beam search once built a version omitting the learned positive
    and negative keys, and the conditioning adapter sliced the query's
    conditioning suffix while leaving it on `emb`, so `pos` and `dual` raised on
    the first batch (REVIEW6 #1).

    Both are the same failure -- an operation that must be applied to several
    members, expressed as several independent operations. `retrieval_only`
    exists so "drop the conditioning" is one named thing.

    Shapes, per *image* (not per row: an image contributes one row per zoom
    step in training and one per live beam in search):

        x, y   (B, K) int64    the neighbour's z16 tile address
        sim    (B, K) float    its cosine similarity to the query
        emb    (B, K, D) float its embedding, RETRIEVAL width -- or None for
                              the modes that do not use keys
    """

    x: "torch.Tensor"
    y: "torch.Tensor"
    sim: "torch.Tensor"
    emb: "torch.Tensor" = None

    @classmethod
    def of(cls, nbrs):
        """Accept this, or any legacy 3- or 4-element sequence."""
        if nbrs is None or isinstance(nbrs, cls):
            return nbrs
        return cls(*nbrs)

    def retrieval_only(self, d_in):
        """Drop a conditioning suffix from the neighbour embeddings.

        Idempotent, so it is safe wherever the width is already right.
        """
        if self.emb is None or self.emb.shape[-1] == d_in:
            return self
        return self._replace(emb=self.emb[..., :d_in])


class GeoAgent(nn.Module):
    def __init__(self, d_street=768, d=512, d_tok=256, n_classes=12,
                 n_actions=256, n_steps=5, dropout=0.1, map_layers=0,
                 pool="mean", n_pool_q=4, pos="learned", sink=False, sink_k=1,
                 map_loop=False, mem="none", d_mem=64, mem_drop=0.0,
                 retr=False, retr_tau=0.07, retr_mode="scalar", d_key=128,
                 nbr_drop=0.0,
                 enc_gate=False, geo="none", d_geo=128, d_cond=0):
        super().__init__()
        self.n_actions = n_actions
        self.n_regions = n_actions          # z4 cells and actions are the same grid
        self.d_tok = d_tok
        self.map_layers = map_layers
        self.pool_kind = pool
        self.pos_kind = pos
        # A 257th action meaning "the truth is not inside this tile".  It is not
        # a place, so it gets its own learned key rather than a grid position:
        # the other 256 keys are the map, this one is a verdict on it.  Because
        # log p(a) = log p(not-sink) + log p(a | not-sink), a beam the model
        # thinks is dead pays for it in its own cumulative score, with no
        # threshold to tune.
        self.sink = nn.Parameter(torch.randn(d_tok) * d_tok ** -0.5) if sink else None

        # Extra sink capacity.  The base sink is a single fixed direction, so
        # its logit is one linear probe on the fused state -- it cannot look at
        # the map the way a real action key does, and one direction has to
        # serve all four zoom steps.  `sink_k > 1` adds K-1 further keys *per
        # step* and takes a logsumexp, which reads as "reject if ANY of K
        # reasons fires" and is piecewise-linear rather than linear.
        #
        # Neutral at init *and* able to learn.  A -20 bias is neutral but
        # dead: it gives each extra about e^-20 of the log-sum-exp gradient, so
        # the keys never move and sink_k is a no-op that looks like a null
        # result.  Instead the extras carry an ordinary bias and a zero-init
        # gate interpolates:
        #     sink = base + g * (logsumexp([base, extras]) - base)
        # g=0 is exactly the base logit, g=1 is the full mixture, and g starts
        # with a real gradient because the bracket is strictly positive. Same
        # contract as w_pos in RetrievalPrior -- it departs only if it pays.
        self.sink_k = int(sink_k) if sink else 1
        self.sink_steps = max(1, n_steps - 1)
        if sink and self.sink_k > 1:
            self.sink_ext = nn.Parameter(
                torch.randn(self.sink_steps, self.sink_k - 1, d_tok)
                * d_tok ** -0.5)
            self.sink_ext_b = nn.Parameter(
                torch.zeros(self.sink_steps, self.sink_k - 1))
            # One gate per step, not one scalar. Sink negatives are sampled
            # only at steps 1 and 2 (dataset._negatives), so the step-0 and
            # step-3 keys see nothing but "do not fire". Under a shared gate
            # those gradients fight the two supervised steps and can drive it
            # negative, which is not the interpolation the formula describes.
            # Per-step, an unsupervised step simply keeps its gate near zero.
            self.sink_ext_g = nn.Parameter(torch.zeros(self.sink_steps))
        else:
            self.sink_ext = None

        # enc_gate: one scalar per (step, encoder block) in front of the street
        # projection.  See StreetProj -- the encoders win at different spatial
        # scales and the steps decide at different spatial scales, so a single
        # shared ratio is leaving something on the table.
        # d_street is the RETRIEVAL width. When d_cond is set the dataset hands
        # a wider tensor -- retrieval block first, conditioning appended -- and
        # StreetProj splits it. Retrieval never sees the conditioning block:
        # the k-NN cache was built on the retrieval file alone, which is the
        # entire point of the decoupling (runs/PYR_LEVELS.md measured that
        # putting extra detail *into* the retrieval vector is churn without
        # gain).
        self.street = StreetProj(d_street, d, n_steps=n_steps if enc_gate else 0,
                                 d_cond=d_cond)
        # A learned key per map tile, added to the policy readout. See GeoMem:
        # absolute geography reaches the logits nowhere else, since the map keys
        # are built from mask content and the query is shared across candidates.
        self.geo = None if geo == "none" else GeoMem(d, d_geo, geo, g=int(
            n_actions ** 0.5))
        self.map = MapTokenizer(n_classes, n_actions, d_tok, d,
                                pool=pool, n_q=n_pool_q, dropout=dropout,
                                pos=pos)
        self.state = StateEncoder(n_steps, 8, 128, d)

        # A persistent prior per z4 region, kept deliberately small.  Only z4
        # earns a table: 77 of its 256 rows are occupied with ~520 training
        # images each, while z8 has 14.6 per row and z12 has 1.4 -- a row there
        # is parameters fitted to one image, and an empirical count table over
        # child cells already beats the model at step 2 on the sequence split,
        # so a deeper table would be scoring co-location rather than geography.
        # Row 256 is the root: step 0 is the whole world and has no z4 ancestor.
        # "rand" freezes the table at init, the control for "does a lookup of
        # the right shape help even when it stores nothing".
        self.mem_kind, self.mem_drop = mem, mem_drop
        if mem == "none":
            self.mem = None
        else:
            self.mem = nn.Embedding(self.n_regions + 1, d_mem)
            if mem == "rand":
                self.mem.weight.requires_grad_(False)
            self.mem_proj = nn.Linear(d_mem, d)

        d_fuse = (4 if self.mem is not None else 3) * d
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_fuse),
            nn.Linear(d_fuse, 2 * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d, d), nn.LayerNorm(d))

        # 0 layers reproduces the linear-readout model exactly, so this is a
        # controlled ablation rather than a replacement.
        self.map_tf = (MapTransformer(map_layers, d_tok, d, 4, dropout,
                                     loop=map_loop)
                       if map_layers else None)

        self.retr = (RetrievalPrior(int(n_actions ** 0.5), retr_tau,
                                    mode=retr_mode, d_street=d_street,
                                    d_key=d_key, n_steps=n_steps - 1,
                                    nbr_drop=nbr_drop)
                     if retr else None)

        self.q_proj = nn.Linear(d, d_tok)
        self.click = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 2))

    def fuse(self, street, tokens, x0, y0, step):
        """street (B,Ds); tokens (B,S,A,C); x0/y0/step (B,S).

        Returns fused (B*S, d) and keys (B*S, A, d_tok), flattened over steps.
        """
        B, S, A, C = tokens.shape
        st = street.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
        return self.fuse_flat(st, tokens.reshape(B * S, A, C),
                              x0.reshape(-1), y0.reshape(-1), step.reshape(-1))

    def fuse_flat(self, street, tokens, x0, y0, step):
        """The one place the fusion input is assembled.  (N,Ds), (N,A,C), (N,).

        Beam search used to inline this cat, so adding a fourth block to the
        fusion silently produced a shape error there while training passed --
        the same duplicate-implementation failure as the split had.  Every
        caller now routes through here.
        """
        k, pooled = self.map(tokens)
        parts = [self.street(street, step), pooled, self.state(x0, y0, step)]
        if self.mem is not None:
            parts.append(self.region_prior(x0, y0, step))
        return self.fusion(torch.cat(parts, dim=-1)), k

    def region_prior(self, x0, y0, step):
        """The z4 ancestor of the current tile, read straight off its corner.

        x0/y0 are the normalised Mercator corner, so the z4 cell is just the
        top nibble of each -- no new dataset column and nothing for beam search
        to thread through, which is what keeps this a clean ablation.
        """
        g = int(self.n_regions ** 0.5)
        ax = (x0 * g).long().clamp_(0, g - 1)
        ay = (y0 * g).long().clamp_(0, g - 1)
        idx = torch.where(step == 0, torch.full_like(ax, self.n_regions), ay * g + ax)
        m = self.mem_proj(self.mem(idx))
        if self.training and self.mem_drop > 0:
            keep = (torch.rand(m.shape[0], 1, device=m.device) >= self.mem_drop)
            m = m * keep
        return m

    def geo_bias(self, fused, x0, y0, step, n_logits):
        """GeoMem's additive logit term, or None when it is not enabled."""
        if self.geo is None:
            return None
        return self.geo(fused, self.geo.rows(x0, y0, step), n_logits)

    def _add_geo(self, prior, fused, x0, y0, step, n_logits):
        """Fold the tile memory into the same additive slot as the retrieval
        prior, so every call site that already threads `prior` gets it -- the
        keyed retrieval branch was once dropped at inference exactly because a
        second optional term had a second path."""
        g = self.geo_bias(fused, x0, y0, step, n_logits)
        if g is None:
            return prior
        return g if prior is None else prior + g

    def policy_logits(self, fused, keys, prior=None, step=None):
        """The action distribution *is* the attention distribution over map tokens."""
        if self.map_tf is not None:
            keys = self.map_tf(keys, fused)
        if self.sink is not None:
            # appended after the map transformer: it is not a spatial token and
            # must not participate in neighbour attention over the grid
            keys = torch.cat(
                [keys, self.sink.expand(keys.shape[0], 1, -1).to(keys.dtype)], dim=1)
        q = self.q_proj(fused)
        logits = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_tok)
        if self.sink_ext is not None:
            if step is None:
                # Silently skipping this trained one network and scored
                # another; make the omission impossible rather than harmless.
                raise ValueError(
                    "policy_logits needs `step` when sink_k > 1: the extra "
                    "sink keys are indexed by step")
            st = step.reshape(-1).clamp(max=self.sink_steps - 1)
            ext = self.sink_ext[st]                       # (B, K-1, d_tok)
            el = torch.einsum("bkd,bd->bk", ext.to(q.dtype), q)
            el = el / math.sqrt(self.d_tok) + self.sink_ext_b[st].to(q.dtype)
            base = logits[:, -1:]
            lse = torch.logsumexp(torch.cat([base, el], dim=1), dim=1,
                                  keepdim=True)
            g = self.sink_ext_g[st].unsqueeze(1).to(q.dtype)
            merged = base + g * (lse - base)
            logits = torch.cat([logits[:, :-1], merged], dim=1)
        if prior is not None:
            logits = logits + prior.to(logits.dtype)
        return logits

    def retr_prior(self, nbrs, street, x0, y0, step, per_image, n_logits):
        """The retrieval bias for a batch of rows, keys included.

        nbrs is per *image* -- (nbr_x, nbr_y, nbr_sim) and optionally the
        neighbours' own embeddings -- while x0/y0/step are per *row*, because an
        image contributes several rows: one per zoom step when training, one per
        live beam when searching. per_image says how many, and the 4,608-d key
        projections run once per image before that expansion.

        Every caller must come through here. When beam.search had its own
        version it omitted a_pos and a_neg, which silently disables the learned
        positive key and the entire negative branch -- so pos/dual models were
        trained with them and evaluated without them, and nothing failed.
        """
        if self.retr is None or nbrs is None:
            return None
        # The learned retrieval keys live in the BANK's space. The conditioning
        # block has no counterpart in the bank -- that is what decoupling means
        # -- so it is dropped before the key projection. Doing it here rather
        # than at each call site is deliberate: this function's docstring
        # records that beam.search once had its own version and silently
        # omitted a_pos and a_neg.
        nb = NeighborBatch.of(nbrs)
        if self.street.d_cond:
            # BOTH sides, in one operation each. The neighbours come from the
            # same combined street file as the query, so they carry the
            # conditioning suffix too, while the key projections were built for
            # the retrieval width.
            street = street[..., :self.street.d_in]
            nb = nb.retrieval_only(self.street.d_in)
        K = nb.x.shape[1]
        rep = lambda t: (t.unsqueeze(1).expand(t.shape[0], per_image, K)
                         .reshape(-1, K))
        a_pos, a_neg = self.retr.weights(nb.sim, street, nb.emb)
        return self.retr(rep(nb.x), rep(nb.y), rep(nb.sim),
                         x0, y0, step, n_logits,
                         a_pos=rep(a_pos),
                         a_neg=None if a_neg is None else rep(a_neg))

    def policy_from(self, street, tokens, x0, y0, step, nbrs=None):
        """Policy logits for arbitrary (image, tile, step) rows -- used for the
        off-path negatives, which have no place in the teacher-forced prefix.

        The negatives are where the retrieval prior should matter most: its
        "outside" bucket answers the same question the sink does, so it is
        passed through here rather than only on the on-path rows.
        """
        f, k = self.fuse(street, tokens, x0, y0, step)
        n_logits = k.shape[1] + (1 if self.sink is not None else 0)
        prior = self.retr_prior(
            nbrs, street, x0.reshape(-1), y0.reshape(-1), step.reshape(-1),
            step.shape[1], n_logits)
        prior = self._add_geo(prior, f, x0.reshape(-1), y0.reshape(-1),
                              step.reshape(-1), n_logits)
        return self.policy_logits(f, k, prior, step)

    def click_uv(self, fused):
        return torch.sigmoid(self.click(fused))

    def forward(self, batch):
        B, S = batch["step"].shape
        f, k = self.fuse(batch["street"], batch["tokens"],
                         batch["x0"], batch["y0"], batch["step"])
        f = f.view(B, S, -1)
        k = k.view(B, S, k.shape[1], k.shape[2])
        n_policy = S - 1
        fp = f[:, :n_policy].reshape(B * n_policy, -1)
        kp = k[:, :n_policy].reshape(B * n_policy, k.shape[2], k.shape[3])
        prior = None
        if "nbr_x" in batch:
            nbrs = NeighborBatch(batch["nbr_x"], batch["nbr_y"],
                                 batch["nbr_sim"], batch.get("nbr_emb"))
            prior = self.retr_prior(
                nbrs, batch.get("street"),
                batch["x0"][:, :n_policy].reshape(-1),
                batch["y0"][:, :n_policy].reshape(-1),
                batch["step"][:, :n_policy].reshape(-1),
                n_policy,
                kp.shape[1] + (1 if self.sink is not None else 0))
        prior = self._add_geo(
            prior, fp,
            batch["x0"][:, :n_policy].reshape(-1),
            batch["y0"][:, :n_policy].reshape(-1),
            batch["step"][:, :n_policy].reshape(-1),
            kp.shape[1] + (1 if self.sink is not None else 0))
        logits = self.policy_logits(
            fp, kp, prior,
            batch["step"][:, :n_policy]).view(B, n_policy, -1)
        uv = self.click_uv(f[:, -1])
        return logits, uv


def param_report(model):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
    total = sum(groups.values())
    lines = ["{:<12} {:>10,}".format(k, v) for k, v in sorted(groups.items())]
    lines.append("{:<12} {:>10,}".format("TOTAL", total))
    return "\n".join(lines), total
