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

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encoders import MapTokenizer, MapTransformer, StateEncoder, StreetProj
from retrieval import RetrievalPrior


class GeoAgent(nn.Module):
    def __init__(self, d_street=768, d=512, d_tok=256, n_classes=12,
                 n_actions=256, n_steps=5, dropout=0.1, map_layers=0,
                 pool="mean", n_pool_q=4, pos="learned", sink=False,
                 map_loop=False, mem="none", d_mem=64, mem_drop=0.0,
                 retr=False, retr_tau=0.07, retr_mode="scalar", d_key=128):
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

        self.street = StreetProj(d_street, d)
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
                                    d_key=d_key, n_steps=n_steps - 1)
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
        parts = [self.street(street), pooled, self.state(x0, y0, step)]
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

    def policy_logits(self, fused, keys, prior=None):
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
        if prior is not None:
            logits = logits + prior.to(logits.dtype)
        return logits

    def policy_from(self, street, tokens, x0, y0, step, nbrs=None):
        """Policy logits for arbitrary (image, tile, step) rows -- used for the
        off-path negatives, which have no place in the teacher-forced prefix.

        The negatives are where the retrieval prior should matter most: its
        "outside" bucket answers the same question the sink does, so it is
        passed through here rather than only on the on-path rows.
        """
        f, k = self.fuse(street, tokens, x0, y0, step)
        prior = None
        if self.retr is not None and nbrs is not None:
            B, S = step.shape
            K = nbrs[0].shape[1]
            rep = lambda t: t.unsqueeze(1).expand(B, S, K).reshape(-1, K)
            a_pos, a_neg = self.retr.weights(nbrs[2], street, nbrs[3]
                                             if len(nbrs) > 3 else None)
            prior = self.retr(rep(nbrs[0]), rep(nbrs[1]), rep(nbrs[2]),
                              x0.reshape(-1), y0.reshape(-1), step.reshape(-1),
                              k.shape[1] + (1 if self.sink is not None else 0),
                              a_pos=rep(a_pos),
                              a_neg=None if a_neg is None else rep(a_neg))
        return self.policy_logits(f, k, prior)

    def retrieval_bias(self, batch_or_none, x0, y0, step, n_logits):
        """Additive logit bias from the query's visual neighbours, or None."""
        if self.retr is None or batch_or_none is None:
            return None
        nx, ny, sim = batch_or_none
        return self.retr(nx, ny, sim, x0, y0, step, n_logits)

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
        if self.retr is not None and "nbr_x" in batch:
            K = batch["nbr_x"].shape[1]
            # the 4,608-d key projections run once per image, then expand
            a_pos, a_neg = self.retr.weights(batch["nbr_sim"],
                                             batch.get("street"),
                                             batch.get("nbr_emb"))
            rep = lambda t: t.unsqueeze(1).expand(B, n_policy, K).reshape(-1, K)
            prior = self.retr(rep(batch["nbr_x"]), rep(batch["nbr_y"]),
                              rep(batch["nbr_sim"]),
                              batch["x0"][:, :n_policy].reshape(-1),
                              batch["y0"][:, :n_policy].reshape(-1),
                              batch["step"][:, :n_policy].reshape(-1),
                              kp.shape[1] + (1 if self.sink is not None else 0),
                              a_pos=rep(a_pos),
                              a_neg=None if a_neg is None else rep(a_neg))
        logits = self.policy_logits(fp, kp, prior).view(B, n_policy, -1)
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
