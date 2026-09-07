"""Content-keyed memory: what the query's visual neighbours say about this tile.

The tile-ID table failed because its key does not exist in a region that never
appeared in training, and because the coordinates it stored were already an
input.  Retrieval fixes both: the key is the street embedding, which exists
everywhere, and the value is empirical -- where images that *look like this one*
actually are.

The prior is deliberately shaped like the decision.  At a tile T the policy
picks one of 256 children, so the neighbours are reduced to exactly that: a
distribution over the 256 child cells, plus one "outside" bucket for neighbours
that do not fall inside T at all.  That last bucket lines up with the sink
class, which asks the same question -- is the answer even in here?

Four modes, each a strict superset of the last, and every gate initialised to
zero so all of them start bit-identical to a model with no retrieval at all.
The bias is always `gate * log(prior + eps)`; what changes is what `gate` is a
function of, and how `prior` is built.

  scalar   `gate` is a rank-0 tensor -- literally one learned number, shared by
           all four steps and shared between the 256 cell logits and the sink.
           That shape is what the name refers to.
  cond     `gate` becomes *conditional*: a per-step base for the cells and the
           sink separately, plus a modulation from six features describing how
           trustworthy this particular retrieval looks (see `quality`).
           Motivated by measurement, not taste: the concentration of the prior
           predicts its own correctness at AUC 0.811 (s0) and 0.728 (s1), and a
           single number cannot use a signal that varies per image.
  pos      adds a learned *positive key*: re-weight the same neighbours by
           `w_pos * cos(q_pos(street), k_pos(neighbour))` before the softmax,
           so the model can depart from raw cosine if it pays.  w_pos starts at
           zero, so it begins exactly at frozen cosine.
  dual     adds a second, separately keyed retrieval trained to find the
           *misleading* neighbours -- visually close, geographically wrong --
           and subtracts its prior.  The two key spaces let one branch
           specialise in proposal and the other in rejection.

The names are on two axes -- `scalar`/`cond` describe the gate, `pos`/`dual`
describe how many keyed branches there are -- which hides the fact that this is
one ladder.  Renaming is worth doing, but `retr_mode` is recorded in every
checkpoint and read back by `evaluate.load_model`, so it needs an alias table
rather than a rename in place.

Everything else is integer tile arithmetic on z16 addresses, so it is exact,
cheap, and works identically in teacher-forced training and in beam search.
"""

import torch
import torch.nn as nn


def child_prior(nbr_x16, nbr_y16, alpha, x0, y0, step, g=16, max_z=12):
    """Distribution over child cells implied by retrieved neighbours.

    nbr_x16/nbr_y16 (N, K) int64   z16 tile of each neighbour
    alpha           (N, K) float   retrieval weights, already normalised
    x0/y0           (N,)   float   normalised Mercator corner of the current tile
    step            (N,)   long    zoom step, 0..3

    Returns (N, g*g + 1): mass on each child cell, then mass outside the tile.
    """
    n_act = g * g
    z = (4 * step).clamp(max=max_z)
    scale = torch.pow(torch.full_like(x0, 2.0), z.to(x0.dtype))
    x = torch.round(x0 * scale).long()
    y = torch.round(y0 * scale).long()

    # Both tests below are truncation of a z16 address, which is why no tile
    # coordinates need to be carried around: a z16 tile lies inside the tile at
    # zoom z exactly when its address right-shifted by (16 - z) equals that
    # tile's address, because one zoom level is one bit of coordinate.
    #
    # Worked example, a neighbour at z16 (32751, 21793) against the current tile
    # at step 2, i.e. z8:
    #     shift = 16 - 8 = 8      32751 >> 8 = 127, 21793 >> 8 = 85
    #     so it is inside iff the current z8 tile is (127, 85)
    #     csh   = 16 - 8 - 4 = 4  (32751 >> 4) & 15 = 15, (21793 >> 4) & 15 = 1
    #     so it votes for child (row 1, col 15) = action 31
    shift = (16 - z).unsqueeze(1)
    inside = ((nbr_x16 >> shift) == x.unsqueeze(1)) & \
             ((nbr_y16 >> shift) == y.unsqueeze(1))

    # which of the 16x16 children it lands in: drop the (16 - z - 4) bits
    # below the child grid, then keep the low 4 bits of what remains.
    # clamp(min=0) matters at step 3, where z is already max_z and there is
    # no finer level left to shift away.
    csh = (16 - z - 4).clamp(min=0).unsqueeze(1)
    cx = (nbr_x16 >> csh) & (g - 1)
    cy = (nbr_y16 >> csh) & (g - 1)
    child = (cy * g + cx).clamp(0, n_act - 1)

    w = alpha * inside.to(alpha.dtype)
    cells = torch.zeros(nbr_x16.shape[0], n_act, dtype=alpha.dtype,
                        device=alpha.device)
    cells.scatter_add_(1, child, w)
    outside = (alpha * (~inside).to(alpha.dtype)).sum(1, keepdim=True)
    return torch.cat([cells, outside], dim=1)


def quality(prior, alpha, sim):
    """How trustworthy this retrieval looks, from the retrieval alone.

    Measured on val: the top-cell mass predicts whether the prior's argmax is
    correct at AUC 0.811 at step 0 and 0.728 at step 1 -- the two steps that
    set the median error.  These are the features that carry it.
    """
    cells, outside = prior[:, :-1], prior[:, -1:]
    inside = cells.sum(1, keepdim=True)
    norm = cells / inside.clamp_min(1e-9)
    cell_ent = -(norm * norm.clamp_min(1e-9).log()).sum(1, keepdim=True)
    a_ent = -(alpha * alpha.clamp_min(1e-9).log()).sum(1, keepdim=True)
    return torch.cat([cells.max(1, keepdim=True).values, inside, outside,
                      cell_ent, a_ent, sim[:, :1]], dim=1)


N_QUALITY = 6


class RetrievalPrior(nn.Module):
    def __init__(self, g=16, tau=0.07, eps=0.01, mode="scalar",
                 d_street=0, d_key=128, n_steps=4, d_hidden=64, nbr_drop=0.0):
        super().__init__()
        self.g, self.mode, self.n_steps = g, mode, n_steps
        # Thin the neighbour set at random during training. The model trained
        # against the 3.40M bank beats the 2.65M one by +2.8 pp on the
        # benchmark and loses 0.76-1.20 pp on held-out KartaView *whichever
        # bank it is given at inference* -- the damage follows the model, not
        # the bank, so it is a training-time dependency on dense retrieval.
        # Off-domain a query gets fewer and worse neighbours (top-1 0.71 vs
        # 0.90); a policy that only ever saw dense coverage over-trusts them.
        self.nbr_drop = float(nbr_drop)
        self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())
        self.log_eps = nn.Parameter(torch.tensor(float(eps)).log())

        if mode == "scalar":
            self.gate = nn.Parameter(torch.zeros(()))
            return

        # per-step base gates for the cell terms and the sink term separately.
        # One scalar had to serve both jobs; the sink bucket answers a different
        # question from the 256 cells and deserves its own weight.
        self.g_cell = nn.Parameter(torch.zeros(n_steps))
        self.g_sink = nn.Parameter(torch.zeros(n_steps))
        # modulation by retrieval quality; last layer zero so it starts inert
        self.cond = nn.Sequential(
            nn.Linear(N_QUALITY + n_steps, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2))
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)

        if mode in ("pos", "dual"):
            # A learned re-weighting of the *same* retrieved neighbours, in its
            # own key space.  w_pos is zero-initialised, so the positive branch
            # starts exactly at frozen cosine and can only depart if it pays.
            self.q_pos = nn.Linear(d_street, d_key, bias=False)
            self.k_pos = nn.Linear(d_street, d_key, bias=False)
            self.w_pos = nn.Parameter(torch.zeros(()))
        if mode == "dual":
            # The negative branch has its own key space entirely.  Its job is
            # the inference-time failure: a neighbour that is visually close and
            # geographically wrong, which is exactly what drags a beam off.
            self.q_neg = nn.Linear(d_street, d_key, bias=False)
            self.k_neg = nn.Linear(d_street, d_key, bias=False)
            self.log_tau_neg = nn.Parameter(torch.tensor(float(tau)).log())
            self.g_neg = nn.Parameter(torch.zeros(n_steps))

    @staticmethod
    def _cos(a, b):
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (a.unsqueeze(1) * b).sum(-1)

    def weights(self, sim, q_emb=None, nbr_emb=None):
        """Neighbour weights for the positive and (optionally) negative branch.

        Inputs are per *image*; the caller expands to rows afterwards, so the
        4,608-d projections run once per image rather than once per step.
        """
        tau = self.log_tau.exp().clamp_min(1e-3)
        score = sim.float() / tau
        if self.mode in ("pos", "dual") and nbr_emb is not None:
            score = score + self.w_pos * self._cos(self.q_pos(q_emb),
                                                   self.k_pos(nbr_emb))
        keep = None
        if self.training and self.nbr_drop > 0:
            keep = torch.rand_like(score) >= self.nbr_drop
            # A row with every neighbour dropped softmaxes -inf to NaN, so
            # rescue one at random rather than always keeping the top-1 --
            # keeping the best one is not the condition being simulated.
            #
            # The three host reads below -- `bool`, `int`, and the `nonzero`
            # that sizes its own output -- each drain the CUDA queue, and
            # REVIEW9 #6 proposes a tensor-only rescue to remove them.
            #
            # This tensor is (B, K) = (64, 16): `retr_prior` calls `weights`
            # on the per-IMAGE similarity and applies `rep()` afterwards, as
            # the docstring above says. A first measurement used (320, 16),
            # the post-expansion row count, and was wrong by more than its own
            # conclusion. Two things follow, pulling opposite ways: there are
            # TWO calls per batch, not one (`model(b)` and `policy_from` for
            # the off-path negatives), so 12,500 an epoch rather than 6,250 --
            # and a row is empty with probability 0.7^16, so the rescue branch
            # fires 19.2% of calls at 64 rows against 65.4% at 320. On four
            # calls in five the only host read is the unconditional
            # `bool(empty.any())`.
            #
            # Re-measured on an idle card at (64, 16), sweeping the queued
            # work a sync has to drain, because the first setup fixed that
            # at one unstated size:
            #
            #   filler     512^2   1024^2   2048^2   4096^2
            #   delta/call 0.022   0.079    0.073    0.049  ms
            #   % of epoch 0.04    0.13     0.12     0.08
            #
            # So 0.04-0.13%, not the 0.29% first recorded. Note it does NOT
            # grow with queue depth: at 4096^2 each matmul is 11 ms, so a sync
            # could drain 11 ms, and costs 0.049. In a throughput-bound loop a
            # host read waits for work that had to finish anyway -- and
            # training here is throughput-bound, 94-96% GPU at a
            # memory-bandwidth limit. That is the reason the saving stays
            # small, and it is more durable than any single number.
            #
            # It is not taken, and the timing was never the deciding half. The
            # tensor-only form must draw one index per ROW every step where
            # this draws `empty.sum()` only when a row is empty. That changes
            # how much of the generator each step consumes, so every seeded run
            # using --retr-drop -- the shipping arm, at 0.7 -- would stop
            # reproducing. Revisit if the rescue ever moves off the critical
            # path or reproducibility of these runs stops mattering.
            empty = ~keep.any(dim=1)
            if bool(empty.any()):
                j = torch.randint(0, score.shape[1], (int(empty.sum()),),
                                  device=score.device)
                keep[empty.nonzero(as_tuple=True)[0], j] = True
            score = score.masked_fill(~keep, float("-inf"))
        a_pos = torch.softmax(score, dim=1)
        a_neg = None
        if self.mode == "dual" and nbr_emb is not None:
            ns = (self._cos(self.q_neg(q_emb), self.k_neg(nbr_emb))
                  / self.log_tau_neg.exp().clamp_min(1e-3))
            if keep is not None:            # same neighbours hidden from both
                ns = ns.masked_fill(~keep, float("-inf"))
            a_neg = torch.softmax(ns, dim=1)
        return a_pos, a_neg

    def forward(self, nbr_x16, nbr_y16, sim, x0, y0, step, n_logits,
                a_pos=None, a_neg=None):
        """Additive bias on the policy logits.  All gates start at zero."""
        if a_pos is None:
            a_pos, a_neg = self.weights(sim)
        eps = self.log_eps.exp()
        p = child_prior(nbr_x16, nbr_y16, a_pos, x0, y0, step, self.g)
        lp = torch.log(p + eps)

        if self.mode == "scalar":
            bias = self.gate * lp
            return bias[:, :-1] if n_logits == p.shape[1] - 1 else bias

        st = step.clamp(max=self.n_steps - 1)
        oh = torch.nn.functional.one_hot(st, self.n_steps).to(lp.dtype)
        delta = self.cond(torch.cat([quality(p, a_pos, sim), oh], dim=1))
        g_cell = (self.g_cell[st] + delta[:, 0]).unsqueeze(1)
        g_sink = (self.g_sink[st] + delta[:, 1]).unsqueeze(1)

        bias = torch.cat([g_cell * lp[:, :-1], g_sink * lp[:, -1:]], dim=1)
        if a_neg is not None:
            n = child_prior(nbr_x16, nbr_y16, a_neg, x0, y0, step, self.g)
            bias = bias - self.g_neg[st].unsqueeze(1) * torch.log(n + eps)
        return bias[:, :-1] if n_logits == p.shape[1] - 1 else bias
