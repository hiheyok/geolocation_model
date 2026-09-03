"""Does the rotary encoding actually give relative position?

An architecture review claimed it does not: standard RoPE rotates queries and
keys *inside* attention, after the Q/K projections. This model rotates the map
token embeddings in `MapTokenizer.forward` and hands the rotated tensors to
`MapBlock`, which applies its own LayerNorm and then arbitrary W_q/W_k. Rotation
does not commute with either, so the property `<q_i, k_j> = f(i - j)` need not
survive.

That is checkable rather than arguable, so this checks it. Feed identical
content at every one of the 256 positions, so the ONLY thing distinguishing
position i from position j is the rotation. Then ask whether the pre-softmax
attention logit between positions depends solely on their offset.

Two arrangements are measured against each other:

  as built   rotate the token embeddings, then LayerNorm, then W_q/W_k
  textbook   LayerNorm, then W_q/W_k, then rotate the projected q and k

With identical content the textbook arrangement must give a logit that is an
exact function of the offset. Any spread within an offset group is the
relative-position property failing.

    OSV_RELEASE=s10 python scripts/rope_probe.py --tag d768-b350-e6-drop70
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import tile_math as tm  # noqa: E402
from encoders import apply_rope, rope2d_tables  # noqa: E402
from evaluate import load_model  # noqa: E402


def offsets(g):
    """(dx, dy) between every ordered pair of grid positions."""
    a = np.arange(g * g)
    col, row = a % g, a // g
    dx = col[:, None] - col[None, :]
    dy = row[:, None] - row[None, :]
    return dx, dy


def spread_by_offset(logits, dx, dy):
    """Std of the logit within each (dx, dy) group, averaged, and the overall std.

    A ratio near zero means the logit is a function of the offset alone, which
    is what relative position means. Near one means position is carrying
    something else entirely.
    """
    key = (dx + dx.max()) * (2 * dy.max() + 1) + (dy + dy.max())
    k = key.ravel()
    v = logits.ravel()
    order = np.argsort(k, kind="stable")
    k, v = k[order], v[order]
    bounds = np.flatnonzero(np.diff(k)) + 1
    within = [g.std() for g in np.split(v, bounds) if len(g) > 1]
    return float(np.mean(within)), float(v.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d768-b350-e6-drop70")
    a = ap.parse_args()

    model, ck, _ = load_model(a.tag, "cpu")
    if ck.get("map_layers", 0) < 1:
        raise SystemExit("{} has no map transformer; nothing to probe"
                         .format(a.tag))
    if ck.get("pos", "learned") not in ("rope", "both"):
        raise SystemExit("{} was trained with pos={!r}; no rotary half"
                         .format(a.tag, ck.get("pos")))

    g, A = tm.G, tm.actions()
    tokr = model.map
    block = model.map_tf.blocks[0]
    d_tok = tokr.proj.out_features
    cos, sin = rope2d_tables(A, d_tok)

    # Identical content at every position: rotation is the only difference.
    torch.manual_seed(0)
    base = torch.randn(1, d_tok)
    h = base.expand(A, d_tok).contiguous()
    k_norm = tokr.norm(h)

    attn = block.attn
    W = attn.in_proj_weight
    b = attn.in_proj_bias
    Wq, Wk = W[:d_tok], W[d_tok:2 * d_tok]
    bq, bk = b[:d_tok], b[d_tok:2 * d_tok]
    heads = attn.num_heads
    dh = d_tok // heads

    def logits_of(q, k):
        """Head-0 pre-softmax logits, the quantity RoPE is supposed to shape."""
        q = q[:, :dh]
        k = k[:, :dh]
        return (q @ k.T / dh ** 0.5).detach().numpy()

    with torch.no_grad():
        # as built: rotate the embeddings, then the block's own norm, then W
        rot = apply_rope(k_norm.unsqueeze(0), cos, sin).squeeze(0)
        x = block.n1(rot)
        built = logits_of(x @ Wq.T + bq, x @ Wk.T + bk)

        # textbook: norm and project first, rotate the projections
        y = block.n1(k_norm)
        q_t = apply_rope((y @ Wq.T + bq).unsqueeze(0), cos, sin).squeeze(0)
        k_t = apply_rope((y @ Wk.T + bk).unsqueeze(0), cos, sin).squeeze(0)
        textbook = logits_of(q_t, k_t)

    dx, dy = offsets(g)
    print("\n{}   map_layers={}  pos={!r}  d_tok={}"
          .format(a.tag, ck.get("map_layers"), ck.get("pos"), d_tok))
    print("identical content at all {} positions; only the rotation differs\n"
          .format(A))
    print("  {:<38} {:>12} {:>12} {:>10}"
          .format("arrangement", "within-offset", "overall", "ratio"))
    for name, lg in (("as built (rotate embeddings, then W)", built),
                     ("textbook (rotate q and k after W)", textbook)):
        w, o = spread_by_offset(lg, dx, dy)
        print("  {:<38} {:>12.4f} {:>12.4f} {:>10.3f}"
              .format(name, w, o, w / o if o else float("nan")))

    print("\nA ratio near 0 means the attention logit is a function of the "
          "offset alone,\nwhich is what relative position means. Near 1 means "
          "it is not.")


if __name__ == "__main__":
    main()
