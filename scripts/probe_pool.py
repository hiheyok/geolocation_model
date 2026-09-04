"""Is MapTokenizer's pooled vector spatially blind?

Positions are added *before* the mean, so without the LayerNorm the pool would
collapse exactly to proj(mean class histogram) + const.  LN keeps that from
being literally true.  This measures how much signal actually survives.

The number that matters is not the raw shuffle delta but its ratio to the
between-tile spread: if permuting a tile's own patches moves `pooled` far less
than swapping in a different tile does, the pooled vector is carrying "how
much" and not "where".
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from evaluate import build_from_ck

ck_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/c3mt1.pt")
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
# This used to rebuild the architecture here, inferring d_street and
# map_layers from the weights and defaulting everything else. `pos` was the
# dangerous one: pos="both" adds rotary positions, which carry no parameters,
# so the strict load succeeded and this probe measured a model with the
# rotary half missing -- while probing exactly the spatial structure rotary
# supplies. build_from_ck reads every field the checkpoint records.
model, d_street = build_from_ck(ck)
print("{}  d_street={}  pool={}  pos={}  map_layers={}".format(
    ck_path.name, d_street, ck.get("pool", "mean"), ck.get("pos", "learned"),
    ck.get("map_layers", 0)))

tokens = np.load(config.map_files()[0], mmap_mode="r")
rng = np.random.default_rng(0)
pick = rng.choice(len(tokens), 512, replace=False)
x = torch.from_numpy(np.asarray(tokens[np.sort(pick)], dtype=np.float32))  # (N,A,C)
N, A, C = x.shape

with torch.no_grad():
    _, pooled = model.map(x)
    # permute the patches within each tile: same histogram, different layout
    perm = torch.stack([torch.randperm(A) for _ in range(N)])
    xs = torch.gather(x, 1, perm.unsqueeze(-1).expand(N, A, C))
    _, pooled_s = model.map(xs)

d_shuf = (pooled_s - pooled).norm(dim=1)
# between-tile spread: distance to a different random tile's pooled vector
other = torch.randperm(N)
d_tile = (pooled[other] - pooled).norm(dim=1)

print("\n||pooled||                {:.3f}".format(pooled.norm(dim=1).mean()))
print("||shuffle delta||          {:.3f}".format(d_shuf.mean()))
print("||different-tile delta||   {:.3f}".format(d_tile.mean()))
print("\nlayout signal / tile signal = {:.1%}".format(
    (d_shuf.mean() / d_tile.mean()).item()))

# how much of pooled is predictable from the histogram alone
hist = x.mean(dim=1)                              # (N,C) the bare class fractions
Xh = torch.cat([hist, torch.ones(N, 1)], dim=1)
W = torch.linalg.lstsq(Xh, pooled).solution
resid = (pooled - Xh @ W).norm() / (pooled - pooled.mean(0)).norm()
print("residual after linear fit from the 12-d histogram: {:.1%}".format(resid.item()))
