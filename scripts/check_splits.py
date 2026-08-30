"""Verify split integrity and size the tile fetch before spending time on it."""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import tile_math as tm

ds = pq.read_table(config.DATASET_PARQUET)
tg = pq.read_table(config.TARGETS_PARQUET)

split = np.asarray(ds["split"].to_pylist(), dtype=object)
cell = np.asarray(ds["cell_z8"])
lat = np.asarray(ds["lat"], dtype=np.float64)
lon = np.asarray(ds["lon"], dtype=np.float64)
seq = np.asarray(ds["sequence"].to_pylist(), dtype=object)

print("=== cell disjointness ===")
sets = {s: set(cell[split == s].tolist()) for s in ("train", "val", "test")}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    ov = sets[a] & sets[b]
    print("  {:<5} n {:<5} overlap with {:<5} {}".format(a, len(sets[a]), b, len(ov)))

print("\n=== sequence disjointness ===")
sseq = {s: set(seq[split == s].tolist()) for s in ("train", "val", "test")}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    print("  {} / {} shared sequences: {}".format(a, b, len(sseq[a] & sseq[b])))

print("\n=== nearest train image to each test image ===")
def unit(la, lo):
    la, lo = np.radians(la), np.radians(lo)
    return np.stack([np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)], 1)

tr = unit(lat[split == "train"], lon[split == "train"])
te = unit(lat[split == "test"], lon[split == "test"])
rng = np.random.default_rng(0)
sample = rng.choice(len(te), size=min(1500, len(te)), replace=False)
best = np.empty(len(sample))
for i, si in enumerate(sample):
    dot = np.clip(tr @ te[si], -1.0, 1.0)
    best[i] = 6371.0088 * np.arccos(dot.max())
for q in (0, 1, 5, 25, 50):
    print("  p{:<3} {:8.2f} km".format(q, np.percentile(best, q)))
print("  fraction under 1 km: {:.3%}".format((best < 1.0).mean()))

print("\n=== step 0 class support (256-way over z4) ===")
step = np.asarray(tg["step"])
act = np.asarray(tg["target_action"])
a0 = act[step == 0]
occ, cnt = np.unique(a0, return_counts=True)
print("  occupied cells      {} of {}".format(len(occ), tm.actions()))
print("  top cell share      {:.1%}".format(cnt.max() / cnt.sum()))
print("  cells with <10 imgs {}".format(int((cnt < 10).sum())))

print("\n=== distinct tiles to fetch ===")
tz = np.asarray(tg["tile_z"]); tx = np.asarray(tg["tile_x"]); ty = np.asarray(tg["tile_y"])
total_sparse = 0
for z in sorted(set(tz.tolist())):
    m = tz == z
    keys = np.unique(tx[m].astype(np.int64) * (1 << 20) + ty[m].astype(np.int64))
    exh = (1 << z) * (1 << z)
    tag = "exhaustive {:,}".format(exh) if z <= 8 else "on true paths"
    print("  z{:<3} distinct {:>7,}   ({})".format(z, len(keys), tag))
    if z > 8:
        total_sparse += len(keys)
print("  z4+z8 exhaustive     {:,}".format(256 + 65536))
print("  sparse z12+z16       {:,}".format(total_sparse))
print("  TOTAL fetches        {:,}".format(256 + 65536 + total_sparse))
mb = (256 + 65536 + total_sparse) * 256 * 12 * 2 / 1e6
print("  token cache          {:.0f} MB fp16".format(mb))
