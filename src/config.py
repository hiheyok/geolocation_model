"""Paths and knobs.  Everything grid-related lives in tile_math."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OSV_ROOT = Path(os.environ.get("OSV_ROOT", "E:/data/osv5m/datasets/osv5m"))
TRAIN_CSV = OSV_ROOT / "train.csv"
TRAIN_ZIPS = OSV_ROOT / "images" / "train"
SHARD = os.environ.get("OSV_SHARD", "00")

# Dataset release -- the set of shards everything downstream was derived from.
#
# The parquet, the street embeddings, the token cache and the kNN bank are all
# indexed by row order in dataset.parquet, so adding shards invalidates every
# one of them simultaneously, and the split hash with them.  Naming the release
# keeps each generation intact side by side: s01 is the 50k benchmark that every
# result so far was measured on and stays reproducible after a larger release
# exists.  Checkpoints record the release they trained on, so a metric can never
# be silently compared across two different datasets.
#
#   s01  shard 00            50,000 images
#   s10  shards 00-09       500,000 images
RELEASE = os.environ.get("OSV_RELEASE", "s01")

PROCESSED = ROOT / "data" / "processed" / RELEASE
CACHE = ROOT / "cache"
STREET_CACHE = CACHE / "street" / RELEASE
MAP_CACHE = CACHE / "map" / RELEASE
CHECKPOINTS = ROOT / "checkpoints"

DATASET_PARQUET = PROCESSED / "dataset.parquet"
TARGETS_PARQUET = PROCESSED / "targets.parquet"

TILE_SERVER = os.environ.get("TILE_SERVER", "http://192.168.50.1:3000")

# Splitting: cells are held out whole, and a sequence never spans splits.
SPLIT_CELL_ZOOM = 8
SPLIT_FRACTIONS = (0.80, 0.10, 0.10)
SPLIT_SEED = 17

for _d in (PROCESSED, STREET_CACHE, MAP_CACHE, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)
