"""Paths and knobs.  Everything grid-related lives in tile_math."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OSV_ROOT = Path(os.environ.get("OSV_ROOT", "E:/data/osv5m/datasets/osv5m"))
TRAIN_CSV = OSV_ROOT / "train.csv"
TRAIN_ZIPS = OSV_ROOT / "images" / "train"
SHARD = os.environ.get("OSV_SHARD", "00")

PROCESSED = ROOT / "data" / "processed"
CACHE = ROOT / "cache"
STREET_CACHE = CACHE / "street"
MAP_CACHE = CACHE / "map"
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
