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


# ---------------------------------------------------------------- artefacts --
#
# Names of the files inside a cache directory, and the conventions for deriving
# one name from another.  These live here because more than one module has to
# agree on them and a disagreement is silent: beam.py reading a token cache
# under a name fetch_tiles.py did not write would simply miss every lookup and
# refetch the whole split over HTTP.

MAP_TOKENS = "tokens.f16.npy"       # [n_tiles, g*g, n_classes] float16
MAP_INDEX = "index.parquet"         # (z, x, y) -> row
MAP_DONE = "done.u8.npy"            # which rows have been fetched; makes it resumable

STREET_DEFAULT = "embeddings.f16.npy"
STREET_IDS = "image_ids.i64.npy"

KNN_K = 32                          # neighbours stored per query


def map_files(cache=None):
    """(tokens, index, done) paths inside a map cache directory.

    `cache` overrides the release's own, which is how fetch_tiles --seed-from
    reads an older release's tiles.
    """
    d = Path(cache) if cache else MAP_CACHE
    return d / MAP_TOKENS, d / MAP_INDEX, d / MAP_DONE


def bank_meta(stem):
    """Metadata for a bank extension: z16 addresses and sequences of the
    bank-only shards, written by build_bank_ext.py and read by build_knn.py and
    dataset.py."""
    return STREET_CACHE / (stem + "_meta.npz")


def knn_name(street_file, mode, k=KNN_K, bank_limit=0, ext=None):
    """Filename of a kNN cache.

    A cache is identified by everything that changes its contents: the
    embeddings it was built from, the split whose train side is the bank, how
    many neighbours it stores, and whether the bank was restricted or extended.
    The stored k is the *cache* size; --retr-k selects a prefix of it at
    training time and does not change the file.
    """
    stem = street_file.replace(".f16.npy", "")
    tail = "" if not bank_limit else "_bank{}k".format(bank_limit // 1000)
    tail += "" if not ext else "_" + ext
    return "knn_{}_{}_k{}{}.npz".format(stem, mode, k, tail)


TILE_SERVER = os.environ.get("TILE_SERVER", "http://192.168.50.1:3000")

# Splitting: cells are held out whole, and a sequence never spans splits.
SPLIT_CELL_ZOOM = 8
SPLIT_FRACTIONS = (0.80, 0.10, 0.10)
SPLIT_SEED = 17

for _d in (PROCESSED, STREET_CACHE, MAP_CACHE, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)
