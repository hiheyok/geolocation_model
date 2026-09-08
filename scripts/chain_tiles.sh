#!/usr/bin/env bash
# Tile the rest of the shipping corpus: bank_ext2, 3 and 4, 750,000 each.
#
# bank_ext is already tiled, so this takes the tiled bank from 1,150,180 rows
# to the full 3,400,180. Tiling is ALL-OR-NOTHING -- one cosine ranks blended
# rows against L0 rows and silently demotes the untiled ones -- so the tile
# cache is unusable with the shipping bank until all three finish.
#
# ~4 h each at the 53.5 img/s measured on the first pass, so ~12 h. Each stem
# is separate and tile_cache keeps a done-mask, so an interrupted run resumes
# rather than restarting.
set -u
cd "$(dirname "$0")/.."
export OSV_RELEASE=s10
L=runs/logs
mkdir -p "$L"

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$L/chain_tiles.log"; }

say "start; 3 extensions x 750,000 images"
for e in 2 3 4; do
  out="tile6_ext$e"
  say "tiling bank_ext$e -> $out"
  py scripts/tile_cache.py --parquet "bank_ext$e.parquet" --out "$out" \
      --n 750000 --grid 3x2 >> "$L/$out.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    say "bank_ext$e FAILED rc=$rc -- stopping so the next one does not queue behind a broken cache"
    exit 1
  fi
  say "bank_ext$e done"
done
say "all three tiled; the corpus is ready to pool and stack"
