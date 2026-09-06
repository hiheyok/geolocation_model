#!/usr/bin/env bash
# Start the tile server. exfat cannot store the exec bit, so run: bash start.sh
set -e
export PATH="$HOME/.local/node/bin:$PATH"
cd /mnt/storage/mapping_service

export MBTILES_PATH="${MBTILES_PATH:-/mnt/storage/mapping_service/maptiler-osm-2020-02-10-v3.11-planet.mbtiles}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/storage/mapping_service/tiles}"
export PORT="${PORT:-3000}"

# The prebuilt maplibre-gl-native binary is GLX-only and needs an X display,
# so headless rendering goes through a virtual framebuffer (software llvmpipe).
exec xvfb-run -a node server.js
