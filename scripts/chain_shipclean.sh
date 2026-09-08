#!/usr/bin/env bash
# The missing comparator: the SHIPPING street file, trained clean.
#
# `d768-b350-e6-drop70` and the `wd29-*` arms were all trained before
# 2026-09-04 04:05, when the file at
# `knn_pca768_bank70_sequence_k32_bank_ext70.npz` was the same-sequence-leaky
# cache. A checkpoint records that path, not its content (AGENTS.md §7), so
# nothing downstream can see it. `pyrL0*` and `pyrL0L1*` were trained on
# 2026-09-07 against the rebuilt one.
#
# Those two street files are RETRIEVAL-IDENTICAL -- every threshold spans zero
# at top-1 and any-of-32 -- yet the leak-trained arms beat the clean-trained
# `pyrL0` by 2.08 to 3.98 pp. Weight-decay grouping is null between them
# ([-0.40, +0.68]) and the split difference is null ([-1.02, +0.92]), so the
# training-time leak is what is left.
#
# This settles it. Same recipe as `tilefull_ladder`'s rung, same seed, same
# limit, same architecture, same --retr-drop; only the street file and its
# neighbour table change, and both are the shipping ones.
#
#   lands near pyrL0-b340-e4 (54.5%)  -> the 57.5% is a leak-training
#                                        artifact, and the honest baseline
#                                        tiles must beat is 54.5%
#   lands near 57.5%                  -> something else explains the gap and
#                                        tiles really are at parity
#
# ~1 h, two rungs.
set -u
cd "$(dirname "$0")/.."
export OSV_RELEASE=s10
L=runs/logs
mkdir -p "$L"

ARCH="--pool attn --pool-q 4 --pos both --map-layers 1 --neg 4 --retr --retr-k 16 --retr-mode dual --d-key 128"
COMMON="--epochs 2 --batch 64 --limit 400000 --select hit --sel-n 2000 --val-n 5000 \
  --split-mode sequence --street-file pca768_bank70.f16.npy \
  --knn-file knn_pca768_bank70_sequence_k32_bank_ext70.npz \
  --seed 0 --neg-random $ARCH --retr-drop 0.7"

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$L/chain_shipclean.log"; }

say "start: shipping street file, clean cache, pyrL0 recipe"
py src/train.py --tag shipclean-e2 $COMMON >> "$L/shipclean-e2.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "shipclean-e2 FAILED rc=$rc"; exit 1; fi
say "shipclean-e2 done"

py src/train.py --tag shipclean-e4 $COMMON --init shipclean-e2 >> "$L/shipclean-e4.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "shipclean-e4 FAILED rc=$rc"; exit 1; fi
say "shipclean-e4 done -- bootstrap it against pyrL0-b340-e4 and wd29-fix-e4"
