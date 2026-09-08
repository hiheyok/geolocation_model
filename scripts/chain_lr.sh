#!/usr/bin/env bash
# Is the sixth epoch bad, or just re-heated?
#
# Every ladder here is separate `--epochs 2` runs chained with `--init`, and
# `--init` restores WEIGHTS ONLY: Adam's moments restart and the LR schedule
# is a fresh cosine. train.py already compensates -- with `--init` the default
# lr drops to 1e-4 from 3e-4 and warmup to 100 steps, "because the previous
# run annealed to zero" -- but 1e-4 is still a re-heat of a model that just
# annealed, three times over.
#
# e6 losing to e4 is real and separated in the tiled arm: [+0.32, +1.62] pp
# for e4 (runs/BOOTSTRAP_tilefull.md). It is the fifth ladder in this project
# to peak before its last rung, which has always been read as "epochs do not
# substitute for data". That reading may be right, but it has never been
# separated from the schedule.
#
# So: the same last rung from the same e4 checkpoint, at 1e-4 (already run,
# `pyrL0L1-b340-e6`), 5e-5 and 3e-5. One knob.
#
#   a lower LR beats e4  -> the regression was the re-heat, and every ladder
#                           in this project has been paying for it
#   all of them tie e4   -> the model is done at 4 epochs; "stop at 4" stands,
#                           for a better reason than before
#
# ~1 h, two rungs, then a four-way bootstrap.
set -u
cd "$(dirname "$0")/.."
export OSV_RELEASE=s10
L=runs/logs
mkdir -p "$L"

ARCH="--pool attn --pool-q 4 --pos both --map-layers 1 --neg 4 --retr --retr-k 16 --retr-mode dual --d-key 128"
COMMON="--epochs 2 --batch 64 --limit 400000 --select hit --sel-n 2000 --val-n 5000 \
  --split-mode sequence --street-file pyr768_l0l1_b340.f16.npy \
  --knn-file knn_pyr768_l0l1_b340_sequence_k32_bank_ext70.npz \
  --seed 0 --neg-random $ARCH --retr-drop 0.7 --init pyrL0L1-b340-e4"

say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$L/chain_lr.log"; }

say "start: last rung re-run at lower LR, from pyrL0L1-b340-e4"
for lr in 5e-5 3e-5; do
  tag="pyrL0L1-b340-e6-lr${lr}"
  say "rung lr=$lr -> $tag"
  py src/train.py --tag "$tag" $COMMON --lr "$lr" >> "$L/$tag.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "$tag FAILED rc=$rc"; exit 1; fi
  say "$tag done"
done

say "bootstrap: e4 against the three sixth-epoch variants"
py scripts/bootstrap.py \
  --tags pyrL0L1-b340-e4,pyrL0L1-b340-e6,pyrL0L1-b340-e6-lr5e-5,pyrL0L1-b340-e6-lr3e-5 \
  --split test --n 5000 --beam 2 --score-steps 3 \
  --out runs/BOOTSTRAP_lr.md >> "$L/boot_lr.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "bootstrap FAILED rc=$rc"; exit 1; fi
say "done -- read runs/BOOTSTRAP_lr.md"
