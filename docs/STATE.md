# Live state — written 2026-09-01 03:15, before a context compaction

Read this first after a compaction. `docs/NEXT.md` is the forward plan;
this file is *what is running right now* and *what came out of the last
session*. Durable findings also live in the memory directory and in the git
log, which is unusually detailed and worth `git log --oneline -25`.

## Running unattended right now

`scripts/tonight.py 8.0`, started 03:11, **deadline 11:11**. Output in
`runs/tonight.out`, per-stage logs in `runs/logs/<stage>.log`, stage markers in
`runs/marks/`. It is resumable: re-running skips completed stages.

| block | stages | needs tile server |
|---|---|---|
| fusion head | `fh_combo`, `fh_pos1`, `fh_tau`, `fh_big` | no |
| pooled vector on cell8 | `pc8_knn`, `pc8_a/b/c`, `pc8_eval` | **yes** |
| corpus extension | `ext2_meta`, `ext2_dino`, `ext2_siglip`, `ext2_join`, `ext2_pool` | no |

**What to check when it finishes**

* `runs/logs/fh_*.log` — does any objective variant make the head beat the mean
  pool at 25 km, rather than only at 750 km and beyond?
* `runs/BOOTSTRAP_pool_cell8.md` — does the pooled 1536-d vector hold on the
  geographic holdout as it did on `sequence`? Baseline to beat is
  `s10_cell8_bank25_lr1e4_c` at 247.5 km / 7.66%.
* `cache/street/s10/bank_ext2_pool.f16.npy` — if it exists, 750k more bank
  images are embedded and pooled. Next steps then are `stack_bank.py` to join
  train + ext + ext2, `build_knn.py`, and one training arm. That is the +20.4 pp
  axis.

Nothing in block 3 touches `dataset.parquet` or the split hash, so the benchmark
is intact whatever happens.

## What ships

`s10_bal_bank25_c6` — **7.7 km median, 386.9 mean, 64.2% within 25 km** on the
`sequence` test split at n=5,000; **247.5 km / 7.66%** on `cell8`. 9,215,444
trainable parameters, 178,609,152 frozen. End-to-end diagram in the roadmap
artifact §9.

## Results from the last session, newest first

### Fusion head over 9 regions x 2 encoders (`scripts/fuse_head.py`)

Residual on the level-weighted mean pool, last layer zero-init, so it starts
*at* the baseline. Trained alone it trades fine accuracy for coarse:

| vs `L0+L1` mean pool | value |
|---|---|
| `<25 km` | **-4.07 [-5.7, -2.4]** |
| `<2500 km` | **+3.93 [+2.4, +5.5]** |

The cause is the objective, not the architecture: positives within 5 km but
in-batch negatives drawn globally, so the task is "same continent?". Restricting
negatives to one z6 bucket made it too hard instead — loss 5.08 against a chance
level of ln(256) = 5.55, i.e. it never learned. Both failure modes recorded.

**The combination works**, and survives an equal-bytes control (PCA back to
1536-d). After *one* epoch, against the mean pool: +3.80 pp [+0.4, +7.0] at
200 km, +6.00 [+2.4, +9.6] at 2500 km. Same pattern as crops+tiles — two
representations that fail differently beat either alone.

### Higher resolution (`scripts/kartaview_harvest.py`, `res_probe.py`)

5,426 US KartaView images at a median 5.3 MP against OSV-5M's 0.35, in
`E:/data/kartaview_us`. KartaView needs **no token**, unlike Mapillary, and is
one of the two sources OSV-5M was built from — so it is the same imagery at
native resolution, not a new distribution.

* **Source resolution alone changes nothing.** `osv_sim` and `crop3_224` agree
  to +0.00 pp; `preprocess` resizes to 224 either way.
* **Encoder input 448 instead of 224 helps**: +2.08 pp [+0.5, +3.7] at `<1 km`
  for DINOv2, +3.50 [+1.0, +6.1] at `<750 km` for SigLIP. Unreachable on OSV-5M,
  whose short side is already under 448.
* **Tiling still loses** with a 15x sharper source. Sharpness was never the
  problem; semantic scope was.
* **Crops + tiles together win**: +1.50 pp [+0.3, +2.8] at 25 km on high-res,
  +1.30 pp [+0.3, +2.3] at 200 km on OSV-5M at 117k bank, at equal bank bytes.
* **The pyramid saturates at two levels.** A third level hurts, and
  level-weighting beats token-weighting by 4 pp at three levels.

### Retracted during the session

* "The two encoders disagree about tiling" — SigLIP `tile6` read +3.75 pp
  [+0.3, +7.5] at 400 queries and +0.58 [-1.6, +2.7] at 1,200. Not a finding.
* "Attention cannot add signal that is not there" — unsupported; Chamfer bounds
  the *index constraint*, not a learned metric.
* The raw tile oracle showing tiles ahead by +2.23 pp — a draw-count artifact,
  inverts to -1.72 pp when matched.
* "Encoders are more redundant with each other than tiles are with crops" — an
  artifact of reading complementarity only at 25 km.
* Host-RAM neighbour table being a speed win — it is not, 1461 s against 1457 s.

## Hazards

**1.45% of OSV-5M frames have their true GPS burned in** as a dashcam overlay,
median 0.05 km from the label. Worth about +1.3 pp on `<25 km` if exploited —
the same magnitude as every effect here. Safe today only because `preprocess`
resizes to 224 and 8-pixel text is unresolvable. **Any higher-resolution
direction walks into it**; screen with `scripts/screen_leak.py` first. The first
100 KartaView images screen clean.

**Noise floor** on this recipe at n=5,000 is about 1 pp of hit rate.

## The habit that earned its keep

**Read the parameters, not the metric.** It caught a `GeoMem` table that was
bit-for-bit zero while producing metrics indistinguishable from an honest null —
and which matched the prediction I had already made, which is what made it
dangerous. `tests/test_geomem.py` encodes the invariant: a module that starts
inert must still receive gradient on the first backward pass.
