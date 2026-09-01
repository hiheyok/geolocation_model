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
  images are embedded and pooled, and the corpus can go to 2.0M. That is the
  +20.4 pp axis. The chain is written out below.

### The corpus chain, once `ext2_pool` lands

`build_knn.py --bank-ext` takes one stem and reads one `_meta.npz`, so two
extensions have to be made to look like one. `scripts/merge_bank_meta.py` does
that; it was written and tested while block 3 was running.

```
set OSV_RELEASE=s10
python scripts/stack_bank.py      --base pool_bal_bank25 --ext bank_ext2_pool                                   --out  pool_bal_bank40
python scripts/merge_bank_meta.py --parts bank_ext,bank_ext2 --out bank_ext40                                   --emb  pool_bal_bank40.f16.npy
python scripts/build_knn.py       --street-file pool_bal_bank40.f16.npy                                   --split-mode sequence --k 32                                   --bank-ext bank_ext40
```

**The part order is a contract.** `stack_bank.py` writes release rows first,
then the extension in the order given, and `build_knn` reads metadata row *i*
for embedding row `n_rel + i`. Listing the parts in the other order does not
fail — every row keeps a valid-looking z16 address, just the wrong one — so
`--emb` checks the row count, which is the only cheap way to catch it. Expect
`2,000,000 rows = 500,000 release + 1,500,000 extension`, 6.14 GB at 1536-d,
against the 3.75M ceiling pooling bought.

Pooling composes with stacking because `pool_street.py` is strictly
row-independent: it averages three crop tokens per row with no normalisation
and no fitted statistic, so pooling a stacked bank and stacking pooled banks
give the same file.

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
1536-d). Same pattern as crops+tiles — two representations that fail
differently beat either alone.

#### The four-arm block, settled at 12 epochs (2026-09-01 03:11–03:52)

All figures are `mean + head` PCA'd back to **1536-d**, against the `L0+L1`
level-weighted mean pool at identical width, so no arm wins on bytes. `~` spans
zero.

| arm | `<25 km` | `<200 km` | `<2500 km` | median km |
|---|---|---|---|---|
| `fh_combo` τ=0.05, d=256, 12 ep | +2.83 [+1.4, +4.4] | +5.73 [+4.1, +7.3] | +7.40 [+6.1, +8.8] | 98.3 |
| `fh_tau` τ=0.02 | **+3.67** [+2.4, +4.9] | **+6.33** [+4.8, +7.7] | +7.03 [+5.8, +8.3] | **95.9** |
| `fh_big` d=384, 24 ep | +2.00 [+0.4, +3.6] | +4.40 [+2.6, +6.2] | +5.70 [+4.3, +7.1] | 110.4 |
| `fh_pos1` 1 km positives | +0.93 [-0.4, +2.2]~ | +1.97 [+0.6, +3.3] | +3.47 [+2.3, +4.7] | 141.4 |

Three things this says, in order of confidence.

**The combination is robust to how the head is trained.** The head *alone* sits
at -3.7 to -5.6 pp at `<25 km` in every variant; the combination sits at +2.0 to
+3.7. Neither temperature nor capacity nor positive radius changes that
structure, which is what makes it a property of the two representations rather
than of one lucky hyperparameter.

**Capacity is not the constraint; the objective is.** `fh_big` has 4.74M
parameters against 2.50M and twice the steps, and is worse at every threshold.
Its head-alone `<25 km` falls to -5.57 [-7.5, -3.8], the worst of the four. More
capacity against globally-drawn negatives buys a better "same continent?"
answer, which is the wrong axis.

**`fh_tau` is not separated from `fh_combo`.** Those intervals are each against
the baseline, not paired against each other, and they overlap heavily.
`fuse_head.py` does not persist per-query errors, so no paired test between arms
is possible from what is on disk. Read τ=0.02 as "no worse, plausibly better at
the fine end", not as a win. `fh_seed1` in `after.py` re-runs it under a second
seed for exactly this reason.

**`fh_pos1`: the confound hypothesis was tested and is false.** Tightening
positives 5 km → 1 km cuts the pair count 377,302 → 25,532, so the 12-epoch run
trained for 1,188 steps against 17,676. I predicted a weaker result was
undertraining and ran `fh_pos1m` at `--epochs 179` to restore the step count
(17,721 steps, matched). It is **much worse, not better**:

| vs `L0+L1` mean pool | `fh_pos1`, 12 ep | `fh_pos1m`, 179 ep |
|---|---|---|
| head alone, `<200 km` | -0.80 [-2.4, +0.8]~ | **-17.63** [-19.6, -15.7] |
| head alone, median km | 141.4 | 681.2 |
| combination, `<25 km` | +0.93 [-0.4, +2.2]~ | **-4.93** [-6.6, -3.4] |

179 passes over 25,532 pairs overfits, and the 12-epoch version was the better
of the two. **Radius and pair count are not separable knobs here**: at 1 km
there are not enough distinct pairs to support a real budget, so the 15× pair
advantage *is* the mechanism by which 5 km wins rather than a confound sitting
on top of it. `fh_combo` and `fh_tau` at 5 km stand as the right configuration.

Worth keeping as a method note: "the losing arm was undertrained" is a
comfortable hypothesis and it was wrong. Matching the budget cost eight minutes
and turned a caveat into a result.

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

## Block 2 result: the pooled vector reaches parity on cell8 (05:07)

`runs/BOOTSTRAP_pool_cell8.md`. Against `s10_cell8_bank25_lr1e4_c`, the 4608-d
baseline at 247.5 km / 7.7%:

| arm | median km | mean | `<25 km` | vs baseline (median, `<25 km`) |
|---|---|---|---|---|
| `s10_pool_c8` (2 ep) | 260.1 | 1057.1 | 7.1% | [-23.5, -3.2] km, [+0.04, +1.08] pp — **separated, worse** |
| `s10_pool_c8_c4` (+2) | 245.8 | 1019.2 | 7.6% | [-7.9, +10.3] km, [-0.40, +0.56] pp — inside noise |
| `s10_pool_c8_c6` (+2) | **243.3** | 1020.6 | 7.6% | [-5.1, +12.6] km, [-0.44, +0.56] pp — inside noise |

**Verdict: pooling is safe on both splits.** The same shape as `sequence`, where
`s10_pool_c6` sat at [-0.5, +0.7] km and [-0.74, +0.94] pp against its 4608-d
baseline. A third of the bank bytes costs nothing measurable on either split,
which is what licenses the 2.0M corpus the third block is embedding for.

Two things worth keeping from the internals. The two-epoch arm *loses*
separably and the ladder recovers it — matched schedule mattered, exactly as
`poolbank.py` argued it would, so a single long cosine would have given the
wrong answer here. And parity on `cell8` means parity **on the coarse steps**,
for the reason in the next section; it is not evidence about fine localisation.

## The cell8 split cannot measure steps 1-3 (found 2026-09-01 04:20)

Validation per-step tile accuracy on `cell8`, over 5,000 images, teacher-forced
(the val loop scores `logits.argmax` against `b["action"]` on rows carrying the
true prefix -- there is no rollout, so s1 is not conditioned on s0 being right):

| run | street vector | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| `pc8_a` | pooled 1536 | 70.2% | **0.0%** | 0.4% | 4.0% |
| `pc8_b` | pooled 1536 | 71.4% | **0.0%** | 0.5% | 4.3% |
| `rc_c8_lr1` | dual 4608 | 70.2% | **0.0%** | 0.4% | 3.8% |
| `rc_c8_lr3` | dual 4608 | 70.0% | **0.1%** | 4.6% | 4.3% |
| `pb_2`, `sequence` split | pooled 1536 | 88.0% | 70.6% | 42.0% | 12.3% |

**Below chance is the finding, not zero.** The action space at each step is 256,
so an uninformed model scores 0.39% and should get about 20 of 5,000 right by
luck. It gets at most two. A model that were merely ignorant of held-out cells
would sit *at* chance; sitting below it means the policy is actively steering
away from the correct cell.

The mechanism follows from how the split is built. `cell8` holds out whole z8
cells; s1 is exactly the step that selects the z8 cell; and both the training
rows and the retrieval bank exclude held-out cells (`build_knn` drops 123,403 of
the extension's 750,000 for this reason). So the correct s1 action is guaranteed
to be a cell with no training data, and a policy that has learned *where
training data exists* -- an occupancy prior -- points away from it every time.

This is the same failure mode as the per-tile key table in
`tile-memory-is-a-leakage-detector`, but in the shipped policy head rather than
an ablation.

Two consequences:

* **The information is there and the model is not using it.** Map tokens for the
  correct child tile are fetched at inference regardless of the split, so "pick
  the child whose map view matches the street view" is learnable in principle.
  At s1 it is not happening at all.
* **`cell8`'s 247.5 km / 7.66% headline is s0 plus the click head.** Any change
  that improves fine localisation is invisible on this split by construction, so
  `cell8` answers "does this hurt the coarse steps?" and not "does this transfer
  geographically?". Read block 2's bootstrap that way.

Not yet measured: the exact count behind "0.0%" (the log rounds to one decimal,
so it is at most two of 5,000), and whether s1 is recoverable by training
against map-token matching rather than cell identity.

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
