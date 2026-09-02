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
  `d4608-b115-e4-cell8` at 247.5 km / 7.66%.
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

**`d1536-b350-e6` — 1.8 km median, 311.0 mean, 76.2% within 25 km** on the
`sequence` test split at n=5,000 (`runs/BOOTSTRAP_bank55.md`, 2026-09-01 13:28).
Pooled 1536-d street vector, **2.65M-image** retrieval bank. The previous best,
`d1536-b190-e6` at 3.6 km / 70.6%, held for four hours. Previous: `d4608-b115-e6` at 7.7 km / 64.2% and its pooled twin
`d1536-b115-e6` at 7.7 / 64.1%.

| contrast | median | `<25 km` |
|---|---|---|
| `d1536-b115-e6` → `d1536-b190-e6` | [+3.4, +4.8] km separated | **+6.50 pp [+5.50, +7.64]** separated |
| `d4608-b115-e6` → `d1536-b190-e6` | [+3.6, +4.9] km separated | +6.50 pp [+5.40, +7.60] separated |
| `d4608-b115-e6` vs `d1536-b115-e6` | inside noise | inside noise |

The last row matters as much as the first two: pooling is *still* free at
4608 → 1536, which is the only reason a 1.90M bank fits in 6.14 GB of host RAM
at all. The ladder converges — `b40` → `c4` is separated at +1.5 pp, `c4` → `c6`
is inside noise — so 6 epochs is the stopping point and the gain is not an
artifact of an undertrained baseline.

### Two corpus steps, two scored predictions (2026-09-01)

| bank | images | median km | `<25 km` | vs previous |
|---|---|---|---|---|
| `pool_c6` | 1.15M | 7.7 | 64.1% | — |
| `b40_c6` | 1.90M | 3.6 | 70.6% | +6.50 pp [+5.50, +7.64] |
| **`b55_c6`** | **2.65M** | **2.5** | **73.8%** | **+3.21 pp [+2.40, +4.02]** |

**+9.7 pp and the median down 3.1x, with no architecture change at all.**

The second prediction was made a better way and worked. For the first step I
log-extrapolated the *metric* and got +9 against +6.5 observed. For the second I
used the *measured* retrieval-similarity increments instead — 0.8926 -> 0.9000 ->
0.9046, whose ratio 0.62 tracks the log-corpus ratio 0.66 almost exactly — and
predicted +3 to +4.5 pp before the arms ran. Observed +3.21. **Predict from the
cheap measurement, not from the expensive one's history.**

Both ladders converge at c6 (`c4` -> `c6` inside noise), so six epochs stays the
stopping point.

### 768-d: free at retrieval level, not free in the agent (so far)

`width_probe.py` put the compression knee at 768: -0.17 pp [-1.00, +0.63] on
any-of-32 `<25 km` against 1536, while 384 was separated at -0.87. A control in
the same sweep matters as much — PCA-768 also beat taking DINOv2's own 768
dimensions (-1.83 pp) at identical width, so the second encoder contributes
something that survives compression rather than the dual-encoder result being a
claim about width.

The agent disagrees, at least at two epochs. Epoch-matched, same 2.65M bank:

| both 2 ep | median km | `<25 km` |
|---|---|---|
| `d1536-b265-e2` 1536-d | 2.7 | 72.8% |
| `d768-b265-e2` 768-d | 3.2 | 70.6% |
| contrast | [-0.7, -0.2] km separated | **+2.20 pp [+1.46, +2.94] separated** |

**The verdict is open, and the precedent says so quantitatively.** Pooling
4608 -> 1536 at its own 2-epoch rung was +0.70 to +2.36 pp worse, separated —
the same signature, the same rung, a very similar magnitude — and recovered to
[-0.74, +0.94] by rung three. Calling pooling at this stage would have been
wrong. `w768_4` and `w768_6` are queued and decide it.

Two corrections from this run:

* **768-d is not faster.** 823 s/epoch against 789 s at 1536-d. I predicted
  halving the width would halve the pages touched per gather; a 1536-d row is
  3 KB and a 768-d row 1.5 KB, and **both fit in one 4 KB page**, so the fault
  count per neighbour is one either way. The memory benefit is real, the I/O
  speedup is not.
* **The real limiter is fault count, not disk speed.** 16,934 page-ins/sec at
  66 MB/s while the disk sits 80% idle at 0.20 ms. C: is NVMe and healthy; E:,
  which holds the shard zips, is a 2 TB **HDD** — which is why sequential
  `slurp()` took `embed_street` from 3 to 83 MB/s.

### 3.50M corpus: 1.8 km / 76.2%, prediction landed (2026-09-02 00:45)

| bank | median km | mean | `<25 km` | vs previous |
|---|---|---|---|---|
| 1.15M | 7.7 | 391.4 | 64.1% | — |
| 1.90M | 3.6 | 367.5 | 70.6% | +6.50 [+5.50, +7.64] |
| 2.65M | 2.5 | 356.7 | 73.8% | +3.21 [+2.40, +4.02] |
| **3.50M** | **1.8** | **311.0** | **76.2%** | **+2.34 [+1.56, +3.12]** |

**Predicted +2.0 to +2.6 pp, centre +2.3, before the arms ran. Observed +2.34.**
Third correct pre-registration from the retrieval-derived method. Across three
steps: 4.3x median reduction, +12.1 pp, architecture untouched.

### The benchmark measures bank coverage as much as it measures the model

1,000 KartaView images through the shipping pipeline and the real beam rollout,
same settings as the benchmark: **442.6 km median / 12.8% `<25 km`**, against
2.7 / 72.8% for the same checkpoint on OSV-5M test.

Two explanations were tested and both are dead:

* **Aspect ratio.** KartaView mixes portrait, square, 4:3 and 16:9 where OSV-5M
  is uniformly 910x512, and for a portrait image the three-crop scheme
  degenerates to one repeated centre crop. But top-1 similarity is flat across
  every bucket -- 0.7010 / 0.7250 / 0.7146 / **0.7129 for 16:9**, the shape that
  matches OSV-5M exactly.
* **Domain shift in the encoder.** Falsified by the self-retrieval control:
  KartaView against the 2.75M OSV-5M bank scores **0.7103**, KartaView against
  *other KartaView images* scores **0.6830**. It retrieves better from OSV-5M
  than from itself, which is what a 146x larger bank should do. There is no
  cross-corpus embedding gap.

What is left is **corpus density**. OSV-5M's 0.90 top-1 comes from queries whose
own streets the bank densely covers -- the `sequence` split holds out a drive
but not the road. That is legitimate geolocation and it is how the system is
meant to work, but it means the headline number is partly a statement about how
well the bank covers the test split's streets. A photo 400 m away facing a
different direction has thin coverage, top-1 falls to 0.71, and the system
degrades with it.

Every comparison here is unaffected -- all arms are measured identically -- but
the absolute numbers are optimistic as a claim about arbitrary photographs. It
also reframes what corpus scaling buys: coverage, not just accuracy.

### 768-d costs about a point, and it is worth paying (2026-09-01 22:00)

Full ladder, both arms on the 2.65M bank. Positive means 768-d is worse.

| rung | 1536-d | 768-d | `<25 km` deficit |
|---|---|---|---|
| e2 | 2.7 km / 376.8 mean / 72.8% | 3.2 / 369.2 / 70.6% | +3.23 [+2.46, +4.00] |
| e4 | 2.6 / 352.0 / 73.6% | 2.8 / 356.4 / 72.3% | +1.49 [+0.76, +2.22] |
| e6 | **2.5 / 356.7 / 73.8%** | 2.7 / **330.8** / 72.8% | **+1.05 [+0.28, +1.82]** |

The gap closes (3.23 -> 1.49 -> 1.05) and both ladders converge, but unlike
pooling it does not reach parity. **768-d costs ~1 pp**, at the noise floor.

Note the mean: 768-d is *better* there at e6, 330.8 against 356.7, while worse
on median and hit rate. PCA discards low-variance directions that sharpen fine
discrimination and keeps the coarse structure that prevents wrong-continent
errors, so it trades the head of the distribution for the tail.

**Adopt it anyway.** Doubling the corpus is worth ~+3.9 pp against a 1.05 pp
width cost, so equal-memory nets ~+2.8 pp -- and at 1536-d a 4.90M table is
15.05 GB, which can never clear `0.4 x free` on a 31.7 GB machine. 768-d is the
only configuration where the whole of OSV-5M is resident (7.53 GB).

### Mean km barely moves while the median collapses

Across two corpus steps: median 7.7 -> 2.5 km (3.1x), mean 386.9 -> 356.7 (8%).
Corpus scaling fixes the typical case and leaves the tail alone -- a 5,000-image
mean near 360 km is dominated by a few hundred images landing on the wrong
continent, and more neighbours do not help an image whose content matches
nowhere. On `cell8` the mean sits at ~1,020 km for every arm, because there
almost everything is tail. The mean is close to useless for selection here,
which is why `--select hit` exists.

### Attention over the 33-token pyramid: null, and confounded

| vs `L0+L1+L2` level mean | `<25 km` | median km |
|---|---|---|
| fusion head alone | **-15.75** [-17.8, -13.8] | 693.7 |
| mean + head, PCA 1536 | **-2.15** [-3.8, -0.5] | 346.7 |

A collapse, not a marginal loss -- and the same head gained +3.67 pp on OSV-5M.
**Do not read this as "attention cannot extract L2".** It trained on 14,938
images against 96,091, with 33 tokens instead of 9. Loss plateaued at 3.35
against a 5.55 chance level: it learned something that does not transfer. The
120k-image harvest is what disambiguates data volume from architecture.

What does stand: mean pooling provably cannot extract L2. Level weighting
recovers two thirds of the three-level penalty (-2.90 -> -0.97 at 25 km) but
still does not make it pay, and the levels are not redundant -- L0 vs L2 cosine
is 0.615.

### Large pages: only as the first big allocation after boot

Confirmed twice in one evening. 10.75 GB succeeds on a freshly booted idle
machine; four hours later with three jobs running, 4.22 GB succeeds and 10.75 GB
fails with ERROR_NO_SYSTEM_RESOURCES. The tier in `dataset.py` falls back
cleanly and says why. To use it on the 3.50M table, reboot and start the ladder
before anything else.

### The corpus axis is not flattening

Bank 1.15M → 1.90M, a factor of 1.65. **Prediction made before the arms
finished**, so it is scoreable: log-extrapolation from the previous step (400k →
1.15M, factor 2.9, worth +20.4 pp) gives ~+9 pp, and I predicted materially less,
naming +2 to +4 pp as the flattening case. Observed **+6.50 [+5.50, +7.64]**.
Below the log line, so the direction was right; the band I named is excluded by
the interval, so the magnitude was wrong. Scaling still pays.

Retrieval-level diagnostics measured on the same 500,000 queries before
training, which is the cheap way to see this coming:

| bank | images | top-1 sim | top-32 sim |
|---|---|---|---|
| `bank25` | 1,150,180 | 0.8926 | 0.8532 |
| `bank40` | 1,900,180 | 0.9000 | 0.8603 |

The new 750k images supply **39.7% of retrieved top-32 neighbours while being
39.5% of the bank** — exactly proportional, so shards 25-39 are the same
distribution and the index is growing by clean addition rather than dilution.
Per-step val accuracy shows where the corpus pays: s0 +2.0, s1 +5.2, **s2 +8.0**,
s3 +5.2 pp. The middle steps gain most, which is what the retrieval prior is
for — more neighbours voting on which child cell to descend into.

`cell8` for this arm is not yet measured.

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
the fine end", not as a win.

**The result is seed-stable, though** (`fh_seed1`, 09:17). Re-running τ=0.02
under seed 1, combination against `L0+L1` at equal width:

| seed | `<25 km` | `<200 km` | `<750 km` | `<2500 km` |
|---|---|---|---|---|
| 0 | +3.67 | +6.33 | +7.87 | +7.03 |
| 1 | +3.47 | +6.30 | +8.10 | +6.93 |

Every threshold reproduces within 0.25 pp. That was worth checking: seeds have
differed by 18 km on the *agent's* metric here
([[select-on-hit-rate-not-median]]). At retrieval level, 3,000 queries against a
fixed bank, run-to-run variance is far below the effect, so the fusion-head win
is not a seed artifact.

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

`runs/BOOTSTRAP_pool_cell8.md`. Against `d4608-b115-e4-cell8`, the 4608-d
baseline at 247.5 km / 7.7%:

| arm | median km | mean | `<25 km` | vs baseline (median, `<25 km`) |
|---|---|---|---|---|
| `d1536-b115-e2-cell8` (2 ep) | 260.1 | 1057.1 | 7.1% | [-23.5, -3.2] km, [+0.04, +1.08] pp — **separated, worse** |
| `d1536-b115-e4-cell8` (+2) | 245.8 | 1019.2 | 7.6% | [-7.9, +10.3] km, [-0.40, +0.56] pp — inside noise |
| `d1536-b115-e6-cell8` (+2) | **243.3** | 1020.6 | 7.6% | [-5.1, +12.6] km, [-0.44, +0.56] pp — inside noise |

**Verdict: pooling is safe on both splits.** The same shape as `sequence`, where
`d1536-b115-e6` sat at [-0.5, +0.7] km and [-0.74, +0.94] pp against its 4608-d
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

* **The prior is good; `cell8` is built to defeat it.** Measured afterwards with
  `scripts/occupancy_probe.py` (which largely reproduces `count_table.py`): on
  `sequence`, a baseline that ignores the image and picks the most-populated
  child of the true parent scores s0 20.5%, **s1 9.7%**, s2 9.0%, s3 4.0% against
  a 1.9M corpus, while the model scores 90.6 / **76.2** / 50.2 / 17.0. So the
  policy is *not* occupancy-driven on the benchmark split — the street image
  does nearly all the work at s1. What it also carries is a prior that empty
  cells are unlikely, which is correct almost everywhere and guaranteed wrong on
  `cell8`, where the answer is always a zero-occupancy cell. Below chance
  follows from a good prior meeting the one case engineered to defeat it, not
  from the model ignoring the photograph.
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
