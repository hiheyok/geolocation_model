# What the tiles gain actually is, in the caches the ladders train on

A **retrieval probe** off the neighbour tables alone -- a lookup and a
great-circle, no GPU, no encoder, no model. 49,788 `sequence` test queries, the
400,180-row bank, same-sequence pairs already excluded by `build_knn`, and both
tables asserted to index the same bank rows under the same split hash.

## 1. The gain survives the production path

Every pyramid number before this was measured on vectors built inside the probe
scripts. `tiledrisk` trains on `knn_*.npz` built by the production path --
`pool_pyramid` -> `project_street` -> `build_knn` -- which applies a **PCA to
768 that the probes never did**. A flat ladder result would then have had two
very different readings: the agent cannot use the gain, or the gain is not in
these caches.

| window | median km | `<1 km` | `<25 km` | `<200 km` |
|---|---|---|---|---|
| top-1, crops | 122.9 | 4.1% | 32.9% | 55.8% |
| top-1, crops+tiles | **91.3** | 4.3% | **35.7%** | **58.9%** |
| gain | | +0.29 [+0.2,+0.4] | **+2.76 [+2.5,+3.0]** | +3.10 [+2.8,+3.4] |
| any-of-16, crops | 10.4 | 10.6% | 64.4% | 88.0% |
| any-of-16, crops+tiles | **9.3** | 11.3% | **66.4%** | 88.7% |
| gain | | +0.67 [+0.5,+0.8] | **+2.03 [+1.8,+2.3]** | +0.71 [+0.5,+0.9] |
| any-of-32, gain | | +0.78 [+0.6,+0.9] | +1.45 [+1.2,+1.7] | +0.36 [+0.2,+0.5] |

The PCA preserves it, and +2.76 against `XBANK.md`'s +2.63 reproduces the
representation result at **49,788 queries instead of 3,000**, so it was not a
small-sample artifact.

**The agent runs `--retr-k 16`, so any-of-16 is the window it sees.** +2.03 pp
is the headroom, and an agent converts only part of any headroom, so the
expected ladder result is smaller. A ladder gain materially above +2.0 pp would
mean something is wrong with the comparison rather than right with the model.

## 2. It is not a clean shift -- it is churn with a favourable ratio

A net figure cannot distinguish a small consistent improvement from the residue
of two large opposing flows, and those imply very different things about
whether a trained consumer can use the change.

| window | threshold | won | lost | net | ratio |
|---|---|---|---|---|---|
| top-1 | `<1 km` | 458 (0.92%) | 315 (0.63%) | +0.29 | 1.45x |
| top-1 | `<25 km` | 3,269 (6.57%) | 1,894 (3.80%) | +2.76 | **1.73x** |
| top-1 | `<200 km` | 4,039 (8.11%) | 2,494 (5.01%) | +3.10 | 1.62x |
| any-of-16 | `<1 km` | 723 (1.45%) | 390 (0.78%) | +0.67 | 1.85x |
| any-of-16 | `<25 km` | 2,476 (4.97%) | 1,467 (2.95%) | +2.03 | 1.69x |
| any-of-16 | `<200 km` | 1,515 (3.04%) | 1,160 (2.33%) | +0.71 | 1.31x |

**The +2.76 pp at top-1 is the residue of 10.4% of queries changing side.** The
two representations are *different*, not one uniformly better: 3.80% of queries
that crops located within 25 km, crops+tiles does not. The ratio is a
consistent 1.6-1.9x in favour across every threshold and both depths, which is
what makes it a real gain rather than noise -- but it is a reshuffle with a
favourable bias, not a lift.

## 3. The wins are rescues, not refinements

Of the 3,269 queries that cross 25 km at top-1, the **crops error they came
from** was:

    median 379 km        p90 6,031 km

So tiles are not nudging near-misses under a line. They are rescuing matches
that were grossly wrong -- continent-scale wrong at the 90th percentile. That
is a different story from "finer features give finer discrimination", and it
fits §2's churn: the tile level changes *which* scene the query looks like,
sometimes catastrophically better.

**And it predicts a smaller agent gain.** At any-of-16 the same rescued
population comes from a median of only **75 km** (p90 561 km), because having
sixteen candidates already covers most gross failures. The agent has sixteen
candidates *and* a beam over map tiles. Much of what tiles fix at top-1 is
therefore already handled, which is visible in the numbers: the `<200 km` gain
collapses from +3.10 at top-1 to +0.71 at any-of-16, while the `<1 km` gain
*doubles*, +0.29 to +0.67.

That last contrast is the useful one. **Depth absorbs the coarse gain and
leaves the fine one.** If the agent transfers anything, expect it at the tight
thresholds rather than the headline 25 km.

## Reproduce

    OSV_RELEASE=s10 py scripts/knn_gap.py \
        --a knn_pyr768_l0_sequence_k32.npz \
        --b knn_pyr768_mix_sequence_k32.npz --ranks 1,16,32 --flows

Seconds, CPU only.

## A bug found writing it

`idx` in the `.npz` already holds **release rows** -- `build_knn` writes
`bank_rows[best_j]`, not `best_j`. Indexing it through `bank_rows` a second time
raised `IndexError` here, which is the lucky version: on a stacked bank the
second lookup would have landed in range and silently scored the wrong images.

See `runs/GAIN_DENSITY.md` for the companion result -- the gain does **not**
track local bank density, which weakens the extrapolation that made +2.03 pp a
"floor".
