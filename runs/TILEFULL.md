# Tiles at the full corpus: 3,400,180 bank rows

> **See `runs/LEAKTRAIN.md` for the shipping question.** `runs/TILESHIP.md`
> briefly concluded tiles only reach parity; its comparator turned out to be
> trained on a leaky cache, and against an honest baseline tiles win by
> +2.38 to +4.54 pp. The banner below is kept for the history.
>
> **SUPERSEDED ON THE SHIPPING QUESTION -- see `runs/TILESHIP.md`.**
> Every contrast below is crops+tiles against `pyrL0`, and a later three-arm
> bootstrap puts `pyrL0` **2.06 to 4.00 pp BELOW the shipping arm**
> `d768-b350-e6-drop70`. Tiles bring it back to parity ([-0.62, +1.68] pp,
> inside noise), not past it. The numbers here are correct and the comparison
> is sound within its own pair; what is wrong is reading them as a reason to
> ship tiles.

`runs/TILEBIG.md` measured crops+tiles against crops at 1,150,180 rows and
found **+3.30 pp [+3.0, +3.6]** at 25 km, on a series that had not flattened:
+1.80 pp at 25k, +2.60 at 400k, +3.30 at 1.15M. That was the argument for
tiling the rest of the corpus, and it was an extrapolation.

`chain_tiles.sh` tiled `bank_ext2`, `bank_ext3` and `bank_ext4` on 2026-09-07,
11 h 51 m at a steady 53.1 img/s. `scripts/tilefull.py` then pooled, stacked,
projected and indexed both arms at the size the shipping bank actually is.
**43 minutes**, against a 10 h window: `build_knn` over 3.4M rows takes 5.9
min, not the hour budgeted for it.

One code path, one flag between the arms, and the PCA bases reused from the
1.15M point rather than refitted -- refitting per bank size would put the two
densities in different spaces and the difference would read as a result.

---

## 1. The benchmark split

49,788 test queries, `sequence`, 3,400,180 bank rows.

| top-1 | median km | <1km | <25km | <200km | <750km | <2500km |
|---|---|---|---|---|---|---|
| crops | 25.7 | 10.4% | 49.7% | 69.5% | 83.5% | 93.0% |
| crops+tiles | 19.1 | 11.4% | 53.4% | 73.0% | 85.7% | 94.1% |
| b − a | | +0.93 | **+3.72 [+3.4, +4.0]** | +3.48 | +2.14 | +1.07 |

Separated at every threshold, and the median falls 25.7 → 19.1 km.

**The gain thins sharply with retrieval depth:**

| | <25km gain |
|---|---|
| top-1 | **+3.72** |
| any-of-16 | +2.12 |
| any-of-32 | +1.51 |

and at 2500 km any-of-32 spans zero. Tiles are improving **ranking, not
coverage** -- 33.2% of queries at 25 km already have a correct row somewhere
in the 32 the ranker does not pick, and a finer feature reorders that list
rather than lengthening it.

---

## 2. The gain still grows, and the growth is decelerating

| bank rows | 25 km gain |
|---|---|
| 25,000 | +1.80 |
| 400,180 | +2.60 |
| 1,150,180 | +3.30 |
| **3,400,180** | **+3.72** |

**An earlier version of this report said the growth had stopped, on the
grounds that the marginal intervals at 1.15M ([+3.0, +3.6]) and 3.4M
([+3.4, +4.0]) overlap. That reasoning is wrong** and the conclusion it
produced was wrong with it. Two overlapping confidence intervals do not test
the difference between the two quantities; the difference has its own
sampling distribution and it is narrower, because the same 49,788 queries
appear in both measurements and most of the variance is shared.

Tested properly -- the per-query tile gain at 3.4M minus the per-query tile
gain at 1.15M, bootstrapped over queries, 5,000 resamples:

| step | growth in the gain | 95% CI | |
|---|---|---|---|
| 1.15M → 3.4M, `sequence` | **+0.42 pp** | **[+0.05, +0.79]** | separated |
| 418k → 2.92M, `cell8` | +0.08 pp | [−0.14, +0.28] | spans zero |

So on the benchmark the gain **is** still growing. The lower bound is +0.05,
which is as close to nothing as a separated result gets, and the step sizes
are falling for comparable multiples of corpus -- +0.70 pp for the 2.9x from
400k to 1.15M, +0.42 pp for the 3.0x from 1.15M to 3.4M. Decelerating, not
stopped.

**This does not settle it against `runs/GAIN_DENSITY.md`, and the earlier
version of this report claimed it did.** That experiment varied *local*
density inside one fixed bank and found the tile gain flat across four orders
of magnitude of it. Uniform corpus growth is a different manipulation, so the
two results still do not contradict each other -- and with the corrected test,
neither one is now evidence for the other. That question is open, exactly as
TILEBIG left it.

---

## 3. The geographic holdout keeps an eighth of it

38,322 test queries, `cell8`, 2,924,292 bank rows -- fewer than the sequence
bank because a cell split drops the extension rows sitting in held-out cells.

| cell8, top-1 | median km | <1km | <25km | <200km |
|---|---|---|---|---|
| crops | 319.0 | 0.1% | 6.6% | 40.5% |
| crops+tiles | 291.7 | 0.1% | 7.1% | 42.6% |
| b − a | | **−0.00 [−0.0, +0.0]~** | **+0.49 [+0.3, +0.7]** | +2.06 [+1.7, +2.4] |

`~` spans zero.

**+0.49 pp on unseen geography against +3.72 pp on the benchmark, a factor of
7.6.** Under 1 km the gain is nothing at all -- four queries won, five lost.
And unlike the benchmark, this one has **not** grown: +0.08 pp [−0.14, +0.28]
across a 7x corpus increase, tested paired.

---

## 4. Why the two splits differ, and what it is NOT

**It is not same-drive near-duplicates.** An earlier version of this report
said the benchmark gain came from a query's own drive being in the bank. That
is false, and cheap to check: across the 49,788 test queries and their 32
neighbours -- **1,593,216 entries in each arm** -- the number drawn from the
query's own sequence is **zero**. `build_knn` excludes them by construction,
after the same-sequence leak that once cost the headline 18 pp. The claim
survived from an older mental model of the bank and should not have been
written.

What the two splits actually differ in is whether the *region* is represented
at all. `sequence` holds out whole sequences, so other traversals of the same
streets -- different drive, different day -- remain in the bank. `cell8` holds
out whole z8 cells, 156 km across, so nothing from that area is present.

The retrieved distances say so directly:

| any-of-16, median error | |
|---|---|
| `sequence` | **3.3 km** |
| `cell8` | **88.2 km** |

For a benchmark query the bank holds something a few km away and the task is
to pick the right one of many; a finer feature helps with exactly that. For a
holdout query the nearest thing the bank has is ~90 km off, and there is
nothing for a finer feature to discriminate between.

That is the honest mechanism, and it is still a coverage story -- just not the
near-duplicate one.

---

## 5. What this says

Tiles are a real ranking improvement whose benefit depends on the bank already
covering the query's area. +3.72 pp on the benchmark is real and separated,
survives to the trained agent at 1.15M (+2.76 to +4.88 pp), and is still
growing slowly with corpus. +0.49 pp on unseen geography is real, separated,
and flat.

**Quote +3.72 for the benchmark and +0.49 for "does it read geography",
and do not let the first stand in for the second. "Keep tiles" stood
here and no longer does:** `runs/TILESHIP.md` shows the arm these are
measured against is itself 2-4 pp below what ships.

**What is not measured here:** whether the 3.4M retrieval gain survives
training. TILEBIG showed it does at 1.15M and grows; nothing here repeats that
at 3.4M, and this project has twice had inference-level reasoning predict the
wrong sign for a trained agent. A ladder is the next thing, and it is a
decision about ~6 h of GPU rather than a formality.

---

## 6. Cost and caveats

* 11 h 51 m tiling (3 x 750,000 at 53.1 img/s) + 43 min of pipeline. 72 GB.
* All three tile caches were verified complete before pooling -- identity
  digest and done-mask, all 750,000 rows each. Tiling is all-or-nothing: one
  cosine ranks blended rows against `L0` rows and silently demotes the
  untiled ones.
* `bank_ext70_meta.npz` was verified to be `bank_ext ++ bank_ext2 ++
  bank_ext3 ++ bank_ext4` id for id before the stack was built. Two
  extensions swapped give a file of exactly the right length.
* Extensions 2-4 were pooled from `_bal` rather than `_dual`, which only
  extension 1 has. Not a seam: `pool_pyramid` normalises per 768-d token
  before pooling, and `L0(dual_c3)` and `L0(dual_bal)` agree at cosine
  0.99999976 despite raw SigLIP norms of 20.53 and 82.73.
* **The cell8 numbers carry a transductive bias that is not shown to cancel.**
  Each arm is projected with its own saved basis -- `pyr768_l0_pca.npz` and
  `pyr768_mix_pca.npz` -- and they differ: `mu` by up to 0.0326, `P` by up to
  0.2773. They must differ, because an `L0` basis cannot project a blended
  vector. Both were fitted on the *sequence* train split, part of which is
  cell8's test side, so **both** the levels and the contrast inherit some of
  it, and there is no argument that two different projections carry it
  equally. An earlier version of this report asserted the contrast was
  unaffected because the arms "share one basis"; they do not. Settling it
  needs both bases refitted on cell8-train rows only and the pair re-run,
  which has not been done.
* `knn_gap` could not run at all until it was taught to resolve a merged
  extension's coordinates from its parts (PR #56). It failed at the last
  stage of the run, after both indexes were built, because `bank_ext70` has a
  `_meta.npz` and no parquet.
* `scripts/tilefull.py` originally ignored `Stage.critical` and its own stage
  results, so a failed pool would have let the stack, projection and index run
  against whatever was already at those paths and published a comparison with
  one arm silently stale. Fixed, with tests; note that `Stage.critical` is
  still inert in `overnight.py` and `tilebig.py`, which is a separate change.
