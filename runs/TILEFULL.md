# Tiles at the full corpus: 3,400,180 bank rows

`runs/TILEBIG.md` measured crops+tiles against crops at 1,150,180 rows and
found **+3.30 pp [+3.0, +3.6]** at 25 km, on a series that had not flattened:
+1.80 pp at 25k, +2.60 at 400k, +3.30 at 1.15M. That was the entire argument
for paying to tile the rest of the corpus, and it was an extrapolation.

`chain_tiles.sh` tiled `bank_ext2`, `bank_ext3` and `bank_ext4` on 2026-09-07,
11 h 51 m at a steady 53.1 img/s. `scripts/tilefull.py` then pooled, stacked,
projected and indexed both arms at the size the shipping bank actually is.
**43 minutes**, against a 10 h window: `build_knn` over 3.4M rows takes 5.9
min, not the hour budgeted for it.

One code path, one flag between the arms, one PCA basis reused from the 1.15M
point rather than refitted -- refitting per bank size would put the two
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
coverage** -- which is what should have been expected: 33.2% of queries at 25
km already have a correct row somewhere in the 32 that the ranker does not
pick, and a finer feature reorders that list rather than lengthening it.

---

## 2. The growth claim does not survive its own test

| bank rows | 25 km gain | 95% CI |
|---|---|---|
| 25,000 | +1.80 | |
| 400,180 | +2.60 | |
| 1,150,180 | +3.30 | [+3.0, **+3.6**] |
| **3,400,180** | **+3.72** | [**+3.4**, +4.0] |

**The last two intervals overlap.** +3.30 → +3.72 across a 3x corpus increase
is not separated, so the curve that motivated this run has flattened at
exactly the point the run was built to test.

This was the one independent check available, and it agrees with the other
one. `runs/GAIN_DENSITY.md` tested the *mechanism* behind the growth claim --
coverage giving way to discrimination -- by varying local density inside one
fixed bank, and found the gain flat across four orders of magnitude of it.
TILEBIG called that "not a refutation, but the only independent check, and it
comes back flat." It now has company.

The absolute numbers still improved a great deal, but that is the corpus, not
the tiles: the crops-only median went 54.8 km at 1.15M to 25.7 km at 3.4M.

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
7.6.** And under 1 km the gain is *nothing* -- four queries won, five lost.
Whatever tiles buy, they do not buy fine resolution in a region the bank has
not seen.

### Compared with the same measurement at a seventh the corpus

TILEBIG §8c ran its cell8 gate on the **un-extended 417,807-row** bank, not on
1.15M rows -- so the honest comparison is the cell8 gain at two bank sizes:

| cell8 bank rows | 25 km gain |
|---|---|
| 417,807 | +0.41 [+0.3, +0.6] |
| **2,924,292** | **+0.49 [+0.3, +0.7]** |

A **7x** corpus increase moves it by 0.08 pp, well inside either interval.
On geography the tile gain is flat, and it was flat before this run too.

---

## 4. What this says

Tiles are a real but modest ranking improvement that is **mostly same-region
matching**. The mechanism is visible in the split between the two tables: on
`sequence`, a query's own drive is usually in the bank, retrieval returns
near-duplicates, and a finer feature helps choose among them. On `cell8` there
are no near-duplicates to choose among, and the gain collapses to +0.49 pp.

That is not a reason to discard tiles -- +3.72 pp on the benchmark is real,
separated, and survives to the trained agent at 1.15M (+2.76 to +4.88 pp). It
is a reason to **stop quoting the benchmark number as if it measured
geography**, and to stop expecting more corpus to buy more tile gain. Both of
those were live assumptions before today.

**What is not measured here:** whether the 3.4M retrieval gain survives
training. TILEBIG showed it does at 1.15M and grows; nothing here repeats that
at 3.4M, and this project has twice had inference-level reasoning predict the
wrong sign for a trained agent. A ladder is the next thing, and it is a
decision about ~6 h of GPU rather than a formality.

---

## 5. Cost and caveats

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
* **The cell8 levels are mildly transductive.** Both arms were projected with
  a PCA basis fitted on the *sequence* train split, part of which is cell8's
  test side. Both arms share one basis, so the paired contrast -- the only
  thing claimed above -- is unaffected; the absolute percentages are slightly
  optimistic.
* `knn_gap` could not run at all until it was taught to resolve a merged
  extension's coordinates from its parts (PR #56). It failed at the last
  stage of the run, after both indexes were built, because `bank_ext70` has a
  `_meta.npz` and no parquet.
