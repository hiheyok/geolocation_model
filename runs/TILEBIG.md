# Tiles at 1,150,180 bank rows: the gain grows, and the series was not flattening

> **Read with `runs/TILESHIP.md`.** The +3.30 pp and +2.76 to +4.88 pp below
> are crops+tiles against `pyrL0`, the same baseline a later three-arm
> bootstrap put 2.06 to 4.00 pp below the shipping arm. The contrasts hold
> within their pair; they are not evidence for shipping tiles.

`scripts/tilebig.py`, 2026-09-06. The 3x2 tile pass over `bank_ext.parquet`
took **233.7 min** for 750,000 images at 53.5 img/s, producing a 13.8 GB
`tile6_ext` cache. Retrieval only -- no gradient step in any number here.

`knn_gap`, 49,788 test queries, sequence split, **1,150,180 bank rows**, the
same rows in both tables:

| top-1 | median km | <1km | <25km | <200km | <750km | <2500km |
|---|---|---|---|---|---|---|
| crops (`pyr_l0`) | 54.8 | 6.9% | 41.8% | 63.0% | 79.7% | 91.5% |
| **crops+tiles (`pyr_l0l1`)** | **38.4** | 7.4% | **45.1%** | 66.2% | 81.8% | 92.4% |
| b - a | | +0.52 | **+3.30 [+3.0,+3.6]** | +3.20 | +2.10 | +0.92 |

| any-of-32 | median km | <1km | <25km | <200km |
|---|---|---|---|---|
| crops | 4.3 | 21.1% | 78.0% | 94.0% |
| crops+tiles | 4.0 | 22.1% | 79.8% | 94.5% |
| b - a | | +0.96 | +1.82 | +0.45 |

Every interval is separated. The median halves, 54.8 -> 38.4 km.

## The question this was built to answer

Does the matched tiles gain keep growing with bank density? The subsample
series ended at 400k and could not say:

| bank rows | crops | crops+tiles | gain pp |
|---|---|---|---|
| 25,000 | 21.0% | 22.8% | +1.80 |
| 50,000 | 24.4% | 26.6% | +2.20 |
| 100,000 | 29.4% | 31.4% | +2.03 |
| 200,000 | 35.2% | 37.7% | +2.53 |
| 400,000 | 40.7% | 43.3% | +2.60 |
| **1,150,180** | **41.8%** | **45.1%** | **+3.30** |

**It grows.** 2.9x more bank than the series' previous end, and the gain rises
+2.60 -> +3.30 rather than saturating. `runs/GAIN_DENSITY.md` found the gain
*flat* across four orders of magnitude of **local** coverage inside one fixed
400k bank; that stands, and the two are not in conflict -- local density is
confounded with region and country, uniform corpus growth is not. What is now
settled is the corpus-growth direction, which is the one a rebuild acts on.

The `<1 km` column is the other new thing: +0.52 pp separated at top-1 and
+0.96 at any-of-32, on a 6.9% base. On the 400k bank `<1 km` was +1.03; it is
smaller here in absolute terms but no longer the near-null it was on the 40k
banks the arm-level probes used.

## The geographic gate, which is the conservative number

Run first, before the 4 h pass, on the un-extended 417,807-row cell8 bank:

| cell8, top-1 | median km | <25km | <200km |
|---|---|---|---|
| crops | 408.2 | 4.6% | 34.5% |
| crops+tiles | 380.7 | 5.0% | 36.4% |
| b - a | | **+0.41 [+0.3,+0.6]** | +1.89 |

Separated and positive, so the gate passes -- but **+0.41 pp on unseen
geography against +3.30 pp on the sequence split, a factor of 8**. Most of the
sequence-split gain is same-region matching that a geographic holdout strips
out. Quote +3.30 for the benchmark and +0.41 for "does it read geography".

## The agent, not the neighbour table

`scripts/bootstrap.py` via `tilebig-boot`, 5,000 test images, 3,000 resamples,
beam k=2, ranked on s0-s2. Two epoch rungs per arm, trained interleaved so a
window that closed early would still leave a comparable pair.

| arm | median km | mean km | <25km |
|---|---|---|---|
| pyrL0 e4 (crops) | 33.6 | 522.9 | 45.9% |
| **pyrL0L1 e4 (crops+tiles)** | **25.4** | **436.8** | **49.7%** |
| pyrL0 e6 | 34.8 | 528.5 | 45.1% |
| pyrL0L1 e6 | 26.2 | 457.0 | 49.0% |

Paired, at matched epochs:

| contrast | median diff | | <25km diff | |
|---|---|---|---|---|
| crops vs crops+tiles, **e4** | [+5.8, +11.1] km | separated | **[+2.76, +4.88] pp** | separated |
| crops vs crops+tiles, **e6** | [+5.8, +11.3] km | separated | **[+2.74, +5.06] pp** | separated |

**The retrieval gain survives training and grows.** +3.30 pp on the neighbour
table becomes **+2.76 to +4.88 pp** for the trained agent, and the median falls
33.6 -> 25.4 km. Both rungs agree, which is the point of training two.

## Six epochs is worse than four, in both arms

| contrast | median diff | | <25km diff | |
|---|---|---|---|---|
| pyrL0 e6 vs e4 | [-0.4, +2.9] km | inside noise | [-1.34, -0.14] pp | separated |
| pyrL0L1 e6 vs e4 | [-0.1, +2.4] km | inside noise | [-1.28, -0.04] pp | separated |

Small, separated, and the same sign in both arms: the extra two epochs cost
about a point of hit rate. Read **e4** as the result. This is the third time
this ladder has peaked before its last rung, so the next run should stop at 4
rather than spend 51 minutes proving it again.

## Provenance

Two tilebig invocations. The first (04:21) completed the 233.7-min tile pass
and then failed 24 downstream stages on the three defects below; the second
(08:36) reused every mark and completed. The `tilebig-knngap` stage was run by
hand after its fix, and its output appended to the stage log.

## Three defects this run cost

Fixed in this branch -- the report and the changes it depends on are the same
commit range, so the numbers above are reproducible from this revision and
were not from the one that produced them.

1. **`ext_stem_for` strips suffixes from the RIGHT.** tilebig named its
   extension embeddings `pyr_l0_ext`, stem-last, which resolves to `None`, so
   `stack_bank` refused every arm with "no metadata found for extension" and
   24 stages failed. Renamed to `bank_ext_pyrl0`.
2. **`pool_pyramid` died at 0xC0000005 with an empty log.** `pq.read_table`
   imports `pyarrow.dataset` lazily, and that DLL load takes an access
   violation once CUDA is initialised **or** a multi-gigabyte mmap is open --
   either alone is enough, measured as a 2x2. The tiles arm had both.
   Importing pyarrow at module scope, before torch, fixes it.
3. **`knn_gap` could not read an extended bank.** `idx` holds bank rows, which
   with `--bank-ext` run past the release, so every extension neighbour raised
   IndexError -- and this is the script that produces the table above. It now
   joins the extension's true lat/lon from its parquet by `image_id`, rather
   than recovering them from z16 addresses, whose 611 m cell is a large
   fraction of the `<1 km` threshold.

The cell8 gate passed only because it compares un-extended caches, so nothing
caught (3) until the final stage of a four-hour run.
