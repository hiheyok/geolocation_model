# Results digest

Generated 2026-09-08 18:35 by `scripts/digest.py`. Every arm below is the test split, 5,000 seeded-random images, beam k=2, ranked on s0-s2 -- the shipping protocol. Read the paired intervals further down before believing any ordering here: the median carries a ~16 km 95% interval at this sample size.

## Arms, best hit rate first

| arm | rel | split | mode | ep | median km | mean km | `<1km` | `<25km` | street file | cache |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `s10_b55_c6` | ? | ? | ? | ? | 2.5 | 356.7 | 38.8% | 73.8% | ? | no checkpoint |
| `s10_b55_c4` | ? | ? | ? | ? | 2.6 | 352.0 | 38.5% | 73.6% | ? | no checkpoint |
| `s10_b55` | ? | ? | ? | ? | 2.7 | 376.8 | 38.1% | 72.8% | ? | no checkpoint |
| `s10_w768_c6` | ? | ? | ? | ? | 2.7 | 330.8 | 38.0% | 72.8% | ? | no checkpoint |
| `s10_w768_c4` | ? | ? | ? | ? | 2.8 | 356.4 | 38.0% | 72.3% | ? | no checkpoint |
| `s10_b40_c6` | ? | ? | ? | ? | 3.6 | 367.5 | 33.6% | 70.6% | ? | no checkpoint |
| `s10_w768` | ? | ? | ? | ? | 3.2 | 369.2 | 36.6% | 70.6% | ? | no checkpoint |
| `s10_b40_c4` | ? | ? | ? | ? | 3.8 | 359.2 | 32.7% | 70.5% | ? | no checkpoint |
| `s10_b40` | ? | ? | ? | ? | 4.1 | 406.1 | 32.1% | 69.0% | ? | no checkpoint |
| `s10_bal_bank25_c6` | ? | ? | ? | ? | 7.7 | 386.9 | 23.1% | 64.2% | ? | no checkpoint |
| `s10_pool_c6` | ? | ? | ? | ? | 7.7 | 391.4 | 23.4% | 64.1% | ? | no checkpoint |
| `s10_pool_c4` | ? | ? | ? | ? | 7.8 | 401.3 | 23.5% | 63.9% | ? | no checkpoint |
| `s10_bal_bank25_c4` | ? | ? | ? | ? | 7.8 | 408.2 | 23.7% | 63.6% | ? | no checkpoint |
| `s10_bal_bank25` | ? | ? | ? | ? | 8.4 | 428.9 | 23.3% | 63.0% | ? | no checkpoint |
| `s10_pool` | ? | ? | ? | ? | 8.4 | 419.6 | 23.5% | 62.6% | ? | no checkpoint |
| `pyrL0L1-b340-c6` | s10 | sequence | dual | 3 | 14.6 | 372.0 | 8.7% | 58.4% | pyr768_l0l1_b340 | clean? |
| `pyrL0L1-b340-e4` | s10 | sequence | dual | 4 | 15.3 | 354.2 | 8.9% | 58.1% | pyr768_l0l1_b340 | clean? |
| `pyrL0L1-b340-e6-lr5e-5` | s10 | sequence | dual | 6 | 15.6 | 369.0 | 8.4% | 57.7% | pyr768_l0l1_b340 | clean? |
| `d768-b350-e6-drop70` | s10 | sequence | dual | 4 | 15.4 | 434.7 | 11.0% | 57.5% | pca768_bank70 | **suspect** |
| `wd29-fix-e4` | s10 | sequence | dual | 3 | 15.1 | 455.6 | 10.5% | 57.5% | pca768_bank70 | **suspect** |
| `pyrL0L1-b340-e6-lr3e-5` | s10 | sequence | dual | 6 | 15.7 | 372.3 | 8.5% | 57.5% | pyr768_l0l1_b340 | clean? |
| `wd29-legacy-e4` | s10 | sequence | dual | 3 | 15.0 | 446.4 | 10.6% | 57.4% | pca768_bank70 | **suspect** |
| `wd29-fix-e6` | s10 | sequence | dual | 4 | 15.1 | 442.6 | 10.5% | 57.3% | pca768_bank70 | **suspect** |
| `pyrL0L1-b340-e6` | s10 | sequence | dual | 5 | 15.7 | 371.8 | 8.4% | 57.1% | pyr768_l0l1_b340 | clean? |
| `pyrL0-b340-c6` | s10 | sequence | dual | 4 | 17.4 | 454.3 | 8.0% | 55.4% | pyr768_l0_b340 | clean? |
| `shipclean-e4` | s10 | sequence | dual | 4 | 18.5 | 458.1 | 8.1% | 54.6% | pca768_bank70 | clean? |
| `clean2-drop70-e4` | s10 | sequence | dual | 4 | 18.7 | 476.7 | 7.9% | 54.6% | pca768_bank70 | clean? |
| `pyrL0-b340-e4` | s10 | sequence | dual | 4 | 19.2 | 472.9 | 8.4% | 54.5% | pyr768_l0_b340 | clean? |
| `clean2-drop70-e2` | s10 | sequence | dual | 2 | 18.6 | 471.9 | 8.1% | 54.4% | pca768_bank70 | clean? |
| `pyrL0-b340-e6` | s10 | sequence | dual | 6 | 19.6 | 480.6 | 8.1% | 53.9% | pyr768_l0_b340 | clean? |
| `shipclean-e6` | s10 | sequence | dual | 6 | 19.6 | 463.4 | 7.7% | 53.8% | pca768_bank70 | clean? |
| `clean-drop70-e6` | s10 | sequence | dual | 5 | 19.7 | 461.6 | 7.6% | 53.8% | pca768_bank70 | clean? |
| `clean2-drop70-e6` | s10 | sequence | dual | 6 | 19.8 | 467.0 | 7.6% | 53.5% | pca768_bank70 | clean? |
| `pyrL0L1-b115-e4` | s10 | sequence | dual | 4 | 25.4 | 436.8 | 5.1% | 49.7% | pyr768_l0l1_b115 | clean? |
| `pyrL0L1-b115-e6` | s10 | sequence | dual | 6 | 26.2 | 457.0 | 4.7% | 49.0% | pyr768_l0l1_b115 | clean? |
| `pyrL0-b115-e4` | s10 | sequence | dual | 4 | 33.6 | 522.9 | 4.9% | 45.9% | pyr768_l0_b115 | clean? |
| `pyrL0-b115-e6` | s10 | sequence | dual | 6 | 34.8 | 528.5 | 4.5% | 45.1% | pyr768_l0_b115 | clean? |
| `pyrMIX-e6` | s10 | sequence | dual | 5 | 47.7 | 514.2 | 2.6% | 40.0% | pyr768_mix | clean? |
| `pyrMIX-e4` | s10 | sequence | dual | 3 | 48.3 | 526.9 | 2.6% | 39.9% | pyr768_mix | clean? |
| `pyrMIX-cond-e8` | s10 | sequence | dual | 7 | 49.3 | 521.2 | 2.5% | 39.7% | pyr768_mix_cond | clean? |
| `pyrMIX-e8` | s10 | sequence | dual | 7 | 49.3 | 537.5 | 2.4% | 39.2% | pyr768_mix | clean? |
| `pyrL0-e4` | s10 | sequence | dual | 4 | 60.7 | 606.2 | 2.1% | 36.7% | pyr768_l0 | clean? |
| `pyrL0-e6` | s10 | sequence | dual | 5 | 62.6 | 631.2 | 2.2% | 36.5% | pyr768_l0 | clean? |
| `pyrL0L1-c8-e2` | s10 | cell8 | dual | 2 | 207.4 | 859.0 | 0.1% | 8.6% | pyr768_l0l1_b340 | clean? |
| `pyrL0L1-c8-e4` | s10 | cell8 | dual | 3 | 202.7 | 884.6 | 0.0% | 8.2% | pyr768_l0l1_b340 | clean? |
| `pyrL0-c8-e4` | s10 | cell8 | dual | 3 | 240.9 | 989.7 | 0.1% | 7.7% | pyr768_l0_b340 | clean? |
| `s10_cell8_bank25_lr1e4_c` | ? | ? | ? | ? | 247.5 | 1024.5 | 0.1% | 7.7% | ? | no checkpoint |
| `pyrL0-c8-e2` | s10 | cell8 | dual | 2 | 241.5 | 977.0 | 0.1% | 7.6% | pyr768_l0_b340 | clean? |
| `s10_pool_c8_c4` | ? | ? | ? | ? | 245.8 | 1019.2 | 0.1% | 7.6% | ? | no checkpoint |
| `s10_pool_c8_c6` | ? | ? | ? | ? | 243.3 | 1020.6 | 0.1% | 7.6% | ? | no checkpoint |
| `s10_cell8_bank25_lr1e4` | ? | ? | ? | ? | 249.0 | 1046.5 | 0.1% | 7.5% | ? | no checkpoint |
| `s10_pool_c8` | ? | ? | ? | ? | 260.1 | 1057.1 | 0.1% | 7.1% | ? | no checkpoint |

**Read the `cache` column before comparing two rows.** Every `knn_*` neighbour cache was rebuilt on 2026-09-04 04:05 after the same-sequence bank leak, and a checkpoint records the cache's path rather than its bytes. Training against the leaky one is worth **2 to 4 pp** -- more than most gaps in this table.

* `clean` -- the cache on disk is the one it trained against, proven by digest.
* `clean?` -- no digest recorded, but the cache is older than the checkpoint, so nothing contradicts it. Most pre-2026-09-08 arms.
* `**suspect**` -- the cache is NEWER than the checkpoint. That is not proof (a copy or a `touch` moves mtime), but it cannot be ruled out.
* `**STALE**` -- the digest disagrees. Proven trained against different bytes.

**A mark is a check on one arm, not a comparison between two.** It says whether a checkpoint trained against the bytes now sitting at the path it recorded -- so `clean` does not mean leak-free: an arm that trained on the leaky cache would still verify as `clean` if nothing had since overwritten that file. Two arms sharing a mark are therefore not thereby comparable; they also need the same split mode, the same evaluation protocol, and the same side of the 2026-09-04 rebuild, and only the first of those is visible in this table. Use the marks to rule an arm OUT, and take comparability from the paired intervals below, which measure it rather than assume it.

**20 of 52 arms are excluded from the headline below**: 20 with no checkpoint left on disk. They remain in the table with their marks.
The best raw hit rate in this file, `s10_b55_c6` at 73.8%, is one of them -- above the best verifiable arm, `pyrL0L1-b340-c6` at 58.4%; its errors were measured 2026-09-01, before the same-sequence bank leak was fixed on 2026-09-04. It is not a result.

**Headline.** Best by hit rate is `pyrL0L1-b340-c6`: 14.6 km median, 58.4% under 25 km.

**Quote this one externally instead.** The best `cell8` arm is `pyrL0L1-c8-e2`: 207.4 km median, 8.6% under 25 km. `cell8` holds whole z8 cells out of training *and* filters the bank to the same cells, so neither the weights nor the corpus has seen the region -- which is the condition the OSV-5M protocol enforces and the `sequence` split does not.

## Paired comparisons

A difference without one of these next to it is not a claim.

### `BOOTSTRAP_bank.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n25k_e38 vs s10_n25k_bank25k | [-616.5, -496.4] km | separated | [+20.22, +23.02] pp | separated |

### `BOOTSTRAP_test.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n25k_e38 vs s10_n50k_e19 | [-3.1, +17.9] km | inside noise | [-1.56, +0.18] pp | inside noise |
| s10_n25k_e38 vs s10_n100k_e10 | [+16.9, +40.8] km | separated | [-2.38, -0.38] pp | separated |
| s10_n25k_e38 vs s10_n200k_e5 | [+17.4, +41.0] km | separated | [-3.56, -1.70] pp | separated |
| s10_n25k_e38 vs s10_n400k_e2 | [+24.0, +47.7] km | separated | [-3.96, -2.12] pp | separated |
| s10_n25k_e38 vs s10_n50k_e80 | [+2.9, +23.6] km | separated | [-1.62, +0.12] pp | inside noise |
| s10_n25k_e38 vs s10_n25k_bank25k | [-371.8, -327.3] km | separated | [+18.08, +20.72] pp | separated |
| s10_n50k_e19 vs s10_n100k_e10 | [+13.6, +31.1] km | separated | [-1.72, +0.26] pp | inside noise |
| s10_n50k_e19 vs s10_n200k_e5 | [+13.7, +29.6] km | separated | [-2.82, -1.10] pp | separated |
| s10_n50k_e19 vs s10_n400k_e2 | [+21.2, +37.0] km | separated | [-3.30, -1.50] pp | separated |
| s10_n50k_e19 vs s10_n50k_e80 | [-1.0, +14.8] km | inside noise | [-0.92, +0.76] pp | inside noise |
| s10_n50k_e19 vs s10_n25k_bank25k | [-379.4, -333.7] km | separated | [+18.72, +21.40] pp | separated |
| s10_n100k_e10 vs s10_n200k_e5 | [-6.5, +6.0] km | inside noise | [-2.08, -0.36] pp | separated |
| s10_n100k_e10 vs s10_n400k_e2 | [+1.4, +12.5] km | separated | [-2.44, -0.80] pp | separated |
| s10_n100k_e10 vs s10_n50k_e80 | [-23.8, -6.9] km | separated | [-0.32, +1.56] pp | inside noise |
| s10_n100k_e10 vs s10_n25k_bank25k | [-402.3, -356.8] km | separated | [+19.48, +22.14] pp | separated |
| s10_n200k_e5 vs s10_n400k_e2 | [+2.0, +12.8] km | separated | [-1.06, +0.26] pp | inside noise |
| s10_n200k_e5 vs s10_n50k_e80 | [-22.6, -7.1] km | separated | [+1.02, +2.72] pp | separated |
| s10_n200k_e5 vs s10_n25k_bank25k | [-399.3, -357.0] km | separated | [+20.70, +23.40] pp | separated |
| s10_n400k_e2 vs s10_n50k_e80 | [-30.1, -14.1] km | separated | [+1.42, +3.16] pp | separated |
| s10_n400k_e2 vs s10_n25k_bank25k | [-407.4, -364.2] km | separated | [+21.08, +23.82] pp | separated |
| s10_n50k_e80 vs s10_n25k_bank25k | [-386.1, -341.0] km | separated | [+18.86, +21.52] pp | separated |

### `BOOTSTRAP_bank25.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_e2 vs s10_n400k_bank25 | [+58.9, +76.0] km | separated | [-21.76, -19.00] pp | separated |

### `BOOTSTRAP_fixed.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n25k_e38 vs s10_n25k_bank25k | [-377.8, -331.5] km | separated | [+18.12, +20.82] pp | separated |
| s10_n25k_e38 vs s10_n400k_e2 | [+49.7, +75.3] km | separated | [-8.68, -6.42] pp | separated |
| s10_n25k_e38 vs s10_n400k_bank25 | [+94.0, +124.1] km | separated | [-29.48, -26.56] pp | separated |
| s10_n25k_bank25k vs s10_n400k_e2 | [+393.9, +439.8] km | separated | [-28.42, -25.54] pp | separated |
| s10_n25k_bank25k vs s10_n400k_bank25 | [+439.2, +483.8] km | separated | [-48.92, -46.02] pp | separated |
| s10_n400k_e2 vs s10_n400k_bank25 | [+39.5, +52.9] km | separated | [-21.92, -19.02] pp | separated |

### `BOOTSTRAP_cell8.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_base vs s10_cell8_bank25 | [-1.4, +24.1] km | inside noise | [-2.32, -1.12] pp | separated |

### `BOOTSTRAP_cell8_cont.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_base_cont vs s10_cell8_bank25_cont | [-7.2, +17.0] km | inside noise | [-2.56, -1.42] pp | separated |

### `BOOTSTRAP_bal4.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25 vs s10_bal_bank25_c4 | [+0.2, +1.0] km | separated | [-1.08, -0.02] pp | separated |
| s10_bal_bank25 vs s10_n400k_bank25_c4 | [-0.5, +0.6] km | inside noise | [-1.22, +0.42] pp | inside noise |
| s10_bal_bank25_c4 vs s10_n400k_bank25_c4 | [-1.0, +0.0] km | inside noise | [-0.68, +0.92] pp | inside noise |

### `BOOTSTRAP_bal6.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c4 vs s10_bal_bank25_c6 | [-0.4, +0.5] km | inside noise | [-1.20, +0.00] pp | separated |

### `BOOTSTRAP_balbank.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25_cont vs s10_bal_bank25 | [-0.3, +0.8] km | inside noise | [-1.24, +0.40] pp | inside noise |

### `BOOTSTRAP_balbank_ep2.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25 vs s10_bal_bank25 | [+1.1, +2.4] km | separated | [-3.84, -2.16] pp | separated |

### `BOOTSTRAP_bank40.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_pool_c6 | [-0.5, +0.6] km | inside noise | [-0.70, +0.90] pp | inside noise |
| s10_bal_bank25_c6 vs s10_b40 | [+3.0, +4.3] km | separated | [-5.88, -3.76] pp | separated |
| s10_bal_bank25_c6 vs s10_b40_c4 | [+3.3, +4.6] km | separated | [-7.50, -5.30] pp | separated |
| s10_bal_bank25_c6 vs s10_b40_c6 | [+3.6, +4.9] km | separated | [-7.60, -5.40] pp | separated |
| s10_pool_c6 vs s10_b40 | [+2.9, +4.2] km | separated | [-6.02, -3.90] pp | separated |
| s10_pool_c6 vs s10_b40_c4 | [+3.1, +4.5] km | separated | [-7.48, -5.46] pp | separated |
| s10_pool_c6 vs s10_b40_c6 | [+3.4, +4.8] km | separated | [-7.64, -5.50] pp | separated |
| s10_b40 vs s10_b40_c4 | [+0.1, +0.5] km | separated | [-2.10, -0.98] pp | separated |
| s10_b40 vs s10_b40_c6 | [+0.3, +0.8] km | separated | [-2.28, -1.00] pp | separated |
| s10_b40_c4 vs s10_b40_c6 | [+0.1, +0.4] km | separated | [-0.58, +0.38] pp | inside noise |

### `BOOTSTRAP_bank55.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_pool_c6 vs s10_b40_c6 | [+3.4, +4.8] km | separated | [-7.64, -5.48] pp | separated |
| s10_pool_c6 vs s10_b55 | [+4.2, +5.7] km | separated | [-9.84, -7.62] pp | separated |
| s10_pool_c6 vs s10_b55_c4 | [+4.3, +5.8] km | separated | [-10.64, -8.38] pp | separated |
| s10_pool_c6 vs s10_b55_c6 | [+4.4, +5.9] km | separated | [-10.88, -8.62] pp | separated |
| s10_b40_c6 vs s10_b55 | [+0.6, +1.1] km | separated | [-3.00, -1.28] pp | separated |
| s10_b40_c6 vs s10_b55_c4 | [+0.7, +1.2] km | separated | [-3.78, -2.14] pp | separated |
| s10_b40_c6 vs s10_b55_c6 | [+0.8, +1.3] km | separated | [-4.02, -2.40] pp | separated |
| s10_b55 vs s10_b55_c4 | [-0.0, +0.3] km | inside noise | [-1.32, -0.28] pp | separated |
| s10_b55 vs s10_b55_c6 | [+0.1, +0.4] km | separated | [-1.66, -0.42] pp | separated |
| s10_b55_c4 vs s10_b55_c6 | [-0.0, +0.2] km | inside noise | [-0.66, +0.18] pp | inside noise |

### `BOOTSTRAP_bank70.md`

| arm | what it is |
|---|---|---|---|---|
| `d1536-b265-e6` | 1536-d / 2.65M bank / 6 ep |
| `d1536-b350-e2` | 1536-d / 3.50M bank / 2 ep |
| `d1536-b350-e4` | 1536-d / 3.50M bank / 4 ep |
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d1536-b265-e6 vs d1536-b350-e2 | [+0.4, +0.8] km | separated | [-2.24, -0.62] pp | separated |
| d1536-b265-e6 vs d1536-b350-e4 | [+0.6, +0.9] km | separated | [-3.00, -1.50] pp | separated |
| d1536-b265-e6 vs d1536-b350-e6 | [+0.6, +0.9] km | separated | [-3.12, -1.56] pp | separated |
| d1536-b350-e2 vs d1536-b350-e4 | [+0.1, +0.3] km | separated | [-1.30, -0.30] pp | separated |
| d1536-b350-e2 vs d1536-b350-e6 | [+0.1, +0.3] km | separated | [-1.50, -0.38] pp | separated |
| d1536-b350-e4 vs d1536-b350-e6 | [-0.1, +0.1] km | inside noise | [-0.54, +0.32] pp | inside noise |

### `BOOTSTRAP_best.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25_c4 vs s10_bal_bank25_c6 | [+0.0, +1.1] km | separated | [-1.58, +0.10] pp | inside noise |

### `BOOTSTRAP_blend.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_key_dual vs s10_bal | [+0.7, +9.3] km | separated | [-1.82, -0.04] pp | separated |

### `BOOTSTRAP_cell8_c.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_bank25_lr1e4 vs s10_cell8_bank25_lr1e4_c | [-3.4, +8.9] km | inside noise | [-0.46, +0.18] pp | inside noise |
| s10_cell8_bank25_lr1e4 vs s10_cell8_bank25_cont | [-20.4, -1.5] km | separated | [-0.94, +0.00] pp | separated |
| s10_cell8_bank25_lr1e4_c vs s10_cell8_bank25_cont | [-23.7, -4.3] km | separated | [-0.82, +0.14] pp | inside noise |

### `BOOTSTRAP_cell8_lr.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_bank25_cont vs s10_cell8_bank25_lr3e4 | [-43.7, -22.3] km | separated | [+0.02, +0.82] pp | separated |
| s10_cell8_bank25_cont vs s10_cell8_bank25_lr1e4 | [+1.5, +20.4] km | separated | [+0.00, +0.94] pp | inside noise |
| s10_cell8_bank25_lr3e4 vs s10_cell8_bank25_lr1e4 | [+32.8, +56.5] km | separated | [-0.40, +0.54] pp | inside noise |

### `BOOTSTRAP_cleantrain.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| arm | sink negatives |
| `clean-drop70-e6` | seeded |
| `wd29-fix-e6` | OS entropy (predates the flag) |
| `d768-b350-e6-drop70` | OS entropy (predates the flag) |
| arm | weight-decay grouping |
| `clean-drop70-e6` | by module type |
| `wd29-fix-e6` | by module type |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| clean-drop70-e6 vs wd29-fix-e6 | [+3.3, +5.9] km | separated | [-4.58, -2.52] pp | separated |
| clean-drop70-e6 vs d768-b350-e6-drop70 | [+3.1, +5.7] km | separated | [-4.84, -2.74] pp | separated |
| wd29-fix-e6 vs d768-b350-e6-drop70 | [-1.2, +0.9] km | inside noise | [-1.26, +0.74] pp | inside noise |

### `BOOTSTRAP_cleantrain2.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| arm | sink negatives |
| `clean2-drop70-e2` | OS entropy |
| `clean2-drop70-e4` | OS entropy |
| `clean2-drop70-e6` | OS entropy |
| `wd29-fix-e6` | OS entropy (predates the flag) |
| `d768-b350-e6-drop70` | OS entropy (predates the flag) |
| arm | weight-decay grouping |
| `clean2-drop70-e2` | by module type |
| `clean2-drop70-e4` | by module type |
| `clean2-drop70-e6` | by module type |
| `wd29-fix-e6` | by module type |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| clean2-drop70-e2 vs clean2-drop70-e4 | [-0.9, +0.8] km | inside noise | [-0.84, +0.52] pp | inside noise |
| clean2-drop70-e2 vs clean2-drop70-e6 | [-2.3, +0.1] km | inside noise | [+0.04, +1.78] pp | separated |
| clean2-drop70-e2 vs wd29-fix-e6 | [+2.4, +4.9] km | separated | [-3.82, -1.96] pp | separated |
| clean2-drop70-e2 vs d768-b350-e6-drop70 | [+2.2, +4.7] km | separated | [-4.08, -2.12] pp | separated |
| clean2-drop70-e4 vs clean2-drop70-e6 | [-1.9, -0.1] km | separated | [+0.46, +1.70] pp | separated |
| clean2-drop70-e4 vs wd29-fix-e6 | [+2.5, +5.0] km | separated | [-3.74, -1.84] pp | separated |
| clean2-drop70-e4 vs d768-b350-e6-drop70 | [+2.2, +4.7] km | separated | [-4.00, -2.00] pp | separated |
| clean2-drop70-e6 vs wd29-fix-e6 | [+3.3, +6.0] km | separated | [-4.86, -2.82] pp | separated |
| clean2-drop70-e6 vs d768-b350-e6-drop70 | [+3.0, +5.9] km | separated | [-5.12, -3.00] pp | separated |
| wd29-fix-e6 vs d768-b350-e6-drop70 | [-1.3, +0.9] km | inside noise | [-1.22, +0.76] pp | inside noise |

### `BOOTSTRAP_cont.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25 vs s10_n400k_bank25_cont | [+0.9, +2.1] km | separated | [-3.34, -1.88] pp | separated |

### `BOOTSTRAP_cont4.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25_cont vs s10_n400k_bank25_c4 | [-0.0, +0.6] km | inside noise | [-1.36, -0.32] pp | separated |

### `BOOTSTRAP_d1536_drop30.md`

| arm | what it is |
|---|---|---|---|---|
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d1536-b350-e6 vs d768-b350-e6-drop30 | [-0.2, +0.1] km | inside noise | [-0.46, +1.02] pp | inside noise |
| d1536-b350-e6 vs d1536-b350-e6-drop30 | [-0.1, +0.1] km | inside noise | [-1.30, -0.10] pp | separated |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop30 | [-0.1, +0.2] km | inside noise | [-1.68, -0.24] pp | separated |

### `BOOTSTRAP_d1536_drop70.md`

| arm | what it is |
|---|---|---|---|---|
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |
| `d1536-b350-e6-drop70` | 1536-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d1536-b350-e6 vs d1536-b350-e6-drop30 | [-0.1, +0.1] km | inside noise | [-1.28, -0.10] pp | separated |
| d1536-b350-e6 vs d1536-b350-e6-drop70 | [-0.8, -0.4] km | separated | [-0.52, +0.98] pp | inside noise |
| d1536-b350-e6 vs d768-b350-e6-drop70 | [-0.5, -0.2] km | separated | [+0.20, +1.70] pp | separated |
| d1536-b350-e6-drop30 vs d1536-b350-e6-drop70 | [-0.8, -0.4] km | separated | [+0.20, +1.62] pp | separated |
| d1536-b350-e6-drop30 vs d768-b350-e6-drop70 | [-0.5, -0.2] km | separated | [+0.90, +2.40] pp | separated |
| d1536-b350-e6-drop70 vs d768-b350-e6-drop70 | [+0.0, +0.4] km | separated | [-0.08, +1.52] pp | inside noise |

### `BOOTSTRAP_drop30.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b265-e6 vs d768-b350-e6 | [+0.6, +1.0] km | separated | [-3.66, -1.98] pp | separated |
| d768-b265-e6 vs d768-b350-e6-drop30 | [+0.7, +1.1] km | separated | [-4.04, -2.26] pp | separated |
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.0, +0.2] km | inside noise | [-0.98, +0.30] pp | inside noise |

### `BOOTSTRAP_encgate.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_bal_gate | [-0.3, +0.4] km | inside noise | [-0.68, +0.36] pp | inside noise |
| s10_bal_bank25_c6 vs s10_bal_c8 | [-0.3, +0.4] km | inside noise | [-0.88, +0.10] pp | inside noise |
| s10_bal_gate vs s10_bal_c8 | [-0.4, +0.4] km | inside noise | [-0.80, +0.30] pp | inside noise |

### `BOOTSTRAP_geomem.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_bal_c8 | [-0.3, +0.4] km | inside noise | [-0.90, +0.10] pp | inside noise |
| s10_bal_bank25_c6 vs s10_geo_bias | [-0.2, +0.6] km | inside noise | [-0.94, +0.12] pp | inside noise |
| s10_bal_bank25_c6 vs s10_geo_key | [-0.2, +0.5] km | inside noise | [-0.76, +0.24] pp | inside noise |
| s10_bal_c8 vs s10_geo_bias | [-0.2, +0.5] km | inside noise | [-0.52, +0.52] pp | inside noise |
| s10_bal_c8 vs s10_geo_key | [-0.2, +0.4] km | inside noise | [-0.26, +0.54] pp | inside noise |
| s10_geo_bias vs s10_geo_key | [-0.5, +0.3] km | inside noise | [-0.42, +0.68] pp | inside noise |

### `BOOTSTRAP_keys.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_key_none vs s10_key_scalar | [+84.0, +111.6] km | separated | [-18.62, -15.90] pp | separated |
| s10_key_none vs s10_key_cond | [+78.5, +105.9] km | separated | [-18.58, -15.74] pp | separated |
| s10_key_none vs s10_key_pos | [+94.6, +121.6] km | separated | [-20.58, -17.82] pp | separated |
| s10_key_none vs s10_key_dual | [+103.9, +131.1] km | separated | [-22.78, -19.92] pp | separated |
| s10_key_scalar vs s10_key_cond | [-10.5, -0.1] km | separated | [-0.60, +0.90] pp | inside noise |
| s10_key_scalar vs s10_key_pos | [+5.2, +16.1] km | separated | [-2.78, -1.10] pp | separated |
| s10_key_scalar vs s10_key_dual | [+13.8, +25.9] km | separated | [-5.04, -3.08] pp | separated |
| s10_key_cond vs s10_key_pos | [+10.1, +21.5] km | separated | [-2.90, -1.30] pp | separated |
| s10_key_cond vs s10_key_dual | [+18.5, +31.4] km | separated | [-5.16, -3.26] pp | separated |
| s10_key_pos vs s10_key_dual | [+5.1, +13.9] km | separated | [-2.98, -1.26] pp | separated |

### `BOOTSTRAP_keys_s01.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| retr_seq vs retr_gate | [-23.1, +10.6] km | inside noise | [-1.51, +0.53] pp | inside noise |
| retr_seq vs retr_lkey | [-18.6, +17.0] km | inside noise | [-1.77, +0.18] pp | inside noise |
| retr_seq vs retr_dual | [-22.7, +12.0] km | inside noise | [-1.51, +0.55] pp | inside noise |
| retr_seq vs retr_dual32 | [-17.3, +17.3] km | inside noise | [-0.49, +1.51] pp | inside noise |
| retr_gate vs retr_lkey | [-12.2, +23.8] km | inside noise | [-1.24, +0.65] pp | inside noise |
| retr_gate vs retr_dual | [-15.7, +17.3] km | inside noise | [-1.00, +1.00] pp | inside noise |
| retr_gate vs retr_dual32 | [-10.3, +21.4] km | inside noise | [+0.02, +2.00] pp | separated |
| retr_lkey vs retr_dual | [-21.5, +11.7] km | inside noise | [-0.63, +1.24] pp | inside noise |
| retr_lkey vs retr_dual32 | [-16.7, +17.0] km | inside noise | [+0.33, +2.28] pp | separated |
| retr_dual vs retr_dual32 | [-10.5, +19.8] km | inside noise | [+0.06, +1.94] pp | separated |

### `BOOTSTRAP_pcurve.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop10` | 768-d / 3.50M bank / 6 ep / drop10 |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d768-b350-e6-drop50` | 768-d / 3.50M bank / 6 ep / drop50 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b350-e6 vs d768-b350-e6-drop10 | [-0.0, +0.2] km | inside noise | [-1.30, -0.02] pp | separated |
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.0, +0.3] km | inside noise | [-0.96, +0.32] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop50 | [-0.1, +0.2] km | inside noise | [-1.04, +0.44] pp | inside noise |
| d768-b350-e6-drop10 vs d768-b350-e6-drop30 | [-0.1, +0.1] km | inside noise | [-0.30, +1.00] pp | inside noise |
| d768-b350-e6-drop10 vs d768-b350-e6-drop50 | [-0.2, +0.0] km | inside noise | [-0.36, +1.08] pp | inside noise |
| d768-b350-e6-drop30 vs d768-b350-e6-drop50 | [-0.2, +0.0] km | inside noise | [-0.64, +0.72] pp | inside noise |

### `BOOTSTRAP_pcurve2.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d768-b350-e6-drop50` | 768-d / 3.50M bank / 6 ep / drop50 |
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.0, +0.2] km | inside noise | [-0.98, +0.34] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop50 | [-0.1, +0.2] km | inside noise | [-1.02, +0.40] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.40, +1.12] pp | inside noise |
| d768-b350-e6 vs d1536-b350-e6-drop30 | [+0.0, +0.3] km | separated | [-2.04, -0.54] pp | separated |
| d768-b350-e6-drop30 vs d768-b350-e6-drop50 | [-0.2, +0.0] km | inside noise | [-0.64, +0.70] pp | inside noise |
| d768-b350-e6-drop30 vs d768-b350-e6-drop70 | [-0.5, -0.2] km | separated | [-0.04, +1.40] pp | inside noise |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop30 | [-0.1, +0.2] km | inside noise | [-1.68, -0.26] pp | separated |
| d768-b350-e6-drop50 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.02, +1.36] pp | inside noise |
| d768-b350-e6-drop50 vs d1536-b350-e6-drop30 | [+0.0, +0.2] km | separated | [-1.68, -0.26] pp | separated |
| d768-b350-e6-drop70 vs d1536-b350-e6-drop30 | [+0.2, +0.5] km | separated | [-2.38, -0.88] pp | separated |

### `BOOTSTRAP_pcurve3.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop50` | 768-d / 3.50M bank / 6 ep / drop50 |
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6-drop90` | 768-d / 3.50M bank / 6 ep / drop90 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b350-e6 vs d768-b350-e6-drop50 | [-0.1, +0.2] km | inside noise | [-1.00, +0.42] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.42, +1.12] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop90 | [-2.4, -1.7] km | separated | [+4.70, +6.72] pp | separated |
| d768-b350-e6-drop50 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.04, +1.36] pp | inside noise |
| d768-b350-e6-drop50 vs d768-b350-e6-drop90 | [-2.4, -1.7] km | separated | [+5.02, +6.98] pp | separated |
| d768-b350-e6-drop70 vs d768-b350-e6-drop90 | [-2.1, -1.5] km | separated | [+4.40, +6.30] pp | separated |

### `BOOTSTRAP_pool.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_pool | [-1.2, -0.1] km | separated | [+0.70, +2.36] pp | separated |
| s10_bal_bank25_c6 vs s10_pool_c4 | [-0.6, +0.5] km | inside noise | [-0.58, +1.04] pp | inside noise |
| s10_bal_bank25_c6 vs s10_pool_c6 | [-0.5, +0.7] km | inside noise | [-0.74, +0.94] pp | inside noise |
| s10_pool vs s10_pool_c4 | [+0.2, +1.1] km | separated | [-1.86, -0.74] pp | separated |
| s10_pool vs s10_pool_c6 | [+0.3, +1.3] km | separated | [-2.14, -0.76] pp | separated |
| s10_pool_c4 vs s10_pool_c6 | [-0.2, +0.5] km | inside noise | [-0.64, +0.34] pp | inside noise |

### `BOOTSTRAP_pool_cell8.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8 | [-23.5, -3.2] km | separated | [+0.04, +1.08] pp | separated |
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8_c4 | [-7.9, +10.3] km | inside noise | [-0.40, +0.56] pp | inside noise |
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8_c6 | [-5.1, +12.6] km | inside noise | [-0.44, +0.56] pp | inside noise |
| s10_pool_c8 vs s10_pool_c8_c4 | [+6.9, +21.7] km | separated | [-0.86, -0.14] pp | separated |
| s10_pool_c8 vs s10_pool_c8_c6 | [+9.0, +25.2] km | separated | [-0.92, -0.08] pp | separated |
| s10_pool_c8_c4 vs s10_pool_c8_c6 | [-2.4, +8.5] km | inside noise | [-0.34, +0.32] pp | inside noise |

### `BOOTSTRAP_seed.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_key_dual vs s10_n400k_e2 | [-6.0, +1.5] km | inside noise | [-0.28, +1.22] pp | inside noise |

### `BOOTSTRAP_seqfix.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| arm | weight-decay grouping |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |
| `d768-b350-e6` | legacy substring (predates the flag) |
| `wd29-fix-e6` | by module type |
| `wd29-legacy-e6` | legacy substring |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b350-e6-drop70 vs d768-b350-e6 | [-0.5, +1.6] km | inside noise | [-0.72, +1.06] pp | inside noise |
| d768-b350-e6-drop70 vs wd29-fix-e6 | [-0.9, +1.3] km | inside noise | [-0.74, +1.22] pp | inside noise |
| d768-b350-e6-drop70 vs wd29-legacy-e6 | [-0.6, +1.4] km | inside noise | [-0.76, +1.14] pp | inside noise |
| d768-b350-e6 vs wd29-fix-e6 | [-1.4, +0.8] km | inside noise | [-0.92, +1.02] pp | inside noise |
| d768-b350-e6 vs wd29-legacy-e6 | [-1.3, +0.9] km | inside noise | [-1.02, +0.92] pp | inside noise |
| wd29-fix-e6 vs wd29-legacy-e6 | [-0.6, +0.9] km | inside noise | [-0.68, +0.62] pp | inside noise |

### `BOOTSTRAP_seqfix_all.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d768-b350-e6-drop90` | 768-d / 3.50M bank / 6 ep / drop90 |
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |
| `d1536-b350-e6-drop70` | 1536-d / 3.50M bank / 6 ep / drop70 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b350-e6-drop70 vs d768-b350-e6 | [-0.5, +1.6] km | inside noise | [-0.72, +1.06] pp | inside noise |
| d768-b350-e6-drop70 vs d768-b265-e6 | [-3.7, -1.3] km | separated | [+1.60, +3.56] pp | separated |
| d768-b350-e6-drop70 vs d768-b350-e6-drop30 | [-0.0, +1.8] km | inside noise | [-1.12, +0.54] pp | inside noise |
| d768-b350-e6-drop70 vs d768-b350-e6-drop90 | [-7.0, -4.3] km | separated | [+3.76, +5.90] pp | separated |
| d768-b350-e6-drop70 vs d1536-b350-e6 | [+0.3, +2.2] km | separated | [-1.34, +0.44] pp | inside noise |
| d768-b350-e6-drop70 vs d1536-b350-e6-drop30 | [+1.1, +3.0] km | separated | [-2.50, -0.72] pp | separated |
| d768-b350-e6-drop70 vs d1536-b350-e6-drop70 | [-0.3, +1.7] km | inside noise | [-1.70, +0.14] pp | inside noise |
| d768-b350-e6 vs d768-b265-e6 | [-4.3, -1.9] km | separated | [+1.44, +3.24] pp | separated |
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.6, +1.3] km | inside noise | [-1.28, +0.32] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop90 | [-7.7, -4.8] km | separated | [+3.56, +5.74] pp | separated |
| d768-b350-e6 vs d1536-b350-e6 | [-0.4, +1.6] km | inside noise | [-1.50, +0.22] pp | inside noise |
| d768-b350-e6 vs d1536-b350-e6-drop30 | [+0.5, +2.5] km | separated | [-2.72, -0.94] pp | separated |
| d768-b350-e6 vs d1536-b350-e6-drop70 | [-1.0, +1.3] km | inside noise | [-1.92, +0.00] pp | inside noise |
| d768-b265-e6 vs d768-b350-e6-drop30 | [+2.2, +4.5] km | separated | [-3.80, -1.88] pp | separated |
| d768-b265-e6 vs d768-b350-e6-drop90 | [-4.7, -1.8] km | separated | [+1.18, +3.44] pp | separated |
| d768-b265-e6 vs d1536-b350-e6 | [+2.5, +4.9] km | separated | [-3.98, -2.00] pp | separated |
| d768-b265-e6 vs d1536-b350-e6-drop30 | [+3.3, +5.7] km | separated | [-5.16, -3.16] pp | separated |
| d768-b265-e6 vs d1536-b350-e6-drop70 | [+1.9, +4.5] km | separated | [-4.34, -2.26] pp | separated |
| d768-b350-e6-drop30 vs d768-b350-e6-drop90 | [-8.0, -5.2] km | separated | [+4.12, +6.26] pp | separated |
| d768-b350-e6-drop30 vs d1536-b350-e6 | [-0.6, +1.2] km | inside noise | [-0.98, +0.72] pp | inside noise |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop30 | [+0.2, +2.1] km | separated | [-2.16, -0.44] pp | separated |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop70 | [-1.3, +0.9] km | inside noise | [-1.44, +0.50] pp | inside noise |
| d768-b350-e6-drop90 vs d1536-b350-e6 | [+5.4, +8.4] km | separated | [-6.42, -4.16] pp | separated |
| d768-b350-e6-drop90 vs d1536-b350-e6-drop30 | [+6.3, +9.1] km | separated | [-7.54, -5.40] pp | separated |
| d768-b350-e6-drop90 vs d1536-b350-e6-drop70 | [+4.9, +7.8] km | separated | [-6.68, -4.56] pp | separated |
| d1536-b350-e6 vs d1536-b350-e6-drop30 | [+0.1, +1.6] km | separated | [-1.82, -0.48] pp | separated |
| d1536-b350-e6 vs d1536-b350-e6-drop70 | [-1.5, +0.4] km | inside noise | [-1.16, +0.54] pp | inside noise |
| d1536-b350-e6-drop30 vs d1536-b350-e6-drop70 | [-2.3, -0.5] km | separated | [-0.02, +1.70] pp | inside noise |

### `BOOTSTRAP_sub2.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b350-e6-drop70 vs d768-b350-e6-drop70-sub2 | [-0.4, -0.0] km | separated | [-1.18, +0.32] pp | inside noise |

### `BOOTSTRAP_w768.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_b40_c6 vs s10_b55_c6 | [+0.8, +1.3] km | separated | [-4.00, -2.36] pp | separated |
| s10_b40_c6 vs s10_w768 | [+0.1, +0.7] km | separated | [-0.88, +1.00] pp | inside noise |
| s10_b55_c6 vs s10_w768 | [-0.9, -0.5] km | separated | [+2.48, +4.04] pp | separated |

### `BOOTSTRAP_w768_b70.md`

| arm | what it is |
|---|---|---|---|---|
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| `d768-b350-e2` | 768-d / 3.50M bank / 2 ep |
| `d768-b350-e4` | 768-d / 3.50M bank / 4 ep |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d1536-b350-e6 vs d768-b265-e6 | [-1.2, -0.7] km | separated | [+2.56, +4.30] pp | separated |
| d1536-b350-e6 vs d768-b350-e2 | [-0.5, -0.2] km | separated | [+1.20, +2.62] pp | separated |
| d1536-b350-e6 vs d768-b350-e4 | [-0.3, -0.0] km | separated | [-0.08, +1.38] pp | inside noise |
| d1536-b350-e6 vs d768-b350-e6 | [-0.3, -0.0] km | separated | [-0.18, +1.32] pp | inside noise |
| d768-b265-e6 vs d768-b350-e2 | [+0.4, +0.9] km | separated | [-2.38, -0.64] pp | separated |
| d768-b265-e6 vs d768-b350-e4 | [+0.6, +1.0] km | separated | [-3.60, -1.88] pp | separated |
| d768-b265-e6 vs d768-b350-e6 | [+0.6, +1.0] km | separated | [-3.66, -1.96] pp | separated |
| d768-b350-e2 vs d768-b350-e4 | [+0.0, +0.3] km | separated | [-1.78, -0.76] pp | separated |
| d768-b350-e2 vs d768-b350-e6 | [+0.0, +0.3] km | separated | [-1.94, -0.72] pp | separated |
| d768-b350-e4 vs d768-b350-e6 | [-0.1, +0.1] km | inside noise | [-0.52, +0.42] pp | inside noise |

### `BOOTSTRAP_w768_full.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_b55_c4 vs s10_b55_c6 | [-0.0, +0.2] km | inside noise | [-0.66, +0.18] pp | inside noise |
| s10_b55_c4 vs s10_w768 | [-0.8, -0.4] km | separated | [+2.26, +3.76] pp | separated |
| s10_b55_c4 vs s10_w768_c4 | [-0.4, -0.0] km | separated | [+0.52, +1.98] pp | separated |
| s10_b55_c4 vs s10_w768_c6 | [-0.3, +0.1] km | inside noise | [+0.06, +1.56] pp | separated |
| s10_b55_c6 vs s10_w768 | [-0.9, -0.5] km | separated | [+2.46, +4.00] pp | separated |
| s10_b55_c6 vs s10_w768_c4 | [-0.5, -0.1] km | separated | [+0.76, +2.22] pp | separated |
| s10_b55_c6 vs s10_w768_c6 | [-0.4, -0.0] km | separated | [+0.28, +1.82] pp | separated |
| s10_w768 vs s10_w768_c4 | [+0.2, +0.5] km | separated | [-2.32, -1.18] pp | separated |
| s10_w768 vs s10_w768_c6 | [+0.2, +0.6] km | separated | [-2.88, -1.48] pp | separated |
| s10_w768_c4 vs s10_w768_c6 | [-0.0, +0.2] km | inside noise | [-0.94, +0.06] pp | inside noise |

### `BOOTSTRAP_w768_matched.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_b55 vs s10_w768 | [-0.7, -0.2] km | separated | [+1.46, +2.94] pp | separated |

### `BOOTSTRAP_w768_means.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b265-e4` | 768-d / 2.65M bank / 4 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b265-e4 vs d768-b265-e6 | [-0.0, +0.2] km | inside noise | [-0.96, +0.06] pp | inside noise |

### `BOOTSTRAP_wd29.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| arm | weight-decay grouping |
| `wd29-fix-e6` | by module type |
| `wd29-legacy-e6` | legacy substring |
| `d768-b350-e6-drop70` | by module type |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| wd29-fix-e6 vs wd29-legacy-e6 | [-0.0, +0.2] km | inside noise | [-1.08, +0.10] pp | inside noise |
| wd29-fix-e6 vs d768-b350-e6-drop70 | [+0.2, +0.6] km | separated | [-1.16, +0.54] pp | inside noise |
| wd29-legacy-e6 vs d768-b350-e6-drop70 | [+0.1, +0.4] km | separated | [-0.66, +1.00] pp | inside noise |

### `BOOTSTRAP_tiledrisk.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrL0-e6 vs pyrMIX-e6 | [+9.7, +20.3] km | separated | [-4.64, -2.22] pp | separated |
| pyrL0-e6 vs pyrL0-e4 | [-1.4, +5.8] km | inside noise | [-0.90, +0.56] pp | inside noise |
| pyrL0-e6 vs pyrMIX-e4 | [+9.4, +20.2] km | separated | [-4.54, -2.16] pp | separated |
| pyrMIX-e6 vs pyrL0-e4 | [-17.5, -7.6] km | separated | [+2.04, +4.40] pp | separated |
| pyrMIX-e6 vs pyrMIX-e4 | [-2.9, +2.7] km | inside noise | [-0.74, +0.90] pp | inside noise |
| pyrL0-e4 vs pyrMIX-e4 | [+7.8, +17.3] km | separated | [-4.26, -2.00] pp | separated |

### `BOOTSTRAP_condladder.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrMIX-e6 vs pyrMIX-e8 | [-3.5, +0.8] km | inside noise | [+0.16, +1.36] pp | separated |
| pyrMIX-e6 vs pyrMIX-cond-e8 | [-3.7, +1.3] km | inside noise | [-0.56, +1.00] pp | inside noise |
| pyrMIX-e8 vs pyrMIX-cond-e8 | [-2.4, +2.6] km | inside noise | [-1.30, +0.24] pp | inside noise |

### `BOOTSTRAP_ship.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| arm | sink negatives |
| `pyrL0L1-b340-e4` | OS entropy |
| `pyrL0-b340-e4` | OS entropy |
| `d768-b350-e6-drop70` | OS entropy (predates the flag) |
| arm | weight-decay grouping |
| `pyrL0L1-b340-e4` | by module type |
| `pyrL0-b340-e4` | by module type |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| pyrL0L1-b340-e4 vs pyrL0-b340-e4 | [-5.4, -2.6] km | separated | [+2.54, +4.70] pp | separated |
| pyrL0L1-b340-e4 vs d768-b350-e6-drop70 | [-1.4, +1.0] km | inside noise | [-0.62, +1.68] pp | inside noise |
| pyrL0-b340-e4 vs d768-b350-e6-drop70 | [+2.4, +5.1] km | separated | [-4.00, -2.06] pp | separated |

### `BOOTSTRAP_tilefull.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrL0-b340-e4 vs pyrL0L1-b340-e4 | [+2.6, +5.4] km | separated | [-4.70, -2.54] pp | separated |
| pyrL0-b340-e4 vs pyrL0-b340-e6 | [-1.3, +0.3] km | inside noise | [-0.04, +1.12] pp | inside noise |
| pyrL0-b340-e4 vs pyrL0L1-b340-e6 | [+2.1, +4.8] km | separated | [-3.68, -1.50] pp | separated |
| pyrL0L1-b340-e4 vs pyrL0-b340-e6 | [-5.7, -3.0] km | separated | [+3.08, +5.28] pp | separated |
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6 | [-1.3, +0.1] km | inside noise | [+0.32, +1.62] pp | separated |
| pyrL0-b340-e6 vs pyrL0L1-b340-e6 | [+2.5, +5.1] km | separated | [-4.26, -2.08] pp | separated |

### `BOOTSTRAP_cell8_tiles.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrL0-c8-e4 vs pyrL0L1-c8-e4 | [+26.7, +45.7] km | separated | [-1.10, -0.06] pp | separated |
| pyrL0-c8-e4 vs pyrL0-c8-e2 | [-7.4, +6.7] km | inside noise | [-0.30, +0.40] pp | inside noise |
| pyrL0-c8-e4 vs pyrL0L1-c8-e2 | [+21.7, +42.0] km | separated | [-1.48, -0.44] pp | separated |
| pyrL0L1-c8-e4 vs pyrL0-c8-e2 | [-47.0, -26.8] km | separated | [+0.12, +1.16] pp | separated |
| pyrL0L1-c8-e4 vs pyrL0L1-c8-e2 | [-11.5, +1.3] km | inside noise | [-0.74, -0.04] pp | separated |
| pyrL0-c8-e2 vs pyrL0L1-c8-e2 | [+21.0, +42.8] km | separated | [-1.52, -0.52] pp | separated |

### `BOOTSTRAP_lr.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6 | [-1.3, +0.1] km | inside noise | [+0.36, +1.60] pp | separated |
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6-lr5e-5 | [-0.9, +0.1] km | inside noise | [-0.10, +0.86] pp | inside noise |
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6-lr3e-5 | [-1.1, -0.0] km | separated | [+0.18, +1.08] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0L1-b340-e6-lr5e-5 | [-0.5, +0.8] km | inside noise | [-1.18, -0.02] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0L1-b340-e6-lr3e-5 | [-0.6, +0.7] km | inside noise | [-0.94, +0.24] pp | inside noise |
| pyrL0L1-b340-e6-lr5e-5 vs pyrL0L1-b340-e6-lr3e-5 | [-0.5, +0.2] km | inside noise | [-0.08, +0.56] pp | inside noise |

### `BOOTSTRAP_schedule.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6 | [-1.3, +0.1] km | inside noise | [+0.36, +1.60] pp | separated |
| pyrL0L1-b340-e4 vs pyrL0L1-b340-c6 | [-0.3, +1.5] km | inside noise | [-1.20, +0.58] pp | inside noise |
| pyrL0L1-b340-e4 vs pyrL0-b340-e4 | [-5.3, -2.6] km | separated | [+2.56, +4.66] pp | separated |
| pyrL0L1-b340-e4 vs pyrL0-b340-e6 | [-5.7, -3.0] km | separated | [+3.08, +5.28] pp | separated |
| pyrL0L1-b340-e4 vs pyrL0-b340-c6 | [-3.6, -1.3] km | separated | [+1.62, +3.76] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0L1-b340-c6 | [+0.2, +2.1] km | separated | [-2.18, -0.38] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0-b340-e4 | [-4.8, -2.0] km | separated | [+1.52, +3.66] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0-b340-e6 | [-5.2, -2.4] km | separated | [+2.08, +4.30] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0-b340-c6 | [-3.1, -0.7] km | separated | [+0.62, +2.84] pp | separated |
| pyrL0L1-b340-c6 vs pyrL0-b340-e4 | [-5.8, -3.1] km | separated | [+2.82, +4.98] pp | separated |
| pyrL0L1-b340-c6 vs pyrL0-b340-e6 | [-6.3, -3.6] km | separated | [+3.32, +5.64] pp | separated |
| pyrL0L1-b340-c6 vs pyrL0-b340-c6 | [-4.2, -1.9] km | separated | [+1.92, +4.10] pp | separated |
| pyrL0-b340-e4 vs pyrL0-b340-e6 | [-1.3, +0.3] km | inside noise | [-0.02, +1.20] pp | inside noise |
| pyrL0-b340-e4 vs pyrL0-b340-c6 | [+0.4, +2.7] km | separated | [-1.78, +0.06] pp | inside noise |
| pyrL0-b340-e6 vs pyrL0-b340-c6 | [+0.8, +3.2] km | separated | [-2.42, -0.52] pp | separated |

### `BOOTSTRAP_shipclean.md`

| arm | sink negatives |
|---|---|---|---|---|
| `shipclean-e4` | OS entropy |
| `pyrL0-b340-e4` | OS entropy |
| `pyrL0L1-b340-e4` | OS entropy |
| `wd29-fix-e4` | OS entropy (predates the flag) |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| shipclean-e4 vs pyrL0-b340-e4 | [-1.8, +0.6] km | inside noise | [-0.78, +1.06] pp | inside noise |
| shipclean-e4 vs pyrL0L1-b340-e4 | [+2.2, +4.7] km | separated | [-4.56, -2.36] pp | separated |
| shipclean-e4 vs wd29-fix-e4 | [+2.4, +4.5] km | separated | [-3.80, -1.96] pp | separated |
| pyrL0-b340-e4 vs pyrL0L1-b340-e4 | [+2.6, +5.3] km | separated | [-4.66, -2.54] pp | separated |
| pyrL0-b340-e4 vs wd29-fix-e4 | [+2.7, +5.2] km | separated | [-3.98, -2.08] pp | separated |
| pyrL0L1-b340-e4 vs wd29-fix-e4 | [-1.2, +1.1] km | inside noise | [-0.50, +1.64] pp | inside noise |

### `BOOTSTRAP_shipclean6.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| shipclean-e4 vs shipclean-e6 | [-1.8, -0.2] km | separated | [+0.20, +1.40] pp | separated |
| shipclean-e4 vs pyrL0-b340-e4 | [-1.7, +0.6] km | inside noise | [-0.78, +1.04] pp | inside noise |
| shipclean-e4 vs pyrL0L1-b340-e4 | [+2.2, +4.7] km | separated | [-4.54, -2.38] pp | separated |
| shipclean-e6 vs pyrL0-b340-e4 | [-0.8, +1.7] km | inside noise | [-1.66, +0.32] pp | inside noise |
| shipclean-e6 vs pyrL0L1-b340-e4 | [+3.0, +5.7] km | separated | [-5.36, -3.16] pp | separated |
| pyrL0-b340-e4 vs pyrL0L1-b340-e4 | [+2.6, +5.3] km | separated | [-4.64, -2.60] pp | separated |

### `BOOTSTRAP_split_clean.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| arm | sink negatives |
| `pyrL0L1-b340-e4` | OS entropy |
| `pyrL0-b340-e4` | OS entropy |
| `wd29-fix-e4` | OS entropy (predates the flag) |
| `wd29-legacy-e4` | OS entropy (predates the flag) |
| `d768-b350-e6-drop70` | OS entropy (predates the flag) |
| arm | weight-decay grouping |
| `pyrL0L1-b340-e4` | by module type |
| `pyrL0-b340-e4` | by module type |
| `wd29-fix-e4` | by module type |
| `wd29-legacy-e4` | legacy substring |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| pyrL0L1-b340-e4 vs pyrL0-b340-e4 | [-5.4, -2.6] km | separated | [+2.54, +4.70] pp | separated |
| pyrL0L1-b340-e4 vs wd29-fix-e4 | [-1.1, +1.1] km | inside noise | [-0.54, +1.68] pp | inside noise |
| pyrL0L1-b340-e4 vs wd29-legacy-e4 | [-0.9, +1.3] km | inside noise | [-0.34, +1.76] pp | inside noise |
| pyrL0L1-b340-e4 vs d768-b350-e6-drop70 | [-1.4, +1.0] km | inside noise | [-0.54, +1.66] pp | inside noise |
| pyrL0-b340-e4 vs wd29-fix-e4 | [+2.7, +5.2] km | separated | [-3.98, -2.08] pp | separated |
| pyrL0-b340-e4 vs wd29-legacy-e4 | [+2.9, +5.4] km | separated | [-3.80, -1.96] pp | separated |
| pyrL0-b340-e4 vs d768-b350-e6-drop70 | [+2.5, +5.0] km | separated | [-4.00, -2.08] pp | separated |
| wd29-fix-e4 vs wd29-legacy-e4 | [-0.4, +0.8] km | inside noise | [-0.40, +0.68] pp | inside noise |
| wd29-fix-e4 vs d768-b350-e6-drop70 | [-1.2, +0.8] km | inside noise | [-1.02, +0.92] pp | inside noise |
| wd29-legacy-e4 vs d768-b350-e6-drop70 | [-1.4, +0.6] km | inside noise | [-1.08, +0.80] pp | inside noise |

## Stages

312 stages completed: `b40_2`, `b40_4`, `b40_6`, `b40_eval`, `b40_knn`, `b40_meta`, `b40_stack`, `b55_2`, `b55_4`, `b55_6`, `b55_eval`, `b55_knn`, `b55_meta`, `b55_stack`, `b70_2`, `b70_4`, `b70_6`, `b70_eval`, `b70_knn`, `b70_meta`, `b70_stack`, `bank25_train`, `boot-d1536-drop30`, `boot-d1536-drop70`, `boot-d768-b350-drop30`, `boot-d768-b350`, `boot-pcurve`, `boot-pcurve2`, `boot-pcurve3`, `boot-sub2`, `bx_concat`, `bx_dino`, `bx_knn`, `bx_meta`, `bx_siglip`, `bx_stack`, `c8_eval`, `c8_eval_cont`, `c8_knn_base`, `c8_knn_ext`, `c8_train_base`, `c8_train_base_cont`, `c8_train_ext`, `c8_train_ext_cont`, `calib`, `concat`, `condladder-boot`, `dataset`, `e3_dino`, `e3_join`, `e3_meta`, `e3_pool`, `e3_siglip`, `e4_dino`, `e4_join`, `e4_meta`, `e4_pool`, `e4_siglip`, `eg_ctrl`, `eg_eval`, `eg_gate`, `embed_dino`, `embed_siglip`, `ext2_dino`, `ext2_join`, `ext2_meta`, `ext2_pool`, `ext2_siglip`, `fh_big`, `fh_combo`, `fh_pos1`, `fh_pos1m`, `fh_seed1`, `fh_tau`, `fu_bal4`, `fu_bal4_eval`, `fu_bal6`, `fu_bal6_eval`, `fuse-attn-pyr47`, `gm_bias`, `gm_eval`, `gm_key`, `hrfix-b265`, `hrfix-b350`, `hrfix-d1536`, `hrfix-d1536drop30`, `hrfix-d1536drop70`, `hrfix-drop10`, `hrfix-drop30`, `hrfix-drop50`, `hrfix-drop70`, `hrfix-drop90`, `hrfix-sub2`, `hrfix-x-m265-b350`, `hrfix-x-m350-b265`, `hrpair-d768-b350-e6-drop70`, `hrpair-wd29-fix-e6`, `kartaview-d1536-b350-e6-n5k`, `kartaview-d1536-drop30-n5k`, `kartaview-d768-b265-e6-n5k`, `kartaview-d768-b265-e6`, `kartaview-d768-b350-e6-n5k`, `kartaview-d768-b350-e6`, `kartaview-drop10-n5k`, `kartaview-drop30-n5k`, `kartaview-drop50-n5k`, `kartaview-drop70-n5k`, `kartaview-drop90-n5k`, `kartaview-sub2-n5k`, `kartaview-x-m265-b350`, `kartaview-x-m350-b265`, `key_eval`, `key_s01_eval`, `key_seed`, `key_train_cond`, `key_train_dual`, `key_train_none`, `key_train_pos`, `key_train_scalar`, `knn`, `kv_screen`, `mar_bal_concat`, `mar_bal_eval`, `mar_bal_knn`, `mar_bal_train`, `mar_balbank_eval`, `mar_balbank_knn`, `mar_balbank_stack`, `mar_balbank_train`, `mar_balext_concat`, `mar_c8_cont`, `mar_c8_eval`, `mar_cont4`, `mar_cont4_eval`, `mar_tile_probe`, `mqfix-drop70`, `mqfix-pow4-rr`, `multiq-d768-b350-e6-diverse-global`, `multiq-d768-b350-e6-diverse-rr`, `multiq-d768-b350-e6-diverse`, `multiq-d768-b350-e6-near-global`, `multiq-d768-b350-e6-near-rr`, `multiq-d768-b350-e6-near`, `multiq-d768-b350-e6-pow4-rr`, `multiq-drop30-pow4-rr`, `night-boot-cell8`, `night-boot-schedule`, `night-boot-shipclean6`, `night-c6-pyrL0`, `night-c6-pyrL0L1`, `night-cell8-pyrL0-e2`, `night-cell8-pyrL0-e4`, `night-cell8-pyrL0L1-e2`, `night-cell8-pyrL0L1-e4`, `night-shipclean-e6`, `pb_2`, `pb_4`, `pb_6`, `pb_eval`, `pc8_a`, `pc8_b`, `pc8_c`, `pc8_eval`, `pc8_knn`, `pyrcache-hr47k`, `pyrL0L1-b340-c6-lr15`, `pyrL0L1-b340-c6x2-lr5e-5`, `rc_c8_eval`, `rc_c8_lr1`, `rc_c8_lr3`, `rc_cont`, `rc_cont_eval`, `rc_depth1`, `rc_depth2`, `rc_depth3`, `rc_depth4`, `rc_width`, `reprobe-pyr-levels`, `reprobe-query-only`, `reprobe-resmatch`, `reprobe-xbank`, `s01_km`, `s01_km_eval`, `s10_n100k_e10`, `s10_n200k_e5`, `s10_n25k_e38`, `s10_n25k_e38_eval_test`, `s10_n400k_e2`, `s10_n50k_e19`, `s10_n50k_e80`, `seqfix-boot`, `seqfix-verify`, `seqfix2-boot-all`, `seqfix2-boot-clean`, `seqfix2-hr-clean`, `seqfix2-knn-b55`, `seqfix2-knn-d1536`, `seqfix2-verify-b55`, `seqfix2-verify-d1536`, `seqfix3-boot`, `seqfix3-hr`, `tilebig-boot`, `tilebig-knn-cell8-pyrL0`, `tilebig-knn-cell8-pyrL0L1`, `tilebig-knn-pyrL0`, `tilebig-knn-pyrL0L1`, `tilebig-knngap-cell8`, `tilebig-knngap`, `tilebig-pool-pyrL0`, `tilebig-pool-pyrL0L1`, `tilebig-proj-pyrL0`, `tilebig-proj-pyrL0L1`, `tilebig-stack-pyrL0`, `tilebig-stack-pyrL0L1`, `tilebig-tiles`, `tiledrisk-boot`, `tilefull-knn-cell8-pyrL0`, `tilefull-knn-cell8-pyrL0L1`, `tilefull-knn-pyrL0`, `tilefull-knn-pyrL0L1`, `tilefull-knngap-cell8`, `tilefull-ladder-boot`, `tilefull-pool-pyrL0-bank_ext2`, `tilefull-pool-pyrL0-bank_ext3`, `tilefull-pool-pyrL0-bank_ext4`, `tilefull-pool-pyrL0L1-bank_ext2`, `tilefull-pool-pyrL0L1-bank_ext3`, `tilefull-pool-pyrL0L1-bank_ext4`, `tilefull-proj-pyrL0`, `tilefull-proj-pyrL0L1`, `tilefull-stack-pyrL0-b190`, `tilefull-stack-pyrL0-b265`, `tilefull-stack-pyrL0-b340`, `tilefull-stack-pyrL0L1-b190`, `tilefull-stack-pyrL0L1-b265`, `tilefull-stack-pyrL0L1-b340`, `tiles`, `tiles2_probe`, `train-clean-e2`, `train-clean-e4`, `train-clean-e6`, `train-clean2-e2`, `train-clean2-e4`, `train-clean2-e6`, `train-d1536-b350-e2-drop30`, `train-d1536-b350-e2-drop70`, `train-d1536-b350-e4-drop30`, `train-d1536-b350-e4-drop70`, `train-d1536-b350-e6-drop30`, `train-d1536-b350-e6-drop70`, `train-d768-b350-e2-drop10`, `train-d768-b350-e2-drop30`, `train-d768-b350-e2-drop50`, `train-d768-b350-e2-drop70-sub2`, `train-d768-b350-e2-drop70`, `train-d768-b350-e2-drop90`, `train-d768-b350-e2`, `train-d768-b350-e4-drop10`, `train-d768-b350-e4-drop30`, `train-d768-b350-e4-drop50`, `train-d768-b350-e4-drop70-sub2`, `train-d768-b350-e4-drop70`, `train-d768-b350-e4-drop90`, `train-d768-b350-e4`, `train-d768-b350-e6-drop10`, `train-d768-b350-e6-drop30`, `train-d768-b350-e6-drop50`, `train-d768-b350-e6-drop70-sub2`, `train-d768-b350-e6-drop70`, `train-d768-b350-e6-drop90`, `train-d768-b350-e6`, `train-pyrL0-b340-e2`, `train-pyrL0-b340-e4`, `train-pyrL0-b340-e6`, `train-pyrL0-e2`, `train-pyrL0-e4`, `train-pyrL0-e6`, `train-pyrL0115-e2`, `train-pyrL0115-e4`, `train-pyrL0115-e6`, `train-pyrL0L1-b340-e2`, `train-pyrL0L1-b340-e4`, `train-pyrL0L1-b340-e6`, `train-pyrL0L1115-e2`, `train-pyrL0L1115-e4`, `train-pyrL0L1115-e6`, `train-pyrMIX-cond-e8`, `train-pyrMIX-e2`, `train-pyrMIX-e4`, `train-pyrMIX-e6`, `train-pyrMIX-e8`, `train-wd29-fix-e2`, `train-wd29-fix-e4`, `train-wd29-fix-e6`, `train-wd29-legacy-e2`, `train-wd29-legacy-e4`, `train-wd29-legacy-e6`, `w70_2`, `w70_4`, `w70_6`, `w70_eval`, `w70_knn`, `w70_proj`, `w768_2`, `w768_4`, `w768_6`, `w768_eval`, `w768_knn`, `w768_proj`, `wd29-boot`, `wd29-hr-fix`, `wd29-hr-legacy`

Skipped or failed:

```
[08:16:26] FAIL   train-pyrL0115-e4  rc=1 after 0.0 min
[08:16:29] FAIL   train-pyrL0115-e4  rc=1 after 0.0 min
[08:16:32] FAIL   train-pyrL0L1115-e4  rc=1 after 0.0 min
[08:16:35] FAIL   train-pyrL0L1115-e4  rc=1 after 0.0 min
[08:16:38] FAIL   train-pyrL0L1115-e4  rc=1 after 0.0 min
[08:16:41] FAIL   train-pyrL0L1115-e4  rc=1 after 0.0 min
[08:16:43] FAIL   train-pyrL0115-e6  rc=1 after 0.0 min
[08:16:46] FAIL   train-pyrL0115-e6  rc=1 after 0.0 min
[08:16:49] FAIL   train-pyrL0115-e6  rc=1 after 0.0 min
[08:16:52] FAIL   train-pyrL0115-e6  rc=1 after 0.0 min
[08:16:54] FAIL   train-pyrL0L1115-e6  rc=1 after 0.0 min
[08:16:57] FAIL   train-pyrL0L1115-e6  rc=1 after 0.0 min
[08:17:00] FAIL   train-pyrL0L1115-e6  rc=1 after 0.0 min
[08:17:03] FAIL   train-pyrL0L1115-e6  rc=1 after 0.0 min
[08:17:05] FAIL   tilebig-boot  rc=1 after 0.0 min
[08:17:07] FAIL   tilebig-boot  rc=1 after 0.0 min
[08:17:10] FAIL   tilebig-boot  rc=1 after 0.0 min
[08:36:10] skip   tilebig-knn-cell8-pyrL0  (already done)
[08:36:10] skip   tilebig-knn-cell8-pyrL0L1  (already done)
[08:36:10] skip   tilebig-knngap-cell8  (already done)
[08:36:10] skip   tilebig-tiles  (already done)
[08:44:03] FAIL   tilebig-knngap  rc=1 after 0.1 min
[08:44:06] FAIL   tilebig-knngap  rc=1 after 0.0 min
[14:54:05] FAIL   tilefull-knngap  rc=1 after 0.0 min
[14:54:07] FAIL   tilefull-knngap  rc=1 after 0.0 min
```
