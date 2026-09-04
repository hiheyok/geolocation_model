# Results digest

> **STALE — every number below is inflated by the same-sequence bank leak.**
> Generated 2026-09-04 00:18, hours before `build_knn` was found to serve
> same-drive frames as retrieval neighbours (41.6% of test queries had a
> same-drive top-1, at a median 0.31 km). The corrected eight-arm table is in
> `runs/BOOTSTRAP_seqfix_all.md`: the headline falls about 18 pp, e.g.
> `d768-b350-e6-drop70` 75.2% -> 57.5% and `d1536-b350-e6-drop30` 76.9% ->
> 59.1%. The ordering of the real axes survives, the absolute numbers do not.
> Regenerate with `scripts/digest.py` once every arm has been re-measured
> against the rebuilt caches; until then read the bootstrap files, not this.

Generated 2026-09-04 00:18 by `scripts/digest.py`. Every arm below is the test split, 5,000 seeded-random images, beam k=2, ranked on s0-s2 -- the shipping protocol. Read the paired intervals further down before believing any ordering here: the median carries a ~16 km 95% interval at this sample size.

## Arms, best hit rate first

| arm | rel | split | mode | ep | median km | mean km | `<1km` | `<25km` | street file |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `d1536-b350-e6-drop30` | s10 | sequence | dual | 6 | 1.8 | 286.5 | 42.2% | 76.9% | pool_bal_bank70 |
| `d768-b350-e6-drop10` | s10 | sequence | dual | 4 | 1.8 | 311.9 | 42.5% | 76.2% | pca768_bank70 |
| `d1536-b350-e6` | s10 | sequence | dual | 5 | 1.8 | 311.0 | 42.6% | 76.2% | pool_bal_bank70 |
| `d1536-b350-e4` | s10 | sequence | dual | 4 | 1.8 | 300.1 | 42.6% | 76.0% | pool_bal_bank70 |
| `d1536-b350-e6-drop70` | s10 | sequence | dual | 6 | 2.3 | 266.5 | 38.8% | 75.9% | pool_bal_bank70 |
| `d768-b350-e6-drop30` | s10 | sequence | dual | 5 | 1.8 | 295.2 | 41.8% | 75.9% | pca768_bank70 |
| `d768-b350-e6-drop50` | s10 | sequence | dual | 4 | 1.9 | 280.1 | 41.8% | 75.9% | pca768_bank70 |
| `d768-b350-e6-drop70-sub2` | s10 | sequence | dual | 4 | 2.3 | 276.3 | 39.4% | 75.6% | pca768_bank70 |
| `d768-b350-e6` | s10 | sequence | dual | 4 | 1.9 | 320.4 | 41.8% | 75.6% | pca768_bank70 |
| `d768-b350-e4` | s10 | sequence | dual | 3 | 1.9 | 334.9 | 41.6% | 75.5% | pca768_bank70 |
| `d1536-b350-e2` | s10 | sequence | dual | 2 | 1.9 | 335.2 | 41.7% | 75.2% | pool_bal_bank70 |
| `d768-b350-e6-drop70` | s10 | sequence | dual | 4 | 2.2 | 285.2 | 40.3% | 75.2% | pca768_bank70 |
| `d768-b350-e2` | s10 | sequence | dual | 2 | 2.1 | 334.0 | 41.2% | 74.3% | pca768_bank70 |
| `d1536-b265-e6` | s10 | sequence | dual | 6 | 2.5 | 356.7 | 38.8% | 73.8% | pool_bal_bank55 |
| `s10_b55_c6` | ? | ? | ? | ? | 2.5 | 356.7 | 38.8% | 73.8% | ? |
| `s10_b55_c4` | ? | ? | ? | ? | 2.6 | 352.0 | 38.5% | 73.6% | ? |
| `s10_b55` | ? | ? | ? | ? | 2.7 | 376.8 | 38.1% | 72.8% | ? |
| `d768-b265-e6` | s10 | sequence | dual | 6 | 2.7 | 330.8 | 38.0% | 72.8% | pca768_bank55 |
| `s10_w768_c6` | ? | ? | ? | ? | 2.7 | 330.8 | 38.0% | 72.8% | ? |
| `d768-b265-e4` | s10 | sequence | dual | 4 | 2.8 | 356.4 | 38.0% | 72.3% | pca768_bank55 |
| `s10_w768_c4` | ? | ? | ? | ? | 2.8 | 356.4 | 38.0% | 72.3% | ? |
| `s10_b40_c6` | ? | ? | ? | ? | 3.6 | 367.5 | 33.6% | 70.6% | ? |
| `s10_w768` | ? | ? | ? | ? | 3.2 | 369.2 | 36.6% | 70.6% | ? |
| `s10_b40_c4` | ? | ? | ? | ? | 3.8 | 359.2 | 32.7% | 70.5% | ? |
| `d768-b350-e6-drop90` | s10 | sequence | dual | 6 | 4.0 | 358.9 | 34.2% | 69.9% | pca768_bank70 |
| `s10_b40` | ? | ? | ? | ? | 4.1 | 406.1 | 32.1% | 69.0% | ? |
| `s10_bal_c8` | s10 | sequence | dual | 7 | 7.7 | 369.3 | 22.9% | 64.6% | dual_bal_bank25 |
| `s10_geo_bias` | s10 | sequence | dual | 6 | 7.6 | 370.1 | 23.5% | 64.6% | dual_bal_bank25 |
| `s10_geo_key` | s10 | sequence | dual | 7 | 7.6 | 373.6 | 22.9% | 64.4% | dual_bal_bank25 |
| `s10_bal_gate` | s10 | sequence | dual | 6 | 7.6 | 372.0 | 23.1% | 64.3% | dual_bal_bank25 |
| `s10_bal_bank25_c6` | ? | ? | ? | ? | 7.7 | 386.9 | 23.1% | 64.2% | ? |
| `s10_pool_c6` | ? | ? | ? | ? | 7.7 | 391.4 | 23.4% | 64.1% | ? |
| `s10_pool_c4` | ? | ? | ? | ? | 7.8 | 401.3 | 23.5% | 63.9% | ? |
| `s10_bal_bank25_c4` | ? | ? | ? | ? | 7.8 | 408.2 | 23.7% | 63.6% | ? |
| `s10_n400k_bank25_c4` | s10 | sequence | dual | 5 | 8.4 | 401.8 | 23.1% | 63.4% | dual_c3_bank25 |
| `s10_bal_bank25` | ? | ? | ? | ? | 8.4 | 428.9 | 23.3% | 63.0% | ? |
| `s10_n400k_bank25_cont` | s10 | sequence | dual | 3 | 8.6 | 430.1 | 23.1% | 62.6% | dual_c3_bank25 |
| `s10_pool` | ? | ? | ? | ? | 8.4 | 419.6 | 23.5% | 62.6% | ? |
| `s10_n400k_bank25` | s10 | sequence | dual | 1 | 10.1 | 510.2 | 22.0% | 60.0% | dual_c3_bank25 |
| `s10_bal` | s10 | sequence | dual | 2 | 48.7 | 604.1 | 3.6% | 40.9% | dual_bal |
| `s10_key_dual` | s10 | sequence | dual | 2 | 54.0 | 629.5 | 3.4% | 40.0% | dual_c3 |
| `s10_n400k_e2` | s10 | sequence | dual | 2 | 55.8 | 644.1 | 3.2% | 39.6% | dual_c3 |
| `s10_key_pos` | s10 | sequence | pos | 2 | 62.5 | 667.1 | 3.1% | 37.9% | dual_c3 |
| `s10_key_scalar` | s10 | sequence | scalar | 2 | 73.5 | 676.8 | 3.5% | 36.0% | dual_c3 |
| `s10_key_cond` | s10 | sequence | cond | 2 | 78.5 | 722.2 | 2.9% | 35.8% | dual_c3 |
| `s10_n200k_e5` | s10 | sequence | dual | 3 | 88.5 | 786.9 | 3.1% | 35.2% | dual_c3 |
| `s10_n100k_e10` | s10 | sequence | dual | 9 | 88.6 | 786.1 | 2.5% | 33.9% | dual_c3 |
| `s10_n50k_e80` | s10 | sequence | dual | 3 | 103.6 | 950.4 | 3.1% | 33.3% | dual_c3 |
| `s10_n50k_e19` | s10 | sequence | dual | 2 | 109.1 | 928.3 | 3.2% | 33.2% | dual_c3 |
| `s10_n25k_e38` | s10 | sequence | dual | 2 | 116.6 | 1038.9 | 3.1% | 32.0% | dual_c3 |
| `retr_lkey` | s01 | sequence | pos | 6 | 229.2 | 1077.2 | 0.5% | 19.2% | dual_c3 |
| `retr_dual` | s01 | sequence | dual | 6 | 234.7 | 1067.0 | 0.6% | 19.0% | dual_c3 |
| `retr_gate` | s01 | sequence | cond | 6 | 236.4 | 1070.4 | 0.5% | 19.0% | dual_c3 |
| `s10_key_none` | s10 | sequence | none | 2 | 170.2 | 853.5 | 0.2% | 18.7% | dual_c3 |
| `retr_seq` | s01 | sequence | None | 6 | 227.2 | 1074.3 | 0.6% | 18.5% | dual_c3 |
| `retr_dual32` | s01 | sequence | dual | 6 | 231.7 | 1090.6 | 0.6% | 18.0% | dual_c3 |
| `s10_n25k_bank25k` | s10 | sequence | dual | 1 | 469.0 | 1677.6 | 0.6% | 12.6% | dual_c3 |
| `s10_cell8_bank25_cont` | s10 | cell8 | dual | 3 | 261.5 | 1084.9 | 0.1% | 8.0% | dual_c3_bank25 |
| `s10_cell8_bank25_lr1e4_c` | ? | ? | ? | ? | 247.5 | 1024.5 | 0.1% | 7.7% | ? |
| `s10_cell8_bank25_lr3e4` | s10 | cell8 | dual | 1 | 294.4 | 1171.1 | 0.1% | 7.6% | dual_c3_bank25 |
| `s10_pool_c8_c4` | ? | ? | ? | ? | 245.8 | 1019.2 | 0.1% | 7.6% | ? |
| `s10_pool_c8_c6` | ? | ? | ? | ? | 243.3 | 1020.6 | 0.1% | 7.6% | ? |
| `s10_cell8_bank25_lr1e4` | ? | ? | ? | ? | 249.0 | 1046.5 | 0.1% | 7.5% | ? |
| `s10_cell8_bank25` | s10 | cell8 | dual | 1 | 288.7 | 1156.2 | 0.1% | 7.5% | dual_c3_bank25 |
| `s10_pool_c8` | ? | ? | ? | ? | 260.1 | 1057.1 | 0.1% | 7.1% | ? |
| `s10_cell8_base_cont` | s10 | cell8 | dual | 3 | 267.5 | 1054.5 | 0.0% | 6.0% | dual_c3 |
| `s10_cell8_base` | s10 | cell8 | dual | 1 | 298.5 | 1145.5 | 0.0% | 5.8% | dual_c3 |

**Headline.** Best by hit rate is `d1536-b350-e6-drop30`: 1.8 km median, 76.9% under 25 km.

**Quote this one externally instead.** The best `cell8` arm is `s10_cell8_bank25_cont`: 261.5 km median, 8.0% under 25 km. `cell8` holds whole z8 cells out of training *and* filters the bank to the same cells, so neither the weights nor the corpus has seen the region -- which is the condition the OSV-5M protocol enforces and the `sequence` split does not.

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

### `BOOTSTRAP_best.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25_c4 vs s10_bal_bank25_c6 | [+0.0, +1.1] km | separated | [-1.58, +0.10] pp | inside noise |

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

### `BOOTSTRAP_seed.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_key_dual vs s10_n400k_e2 | [-6.0, +1.5] km | inside noise | [-0.28, +1.22] pp | inside noise |

### `BOOTSTRAP_balbank.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25_cont vs s10_bal_bank25 | [-0.3, +0.8] km | inside noise | [-1.24, +0.40] pp | inside noise |

### `BOOTSTRAP_balbank_ep2.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25 vs s10_bal_bank25 | [+1.1, +2.4] km | separated | [-3.84, -2.16] pp | separated |

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

### `BOOTSTRAP_cont.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25 vs s10_n400k_bank25_cont | [+0.9, +2.1] km | separated | [-3.34, -1.88] pp | separated |

### `BOOTSTRAP_cont4.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25_cont vs s10_n400k_bank25_c4 | [-0.0, +0.6] km | inside noise | [-1.36, -0.32] pp | separated |

### `BOOTSTRAP_encgate.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_bal_gate | [-0.3, +0.4] km | inside noise | [-0.68, +0.36] pp | inside noise |
| s10_bal_bank25_c6 vs s10_bal_c8 | [-0.3, +0.4] km | inside noise | [-0.88, +0.10] pp | inside noise |
| s10_bal_gate vs s10_bal_c8 | [-0.4, +0.4] km | inside noise | [-0.80, +0.30] pp | inside noise |

### `BOOTSTRAP_pool.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_pool | [-1.2, -0.1] km | separated | [+0.70, +2.36] pp | separated |
| s10_bal_bank25_c6 vs s10_pool_c4 | [-0.6, +0.5] km | inside noise | [-0.58, +1.04] pp | inside noise |
| s10_bal_bank25_c6 vs s10_pool_c6 | [-0.5, +0.7] km | inside noise | [-0.74, +0.94] pp | inside noise |
| s10_pool vs s10_pool_c4 | [+0.2, +1.1] km | separated | [-1.86, -0.74] pp | separated |
| s10_pool vs s10_pool_c6 | [+0.3, +1.3] km | separated | [-2.14, -0.76] pp | separated |
| s10_pool_c4 vs s10_pool_c6 | [-0.2, +0.5] km | inside noise | [-0.64, +0.34] pp | inside noise |

### `BOOTSTRAP_geomem.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_bal_c8 | [-0.3, +0.4] km | inside noise | [-0.90, +0.10] pp | inside noise |
| s10_bal_bank25_c6 vs s10_geo_bias | [-0.2, +0.6] km | inside noise | [-0.94, +0.12] pp | inside noise |
| s10_bal_bank25_c6 vs s10_geo_key | [-0.2, +0.5] km | inside noise | [-0.76, +0.24] pp | inside noise |
| s10_bal_c8 vs s10_geo_bias | [-0.2, +0.5] km | inside noise | [-0.52, +0.52] pp | inside noise |
| s10_bal_c8 vs s10_geo_key | [-0.2, +0.4] km | inside noise | [-0.26, +0.54] pp | inside noise |
| s10_geo_bias vs s10_geo_key | [-0.5, +0.3] km | inside noise | [-0.42, +0.68] pp | inside noise |

### `BOOTSTRAP_pool_cell8.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8 | [-23.5, -3.2] km | separated | [+0.04, +1.08] pp | separated |
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8_c4 | [-7.9, +10.3] km | inside noise | [-0.40, +0.56] pp | inside noise |
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8_c6 | [-5.1, +12.6] km | inside noise | [-0.44, +0.56] pp | inside noise |
| s10_pool_c8 vs s10_pool_c8_c4 | [+6.9, +21.7] km | separated | [-0.86, -0.14] pp | separated |
| s10_pool_c8 vs s10_pool_c8_c6 | [+9.0, +25.2] km | separated | [-0.92, -0.08] pp | separated |
| s10_pool_c8_c4 vs s10_pool_c8_c6 | [-2.4, +8.5] km | inside noise | [-0.34, +0.32] pp | inside noise |

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

### `BOOTSTRAP_w768.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_b40_c6 vs s10_b55_c6 | [+0.8, +1.3] km | separated | [-4.00, -2.36] pp | separated |
| s10_b40_c6 vs s10_w768 | [+0.1, +0.7] km | separated | [-0.88, +1.00] pp | inside noise |
| s10_b55_c6 vs s10_w768 | [-0.9, -0.5] km | separated | [+2.48, +4.04] pp | separated |

### `BOOTSTRAP_w768_matched.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_b55 vs s10_w768 | [-0.7, -0.2] km | separated | [+1.46, +2.94] pp | separated |

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

### `BOOTSTRAP_w768_means.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b265-e4` | 768-d / 2.65M bank / 4 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b265-e4 vs d768-b265-e6 | [-0.0, +0.2] km | inside noise | [-0.96, +0.06] pp | inside noise |

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

### `BOOTSTRAP_sub2.md`

| arm | what it is |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
| d768-b350-e6-drop70 vs d768-b350-e6-drop70-sub2 | [-0.4, -0.0] km | separated | [-1.18, +0.32] pp | inside noise |

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

## Stages

212 stages completed: `b40_2`, `b40_4`, `b40_6`, `b40_eval`, `b40_knn`, `b40_meta`, `b40_stack`, `b55_2`, `b55_4`, `b55_6`, `b55_eval`, `b55_knn`, `b55_meta`, `b55_stack`, `b70_2`, `b70_4`, `b70_6`, `b70_eval`, `b70_knn`, `b70_meta`, `b70_stack`, `bank25_train`, `boot-d1536-drop30`, `boot-d1536-drop70`, `boot-d768-b350-drop30`, `boot-d768-b350`, `boot-pcurve`, `boot-pcurve2`, `boot-pcurve3`, `boot-sub2`, `bx_concat`, `bx_dino`, `bx_knn`, `bx_meta`, `bx_siglip`, `bx_stack`, `c8_eval`, `c8_eval_cont`, `c8_knn_base`, `c8_knn_ext`, `c8_train_base`, `c8_train_base_cont`, `c8_train_ext`, `c8_train_ext_cont`, `calib`, `concat`, `dataset`, `e3_dino`, `e3_join`, `e3_meta`, `e3_pool`, `e3_siglip`, `e4_dino`, `e4_join`, `e4_meta`, `e4_pool`, `e4_siglip`, `eg_ctrl`, `eg_eval`, `eg_gate`, `embed_dino`, `embed_siglip`, `ext2_dino`, `ext2_join`, `ext2_meta`, `ext2_pool`, `ext2_siglip`, `fh_big`, `fh_combo`, `fh_pos1`, `fh_pos1m`, `fh_seed1`, `fh_tau`, `fu_bal4`, `fu_bal4_eval`, `fu_bal6`, `fu_bal6_eval`, `fuse-attn-pyr47`, `gm_bias`, `gm_eval`, `gm_key`, `hrfix-b265`, `hrfix-b350`, `hrfix-d1536`, `hrfix-d1536drop30`, `hrfix-d1536drop70`, `hrfix-drop10`, `hrfix-drop30`, `hrfix-drop50`, `hrfix-drop70`, `hrfix-drop90`, `hrfix-sub2`, `hrfix-x-m265-b350`, `hrfix-x-m350-b265`, `kartaview-d1536-b350-e6-n5k`, `kartaview-d1536-drop30-n5k`, `kartaview-d768-b265-e6-n5k`, `kartaview-d768-b265-e6`, `kartaview-d768-b350-e6-n5k`, `kartaview-d768-b350-e6`, `kartaview-drop10-n5k`, `kartaview-drop30-n5k`, `kartaview-drop50-n5k`, `kartaview-drop70-n5k`, `kartaview-drop90-n5k`, `kartaview-sub2-n5k`, `kartaview-x-m265-b350`, `kartaview-x-m350-b265`, `key_eval`, `key_s01_eval`, `key_seed`, `key_train_cond`, `key_train_dual`, `key_train_none`, `key_train_pos`, `key_train_scalar`, `knn`, `kv_screen`, `mar_bal_concat`, `mar_bal_eval`, `mar_bal_knn`, `mar_bal_train`, `mar_balbank_eval`, `mar_balbank_knn`, `mar_balbank_stack`, `mar_balbank_train`, `mar_balext_concat`, `mar_c8_cont`, `mar_c8_eval`, `mar_cont4`, `mar_cont4_eval`, `mar_tile_probe`, `mqfix-drop70`, `mqfix-pow4-rr`, `multiq-d768-b350-e6-diverse-global`, `multiq-d768-b350-e6-diverse-rr`, `multiq-d768-b350-e6-diverse`, `multiq-d768-b350-e6-near-global`, `multiq-d768-b350-e6-near-rr`, `multiq-d768-b350-e6-near`, `multiq-d768-b350-e6-pow4-rr`, `multiq-drop30-pow4-rr`, `pb_2`, `pb_4`, `pb_6`, `pb_eval`, `pc8_a`, `pc8_b`, `pc8_c`, `pc8_eval`, `pc8_knn`, `pyrcache-hr47k`, `rc_c8_eval`, `rc_c8_lr1`, `rc_c8_lr3`, `rc_cont`, `rc_cont_eval`, `rc_depth1`, `rc_depth2`, `rc_depth3`, `rc_depth4`, `rc_width`, `s01_km`, `s01_km_eval`, `s10_n100k_e10`, `s10_n200k_e5`, `s10_n25k_e38`, `s10_n25k_e38_eval_test`, `s10_n400k_e2`, `s10_n50k_e19`, `s10_n50k_e80`, `tiles`, `tiles2_probe`, `train-d1536-b350-e2-drop30`, `train-d1536-b350-e2-drop70`, `train-d1536-b350-e4-drop30`, `train-d1536-b350-e4-drop70`, `train-d1536-b350-e6-drop30`, `train-d1536-b350-e6-drop70`, `train-d768-b350-e2-drop10`, `train-d768-b350-e2-drop30`, `train-d768-b350-e2-drop50`, `train-d768-b350-e2-drop70-sub2`, `train-d768-b350-e2-drop70`, `train-d768-b350-e2-drop90`, `train-d768-b350-e2`, `train-d768-b350-e4-drop10`, `train-d768-b350-e4-drop30`, `train-d768-b350-e4-drop50`, `train-d768-b350-e4-drop70-sub2`, `train-d768-b350-e4-drop70`, `train-d768-b350-e4-drop90`, `train-d768-b350-e4`, `train-d768-b350-e6-drop10`, `train-d768-b350-e6-drop30`, `train-d768-b350-e6-drop50`, `train-d768-b350-e6-drop70-sub2`, `train-d768-b350-e6-drop70`, `train-d768-b350-e6-drop90`, `train-d768-b350-e6`, `w70_2`, `w70_4`, `w70_6`, `w70_eval`, `w70_knn`, `w70_proj`, `w768_2`, `w768_4`, `w768_6`, `w768_eval`, `w768_knn`, `w768_proj`

Skipped or failed:

```
[06:25:04] skip   train-d768-b350-e6-drop90  (already done)
[06:25:04] skip   kartaview-drop90-n5k  (already done)
[06:25:04] skip   boot-pcurve3  (already done)
[06:25:04] skip   train-d768-b350-e2-drop70-sub2  (already done)
[06:25:04] skip   train-d768-b350-e4-drop70-sub2  (already done)
[06:25:04] skip   train-d768-b350-e6-drop70-sub2  (already done)
[06:25:04] skip   kartaview-sub2-n5k  (already done)
[06:25:04] skip   boot-sub2  (already done)
[06:25:04] skip   pyrcache-hr47k  (already done)
[06:25:04] skip   hrfix-b265  (already done)
[06:25:04] skip   hrfix-b350  (already done)
[06:25:04] skip   hrfix-drop10  (already done)
[06:25:04] skip   hrfix-drop30  (already done)
[06:25:04] skip   hrfix-drop50  (already done)
[06:25:04] skip   hrfix-drop70  (already done)
[06:25:04] skip   hrfix-drop90  (already done)
[06:25:04] skip   hrfix-sub2  (already done)
[06:25:04] skip   hrfix-d1536  (already done)
[06:25:04] skip   hrfix-d1536drop30  (already done)
[06:50:32] skip   train-d1536-b350-e2-drop70  (already done)
[06:50:32] skip   train-d1536-b350-e4-drop70  (already done)
[06:50:32] skip   train-d1536-b350-e6-drop70  (already done)
[06:50:32] skip   boot-d1536-drop70  (already done)
[06:50:32] skip   hrfix-d1536drop70  (already done)
[06:50:32] skip   fuse-attn-pyr47  (already done)
```
