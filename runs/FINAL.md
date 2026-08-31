# Results digest

Generated 2026-08-31 02:19 by `scripts/digest.py`. Every arm below is the test split, 5,000 seeded-random images, beam k=2, ranked on s0-s2 -- the shipping protocol. Read the paired intervals further down before believing any ordering here: the median carries a ~16 km 95% interval at this sample size.

## Arms, best hit rate first

| arm | rel | split | mode | ep | median km | mean km | `<1km` | `<25km` | street file |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `s10_n400k_bank25_cont` | s10 | sequence | dual | 3 | 8.6 | 430.1 | 23.1% | 62.6% | dual_c3_bank25 |
| `s10_n400k_bank25` | s10 | sequence | dual | 1 | 10.1 | 510.2 | 22.0% | 60.0% | dual_c3_bank25 |
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
| `s10_cell8_bank25` | s10 | cell8 | dual | 1 | 288.7 | 1156.2 | 0.1% | 7.5% | dual_c3_bank25 |
| `s10_cell8_base_cont` | s10 | cell8 | dual | 3 | 267.5 | 1054.5 | 0.0% | 6.0% | dual_c3 |
| `s10_cell8_base` | s10 | cell8 | dual | 1 | 298.5 | 1145.5 | 0.0% | 5.8% | dual_c3 |

**Headline.** Best by hit rate is `s10_n400k_bank25_cont`: 8.6 km median, 62.6% under 25 km.

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

### `BOOTSTRAP_cont.md`

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n400k_bank25 vs s10_n400k_bank25_cont | [+0.9, +2.1] km | separated | [-3.34, -1.88] pp | separated |

## Stages

46 stages completed: `bank25_train`, `bx_concat`, `bx_dino`, `bx_knn`, `bx_meta`, `bx_siglip`, `bx_stack`, `c8_eval`, `c8_eval_cont`, `c8_knn_base`, `c8_knn_ext`, `c8_train_base`, `c8_train_base_cont`, `c8_train_ext`, `c8_train_ext_cont`, `calib`, `concat`, `dataset`, `embed_dino`, `embed_siglip`, `key_eval`, `key_s01_eval`, `key_seed`, `key_train_cond`, `key_train_dual`, `key_train_none`, `key_train_pos`, `key_train_scalar`, `knn`, `rc_cont`, `rc_cont_eval`, `rc_depth1`, `rc_depth2`, `rc_depth3`, `rc_depth4`, `rc_width`, `s01_km`, `s01_km_eval`, `s10_n100k_e10`, `s10_n200k_e5`, `s10_n25k_e38`, `s10_n25k_e38_eval_test`, `s10_n400k_e2`, `s10_n50k_e19`, `s10_n50k_e80`, `tiles`

Skipped or failed:

```
[07:30:23] skip   dataset  (already done)
[07:30:23] skip   embed_dino  (already done)
[07:30:23] skip   embed_siglip  (already done)
[07:30:23] skip   concat  (already done)
[07:30:23] skip   tiles  (already done)
[07:30:23] skip   knn  (already done)
[07:30:23] skip   s01_km  (already done)
[07:30:23] skip   s01_km_eval  (already done)
[07:30:23] skip   calib  (already done)
[07:30:23] skip   s10_n25k_e38  (already done)
[07:30:23] skip   s10_n50k_e19  (already done)
[15:49:04] FAIL   bx_knn  rc=1 after 0.1 min
[15:49:07] FAIL   bx_knn  rc=1 after 0.1 min
[15:49:10] FAIL   bx_knn  rc=1 after 0.1 min
[16:38:43] skip   bx_meta  (already done)
[16:38:43] skip   bx_dino  (already done)
[16:38:43] skip   bx_siglip  (already done)
[16:38:43] skip   bx_concat  (already done)
[16:38:43] skip   bx_stack  (already done)
[17:09:26] FAIL   bank25_eval  rc=1 after 0.1 min
[17:09:30] FAIL   bank25_eval  rc=1 after 0.1 min
[19:09:26] skip   c8_knn_base  (already done)
[19:09:26] skip   c8_knn_ext  (already done)
[19:24:06] skip   c8_knn_base  (already done)
[19:24:06] skip   c8_knn_ext  (already done)
```
