# Overnight run: data scale vs epochs

> **Corrected 2026-08-30.** Every rollout number in this file was first measured
> while `beam.search` was dropping the learned retrieval keys, and against a
> *leading* rather than random sample of the test split. Both are fixed; the
> `test km` / `test <25km` columns below are re-measured on the same weights, on
> 5,000 seeded-random test images. Anything else here that came from a beam
> rollout -- the per-step rollout accuracies in `SUMMARY.md` in particular --
> predates the fix and is not comparable to them.

Selection criterion is greedy-decode **val median km**, not val loss. Beam ranking is on s0-s2 (`--score-steps 3`), the shipping depth.


## Arms

| arm | release | images | epochs | steps | selected ep | val km (greedy, sel) | test km k=2 | test <25km | s/epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `s01_km` | s01 | all | 6 | - | 6 | 268.5 | n/a | n/a | nan |
| `s10_n100k_e10` | s10 | 100,000 | 10 | 15,625 | 9 | 53.6 | 88.6 | 33.9% | nan |
| `s10_n200k_e5` | s10 | 200,000 | 5 | 15,625 | 3 | 49.9 | 88.5 | 35.2% | nan |
| `s10_n25k_bank25k` | s10 | all | 1 | - | 1 | 388.1 | 469.0 | 12.6% | nan |
| `s10_n25k_e38` | s10 | 25,000 | 38 | 14,843 | 2 | 81.9 | 116.6 | 32.0% | nan |
| `s10_n400k_e2` | s10 | 400,000 | 2 | 12,500 | 2 | 52.2 | 55.8 | 39.6% | nan |
| `s10_n50k_e19` | s10 | 50,000 | 19 | 14,843 | 2 | 77.9 | 109.1 | 33.2% | nan |
| `s10_n50k_e80` | s10 | 50,000 | 80 | 62,500 | 3 | 54.9 | 103.6 | 33.3% | nan |

### Arms from the bank-extension and cell8 runs

Run by `scripts/bank25.py` and `scripts/cell8.py`, which keep their own state, so
`scripts/report.py` does not see them. Same protocol: test split, 5,000
seeded-random images, k=2, ranked on s0-s2.

| arm | split | training images | bank | median km | mean km | `<25 km` |
|---|---|---:|---:|---:|---:|---:|
| `s10_n400k_bank25` | sequence | 400,000 | 1,150,180 | 10.1 | 510.2 | 60.0% |
| `s10_cell8_base` | cell8 | 400,000 | 400,180 | 298.5 | 1145.5 | 5.8% |
| `s10_cell8_base_cont` | cell8 | 400,000 | 400,180 | 267.5 | 1054.5 | 6.0% |
| `s10_cell8_bank25` | cell8 | 400,000 | 1,044,404 | 288.7 | 1156.2 | 7.5% |
| `s10_cell8_bank25_cont` | cell8 | 400,000 | 1,044,404 | 261.5 | 1084.9 | 8.0% |

`_cont` continues its base arm for two further epochs from the saved weights at
lr 1e-4 (`--init`). Both cell8 arms improved monotonically under that, which says
the original decline was the cosine's high-LR phase rather than overfitting, and
that these arms are mistuned rather than at their ceiling.

## Beam width sweep (val)

| arm | k=1 | k=2 | k=4 |
|---|---:|---:|---:|

## System utilisation by stage

| stage | samples | GPU % (peak) | VRAM MB | shared MB | W | degC | CPU % | RAM GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bank25_eval | 1 | 1 (1) | 1211 | 365 | 11 | 39 | 14 | 11.6 |
| bank25_train | 151 | 1 (1) | 4035 | 374 | 11 | 39 | 16 | 29.9 |
| bx_concat | 3 | 1 (1) | 983 | 139 | 11 | 39 | 11 | 21.7 |
| bx_dino | 543 | 1 (1) | 5196 | 200 | 11 | 39 | 18 | 21.2 |
| bx_knn | 26 | 1 (1) | 3291 | 210 | 11 | 39 | 14 | 25.2 |
| bx_siglip | 464 | 1 (1) | 4437 | 213 | 11 | 39 | 18 | 19.9 |
| bx_stack | 9 | 1 (1) | 983 | 112 | 11 | 39 | 10 | 30.5 |
| c8_eval | 8 | 1 (1) | 1306 | 204 | 11 | 39 | 13 | 11.0 |
| c8_eval_cont | 4 | 1 (1) | 1391 | 307 | 11 | 39 | 16 | 11.1 |
| c8_knn_base | 9 | 1 (1) | 4130 | 158 | 11 | 39 | 12 | 22.9 |
| c8_knn_ext | 23 | 1 (1) | 3819 | 209 | 11 | 39 | 12 | 24.6 |
| c8_train_base | 161 | 1 (1) | 3884 | 374 | 11 | 39 | 14 | 22.9 |
| c8_train_base_cont | 153 | 1 (1) | 3675 | 347 | 11 | 39 | 14 | 21.0 |
| c8_train_base_e4 | 76 | 1 (1) | 3673 | 340 | 11 | 39 | 14 | 19.2 |
| c8_train_ext | 160 | 1 (1) | 3744 | 364 | 11 | 39 | 16 | 25.6 |
| c8_train_ext_cont | 157 | 1 (1) | 3683 | 359 | 11 | 39 | 16 | 26.0 |
| calib | 51 | 1 (7) | 2465 | 259 | 13 | 39 | 25 | 22.5 |
| concat | 3 | 1 (2) | 705 | 127 | 13 | 44 | 16 | 21.5 |
| embed_dino | 325 | 79 (100) | 4763 | 186 | 185 | 62 | 19 | 19.6 |
| embed_siglip | 285 | 68 (100) | 4008 | 182 | 180 | 61 | 20 | 19.6 |
| knn | 8 | 84 (96) | 6643 | 184 | 196 | 57 | 20 | 19.0 |
| s01_km | 149 | 92 (98) | 3954 | 357 | 186 | 62 | 18 | 30.1 |
| s01_km_eval | 12 | 24 (93) | 3586 | 257 | 58 | 45 | 25 | 15.5 |
| s10_n100k_e10 | 368 | 1 (1) | 3482 | 356 | 11 | 39 | 18 | 18.6 |
| s10_n200k_e5 | 186 | 1 (1) | 3482 | 360 | 11 | 39 | 17 | 19.3 |
| s10_n25k_e15 | 19 | 1 (1) | 3371 | 367 | 11 | 39 | 24 | 15.8 |
| s10_n25k_e26 | 6 | 1 (1) | 3068 | 181 | 11 | 39 | 23 | 13.5 |
| s10_n25k_e38 | 186 | 1 (1) | 3497 | 352 | 11 | 39 | 18 | 16.7 |
| s10_n25k_e38_eval_test | 5 | 1 (1) | 2782 | 251 | 11 | 39 | 24 | 9.2 |
| s10_n25k_e38_eval_val | 1 | 1 (1) | 1059 | 178 | 11 | 39 | 23 | 9.5 |
| s10_n400k_e2 | 143 | 1 (1) | 3473 | 360 | 11 | 39 | 17 | 19.6 |
| s10_n50k_e19 | 176 | 1 (1) | 3495 | 356 | 11 | 39 | 18 | 17.7 |
| s10_n50k_e80 | 766 | 1 (1) | 3508 | 358 | 11 | 39 | 22 | 18.5 |
| tiles | 76 | 1 (2) | 705 | 96 | 11 | 38 | 24 | 17.6 |

`shared MB` is GPU memory that spilled into system DRAM. The desktop baseline here is 230-290 MB; materially above that means a process is over-committed and every access is crossing PCIe while nvidia-smi still reports 100% utilisation.


## Stage timings

| stage | minutes |
|---|---:|
| c8_knn_base | 1.7 |
| c8_knn_ext | 3.9 |
| c8_train_base | 27.6 |
| c8_train_ext | 27.6 |
| c8_eval | 1.4 |
| c8_train_base_cont | 26.4 |
| c8_train_ext_cont | 27.2 |
| c8_eval_cont | 0.7 |
