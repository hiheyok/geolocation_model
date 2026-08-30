# Overnight run: data scale vs epochs

Selection criterion is greedy-decode **val median km**, not val loss. Beam ranking is on s0-s2 (`--score-steps 3`), the shipping depth.


## Arms

| arm | release | images | epochs | steps | selected ep | val km (greedy, sel) | test km k=2 | test <25km | s/epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `s01_km` | s01 | all | 6 | - | 6 | 268.5 | n/a | n/a | nan |
| `s10_n100k_e10` | s10 | 100,000 | 10 | 15,625 | 9 | 53.6 | n/a | n/a | nan |
| `s10_n200k_e5` | s10 | 200,000 | 5 | 15,625 | 3 | 49.9 | n/a | n/a | nan |
| `s10_n25k_bank25k` | s10 | all | 1 | - | 1 | 388.1 | n/a | n/a | nan |
| `s10_n25k_e38` | s10 | 25,000 | 38 | 14,843 | 2 | 81.9 | 72.6 | 38.8% | nan |
| `s10_n400k_e2` | s10 | 400,000 | 2 | 12,500 | 2 | 52.2 | n/a | n/a | nan |
| `s10_n50k_e19` | s10 | 50,000 | 19 | 14,843 | 2 | 77.9 | n/a | n/a | nan |
| `s10_n50k_e80` | s10 | 50,000 | 80 | 62,500 | 3 | 54.9 | n/a | n/a | nan |

## Beam width sweep (val)

| arm | k=1 | k=2 | k=4 |
|---|---:|---:|---:|

## System utilisation by stage

| stage | samples | GPU % (peak) | VRAM MB | shared MB | W | degC | CPU % | RAM GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bx_concat | 3 | 1 (1) | 983 | 139 | 11 | 39 | 11 | 21.7 |
| bx_dino | 543 | 1 (1) | 5196 | 200 | 11 | 39 | 18 | 21.2 |
| bx_knn | 1 | 1 (1) | 982 | nan | 11 | 39 | 11 | 12.3 |
| bx_siglip | 464 | 1 (1) | 4437 | 213 | 11 | 39 | 18 | 19.9 |
| bx_stack | 9 | 1 (1) | 983 | 112 | 11 | 39 | 10 | 30.5 |
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
| bx_meta | 0.2 |
| bx_dino | 93.3 |
| bx_siglip | 80.1 |
| bx_concat | 0.5 |
| bx_stack | 2.2 |

**Not completed:** bx_knn (rc=1)

