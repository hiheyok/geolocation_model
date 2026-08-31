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
| key_eval | 8 | 1 (1) | 1717 | 352 | 11 | 39 | 22 | 13.9 |
| key_s01_eval | 10 | 1 (1) | 2265 | 368 | 11 | 39 | 35 | 12.7 |
| key_train_cond | 156 | 1 (1) | 4383 | 497 | 11 | 39 | 22 | 23.0 |
| key_train_dual | 148 | 1 (1) | 4004 | 455 | 11 | 39 | 23 | 24.3 |
| key_train_none | 154 | 1 (1) | 4269 | 526 | 11 | 39 | 22 | 21.6 |
| key_train_pos | 156 | 1 (1) | 4372 | 475 | 11 | 39 | 23 | 23.8 |
| key_train_scalar | 154 | 1 (1) | 4380 | 508 | 11 | 39 | 22 | 22.6 |
| knn | 8 | 84 (96) | 6643 | 184 | 196 | 57 | 20 | 19.0 |
| rc_c8_eval | 4 | 1 (1) | 1214 | 206 | 11 | 39 | 28 | 10.3 |
| rc_c8_lr1 | 297 | 1 (1) | 3566 | 379 | 11 | 39 | 23 | 26.2 |
| rc_c8_lr3 | 303 | 1 (1) | 3574 | 384 | 11 | 39 | 31 | 26.7 |
| rc_cont | 146 | 1 (1) | 3507 | 380 | 11 | 39 | 37 | 25.9 |
| rc_cont_eval | 4 | 1 (1) | 1060 | 404 | 11 | 39 | 100 | 13.5 |
| rc_depth1 | 4 | 1 (1) | 1361 | 215 | 11 | 39 | 36 | 9.6 |
| rc_depth2 | 3 | 1 (1) | 1570 | 215 | 11 | 39 | 34 | 10.0 |
| rc_depth3 | 4 | 1 (1) | 1361 | 214 | 11 | 39 | 35 | 9.7 |
| rc_depth4 | 3 | 1 (1) | 1570 | 213 | 11 | 39 | 35 | 10.0 |
| rc_width | 16 | 1 (1) | 2519 | 215 | 11 | 39 | 35 | 10.7 |
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
| key_s01_eval | 1.9 |
| key_train_none | 26.8 |
| key_train_scalar | 26.4 |
| key_train_cond | 26.8 |
| key_train_pos | 26.8 |
| key_train_dual | 25.5 |
| key_eval | 1.3 |
| key_seed | 0.1 |
| rc_width | 2.9 |
| rc_depth1 | 0.6 |
| rc_depth2 | 0.6 |
| rc_depth3 | 0.6 |
| rc_depth4 | 0.6 |
| rc_cont | 25.0 |
| rc_cont_eval | 0.7 |
| rc_c8_lr3 | 52.2 |
| rc_c8_lr1 | 50.9 |
| rc_c8_eval | 0.6 |
