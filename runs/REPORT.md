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

### Retrieval head ablation, re-run with the keys live

`scripts/keys.py`, 2026-08-31. Five arms, identical except for the retrieval
head: 400,000 training images, 2 epochs, sequence split, 400,180-image bank.
Test split, 5,000 seeded-random images, k=2, ranked s0-s2.

| arm | `--retr-mode` | median km | mean km | `<25 km` |
|---|---|---:|---:|---:|
| `s10_key_none` | *no retrieval at all* | 170.2 | 853.5 | 18.7% |
| `s10_key_scalar` | `scalar` | 73.5 | 676.8 | 36.0% |
| `s10_key_cond` | `cond` | 78.5 | 722.2 | 35.8% |
| `s10_key_pos` | `pos` | 62.5 | 667.1 | 37.9% |
| `s10_key_dual` | `dual` | **54.0** | **629.5** | **40.0%** |

Paired 95% intervals on `<25 km`:

| contrast | | |
|---|---|---|
| none -> scalar | **+17.3 pp** [+15.9, +18.6] | separated |
| scalar -> cond | -0.2 pp [-0.9, +0.6] | inside noise |
| scalar -> pos | **+1.9 pp** [+1.1, +2.8] | separated |
| pos -> dual | **+2.1 pp** [+1.3, +3.0] | separated |
| scalar -> dual | **+4.1 pp** [+3.1, +5.0] | separated |

Three things this settles, none of which the previous ablation could see, because
it ran while `beam.search` was dropping the keys and therefore decoded `pos` and
`dual` identically to `cond`:

1. **The retrieval prior is most of the system.** A single zero-initialised
   scalar gate is worth +17.3 pp and halves the median. Everything else on this
   page is a rounding error next to it.
2. **The learned keys do pay, and the ladder is monotone.** `pos` beats `scalar`
   and `dual` beats `pos`, both separated. The old conclusion -- that the keys
   were noise and `w_pos` was an unidentified parameter -- was an artefact of
   evaluating a model without the parameter it was trained with. `w_pos` reaches
   1.29-1.44 here; only its *sign* is unidentifiable, and for a real reason:
   negating `w_pos` and negating `q_pos` is the same function.
3. **`cond` alone buys nothing.** Per-step, quality-modulated gates tie with one
   scalar on the hit rate and are slightly worse on median. `cond` is only worth
   carrying as the substrate the keyed branches sit on.

`g_neg` came out `[+0.41, +0.81, +1.26, +0.25]` -- **positive at step 0**, where
every pre-fix run had it negative. That sign flip is the negative branch finally
being used to reject rather than being inverted into a second proposal signal,
which is what it was designed for and had never been credited with.

### One same-config replicate

`s10_key_dual` repeats `s10_n400k_e2`'s configuration exactly and selected the
same epoch, so the pair differs only in run-to-run randomness:

| | median km | `<25 km` |
|---|---:|---:|
| `s10_key_dual` | 54.0 | 40.0% |
| `s10_n400k_e2` | 55.8 | 39.6% |

Paired: median [-6.0, +1.5] km, `<25 km` [-0.28, +1.22] pp -- inside noise on
both. Two runs is not a variance estimate, but it does bound this replicate, and
the +1.9 and +2.1 pp steps in the ladder above sit outside it. It is also much
tighter than the 18 km spread seen between replicates at 50k, which suggests
seed sensitivity falls with training-set size.

### Overnight of 2026-08-31 --- tuning, and two negative results

Everything below is the test split, 5,000 seeded-random images, k=2, ranked on
s0-s2. `scripts/recheck.py`, `scripts/marathon.py`, `scripts/followup.py` and
`scripts/tiles2.py`; 45 of 48 stages, the three failures a bad `--bank-ext`
argument that was fixed and rerun. Generated per-arm tables in `runs/FINAL.md`
and `runs/ARMS.md`.

**The compute curve, re-measured with the retrieval keys live.**

| beam k | median km | mean km | `<25 km` | s / 5,000 |
|---:|---:|---:|---:|---:|
| 1 | 11.3 | 526.4 | 58.5% | 16.2 |
| 2 | 10.1 | 510.2 | 60.0% | 13.5 |
| 4 | 9.8 | 542.8 | 60.2% | 22.5 |
| 8 | 9.8 | 550.3 | 60.2% | 41.4 |
| 16 | 9.8 | 549.8 | 60.2% | 74.2 |

The recorded finding was that the curve *slopes down* past k=2, which was a
direct problem for a thesis whose headline figure is accuracy against test-time
compute. It does not: it rises and flattens, saturating at k=4. **Saturation,
not degradation.** Ranking depth re-swept at k=4 confirms `--score-steps 3`
(58.5, 59.3, **60.2**, 60.0% at depths 1-4), though depth 3 vs 4 is inside
noise and depth 4 is better on `<1 km`. Some of this is the keys fix and some is
a better-calibrated model; this run cannot separate them.

**Encoder blend.** `concat_street.py` normalised only the joined vector, so the
two encoders entered the cosine at their raw activation norms: 82.96 against
20.58, an 81/19 split nobody chose. Equal-norm is worth **+3.0 pp**
[+2.16, +3.84] at matched epochs and roughly two epochs of training at
convergence, where the unbalanced arm catches up to within noise.

**Continuation.** Two epochs was never checked against three. Gains halve each
step and the mean keeps improving after the median stops:

| arm | epochs | median km | mean km | `<25 km` |
|---|---:|---:|---:|---:|
| `s10_n400k_bank25` | 2 | 10.1 | 510.2 | 60.0% |
| `s10_bal_bank25` | 2 | 8.4 | 428.9 | 63.0% |
| `s10_bal_bank25_c4` | 4 | 7.8 | 408.2 | 63.6% |
| `s10_bal_bank25_c6` | 6 | **7.7** | **386.9** | **64.2%** |

**cell8 was mistuned, and is now converged.** 3e-4 gives 294.4 km, 1e-4 gives
247.5 ([+32.8, +56.5] km, separated); two further epochs then changed nothing.
The best cell8 arm is **247.5 km / 7.66%**, and that is the figure to quote
externally.

**Negative: image tiling.** `preprocess` scales the short side to 224 before
cropping, so a 682x512 frame loses 2.3x of linear resolution. A 3x2 grid of 224
tiles doubles the pixels at near-native resolution -- and does not help. At a
118,500-image bank, 3,000 queries:

| scheme | dims | top-1 `<25 km` | any-of-32 `<25 km` |
|---|---:|---:|---:|
| crop3 | 4608 | 8.0% | 21.4% |
| tile6 | 9216 | 7.8% | 21.8% |
| crop3 | 512 | **7.7%** | **20.9%** |
| tile6 | 512 | 7.3% | 20.1% |

At equal bank bytes tiling loses, consistently at every compressed width; at
full width it is a wash for twice the storage. Two likely reasons: the 3-crop
scheme takes full-height slices that contain sky, horizon and road -- scene-like
framing the encoders were trained on -- while a tile can be pure road or pure
sky; and at 512-d tile6 is compressed 18x against crop3's 9x, which is why the
equal-bytes gap exceeds the full-width one.

**A methodological note that nearly cost a wrong conclusion.** The first version
of this probe used a 30,000-image bank, where the hit rate is 0.7% -- about ten
successful queries out of 1,500 -- and it showed tiling winning at every width.
The same cached embeddings score 0.67% and 5.47% at bank sizes of 28,500 and
40,000, because each draw changes the queries as well as the corpus. Sampling
variance at that scale dwarfed the effect, and the sign flipped once the bank
was large enough to resolve it.

**Offline compression, measured but not yet shipped.** int8 with a single global
scale halves the bank (10.60 -> 5.30 GB) with the top-1 neighbour unchanged in
512/512 queries, recall@32 0.9945, and reconstruction cosine 0.99981 -- and there
is no outlier-channel problem, the worst column being 2.0x the median. PCA to
512-d preserves geographic retrieval quality while recall@32 against the full
space falls to 0.82, so the discarded dimensions carry visual detail with no
geographic content. Both matter because bank RAM is what caps corpus size.

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
| mar_bal_concat | 5 | 1 (1) | 735 | 212 | 11 | 39 | 20 | 13.9 |
| mar_bal_eval | 2 | 1 (1) | 1121 | 376 | 11 | 39 | 26 | 9.9 |
| mar_bal_knn | 8 | 1 (1) | 4265 | 214 | 11 | 39 | 22 | 18.5 |
| mar_bal_train | 143 | 1 (1) | 3561 | 373 | 11 | 39 | 23 | 20.2 |
| mar_balbank_eval | 2 | 1 (1) | 1262 | 376 | 11 | 39 | 28 | 10.6 |
| mar_balbank_knn | 24 | 1 (1) | 3247 | 203 | 11 | 39 | 21 | 20.2 |
| mar_balbank_stack | 8 | 1 (1) | 735 | 128 | 11 | 39 | 19 | 26.7 |
| mar_balbank_train | 143 | 1 (1) | 3561 | 376 | 11 | 39 | 23 | 25.8 |
| mar_balext_concat | 8 | 1 (1) | 735 | 126 | 11 | 39 | 19 | 19.1 |
| mar_c8_cont | 151 | 1 (1) | 3546 | 380 | 11 | 39 | 23 | 24.9 |
| mar_c8_eval | 2 | 1 (1) | 1262 | 378 | 11 | 39 | 27 | 10.3 |
| mar_cont4 | 143 | 1 (1) | 3545 | 371 | 11 | 39 | 24 | 26.0 |
| mar_cont4_eval | 2 | 1 (1) | 1105 | 294 | 11 | 39 | 27 | 9.9 |
| mar_tile_probe | 281 | 1 (1) | 1899 | 197 | 11 | 39 | 41 | 11.5 |
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
