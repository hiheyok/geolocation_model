# Overnight run — data scale, epochs, and what actually carries the win

Run window 03:07–12:15, 30 August 2026. 16 stages, zero failures or drops.
Release **s10** built from OSV-5M shards 00–09: 500,000 images, 400,180 train /
50,032 val / 49,788 test, sequence split, hash `59e4b597281a`.

All numbers below are the **test** split, 5,000 images, beam k=2, ranked on
s0–s2 (the shipping configuration), with 3,000-resample paired bootstrap
intervals. Arms see the same images in the same order, so contrasts are paired.

---

## Headline

| | median km | `<25 km` |
|---|---:|---:|
| `s01_km` — 50k release, everything at 50k | 242.7 | 20.1% |
| `s10_n400k_e2` — 500k release | **81.6** | **35.6%** |

A 3.0× reduction in median error. The rest of this report is about which half
of the scale-up caused it, and the answer is not the one the grid was built to
measure.

---

## 1. It is the retrieval bank, not the training set

The grid varies training-set size with `--limit`, but the kNN bank was built
from every train image regardless — so every arm carried a full 400k
non-parametric memory over a 25k–400k parametric one. The control isolates it:
same 25,000 training images, same config, same schedule, **only the bank
differs**.

| training images | bank | median km | `<25 km` |
|---:|---:|---:|---:|
| 25,000 | 400,180 | 115.8 | 32.5% |
| 25,000 | 25,000 | 465.1 | 13.1% |

Paired 95% CI: **[+18.1, +20.7] pp** on the hit rate, [−372, −327] km on the
median. Separated by a wide margin.

So, decomposed:

- scaling the **bank** 25k → 400k: **+19.4 pp**
- scaling the **training set** 25k → 400k at a fixed full bank: **+3.1 pp**

**The memory is worth about six times the training data.** An independent
cross-check falls out of it: the control (25k bank, 25k train) scores 13.1% and
`s01_km` (40k bank, 40k train) scores 20.1% on its own, easier, fully-evaluated
test split — two different releases landing
where that relationship predicts.

This reframes the standing "data is the binding constraint" finding. Data is
binding, but as *retrieval corpus*, not as gradient signal.

---

## 2. The data-scaling curve is still rising at 400k

Compute-matched: every arm gets ~15,000 optimizer steps at batch 64, so epochs
scale inversely with the training-set size.

| training images | epochs | steps | median km | `<25 km` |
|---:|---:|---:|---:|---:|
| 25,000 | 38 | 14,843 | 115.8 | 32.5% |
| 50,000 | 19 | 14,843 | 109.1 | 33.2% |
| 100,000 | 10 | 15,625 | 88.6 | 33.9% |
| 200,000 | 5 | 15,625 | 88.5 | 35.2% |
| 400,000 | 2 | 12,500 | **81.6** | **35.6%** |

Paired verdicts: 25k↔50k **inside noise**; 50k→100k **separated**;
100k↔200k **inside noise** on the median but **separated** on the hit rate
([+0.36, +2.08] pp); 200k→400k **separated** ([+1.4, +12.5] km, [+0.80, +2.44] pp). The increments are non-monotonic, which is what ~1 pp
resolution around a slowly rising curve looks like — but the endpoints are
separated, so **the curve has not saturated**. 400k also received the *fewest*
steps of any arm, so its ceiling is not measured.

---

## 3. Many epochs on less data do not substitute for data

The non-compute-matched arm: 50,000 images for 80 epochs, 62,500 steps, 4.2× the
budget of every other arm.

| epoch | `<25 km` | median km | val loss |
|---:|---:|---:|---:|
| 3 | **41.2%** | **54.9** | 12.669 |
| 10 | 37.5% | 82.4 | 13.064 |
| 20 | 32.0% | 202.4 | 18.890 |
| 40 | 29.8% | 229.7 | 32.290 |
| 80 | 29.4% | 283.5 | 39.451 |

**It peaks at epoch 3 of 80**; the other 77 are actively destructive. On test it
scores 33.3% against 50k×19's 33.2% -- **inside noise, so 61 extra epochs
bought nothing measurable** -- and loses to 100k×10 (33.9%) on the median
([-23.8, -6.9] km, separated).

**4.2× the compute on fixed data buys less than 2× the data at 1× compute.**
The small arms cannot absorb their budget at all: 25k and 50k peak at epoch 2,
around 780 of their 14,843 steps, and spend the remaining 95% overfitting.

---

## 4. Six defects the 10× exposed, none of which raised an error

Every one degraded silently at 500k and had been harmless-but-latent at 50k.

| defect | at 50k | at 500k |
|---|---|---|
| kNN bank built as fp32 on-card | 0.7 GB, fine | 7.4 GB on an 8 GB card → Windows spills to shared memory and runs the matmul over PCIe |
| dataset pickled to every worker | 1.4 GB, merely slow | 8.5 GB → `OSError 22`; `np.memmap` pickles **by value**, and Windows spawns rather than forks |
| per-epoch val loss pass | 79 steps | 782 steps, against as few as 391 training steps |
| selection on median km | — | selects on noise; see §5 |
| runner re-sized the grid on restart | — | renamed every tag and recomputed finished arms at a different step budget |
| `--select hit` with no rollout | — | criterion NaN every epoch, **no checkpoint ever written**, exit 0, done-marker written |

The last is the worst shape of bug in this project: a stage that reports success
and produces nothing. `train.py` now exits non-zero if it never saved.

---

## 5. Selection: val loss → median km was half a fix

The plan specifies early-stopping on val median km rather than loss. Replacing
the criterion fixed the wrong half.

- **Across epochs within a run, val loss and median km agree.** On `s01_km` they
  correlate **+0.942** and select the same epoch (6). The inherited
  "anti-correlated" finding was an across-*arms* claim, which is a different
  question from the one early stopping decides.
- **The median is far too noisy to select on.** Over 1,000 val images it swung
  82 → 151 → 158 → 213 → 132 → 365 km across consecutive epochs while `<25 km`
  on the same predictions moved smoothly 35.3 → 31.1 → 30.8 → 29.3 → 30.4 →
  26.3%.

`--select hit` is now the default: the `<25 km` rate, median only as tie-break.
This is the project's own error-bar finding applied to model selection.

---

## 6. Reproduction is as noisy as the error bar

`retr_dual` (the 224.7 km headline) and `s01_km` are the same configuration,
release, split and selected epoch. They land **18 km apart** (224.7 vs 242.7),
about the width of the median's own 95% interval at n≈5,000. The `<25 km` rates
agree far better (20.8% vs 20.1%).

Attribution is incomplete: it is either seed variance or a difference in total
`--epochs`, which changes the cosine schedule and which the checkpoint does not
record. **Checkpoints should record the full training argv.**

Consequence: single-seed median comparisons under ~20 km in this project are not
claims. One seed per arm is also the main limit on everything above.

---

## 7. Machine

| stage | GPU | VRAM | shared | W | wall |
|---|---:|---:|---:|---:|---:|
| embed DINOv2 | 79% | 4,763 MB | 186 MB | 185 | 55.1 min |
| embed SigLIP | 68% | 4,008 MB | 182 MB | 180 | 46.8 min |
| tile fetch | 1% | 705 MB | 96 MB | 11 | 13.6 min |
| kNN bank | 84% | 6,643 MB | 184 MB | 196 | 1.4 min |
| training (`s01_km`) | 92% | 3,954 MB | 357 MB | 186 | 25.6 min |

No DRAM spill anywhere: shared memory stays in the 96–357 MB desktop baseline,
including the kNN stage peaking at 6.6 GB dedicated. The tile fetch at 1% GPU is
correct — it is network-bound at 643 tiles/s. Embedding averages of 79%/68%
reflect ~30 s of GPU idle per shard while the next 2.5 GB is read from the
spinning disk at 85 MB/s; prefetching it on a thread would recover ~10 min of
the ~100.

**GPU telemetry wedged at 05:45**, immediately after the first hard kill of a
CUDA process, and reported a frozen `1 %, 11.37 W, 210 MHz` for the rest of the
run. A saturating matmul measured **45.5 TFLOP/s bf16** against that reading, so
the counter was stuck, not the card. Utilisation rows after 05:45 in
`runs/util.csv` are invalid; step rate from the training logs is the reliable
health signal and stayed normal at 0.12 s/step throughout.

---


---

## 8. Correction: every absolute number above was measured on a biased sample

`evaluate(..., n=5000)` took `np.arange(5000)` — the **first** 5,000 rows of the
split, in parquet order. That order is the DuckDB join order over the shards, so
the head of the file is not a random draw over geography. Measured, the first
5,000 test rows give a test loss 0.87 lower than a random 5,000 of the same
split, and a median error 34 km lower.

Every absolute figure in the first version of this report was optimistic:
the headline read 49.4 km rather than 81.6, and 42.4% rather than 35.6%.

**Arm-vs-arm conclusions were never affected**, because every arm was scored on
the same rows, so the paired contrasts — the bank control, the scaling curve,
the long-epoch result — were unbiased throughout. The conclusions in §1–§3 all
survive re-measurement; only their absolute levels moved.

The per-epoch selection rollout drew from the same biased head of the val
split. Selection stays internally consistent -- every epoch of a run was
compared on the same images -- but the epoch each arm selected may differ
from what a representative sample would have chosen, so the arms carry that
as an extra source of noise on top of seed variance.

It stayed invisible until train and test accuracy were computed with two
different samplers and disagreed by more than the train/test gap could explain.
The sampler now draws a seeded random subset, so it is still identical across
arms and the paired bootstrap stays valid.


## What to do next

1. **Scale the bank, not the training set.** §1 says the corpus is worth ~6× the
   gradient signal per image, and the bank is far cheaper: it needs embeddings,
   not optimizer steps. Now running: 15 more shards into the bank alone.
2. **Two seeds per arm, minimum.** §6 means one seed cannot support a median
   ordering, and several conclusions here rest on ~1 pp.
3. **Record the full training argv in every checkpoint.** §6 was only partly
   attributable because `--epochs` is not stored.
4. **Prefetch the next shard during embedding** — ~10% of prep wall clock.
