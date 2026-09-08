# The shipping baseline was trained on a leaky cache, and tiles beat the honest one

`runs/TILESHIP.md` (PR #59, merged) concluded that tiles **reach parity** with
the shipping arm and should not ship. That conclusion is wrong, and wrong in
the direction its own three-arm design was meant to protect against: the
comparator was inflated.

**Tiles beat an honestly-trained shipping baseline by +2.38 to +4.54 pp**,
separated.

---

## 1. What was wrong with the comparator

Every shipping-family checkpoint -- `d768-b350-*`, `wd29-*` -- was trained
**before 2026-09-04 04:05**, which is when
`knn_pca768_bank70_sequence_k32_bank_ext70.npz` was rebuilt after the
same-sequence bank leak was found. The file at that path was the leaky cache
when they trained. `pyrL0*` and `pyrL0L1*` were trained on 09-07, against the
rebuilt one.

A checkpoint records the **path**, not the content. `AGENTS.md` §7 --
"artifacts must be bound by content, not by name" -- is written about exactly
this, and nothing enforces it for the k-NN cache. Two checkpoints can record
the same filename and have been trained on different data.

`scripts/chain_shipclean.sh` trains the shipping street file against the
**clean** cache, same recipe, same seed, same split:

| arm | street file | cache at train time | `<25km` |
|---|---|---|---|
| `wd29-fix-e4` | `pca768_bank70` | **leaky** | 57.5% |
| `shipclean-e4` | `pca768_bank70` | **clean** | **54.6%** |

**-3.80 to -1.96 pp, separated.** One variable. Training on a leaky retrieval
cache is worth two to four points at test time, even though the test-time
cache is clean for both. `same-sequence-leak-in-the-bank` recorded that
training on the leak "HELPED"; this is how much.

---

## 2. Tiles against the honest baseline

| contrast | `<25km` | |
|---|---|---|
| **`pyrL0L1-b340-e4` vs `shipclean-e4`** | **+2.38 to +4.54 pp** | **separated** |
| `pyrL0L1-b340-e4` vs `wd29-fix-e4` | [-0.50, +1.64] pp | inside noise |
| `shipclean-e4` vs `pyrL0-b340-e4` | [-0.78, +1.04] pp | inside noise |

The second row is what #59 measured and read as parity. The first row is the
same question against a baseline trained the same way as the treatment.

The third row closes a loop worth stating carefully. `knn_gap` puts
`pca768_bank70` and `pyr768_l0_b340` inside noise at every threshold, top-1
and any-of-32, and their trained agents are indistinguishable too. That is
**no detected difference on these metrics**, which is not the same as the
same features -- an earlier draft of this file said "retrieval-identical" and
that was wrong. The tables disagree on **top-1 for 19,837 of 500,000 queries
(3.97%)**, and only **35.1%** of queries have an identical top-32 set. The
per-query differences are real and simply do not aggregate into any metric
reported here.

What follows is therefore weaker than "the pooling is nothing": on everything
measured, the pooling choice does not move the result, and the cache does.

**Absolute levels, 5,000 test images, beam k=2, s0-s2:**

| arm | median km | mean km | `<25km` |
|---|---|---|---|
| `pyrL0L1-b340-c6` (tiles, one cosine) | **14.6** | 372.0 | **58.4%** |
| `pyrL0L1-b340-e4` (tiles) | 15.3 | **354.2** | 58.1% |
| `wd29-fix-e4` (leak-trained) | 15.1 | 455.6 | 57.5% |
| `shipclean-e4` (honest baseline) | 18.5 | 458.1 | 54.6% |
| `pyrL0-b340-e4` (crops) | 19.2 | 472.9 | 54.5% |

---

## 3. The sixth epoch was partly the schedule

Every ladder here is separate `--epochs 2` runs chained with `--init`, which
restores **weights only**: Adam's moments restart and the LR schedule is a
fresh cosine. `train.py` compensates -- with `--init` the default lr drops to
1e-4 and warmup to 100 steps -- but that is still a re-heat of a just-annealed
model, three times over. Five ladders have peaked before their last rung and
it has always been read as `epochs-do-not-substitute-for-data`, never
separated from the schedule.

Two experiments, one knob each. `scripts/chain_lr.sh` lowers the last rung's
peak; `night0908.py` runs a single `--epochs 6` cosine from scratch.

**Read the epoch column first.** `--select hit` keeps the best validation
epoch, so a tag names the epoch BUDGET, not the weights. These are
validation-selected checkpoints under different schedule budgets, not
sixth-epoch models:

| arm | budget | epoch selected | `<25km` | vs e4 |
|---|---|---|---|---|
| `pyrL0L1-b340-e4` | 4 | 4 | 58.1% | -- |
| `pyrL0L1-b340-e6` @ 1e-4 | 6 | 5 | 57.1% | e4 better, **separated** [+0.36, +1.60] |
| `pyrL0L1-b340-e6-lr5e-5` | 6 | **6** | 57.7% | inside noise |
| `pyrL0L1-b340-e6-lr3e-5` | 6 | **6** | 57.5% | e4 better, separated |
| **`pyrL0L1-b340-c6`**, one cosine | 6 | **3** | **58.4%** | inside noise |
| `pyrL0-b340-e6` | 6 | **6** | 53.9% | -- |
| `pyrL0-b340-c6`, one cosine | 6 | **4** | -- | -- |
| `shipclean-e6` | 6 | **6** | 53.8% | -- |

**A draft of this section claimed "no arm given six epochs ever selected the
sixth". That is false: four of the seven did**, and two of them were printed
in the tool output the claim was written from. Retracted.

What the column does support, and only this:

* **The two single-cosine arms peak earliest** -- epochs 3 and 4 of 6.
  A single cosine over six epochs is still far from annealed at epoch 3, so
  the best validation point falls mid-run and the remaining budget is spent
  getting worse.
* **The chained arms mostly run to the end of their budget**, which is what a
  schedule that anneals to zero inside every 2-epoch rung should do -- each
  rung's last epoch is its own best.
* **None of it beats e4 on the test bootstrap**, including the four arms that
  did use all six epochs. That is the claim that survives, and it rests on
  the paired intervals rather than on where validation happened to stop.

* **The re-heat was real and costly.** 5e-5 beats 1e-4 [-1.18, -0.02],
  separated, and the single cosine beats the chained e6 [-2.18, -0.38],
  separated.
* **No schedule beats four epochs.** `c6` has the best hit rate and the best
  median of any arm and is still inside noise against e4 -- reached at its own
  epoch 3, having been given six.

So **"stop at 4" survives, for a properly separated reason** rather than five
repetitions of a confounded one. And the chained ladder was costing about a
point at its last rung, which every previous ladder here paid.

---

## 4. The first cell8 agent ladder with tiles

TILEFULL put the tile gain on the geographic holdout at +0.49 pp, but that is
retrieval. No agent had ever been trained on `cell8` with tiles.

| `cell8`, e4 | median km | `<25km` |
|---|---|---|
| crops | 240.9 | 7.7% |
| crops+tiles | **202.7** | 8.2% |

| contrast | | |
|---|---|---|
| tiles vs crops, `<25km` | **+0.06 to +1.10 pp** | separated |
| tiles vs crops, median | **26.7 to 45.7 km better** | separated |

**The hit-rate ratio matches retrieval.** +0.6 pp on geography against +3.5 pp
on the benchmark is about 6x, close to retrieval's 7.6x. Training does not
rescue the transfer.

**But the median moves a long way** -- 38 km, separated -- while the 25 km
threshold barely shifts. On unseen geography tiles move the body of the
distribution without converting many queries into fine localisations. That is
a different effect from the benchmark split and is not visible in a hit rate.

`cell8` also peaks at **e2**: `pyrL0L1-c8-e4` is worse than `-c8-e2`
([-0.74, -0.04] pp, separated). Earlier than the benchmark's e4.

---

## 5. What this changes

* **`runs/TILESHIP.md`'s conclusion is retracted.** Tiles beat the honest
  baseline. Its numbers are all correct; the comparator was not.
* **Every published comparison against a pre-2026-09-04 checkpoint is
  suspect**, including `runs/FINAL.md` and `runs/BOOTSTRAP_seqfix_all.md`
  (66 of the sequence checkpoints on disk predate the rebuild). They are not
  wrong about arms trained in the same era as each other; they are wrong the
  moment a post-rebuild arm is compared against one.
* **Nothing binds a checkpoint to its cache's content.** `bootstrap.py` warns
  about seed provenance and weight-decay grouping and says nothing about the
  cache. That is the fix this run argues for, and it is not made here (PR
  #61 makes it).

  An earlier draft added "and does not compare `split_hash`". That was wrong:
  `bootstrap.provenance` calls `evaluate.check_split`, and `main` refuses
  arms whose provenance differs before it loads any errors. It also
  recognises the pre-2026-09-03 weak digest, which is why the 66 checkpoints
  carrying that digest measure null against their current-hash twins -- a
  hash-function change, not a resplit.

Ruled out along the way, each with a paired test: the street file (no
detected difference at any retrieval threshold, agents inside noise -- though
their top-1 neighbours differ on 3.97% of queries, so this is parity on the
measured metrics, not identity), the sequence split ([-1.02, +0.92] pp), and
the weight-decay grouping ([-0.40, +0.68] pp).
