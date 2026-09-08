# Tiles against what actually ships

> **CONCLUSION RETRACTED -- see `runs/LEAKTRAIN.md`.**
> This says tiles reach parity with the shipping arm and should not ship. The
> comparator was inflated: every shipping-family checkpoint was trained before
> 2026-09-04 04:05, when the file at
> `knn_pca768_bank70_sequence_k32_bank_ext70.npz` was the same-sequence-LEAKY
> cache. Trained against the rebuilt cache, the same street file scores
> **54.6%, not 57.5%** ([-3.80, -1.96] pp, separated), and tiles beat that
> honest baseline by **+2.38 to +4.54 pp**. Every number below is correct;
> the arm they are compared against was not.

`runs/TILEFULL.md` reported crops+tiles beating crops by **+3.72 pp** on the
neighbour tables at 3,400,180 rows, and `scripts/tilefull_ladder.py` showed it
surviving training at **+2.54 to +4.70 pp**. Both are true. Neither answers
the question that decides anything, which is whether tiles beat the arm that
is already shipping.

They do not.

---

## 1. Three arms, one bootstrap

5,000 test images, beam k=2, ranked s0-s2, 3,000 resamples.
`runs/BOOTSTRAP_ship.md`.

| arm | median km | mean km | `<25km` |
|---|---|---|---|
| `pyrL0-b340-e4` (crops, pyramid) | 19.2 | 472.9 | 54.5% |
| `pyrL0L1-b340-e4` (crops+tiles, pyramid) | 15.3 | 354.2 | 58.1% |
| `d768-b350-e6-drop70` (**shipping**) | 15.4 | 434.7 | 57.5% |

| contrast | `<25km` diff | |
|---|---|---|
| tiled pyramid vs crops pyramid | **+2.54 to +4.70 pp** | separated |
| **tiled pyramid vs shipping** | **[-0.62, +1.68] pp** | **inside noise** |
| crops pyramid vs shipping | **-4.00 to -2.06 pp** | separated, worse |

---

## 2. What the three rows say together

The pyramid arm **without** tiles is 2.06 to 4.00 pp *below* the shipping arm.
Tiles add back 2.54 to 4.70 pp. The result is statistically indistinguishable
from what already ships, on both the hit rate and the median.

**Tiles are buying back a deficit, not adding on top of the best
configuration.** Every tile number this project has published --
`runs/XBANK.md`'s +1.80, `runs/TILEBIG.md`'s +3.30 and +2.76 to +4.88,
`runs/TILEFULL.md`'s +3.72 -- is measured against `pyrL0` or its ancestors,
which is the arm this table puts 2-4 pp below the bar.

This is a shape the project has hit before: `runs/PYR_FUSE.md` was "falsified
at step 2 ... a handicapped baseline had hidden it". The difference is that
there the baseline was handicapped by an obvious bug; here `pyrL0` is a
perfectly reasonable arm that simply is not the best one.

**A two-arm bootstrap would have shipped this.** Run with only the tiled
pyramid and the shipping arm, the answer is "inside noise" and the story is
"tiles reach parity". Run with only the two pyramid arms -- which is what
every previous tiles experiment did -- the answer is "+2.54 to +4.70 pp,
separated" and the story is "ship tiles". The third arm is what makes the
first two interpretable.

---

## 3. What is NOT established

**Why `pyrL0` sits below the shipping arm.** The two differ in at least five
ways, and this run separates none of them:

* pooling -- `pool_pyramid`'s per-token normalised `L0` against
  `pool_bal` -> `pca768`;
* projection basis -- `pyr768_l0_pca.npz` against `pca768_bank55_pca.npz`;
* epochs -- e4 against e6;
* **weight-decay grouping** -- the shipping arm predates the flag and uses the
  legacy substring rule, which exempted the learned retrieval keys, 196,608
  parameters or 3.7% of the model, from decay. `bootstrap.py` flags this
  itself.
* **sink negatives** -- both arms draw from OS entropy rather than a seed, so
  neither sees the same off-path tiles on a re-run.

So "the pyramid family is worse" is a hypothesis this supports, not a finding.
The last two confounds also mean the parity row is not clean either.

**The mean.** The tiled arm's mean error is 354.2 km against the shipping
arm's 434.7 -- 80 km better, while median and hit rate are level. `bootstrap.py`
does not resample the mean, so this has no interval and is not a claim. It is
the one hint that the two arms differ in the tail rather than the body, and it
is worth a paired test before tiles are written off entirely.

---

## 4. What would settle it

**Tiles on the shipping configuration.** Blend `L1` into the `pool_bal` ->
`pca768` family rather than the pyramid one, hold everything else at the
shipping recipe, and run the same ladder. That is the arm nobody has built,
and it is the only one that can show tiles adding to the best configuration
rather than repairing a worse one.

Cost: a pool, three stacks, a projection, an index and a six-rung ladder --
about a day, most of it the ladder. The retrieval half alone (`knn_gap`
against the shipping `pca768_bank70`) is under an hour and would say whether
the ladder is worth starting.

**That experiment is a low priority now, and the cheap half is why.**
`knn_gap` puts `pca768_bank70` and `pyr768_l0_b340` inside noise at every
threshold, top-1 and any-of-32, and `shipclean-e4` then showed their trained
agents indistinguishable too. So the pooling choice does not move any metric
measured here, and the 2-4 pp this file attributes to the family was the
cache.

It is **not** that they are the same features: the two tables disagree on
top-1 for 19,837 of 500,000 queries (3.97%), and only 35.1% of queries share
an identical top-32 set. An earlier draft called them identical and called
the experiment moot. Neither is established -- what is established is that
nothing measured so far distinguishes them.

~~**Until then, do not ship tiles**~~ -- retracted. Against a baseline
trained the same way as the treatment, tiles win by +2.38 to +4.54 pp
(`runs/LEAKTRAIN.md`). The second half stands: do not quote a tile number
without saying what it is measured against, which is exactly how this file
reached the wrong conclusion.
