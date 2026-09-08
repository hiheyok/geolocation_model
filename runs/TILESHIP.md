# Tiles against what actually ships

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

**Until then, do not ship tiles**, and do not quote +3.72 pp or +2.54 to
+4.70 pp without saying what they are measured against.
