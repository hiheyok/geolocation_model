# Resolution or tiles? Re-run with an encoder that is actually native

`scripts/resmatch.py --rows 50000 --queries 3000`, 39,831 bank rows, sequence
split, seeded random cohort. 3,540 s.

The previous run used `vit_base_patch16_siglip_224.v2_webli` at `img_size=512`
-- a 224 checkpoint with interpolated position embeddings, which is **out of
distribution, not native** (REVIEW5 #10). That made the two resolution arms
unreadable: "more pixels do not help" and "this checkpoint cannot use them"
produce the same table. `timm` ships a real 512 checkpoint; this run uses it.

Arms 1 and 2 never touch the native encoder and reproduce **to the digit**,
which is what makes this a controlled comparison rather than two runs.

## The arms

| arm | median km | <1km | <25km | <200km | <750km | <2500km |
|---|---|---|---|---|---|---|
| 1  crops @224 (incumbent) | 385.2 | 0.7% | 17.4% | 38.7% | 62.9% | 83.5% |
| 2  crops+tiles @224 | 348.4 | 1.0% | 18.2% | 41.2% | 65.0% | 85.1% |
| 3  crops @518/512 native | 336.4 | 1.3% | 18.4% | 40.8% | 67.5% | 86.0% |
| 4  native crops + 224 tiles | **334.6** | 1.3% | **19.0%** | **41.6%** | 66.5% | 86.0% |

Paired against arm 1, `~` spans zero:

| arm | <1km | <25km | <200km | <750km | <2500km |
|---|---|---|---|---|---|
| 2 | +0.23~ | +0.83~ | +2.50 | +2.10 | +1.60 |
| 3 | +0.53 | +0.97~ | +2.10 | **+4.57** | +2.50 |
| 4 | +0.53 | **+1.63** | +2.97 | +3.57 | +2.50 |

## What the confound was worth

Arm 3 is the whole story. Same cohort, same bank, same arms 1 and 2 -- only the
SigLIP checkpoint changed:

| arm 3, crops at native size | median | <25km | <200km | <750km | <2500km |
|---|---|---|---|---|---|
| 224 checkpoint stretched to 512 | 390.8 | -0.30~ | -0.43~ | +0.47~ | -0.47~ |
| real 512 checkpoint | **336.4** | +0.97~ | **+2.10** | **+4.57** | **+2.50** |

Flat-to-negative at every threshold becomes separated at three of them, and the
median falls 54 km. **The earlier reading was measuring an encoder
extrapolating past its training grid, not measuring resolution.**

`runs/RESOLUTION.md`'s conclusion -- "native 518/512 is flat-to-negative from
25 km up" -- is retracted for the SigLIP half. It was drawn from the stretched
checkpoint.

## The RESMATCH decision

`runs/REPROBE.md` predicted, from the random-cohort re-measurement, that arm 4
would separate and arm 2 would not. That is exactly what happened, and the
gap widened with the encoder fixed:

| | REPROBE prediction | this run |
|---|---|---|
| arm 2, `<25km` | +0.83 [-0.1, +1.7] ~ | +0.83 [-0.1, +1.7] ~ |
| arm 4, `<25km` | +1.27 [+0.2, +2.3] | **+1.63 [+0.6, +2.7]** |

**Resolution and tiles are complements, not alternatives**, and they act at
different scales: arm 3 buys most of its gain at 750 km (+4.57), arm 4 buys
its extra at 25 km (+1.63 against arm 3's +0.97). That is the same
scale-splitting the encoders themselves show.

## Two limits on reading this

**The bank is 39,831 rows, not 400,180.** Absolute levels are far below the
`xbank` table (17.4% vs 40.7% at 25 km) and the tiles gain is known to grow
with density -- +1.80 pp at 25k rows to +2.60 pp at 400k. Arm 2 failing to
separate here is consistent with that curve, not evidence against tiles.

**Arm 4 versus arm 3 is not tested.** Every interval above is paired against
arm 1. Arm 4 leads arm 3 on the median and at 25 km and trails it at 750 km,
and nothing here says whether any of that is separable. The script pairs only
against the incumbent; deciding "is the tile cache worth building on top of
native crops" needs that contrast added.
