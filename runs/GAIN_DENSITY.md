# The tiles gain does not track density, and that weakens my "floor" framing

A **retrieval probe** run off the neighbour tables alone -- a lookup and a
great-circle, no GPU, no encoder, no model. 49,788 test queries against the
400,180-row bank the paired ladders are training on.

## What was being tested, and why it mattered

The case for a bigger *tiled* corpus rests on an extrapolation. `XBANK.md` §3
subsampled the bank and found the matched crops+tiles gain growing at every
step:

| bank rows | crops | crops+tiles | gain |
|---|---|---|---|
| 25,000 | 21.0% | 22.8% | +1.80 |
| 50,000 | 24.4% | 26.6% | +2.20 |
| 100,000 | 29.4% | 31.4% | +2.03 |
| 200,000 | 35.2% | 37.7% | +2.53 |
| 400,000 | 40.7% | 43.3% | **+2.60** |

I offered a mechanism for that: a sparse bank is limited by **coverage**, where
no representation can invent a match, and a dense bank by **discrimination**,
which is what finer features supply. On that reading the +2.03 pp measured at
400k is a floor and 3.4M should do better -- and I have been quoting it as a
floor.

If density really is the variable, it should also work **inside one fixed
bank**: queries landing in well-covered places should gain more than queries in
empty ones, with the bank, the split, the encoders and the projection all held
constant. That is a stronger test of the mechanism than the subsample series,
because nothing varies except where the query happens to be.

## It does not

Bank rows sharing the query's 1-degree cell, against the crops+tiles gain at
`<25 km`:

**any-of-16** (the agent's retrieval window)

| bank rows in cell | queries | crops | +tiles | gain pp |
|---|---|---|---|---|
| 0 | 104 | 8.7% | 8.7% | +0.00 ~ |
| 1 - 9 | 925 | 38.9% | 41.4% | +2.49 [+1.1, +4.0] |
| 10 - 99 | 9,532 | 60.6% | 62.5% | +1.93 [+1.4, +2.5] |
| 100 - 999 | 37,575 | 65.5% | 67.6% | +2.05 [+1.8, +2.3] |
| 1,000 - 9,999 | 1,652 | 79.5% | 81.4% | +1.88 [+0.7, +3.1] |

**top-1**, matching `XBANK.md`'s own protocol

| bank rows in cell | queries | crops | +tiles | gain pp |
|---|---|---|---|---|
| 0 | 104 | 5.8% | 6.7% | +0.96 ~ |
| 1 - 9 | 925 | 22.6% | 24.8% | +2.16 [+0.5, +3.8] |
| 10 - 99 | 9,532 | 33.0% | 35.7% | +2.71 [+2.1, +3.3] |
| 100 - 999 | 37,575 | 33.0% | 35.8% | +2.82 [+2.5, +3.2] |
| 1,000 - 9,999 | 1,652 | 38.9% | 41.2% | +2.24 [+0.4, +4.0] |

**Flat, at both depths, across four orders of magnitude of local coverage** --
while the hit rate itself moves from 22.6% to 38.9% (top-1) and from 38.9% to
79.5% (any-of-16). The two large buckets carry the weight: 10-99 and 100-999
rows per cell differ by 10x in density, hold 9,532 and 37,575 queries, and
their gains are +2.71 [+2.1, +3.3] and +2.82 [+2.5, +3.2]. Indistinguishable.

The 0-row bucket is degenerate and should not be read: 104 queries whose cell
holds no bank rows at all, so every match comes from somewhere else entirely.

## What this does and does not establish

**It does not refute the subsample series.** Local and global density are
different manipulations. Subsampling removes 94% of the bank *uniformly*, with
geography held statistically constant; local density varies *with* geography,
and so is confounded with region, country and image type. A gain that is flat
in one and rising in the other is not a contradiction.

**But it is the only independent test of the mechanism I offered, and the
mechanism does not reproduce.** "Coverage giving way to discrimination"
predicts the gain should rise as local coverage rises. It does not move at all.
So the story that made +2.03 pp a *floor* rather than an *estimate* is
unsupported.

**Correction, therefore.** I have been writing "read +2.03 as a floor, the gain
grows with density". That should read: **+2.03 pp is the estimate at 400k, and
whether it grows at 3.4M is open.** Four points along one axis, with the one
independent check of their mechanism coming back flat, is not enough to promise
a larger number.

**It raises the value of `tilebig`, which is already queued.** That measures
the gain at 1,150,180 rows *directly* instead of extrapolating to it -- which
is now the only way to settle this, and it costs one 4 h tile pass.

One genuinely reassuring reading survives: the gain is **remarkably stable**.
Between 1 and 10,000 bank rows per cell, across a hit rate that nearly doubles,
crops+tiles is worth about +2 pp everywhere. Nothing here suggests it decays.

## Reproduce

    OSV_RELEASE=s10 py scripts/gain_density.py \
        --a knn_pyr768_l0_sequence_k32.npz \
        --b knn_pyr768_mix_sequence_k32.npz --ranks 1

Seconds, CPU only. The grid is deliberately crude -- the question is which
order of magnitude of local coverage a query sits in, not a kernel estimate --
and a degree of longitude shrinks with latitude, which the buckets do not
correct for.
