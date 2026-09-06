# Resolution or tiles? The rebuild has one answer, and it is tiles

A **retrieval probe** -- encode a query, cosine against a bank, take the nearest
neighbour, measure the great-circle error to its true location. No gradient
step, no beam, no map, no click head. Every number here is a claim about the
representation, not about the agent.

Everything in `XBANK.md` pointed at a bank rebuild: query-only changes are a
dose-response failure, and only the matched arm gains. So the question stopped
being *whether* to rebuild and became *what to build*, between two candidates
that had never been compared on the same bank:

    tiles       add detail by fragmenting the scene -- 9 forwards @224
    resolution  add detail by sampling the scene properly -- 3 large forwards

Every arm is **matched** (query and bank built identically), because the 2x2 in
`XBANK.md` established that nothing else is informative. 50,000 release rows ->
39,831 bank rows, 3,000 held-out `sequence`-split queries.

**Each encoder runs at its own native size.** An earlier sweep held both to one
shared input, which forces multiples of `lcm(14, 16) = 112`, lands on 448, and
puts DINOv2 14% below its native grid while upsampling SigLIP. Nothing requires
a shared size -- the encoders run independently and only their 768-d outputs are
concatenated. Frames are 512 tall, so SigLIP `/16` at 512 is the frame's own
pixels with no resampling, and DINOv2 `/14` at 518 is its exact training
resolution.

## The table

| arm | median km | `<1km` | `<25km` | `<200km` | `<750km` | `<2500km` |
|---|---|---|---|---|---|---|
| 1  crops @224 (incumbent) | 392.4 | 1.2% | 19.8% | 39.9% | 60.2% | 80.0% |
| **2  crops+tiles @224** | **342.1** | 1.3% | **21.1%** | **42.9%** | **62.8%** | **81.3%** |
| 3  crops @518/512 native | 418.5 | **1.5%** | 19.4% | 39.3% | 59.9% | 79.6% |
| 4  native crops + 224 tiles | 370.2 | **1.6%** | 20.8% | 41.8% | 62.1% | 80.8% |

Paired bootstrap against the incumbent, `~` spans zero:

```
                              <1km              <25km             <200km            <750km            <2500km
2  crops+tiles @224      +0.17[-0.1,+0.4]~ +1.27[+0.4,+2.2]  +3.00[+1.7,+4.3]  +2.57[+1.3,+3.8]  +1.37[+0.3,+2.4]
3  crops @518/512 native +0.37[+0.1,+0.7]  -0.37[-1.4,+0.7]~ -0.57[-2.0,+0.9]~ -0.33[-1.7,+1.2]~ -0.33[-1.5,+0.9]~
4  native crops + 224 t. +0.40[+0.1,+0.7]  +0.97[-0.1,+2.0]~ +1.90[+0.6,+3.3]  +1.93[+0.5,+3.4]  +0.83[-0.3,+2.0]~
```

## What arm 3 settles

Arm 3 is the clean verdict on resolution, and it closes two questions that were
left open in `XBANK.md` as possible explanations for the -3.27 pp at 448:

* **it is matched**, so "the query was mismatched against a 224 bank" is out;
* **it has no PCA anywhere in the path** (1536-d throughout), so "the projection
  was fitted on 224-derived vectors" is out.

With both removed, native resolution is still **flat to negative at every
threshold from 25 km up**, and its median is *worse* than the incumbent's
(418.5 against 392.4 km). The earlier penalty was not an artifact of the
comparison. More pixels of the same framing do not carry more locatable signal.

This also rules out the "the encoder was never trained to process more pixels"
reading **for this arm specifically**: 518 is exactly DINOv2's training
resolution, so arm 3 is the encoder at home rather than extrapolating, and it
still does not pay. (The reading may still hold for the *agent*, which has only
ever consumed 224-derived vectors -- but nothing here tests that.)

## The one place resolution wins

`<1km` is the only threshold where resolution helps, and it helps in **both**
arms that use it (+0.37 and +0.40, both separated) while tiles do nothing there
(+0.17, spans zero). Finer sampling buys discrimination among near-duplicates.

That is a 1.2% base on a 39,831-row bank, where near-duplicates barely exist.
The shipping bank is 3.4M rows and spends much more of its time in exactly that
regime -- top-1 alone already scores 56.4% `<25 km` there. So this is the one
result on this page that a **denser bank could amplify rather than dilute**, and
it should be re-measured after the rebuild rather than treated as closed. It
does not change the build: the effect is small, and buying it costs ~3x.

## Why not both

Arm 4 answers that. Adding native crops to tiles gives **+0.97 pp** at 25 km,
*below* arm 2's +1.27 and no longer separated. Resolution does not complement
tiles, it dilutes them -- both add detail, and past the first helping the
marginal return is negative while the cost is not.

Caveat in arm 4's favour: it mixes native crops with **224** tiles, because
tiling at native size is another six large forwards. Its true ceiling is
therefore higher than measured. But it would have to clear arm 2 by enough to
justify roughly 3x the embedding cost, and it currently does not clear it at
all.

## Cost, measured rather than argued

| scheme | pixels/image | measured |
|---|---|---|
| crops + tiles @224 | 9 x 224^2 = 451,584 | 43.5 img/s |
| crops @518/512 | 3 x 518^2 = 804,492 | ~33.5 img/s |

Native is 1.8x the pixels and attention is superlinear in tokens, so it is
roughly **3x more expensive than tiling**. An earlier estimate of mine had this
backwards -- "3 forwards against 9, so resolution is cheaper" counted forwards
as if they were the same size. See `docs/STATE.md` §8g.

## Decision

**Build tiles at 224.** The ~19 h extension pass, not the ~60 h one.

Read **+1.27 pp as a floor, not the estimate.** `XBANK.md` §3 measured the
matched gain *growing* with bank density -- +1.80 at 25k rows, +2.60 at 400k --
and this bank is 39,831 rows, so arm 2 is measured near its weakest. Do not
splice the two series numerically: different subset, split and query set.

## Reproduce

    OSV_RELEASE=s10 py scripts/resmatch.py --rows 50000 --queries 3000

43 min on an RTX 3070, almost all of it arms 3 and 4 (the 224 levels are read
from `dual_c3` and `tile6`, which are already on disk). Log
`runs/logs/resmatch.log`.
