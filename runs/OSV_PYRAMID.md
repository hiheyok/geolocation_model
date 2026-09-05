# The pyramid, asked on OSV-5M's own data and split

`scripts/osv_pyramid.py`. Everything previously measured about the pyramid ran
on `pyr47` -- KartaView images, their own bank. This uses **OSV-5M release rows
and the benchmark's own splits**, so the corpus and the split gaps are closed.
What remains open: it is still a retrieval probe, not the agent.

**Only two levels exist here, by construction.** OSV-5M frames are 682x512, so
224 px supports 3 crops and a 3x2 tile grid. The 24-tile L2 of `pyr47` would
need 1344x896. The high-resolution finding -- deepest level wants the most
weight -- is therefore untestable on this corpus, and that is a fact about the
data rather than a gap in the experiment.

## The baseline is the real one

`L0 alone` is verified against the production vector on the same rows:

| vector | median km | `<25 km` | any32 `<25 km` |
|---|---|---|---|
| `pool_bal.f16.npy` (what ships) | 226.0 | 27.8% | 66.9% |
| `L0` here (3 crops, per-encoder L2) | 223.0 | 27.7% | 66.9% |
| raw crop mean, no per-encoder L2 | 274.5 | 26.2% | 64.8% |

## Result: adding the tile level helps, on both splits

`L0:L1 = 1:1` against `L0` alone, reported on the held-out half:

| split | `<25 km` | `<200 km` | median |
|---|---|---|---|
| `sequence` (the benchmark's split) | **+4.47** [+2.9, +6.0] | **+4.34** [+2.6, +6.0] | 213 -> 139 km |
| `cell8` (geographic holdout) | **+0.80** [+0.3, +1.3] | **+3.00** [+1.3, +4.7] | 515 -> 464 km |

Both separated at 25 km and 200 km. **The 25 km gain is 5.6x larger on
`sequence` than on `cell8`**, so much of it is same-region matching that a
geographic holdout removes; the coarse 200 km gain transfers far better. Read
the cell8 row as the conservative estimate.

The peak is flat in both -- selection and reporting halves disagree about the
best ratio -- so equal weighting needs no tuning.

## Cost

**No extra bank bytes.** The level mean returns the same 1536-d vector, so the
bank, the k-NN cache and the model input are unchanged. The cost is 9 encoder
forwards per image instead of 3, and a tile cache over all 500,000 release rows
where only 120,000 exist today.

## An earlier measurement I cannot reconcile

`runs/logs/tile_match.log` records `crop3 mean` at **4944.5 km top-1, 7.8%
`<25 km`, 23.3% any32** on a 117,000 bank, and
[[tiling-does-not-beat-crops]] draws on that run. The production vector
measured here on the same cache is **226 km, 27.8%, 66.9%** -- a 3.6x gap in
hit rate that per-encoder normalisation does not explain (the un-normalised
crop mean is 274 km / 26.2%, not 4944 km / 7.8%).

I could not reproduce that scale under either split. Treating the direction it
recorded as settled would be unsafe until someone re-derives it; the numbers
here are grounded against the artifact the agent actually uses.

---

## sequence split

```text
OSV-5M rows with tiles cached : 120,000
split 'sequence': 96,091 train (bank), 11,985 test
3,000 queries used   1,478 choose / 1,522 report
0 same-sequence bank cells masked

tokens (120000, 9, 2, 768)  levels [3 6]  in 17s
similarities in 22s

arm                        median km     <1km    <25km   <200km   <750km  <2500km
---------------------------------------------------------------------------------
L0:L1 = 1:0   <- L0 only, what ships     213.4     2.7%    28.3%    49.6%    63.7%    77.2%
L0:L1 = 0:1                    166.6     2.7%    32.4%    51.8%    64.5%    76.2%
L0:L1 = 3:1                    157.7     2.8%    30.9%    52.4%    65.8%    77.5%
L0:L1 = 2:1                    146.8     2.9%    31.9%    53.2%    66.0%    77.9%
L0:L1 = 3:2                    139.1     2.8%    32.3%    53.6%    66.4%    78.3%
L0:L1 = 1:1   <- equal, the baseline     138.6     2.8%    32.7%    53.9%    66.4%    78.6%
L0:L1 = 2:3                    144.8     2.8%    33.0%    53.3%    65.8%    78.0%
L0:L1 = 1:2                    145.0     2.7%    32.9%    53.2%    65.8%    77.7%
L0:L1 = 1:3                    146.3     2.7%    33.2%    53.1%    65.6%    77.1%

best on <25 km: selection half (1, 1), reporting half (1, 3)   DISAGREE, the peak is flat

--- against L0 only (the shipping representation), 1,522 held-out queries ---
L0:L1 = 0:1    +0.00[-0.7,+0.6]~ +4.14[+2.0,+6.3]  +2.17[-0.1,+4.5]~ +0.79[-1.3,+3.0]~ -0.99[-3.0,+1.0]~
L0:L1 = 3:1    +0.13[-0.1,+0.4]~ +2.69[+1.6,+3.9]  +2.76[+1.4,+4.1]  +2.17[+0.9,+3.4]  +0.26[-0.8,+1.2]~
L0:L1 = 2:1    +0.20[-0.1,+0.6]~ +3.61[+2.2,+4.9]  +3.61[+2.2,+5.1]  +2.37[+1.1,+3.8]  +0.66[-0.4,+1.8]~
L0:L1 = 3:2    +0.13[-0.2,+0.5]~ +4.01[+2.6,+5.5]  +4.01[+2.5,+5.7]  +2.69[+1.2,+4.2]  +1.05[-0.3,+2.4]~
L0:L1 = 1:1    +0.13[-0.3,+0.5]~ +4.47[+2.9,+6.0]  +4.34[+2.6,+6.0]  +2.69[+1.1,+4.2]  +1.45[+0.1,+2.8] 
L0:L1 = 2:3    +0.13[-0.3,+0.5]~ +4.80[+3.0,+6.4]  +3.68[+1.9,+5.8]  +2.17[+0.5,+3.9]  +0.79[-0.7,+2.3]~
L0:L1 = 1:2    +0.00[-0.4,+0.5]~ +4.66[+3.0,+6.6]  +3.55[+1.6,+5.5]  +2.17[+0.4,+3.9]  +0.46[-1.2,+2.0]~
L0:L1 = 1:3    +0.00[-0.5,+0.5]~ +4.93[+3.1,+6.9]  +3.48[+1.4,+5.6]  +1.91[+0.1,+3.7]  -0.07[-1.8,+1.5]~

~ marks an interval spanning zero. 52s total
```

## cell8 split

```text
OSV-5M rows with tiles cached : 120,000
split 'cell8': 100,297 train (bank), 9,246 test
3,000 queries used   1,500 choose / 1,500 report
8 same-sequence bank cells masked

tokens (120000, 9, 2, 768)  levels [3 6]  in 26s
similarities in 31s

arm                        median km     <1km    <25km   <200km   <750km  <2500km
---------------------------------------------------------------------------------
L0:L1 = 1:0   <- L0 only, what ships     515.2     0.0%     3.3%    29.7%    58.2%    77.3%
L0:L1 = 0:1                    518.6     0.0%     3.8%    31.5%    57.1%    74.7%
L0:L1 = 3:1                    510.2     0.0%     3.8%    30.2%    58.6%    77.5%
L0:L1 = 2:1                    474.8     0.0%     3.9%    31.6%    60.1%    78.4%
L0:L1 = 3:2                    463.9     0.0%     4.0%    32.3%    59.6%    78.6%
L0:L1 = 1:1   <- equal, the baseline     464.3     0.0%     4.1%    32.7%    60.0%    78.7%
L0:L1 = 2:3                    464.3     0.0%     4.1%    33.5%    59.9%    78.5%
L0:L1 = 1:2                    470.4     0.0%     3.9%    33.1%    59.7%    77.8%
L0:L1 = 1:3                    462.7     0.0%     4.0%    33.0%    59.7%    77.3%

best on <25 km: selection half (3, 2), reporting half (1, 1)   DISAGREE, the peak is flat

--- against L0 only (the shipping representation), 1,500 held-out queries ---
L0:L1 = 0:1    +0.00[+0.0,+0.0]~ +0.47[-0.2,+1.1]~ +1.87[-0.3,+3.9]~ -1.07[-3.4,+1.3]~ -2.60[-4.7,-0.6] 
L0:L1 = 3:1    +0.00[+0.0,+0.0]~ +0.47[+0.1,+0.9]  +0.53[-0.7,+1.7]~ +0.40[-1.0,+1.8]~ +0.20[-0.9,+1.3]~
L0:L1 = 2:1    +0.00[+0.0,+0.0]~ +0.60[+0.2,+1.1]  +1.93[+0.5,+3.4]  +1.93[+0.3,+3.5]  +1.13[-0.1,+2.3]~
L0:L1 = 3:2    +0.00[+0.0,+0.0]~ +0.67[+0.3,+1.1]  +2.67[+1.2,+4.1]  +1.40[-0.2,+3.1]~ +1.33[+0.0,+2.6]~
L0:L1 = 1:1    +0.00[+0.0,+0.0]~ +0.80[+0.3,+1.3]  +3.00[+1.3,+4.7]  +1.80[-0.1,+3.7]~ +1.47[+0.1,+2.8] 
L0:L1 = 2:3    +0.00[+0.0,+0.0]~ +0.73[+0.2,+1.3]  +3.87[+2.1,+5.6]  +1.67[-0.3,+3.7]~ +1.20[-0.3,+2.7]~
L0:L1 = 1:2    +0.00[+0.0,+0.0]~ +0.60[+0.1,+1.1]  +3.40[+1.7,+5.3]  +1.53[-0.4,+3.6]~ +0.53[-1.1,+2.1]~
L0:L1 = 1:3    +0.00[+0.0,+0.0]~ +0.67[+0.1,+1.3]  +3.33[+1.5,+5.2]  +1.53[-0.7,+3.7]~ +0.07[-1.7,+1.9]~

~ marks an interval spanning zero. 61s total
```
