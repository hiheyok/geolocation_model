# Pyramid levels and the fusion head: the blend weight nobody set

`scripts/pyr_blend.py`, pyr47 (47,646 high-resolution KartaView images),
3,000 test queries against a 38,009 bank, sequences split by crc32 so no
same-sequence pair crosses.

`mean` here reproduces the published level-weighted pool to within rounding
(178.6 km / 33.8% against the recorded 179.1 / 33.8%), which is what makes the
rest comparable to the artifact's session-14 table.

**The sweep tables are exploratory and cover every query. The paired intervals
at the end are reported on a HELD-OUT half**, because an arm picked as the best
of a sweep and then scored on the same queries carries whatever noise favoured
it. That is not a hypothetical correction: the 25 km gain for the level-tuned
arm went from +2.10 pp "separated" to +1.64 pp when the halves were separated,
and a per-step variant that looked uniformly positive turned negative at two of
five thresholds (`scripts/pyr_perstep.py`).

**Three findings.**

1. The published concat is an exact 50/50 average of the two cosines, and 0.5
   is roughly ten times too much head. The optimum is w = 0.05-0.10.
2. Levels want the deepest one weighted up -- the 24 tiles that exist only
   because the source is high-resolution are individually the *weakest*
   (31.3% at 25 km against L0's 32.3%) and still want more weight, because
   their errors are the least correlated with the others.
3. Cascade loses to blending. The head's recall is too low to be a first
   stage: it discards the right region before the re-ranker ever sees it.

**Best supported arm is the simplest one.** `blend w=0.05` -- the published
pool with a 5% head contribution, one number changed -- is separated at all
five thresholds on held-out queries: +0.79 pp at 1 km, +1.18 at 25 km,
+2.17 at 200 km, +2.89 at 750 km, +2.50 at 2500 km.

---

```text
pyr47  47,646 rows, levels [ 3  6 24]
3,000 test queries  38,009 bank  0 same-sequence masked

components built in 21s
similarities in 23s

arm                            median km     <1km    <25km   <200km   <750km  <2500km
-------------------------------------------------------------------------------------
L0 alone                           217.0    17.2%    32.3%    49.3%    68.3%    86.2%
L1 alone                           242.3    15.9%    31.4%    48.2%    68.2%    85.7%
L2 alone                           248.4    14.5%    31.3%    48.3%    66.0%    83.7%
mean                               178.6    17.8%    33.8%    51.1%    70.3%    86.1%
head                               670.5     4.9%    12.9%    25.5%    53.6%    81.1%

--- mean/head blend:  cos = (1-w) * cos_mean + w * cos_head ---
arm                            median km     <1km    <25km   <200km   <750km  <2500km
-------------------------------------------------------------------------------------
w = 0.00                           178.6    17.8%    33.8%    51.1%    70.3%    86.1%
w = 0.05                           146.1    18.4%    34.8%    53.2%    73.2%    88.3%
w = 0.10                           153.2    18.3%    34.7%    52.9%    73.4%    88.3%
w = 0.15                           162.8    18.4%    34.8%    52.7%    73.5%    88.5%
w = 0.20                           166.3    18.3%    34.4%    52.3%    73.4%    88.6%
w = 0.30                           195.9    17.4%    33.1%    50.2%    72.1%    88.6%
w = 0.40                           247.5    16.1%    31.0%    47.7%    70.0%    87.5%
w = 0.50   <- the concat           300.4    14.4%    28.2%    44.7%    68.2%    87.1%
w = 0.70                           412.2    11.7%    23.3%    38.6%    64.1%    85.8%
w = 1.00                           670.5     4.9%    12.9%    25.5%    53.6%    81.1%

--- level weights (head excluded); equal is the published mean ---
arm                            median km     <1km    <25km   <200km   <750km  <2500km
-------------------------------------------------------------------------------------
L0:L1:L2 = 1:0:0                   217.0    17.2%    32.3%    49.3%    68.3%    86.2%
L0:L1:L2 = 0:1:0                   242.3    15.9%    31.4%    48.2%    68.2%    85.7%
L0:L1:L2 = 0:0:1                   248.4    14.5%    31.3%    48.3%    66.0%    83.7%
L0:L1:L2 = 1:1:1   <- equal        149.9    17.9%    34.4%    52.4%    71.3%    86.4%
L0:L1:L2 = 2:1:1                   169.2    18.0%    34.2%    51.8%    70.8%    86.5%
L0:L1:L2 = 1:2:1                   153.8    17.8%    33.9%    52.3%    71.8%    86.9%
L0:L1:L2 = 1:1:2                   144.2    17.9%    34.9%    53.0%    71.8%    86.8%
L0:L1:L2 = 3:2:1                   174.7    18.0%    33.7%    51.5%    70.7%    86.3%
L0:L1:L2 = 1:2:3                   165.6    17.3%    33.8%    51.8%    70.7%    86.4%
L0:L1:L2 = 2:2:1                   172.6    17.9%    33.9%    51.5%    70.7%    86.3%
L0:L1:L2 = 1:2:2                   158.9    17.5%    34.0%    52.0%    70.9%    86.4%
L0:L1:L2 = 0:1:1                   212.1    16.0%    32.4%    49.5%    68.7%    85.5%
L0:L1:L2 = 1:1:0                   178.8    18.0%    33.4%    51.0%    70.8%    86.5%
L0:L1:L2 = 1:0:1                   156.8    17.9%    34.8%    52.3%    71.4%    86.8%

--- best level mix (1, 1, 2) + head ---
arm                            median km     <1km    <25km   <200km   <750km  <2500km
-------------------------------------------------------------------------------------
levels 1:1:2, head w = 0.00        144.2    17.9%    34.9%    53.0%    71.8%    86.8%
levels 1:1:2, head w = 0.05        126.2    18.2%    35.3%    54.6%    73.7%    88.4%
levels 1:1:2, head w = 0.10        125.3    18.3%    35.9%    54.7%    74.0%    88.9%
levels 1:1:2, head w = 0.15        128.2    18.1%    35.6%    54.1%    74.2%    89.1%
levels 1:1:2, head w = 0.20        133.7    18.0%    35.7%    53.8%    74.1%    89.0%
levels 1:1:2, head w = 0.30        174.4    17.0%    33.9%    51.5%    72.4%    88.9%
levels 1:1:2, head w = 0.50        291.7    14.6%    29.1%    45.8%    68.8%    87.0%

--- cascade: top-200 by A, re-ranked by B ---
arm                            median km     <1km    <25km   <200km   <750km  <2500km
-------------------------------------------------------------------------------------
head retrieves -> mean ranks       189.8    17.4%    33.3%    50.5%    71.9%    87.7%
mean retrieves -> head ranks       461.1     8.2%    19.3%    34.7%    63.0%    85.4%
L2 retrieves -> mean ranks         180.0    17.6%    33.5%    51.0%    70.5%    86.1%
mean retrieves -> L0 ranks         217.0    17.1%    32.2%    49.3%    68.4%    86.3%

--- paired against `mean` (the published level-weighted pool) ---
(reported on 1,522 held-out queries; the arms were chosen on the other 1,478)
blend w=0.05             +0.79[+0.1,+1.6]  +1.18[+0.1,+2.3]  +2.17[+0.7,+3.5]  +2.89[+1.5,+4.3]  +2.50[+1.4,+3.6] 
blend w=0.1              +0.20[-0.7,+1.1]~ +0.92[-0.5,+2.3]~ +2.10[+0.4,+3.7]  +3.29[+1.4,+5.1]  +2.50[+1.1,+3.8] 
blend w=0.5 (concat)     -3.81[-5.2,-2.4]  -5.72[-7.7,-3.7]  -6.04[-8.6,-3.6]  -2.37[-4.8,+0.1]~ +0.26[-1.4,+2.1]~
levels 1:1:2             +0.07[-0.7,+0.9]~ +1.25[+0.0,+2.4]~ +1.51[+0.1,+3.0]  +1.51[+0.1,+3.0]  +0.39[-0.8,+1.5]~
levels 1:1:2 + head 0.10 +0.53[-0.5,+1.5]~ +1.64[+0.1,+3.2]  +3.02[+1.3,+4.8]  +3.09[+1.1,+5.0]  +3.09[+1.6,+4.5] 

~ marks an interval spanning zero. 66s total
```
