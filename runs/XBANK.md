# Which side has to change, and does the gain survive a denser bank?

Two runs, both **retrieval probes** -- one matmul to a nearest neighbour. No
gradient step anywhere, no beam, no map, no click head. Everything here is a
claim about the representation, not about the agent.

## 1. Query-only, against the live 3.4M shipping bank

The goal was to leave `pca768_bank70` alone and change only how the uploaded
image is read. **It does not work, and the failure is a dose-response.**

| query encoding | median | `<25 km` | vs shipping |
|---|---|---|---|
| 3 crops @224 (ships) | 13.2 km | 56.4% | -- |
| 3 crops @336 | 15.1 km | 54.8% | **-1.60 [-2.9, -0.3]** |
| 3 crops @448 | 17.1 km | 53.2% | **-3.27 [-4.8, -1.7]** |

Framing is identical at every size -- each crop covers 56% of the width -- so
only sampling density changes. The further the query encoding sits from the
bank's own 224, the worse it scores. Combined with the blended-tiles query at
+0.2 pp, **the bank's encoding is a contract that cannot be improved from one
side.**

Caveat worth keeping: the projection in that path (`pca768_bank55_pca.npz`) was
fitted on 224-derived vectors, so part of this penalty may be the basis rather
than the encoder features. Untested -- see the open questions below.

## 2. The 2x2, on a 400k bank

|  | bank: crops | bank: crops+tiles |
|---|---|---|
| **query: crops** | 40.7% (incumbent) | 40.1% |
| **query: crops+tiles** | 39.3% | **43.3%** |

| arm | `<25 km` vs incumbent |
|---|---|
| query only | **-1.43 [-2.6, -0.3]** |
| bank only | -0.57 [-1.8, +0.6] ~ |
| **both** | **+2.63 [+1.3, +4.0]** |

Both single-side changes fail; only the matched rebuild works. This **corrects**
an earlier reading on a 96k bank where bank-only looked better than query-only
(+0.9 against +0.2): at 400k that ordering reverses and query-only is separated
*harmful*. The 96k numbers were too noisy to rank.

## 3. The gain grows with bank density

| bank rows | crops | crops+tiles | gain |
|---|---|---|---|
| 25,000 | 21.0% | 22.8% | +1.80 |
| 50,000 | 24.4% | 26.6% | +2.20 |
| 100,000 | 29.4% | 31.4% | +2.03 |
| 200,000 | 35.2% | 37.7% | +2.53 |
| 400,000 | 40.7% | 43.3% | **+2.60** |

**This refutes the concern that motivated the sweep.** The argument was that a
dense bank already holds a near-duplicate for most queries, so a representation
gain measured on a sparse bank would shrink at 3.4M. It grows instead.

The reading that fits: a sparse bank is limited by **coverage** -- no close
match exists and no representation can invent one. A dense bank is limited by
**discrimination** -- the match is present among many near-misses, and finer
features separate them. Hypothesis, but consistent with all five points.

Do not splice the earlier "+3.0 pp at 96k" into this series; it used a
different subset, split and query set.

## Open, and material

* Is the 448 penalty the encoder features or the **PCA basis fitted at 224**?
  Testable at 1536-d with no projection in the path.
* DINOv2's native input is **518** and SigLIP's is **224**, so 448 moves one
  encoder toward its training distribution and the other away. Scored jointly,
  a win and a loss would cancel. Testable per encoder.
* None of this is the agent. The retrieval prior consumes 16 neighbours through
  a trained module; only a retrain says whether a retrieval gain becomes an
  agent gain, and this project has twice found that inference-level reasoning
  predicted the wrong sign.

---

## query_only

```text
bank   pca768_bank70.f16.npy  3,400,180 rows of 3,500,000, 768-d
query  3,000 held-out release images

basis  pca768_bank55_pca.npz (recorded)
query encoding               median km     <1km    <25km   <200km   <750km  <2500km
----------------------------------------------------------------------------
Q1 3 crops @224 (ships)           13.2    13.9%    56.4%    70.5%    77.6%    83.6%
Q3 3 crops @336                   15.1    13.9%    54.8%    69.1%    76.1%    82.6%
Q3 3 crops @448                   17.1    13.1%    53.2%    67.8%    75.1%    82.3%

--- against the shipping query encoding ---
Q3 3 crops @336        +0.00[-0.9,+0.9]~ -1.60[-2.9,-0.3]  -1.40[-2.6,-0.2]  -1.50[-2.6,-0.4]  -1.03[-2.1,+0.0]~
Q3 3 crops @448        -0.80[-1.9,+0.2]~ -3.27[-4.8,-1.7]  -2.70[-4.1,-1.4]  -2.50[-3.9,-1.3]  -1.33[-2.5,-0.1] 

~ spans zero. 799s total
```

## xbank

```text
500,000 tiled release rows: 400,180 bank, 3,000 queries

both representations built in 21s
query / bank                       median km     <1km    <25km   <200km   <750km  <2500km
-----------------------------------------------------------------------------------------
crops / crops   <- the incumbent        77.9     6.3%    40.7%    58.2%    66.9%    76.0%
crops+tiles / crops   <- query only      87.4     6.4%    39.3%    57.3%    66.1%    75.1%
crops / crops+tiles   <- bank only      82.7     6.5%    40.1%    57.2%    66.1%    75.4%
crops+tiles / crops+tiles   <- both      52.6     7.3%    43.3%    61.5%    69.8%    78.5%

--- against the incumbent (crops / crops) ---
crops+tiles / crops        +0.10[-0.5,+0.7]~ -1.43[-2.6,-0.3]  -0.87[-2.0,+0.3]~ -0.73[-1.9,+0.4]~ -0.90[-2.0,+0.2]~
crops / crops+tiles        +0.23[-0.3,+0.8]~ -0.57[-1.8,+0.6]~ -0.93[-2.1,+0.3]~ -0.77[-2.0,+0.5]~ -0.60[-1.7,+0.6]~
crops+tiles / crops+tiles  +1.03[+0.4,+1.6]  +2.63[+1.3,+4.0]  +3.33[+2.1,+4.7]  +2.97[+1.7,+4.3]  +2.50[+1.3,+3.7] 

--- matched gain against bank density ---
bank rows         crops  crops+tiles    gain pp
------------------------------------------------
25,000            21.0%        22.8%      +1.80
50,000            24.4%        26.6%      +2.20
100,000           29.4%        31.4%      +2.03
200,000           35.2%        37.7%      +2.53
400,000           40.7%        43.3%      +2.60

~ spans zero. 169s total
```
