# Live state — updated 2026-09-05, before a compaction

Read this first. `docs/REVIEW.md` through `REVIEW4.md` hold four
Codex reviews; `docs/BACKLOG.md` the code and performance work;
`docs/ARCHITECTURE_NEXT.md` the architecture directions. Durable findings are in
the memory directory. Branch `retrieval-dropout`, PR #13, all pushed.
**383 tests pass. REVIEW3 closed; REVIEW4 arrived with 22 more, 7 fixed (§4).
§7 is ANSWERED: training against the leaking bank *helped*, so 57.5% is the
honest number and not a lower bound. Tile server up.**

---

## 1. The headline finding: the retrieval bank was leaking

`build_knn` offset the bank extension's sequence ids into a separate namespace,
making the release and the extension disjoint **by construction**. They are not
— the extension is the same OSV-5M shards with the same sequence naming, and
**75.5% of test sequences also appear in `bank_ext70`**. Same-sequence exclusion
therefore never crossed the corpus boundary, and the bank served near-duplicate
frames of the query's own drive as retrieval neighbours.

* **41.6% of test queries had a same-drive top-1**, at a median **0.31 km**
* legitimate top-1 sat at a median 28.4 km

Fixed in `build_knn` (factorise both corpora together). **All three caches
rebuilt and verified clean** — 0 same-sequence neighbours across all 32 ranks:

| cache | pairs excluded | per query | top-1 cosine |
|---|---|---|---|
| `pca768_bank70` | 3,722,574 | 7.4 | 0.8261 → 0.7987 |
| `pool_bal_bank70` | 3,722,574 | 7.4 | 0.9080 → 0.8948 |
| `pca768_bank55` | 2,988,188 | 6.0 | (no comparable prior) |

The two bank70 counts being identical is the check that matters: same corpus,
same sequences, so the exclusion must not depend on embedding width.

Old caches kept as `knn_*.LEAKY-preseqfix.npz`, so every published number stays
reproducible. `scripts/seqleak.py` measures; `--verify` asserts a rebuild is
clean and exits non-zero if not.

---

## 2. The corrected benchmark — and a claim of mine to retract

**~18 pp of the OSV-5M headline was the leak.** 5,000 paired test images:

| arm | `<25 km` | median | (was) |
|---|---|---|---|
| `d1536-b350-e6-drop30` | **59.1%** | 13.4 km | 76.9% |
| `d1536-b350-e6-drop70` | 58.3% | 14.6 km | 75.9% |
| `d1536-b350-e6` | 58.0% | 14.1 km | 76.2% |
| `d768-b350-e6-drop30` | 57.8% | 14.4 km | — |
| **`d768-b350-e6-drop70`** (shipping) | 57.5% | 15.4 km | 75.2% |
| `d768-b350-e6` | 57.3% | 14.7 km | 75.6% |
| `d768-b265-e6` | 55.0% | 17.8 km | 72.8% |
| `d768-b350-e6-drop90` | 52.7% | 21.1 km | 69.9% |

**RETRACTION.** I wrote that "the OSV-5M benchmark no longer separates arms."
That was measured on four arms which are near-identical by construction
(same bank, same width, similar dropout) and it does not generalise. The full
eight-arm table spans **52.7% to 59.1%**, and the axes still separate:

* **corpus still pays** — b265 55.0% vs b350 57.3%, +2.3 pp. This survives the
  fix, which I had flagged as needing re-measurement. It does.
* **width still pays** — d1536 58.0% vs d768 57.3%
* **drop90 is clearly bad** — 52.7%, worst of the eight

What is true: among arms differing only slightly, the benchmark cannot separate
them, and it never could. Best on OSV-5M is `d1536-b350-e6-drop30`; best
externally is `d768-b350-e6-drop70`. That split is unchanged by the fix.

**57.5% is the honest number, not a bound.** These arms were *trained* against
the leaky cache, and I expected that to make 57.5% a floor — a model that
learned to lean on near-duplicates should be worse at using honest neighbours.
**Measured, and it is the other way round** (§7): retraining the ladder against
the clean bank scores 53.5–54.6%, separated. Training against the leak helped.
The "lower bound" framing is retracted.

**KartaView is unaffected** (no same-drive frames in the bank) and is the
selection benchmark. Full table, every row on the corrected protocol —
**3,400,180 bank rows**, i.e. the release's val and test images excluded from
the corpus:

| arm | `<25 km` | median |
|---|---|---|
| `d768-b350-e6-drop90` | 14.0% | 390.0 km |
| **`clean2-drop70-e6`** (clean bank, §7) | **13.6%** | **420.0 km** |
| `clean-drop70-e6` (clean bank, confounded, §7) | 13.5% | 420.6 km |
| `d768-b350-e6-drop70` (shipping) | 13.4% | 435.7 km |
| `d1536-b350-e6-drop70` | 12.8% | 467.4 km |
| `d768-b350-e6-drop50` | 12.6% | 468.1 km |
| `d768-b350-e6-drop70-sub2` | 12.5% | 459.7 km |
| `wd29-legacy-e6` | 12.4% | 466.2 km |
| `wd29-fix-e6` | 12.3% | 462.0 km |
| `d1536-b350-e6-drop30` | 12.3% | 492.0 km |
| `d768-b350-e6-drop30` | 12.1% | 503.1 km |
| `d768-b265-e6` (on bank70) | 12.0% | 486.9 km |
| `d1536-b350-e6` | 11.2% | 532.5 km |
| `d768-b350-e6-drop10` | 11.1% | 515.9 km |

**On this benchmark the clean-trained arm does not win either.** It ties the
shipping arm at `<25 km` (13.6% vs 13.4%) and is clearly worse at `<1 km`
(1.0% vs 2.1%). Its visible edge is only over `wd29-fix-e6`, which is a weaker
arm than the shipping one to begin with — so the sign disagreement I read into
an earlier version of this table was mostly a choice of comparator. See §7.

**The `runs/hr_*.npz` exports are all from the SUPERSEDED protocol** — written
09-02, searching all 3,500,000 bank rows. The `hrfix-*` re-runs on 09-03
excluded val/test and are the numbers above, but they exported no per-image
errors. So **no paired KartaView interval can be computed for any arm on
record**, and any bootstrap mixing an `hr_*.npz` with a current run is
comparing across protocols. I made exactly that mistake once; the `<1 km`
"separated" result it produced was an artifact. `scripts/hr_pairs.py` is
re-exporting `wd29-fix-e6` and `d768-b350-e6-drop70` at the corrected protocol
so the §7 KartaView comparison can carry a paired interval.

Note also `hrfix-b265.log` (2,750,000 rows) and `hrfix-b350.log` (3,500,000)
kept the old bank despite the name; their 12.1% / 11.6% are not on this
protocol and are excluded from the table.

---

## 3. #29 weight decay: answered, no effect

The rule `"pos" in name` exempted `retr.q_pos.weight` and `retr.k_pos.weight`
— `nn.Linear` projections, the **learned retrieval keys**, 196,608 params,
3.7% of the model — from weight decay. Now classified by module type.

Two full ladders at `--seed 0`, one flag apart. **Every contrast inside noise**:
OSV-5M `<25 km` [−1.08, +0.10] pp; KartaView `<1 km` −0.18 pp, `<25 km`
+0.16 pp, `<200 km` −0.30 pp, median +4.1 km. No re-baseline needed.

Two things the paired design caught:
* the 2,000-image selection set pointed the **wrong way, monotonically**
  (+0.15 / +0.30 / +0.55 pp) — it picks checkpoints, it does not measure them
* **the seed is the larger effect** — both wd29 arms are separated worse than
  the shipping arm while differing from it only in the seed

`--wd-legacy` restores the old rule. Also corrected: STATE.md had said the
substring caught "exactly two" tensors; it caught **three** — the third is
`map.pos.weight`, a real `nn.Embedding` that must stay exempt, so deleting the
clause would have been a second bug.

---

## 4. Reviews

`REVIEW.md` 86 items: **68 fixed, 8 refuted, 7 deferred, 2 open** (24, 39 —
both need a re-baseline decision, §5).
`REVIEW2.md` 22 items: **all 22 fixed.**
`REVIEW3.md` 10 items: **all 10 fixed**, commit `4fbe408`. Tests 321 → 355.
`REVIEW4.md` 22 items: **7 fixed** (#3, #4, #5, #11, #12, #17, #20), 15 open.
Its own triage header says which. Everything still open is in a file the
clean-bank chain was importing; that chain is finished now, so the block is
workable. #6 and #7 are the reviewer correcting my round-three marker fix:
generic argv-based input/output discovery covers almost no real stage
(`build_knn` records no outputs at all), and per-stage declaration is the fix I
rejected as too large.

### REVIEW3 — closed 2026-09-04, two of them defects in my own work that day

Each was reproduced against the old implementation before being fixed, so the
new tests are known to fail without the fix. The measured reproductions:

| # | the bug, measured | |
|---|---|---|
| 7 | `file_stamp` size discarded — 1 byte and 999,999,999 bytes stamp identically at equal mtime | confirmed |
| 1 | old rule on a 750k extension-only cache | `ACCEPTED UNCHECKED` |
| 9 | old mask rule calls a `2` fetched | `True` |
| 8 | old `marker_matches("{not json}")` → `None`, which the caller reads as satisfied | confirmed |
| 5 | old identity unchanged by a rebuilt bank | `True` |
| 3 | old PCA fit sample: **39,903 of 200,000 rows (20.0%) were val or test** | confirmed |

**#7 and #2 were mine, from earlier the same day.** #7 weakened the error-cache
key I added that morning to fix a *different* staleness bug. #2 contradicts a
claim I wrote into `REVIEW2.md` **and** into a source comment at `train.py:448`;
the *train* set took `neg_random=True` and drew off-path tiles from OS entropy,
so every `--neg 4` run was unseeded. The wd29 conclusion survives — both ladders
were equally unseeded, so it adds noise to a result already inside noise — but
the recorded seed overstated what it covered. Checkpoints now record
`neg_random`, and `bootstrap.py` prints it whenever arms disagree.

**The pattern, which is the useful part:** every one of the ten sits at a
*seam*. One round fixed a producer, the next fixed a consumer, and the hole was
in whichever of the two nobody looked at that day — the completion mask is
validated on write and read back as "nonzero means done"; row provenance is
checked on the one cache length that happens to be the release's.

**What the fixes deliberately do not do:**

* The `pca768_*_pca.npz` bases on disk are **still the transductive ones** (#3).
  Refitting changes the coordinate system, so every 768-d arm would need
  re-measuring against a rebuilt bank before its number could be compared to
  the others. Script fixed; artifacts not rebuilt. **This is a re-baseline
  decision — see §5.**
* Marker input stamps cover the entry script, not its imports (#5). Stamping
  all of `src/` would invalidate every marker on any edit, which in a research
  runner means re-running finished work several times a day.
* `runs/FINAL.md` was regenerated at 00:18, **before** the leak was found, so it
  carries the inflated headline. Marked stale in place; needs regenerating once
  every arm is re-measured.

---

## 5. Needs your decision — do not fix silently

Seven items change what a trained arm *is*. #29 is closed (§3); #24 joined.

* **#24 multi-photo scoring** — one query embedding per group, so candidates a
  *secondary* photo retrieved are scored against a photo that did not retrieve
  them. Fixing changes the prior's interface and invalidates "+1.2 pp for a
  second angle". `multiquery` prints the caveat at N>1.
* **#39 RoPE** gives no relative position (measured: within-offset spread
  0.845 of overall, vs 0.000 for the textbook arrangement). Live for 108 of 122
  checkpoints including the shipping arm.
* **#28** `quality()` sees the unmasked top-1 even when `--retr-drop` hid it.
* **#27** `--save-opt` writes optimizer state nothing loads.
* **#2** `FuseHead.baseline` omits a per-encoder renormalisation.
* **#31** sink coefficient unconstrained (dormant, all `sink_k=1`).
* **#32** memory dropout lacks inverted-dropout scaling.

**A checkpoint cannot say which bank it trained against.** `clean2-drop70-e2`,
`wd29-fix-e6` and `d768-b350-e6-drop70` all record
`knn_file=knn_pca768_bank70_sequence_k32_bank_ext70.npz`. That file was
rebuilt in place at ~10:40 and ~12:00 today, so the same name means the leaky
cache for the arms trained before and the clean one for those after. The only
evidence separating them is the checkpoint mtime against the rebuild time,
which is exactly the "identifier rather than content" failure in §6. The fix is
for `train.py` to stamp the *contents* of its street and k-NN inputs into the
checkpoint (REVIEW4 #6's argument, applied one layer up). It touches
`train.py`, so it waits for the chain — but until it lands, the leaky/clean
provenance of every arm on record rests on file timestamps.

Added by REVIEW3 (§4):

* **REVIEW3 #3 — the PCA bases on disk are transductive.** 20.0% of the
  200,000-row fit sample was val/test (39,903 rows, measured). The *script* now
  fits on the training split; the `pca768_*_pca.npz` files are unchanged.
  Refitting changes the coordinate system, so it is not a drop-in: the bank
  must be re-projected and **every 768-d arm re-measured** before its number is
  comparable to anything on record. That is the entire d768 family, which is
  most of the recent work. PCA is unsupervised so the effect is likely small —
  but "likely small" is a claim, not a measurement, and the cheap version of
  the measurement does not exist. Your call whether to spend the rebuild.

---

## 6. Corrections I made to my own claims today

Recorded because the pattern matters more than any one of them.

1. **"the benchmark no longer separates arms"** — retracted, §2. Generalised
   from four near-identical arms.
2. **top-1 cosine "0.9080 → 0.7987"** — compared the 768-d bank's new value to
   the **1536-d** bank's old one. Real drop is 0.8261 → 0.7987 (−0.027). The
   18 pp result is unaffected; it was measured, not inferred.
3. **"#29 exempts exactly two tensors"** — it was three (§3).
4. **"training passes `neg_random=False`"** — that is val, not train (§4 #2).
5. **the error cache did not depend on the retrieval cache** — the first
   re-measurement returned in 6 s with pre-rebuild numbers to 4 s.f. Third
   instance today of one shape: a cache key that names an identifier rather
   than everything the value depends on.
6. **`eval_highres` printed "a gap here is domain shift, not overfitting"** —
   half wrong; much of the gap was the OSV-5M side inflated.
7. **`bootstrap` labelled every pre-flag checkpoint "by module type"** — a
   missing field read as falsy, so the table added to prevent misreading
   mislabelled all 122 arms.
8. **`file_stamp` "includes size and mtime"** — it never included the size.
   The `[-12:]` slice keeps only the low bits of mtime, so the field I relied
   on that morning to fix a staleness bug was doing half of what its docstring
   said. Same day, same file, one function apart.

### Added 2026-09-05, from the representation line (§8)

9. **"bank-only helps more than query-only"** — measured on a 96k bank
   (+0.9 against +0.2) and stated as an ordering. At 400k it *reverses*:
   query-only is separated harmful, bank-only spans zero. The 96k numbers
   never had the resolution to rank those two and I ranked them anyway.
10. **"the gain will shrink as the bank gets denser"** — my main argument
    against the rebuild, and the sweep I designed to test it refuted it.
    It grows: +1.80 at 25k to +2.60 at 400k.
11. **"native resolution is cheaper than tiling, 3 forwards against 9"** —
    counted forwards as if they were the same size. Native is 1.8x the pixels
    and ~3x the cost, not 4x cheaper. Wrong by an order of magnitude in the
    decision-relevant direction.
12. **"a shared input size must be a multiple of lcm(14,16)=112"** — an
    invented constraint. The encoders run independently and only their 768-d
    outputs are joined, so each can take its own native size. That forced
    every resolution test through 448, a compromise neither encoder wanted.

Numbers 9-12 share a shape too, and it is not the §6 one: **each was a claim
made from a smaller or cheaper measurement than the decision it was feeding.**
The fix is the same each time — measure at the scale of the decision.

Six of the eight are one shape: **a check that names an identifier rather than
everything the value depends on** — a tag, a filename, a length, a prefix, a
substring, a size that gets sliced off. When something here is wrong, that is
the first thing to test for.

---

## 7. ANSWERED: training against the leaking bank helped

The question §2 left open — every arm on record was *trained* against a bank
that served same-drive frames, so what does an arm trained against a clean one
score? Measured twice, and the second run is the one that counts.

`clean2-drop70-e2/e4/e6` (`scripts/seqfix3.py`) retrained the shipping ladder
against the rebuilt cache at `--seed 0` with `--neg-random`, differing from
`wd29-fix-e6` in **nothing but the bank**: same seed, schedule, width, corpus,
weight-decay rule and negative draw.

**OSV-5M**, 5,000 paired test images, every arm scored against the clean bank:

| arm | trained against | `<25 km` | median |
|---|---|---|---|
| `d768-b350-e6-drop70` | leaky | **57.5%** | 15.4 km |
| `wd29-fix-e6` | leaky | 57.3% | 15.1 km |
| `clean2-drop70-e4` | clean | 54.6% | 18.7 km |
| `clean2-drop70-e2` | clean | 54.4% | 18.6 km |
| `clean2-drop70-e6` | clean | 53.5% | 19.8 km |

Every clean-vs-leaky contrast is **separated**: −2.0 to −3.8 pp at the clean
ladder's best rung, −2.8 to −4.9 pp at the matched e6.

**KartaView**, corrected protocol, frozen cohort `14680b9ed911`:

| arm | `<1 km` | `<25 km` | median |
|---|---|---|---|
| `d768-b350-e6-drop70` | 2.1% | 13.4% | 435.7 km |
| `clean2-drop70-e6` | 1.0% | **13.6%** | 420.0 km |
| `wd29-fix-e6` | 1.7% | 12.3% | 462.0 km |

The external set does not rescue it: the clean arm ties the shipping arm at
`<25 km` and is clearly worse at `<1 km`. Its apparent +1.3 pp is only against
`wd29-fix-e6`, which is a weaker arm than the shipping one to begin with.
`scripts/hr_pairs.py` is re-exporting both references so this can carry a
paired interval rather than three point estimates.

### What this means

**57.5% is the honest number for the shipping arm, and it is NOT a lower
bound.** §2 said the opposite and that framing is retracted — an arm trained
on a clean bank scores *lower*, not higher.

A plausible mechanism, untested: near-duplicate neighbours are a strong,
low-noise retrieval signal that teaches the model to *use* the prior, whereas
the clean bank's noisier neighbours teach it to lean on retrieval less. The
leak acted as a curriculum. That is a hypothesis; the table is the measurement.

The clean ladder peaks at **2 epochs** and declines by 6 — in the confounded
run and the corrected one alike — so the reference arms are being compared
against a clean arm past its own peak, and it still loses.

### The first attempt, and what voiding it was worth

`clean-drop70-e2/e4/e6` was void because I committed the sink-negative seeding
change to `train.py` at 12:32:53 and `train-clean-e4` launched at 12:33:38,
so the ladder's rungs were not the same arm (§11).

**The confound turned out to be immaterial.** Confounded `clean-drop70-e6`
measured −2.5 to −4.6 pp against `wd29-fix-e6`; corrected `clean2-drop70-e6`
measures −2.8 to −4.9 pp. Same conclusion, nearly the same interval. Voiding it
was still right — the size of a confound is not knowable before it is removed,
and a result defended by "it probably did not matter" is not a result — but the
honest record is that the re-run reproduced the original rather than overturning
it. The `clean-drop70-*` checkpoints are kept.

---

## 8. The representation line — where the pyramid work landed

All of this is **retrieval probes**, not the agent: encode a query, cosine
against a bank, take the nearest neighbour, measure the great-circle error to
its true location. **No gradient step anywhere.** Every number below is a claim
about the representation. The agent consumes 16 neighbours through a *trained*
retrieval prior, so only a retrain converts any of it into an agent number —
and this project has twice had inference-level reasoning predict the wrong
sign, most recently §7.

### 8a. The blend weight was never set (merged, PR #16)

`fuse_head` combines the level-mean and the learned head as
`concat([l2(base_un), l2(Z)])`. Both blocks are unit-normalised first, so the
cosine is **exactly** `0.5*cos_mean + 0.5*cos_head`. The 0.5 fell out of
concatenating two unit blocks; nobody chose it, and it is ~10x too much head.

* the optimum is a **flat plateau from w=0.02 to 0.15** — the two halves of a
  held-out split disagree about the argmax (0.125 vs 0.02) and the SE at 34%
  over 1,522 queries is ~1.2 pp. **0.03, 0.04, 1/33 and 1/24 are not
  separable.** Use 0.04 as a round number mid-plateau, not as an optimum.
* w=0.05 held out: +0.79 / +1.18 / +2.17 / +2.89 / +2.50 pp at
  1/25/200/750/2500 km, separated at all five.
* **Third instance of the same shape**: raw DINOv2/SigLIP norms 82.96 vs 20.58
  gave DINOv2 81% of the cosine (`--scale-b 4.03` fixed it); `FuseHead.baseline`
  averages levels 1:1:1. See [[blend-weight-is-set-by-block-norms]].

### 8b. Two clean negatives (PR #17 merged, #18 open)

* **Per-step weights do not transfer.** Best-per-threshold over one global
  weighting: +0.46 / 0.00 / **-0.59** / +1.51 / **-0.07** pp held out. Three of
  five zero or negative. Scored *without* the split it looked uniformly
  positive — that difference is the whole result. **Do not build w_t.**
  Useful corollary: where headroom exists, re-ranking the cached top-32 under
  one retrieval captures 100% of it, so four banks were never the obstacle.
* **GeM within a level is inert.** Every p inside noise against the mean, and
  the one separated result is a *loss* (p=2 at 200 km, -1.18 pp). The halves
  disagree on the best p. **Equal averaging of 24 tiles is already right** —
  the first of the three unswept constants that was already at its optimum.

### 8c. crops+tiles on OSV-5M's own data (PR #19 open)

The `sequence` and `cell8` splits, the release's own rows. `L0 alone` was
verified to *be* the shipping representation: `pool_bal` 226.0 km / 27.8%
against `L0` 223.0 km / 27.7%, identical any32.

| split | `<25 km` | `<200 km` | median |
|---|---|---|---|
| `sequence` | **+4.47 [+2.9, +6.0]** | +4.34 | 213 -> 139 km |
| `cell8` (geographic holdout) | **+0.80 [+0.3, +1.3]** | +3.00 | 515 -> 464 km |

The 25 km gain is **5.6x larger on `sequence`**, so much of it is same-region
matching that a geographic holdout strips out. **Quote `cell8` as the
conservative number.**

### 8d. Only the matched rebuild works — the 2x2 at 400k

|  | bank: crops | bank: crops+tiles |
|---|---|---|
| **query: crops** | 40.7% (incumbent) | 40.1% |
| **query: crops+tiles** | 39.3% | **43.3%** |

query-only **-1.43 [-2.6, -0.3]** (separated, *harmful*); bank-only -0.57 ~;
**both +2.63 [+1.3, +4.0]**.

**This corrects an earlier reading of mine.** On the 96k subset bank-only had
looked better than query-only (+0.9 against +0.2) and I reported that ordering.
At 400k it reverses. The 96k numbers were too noisy to rank and I ranked them.

### 8e. Query-only is a dose-response failure — the fixed-bank goal is dead

Against the **live 3.4M bank**, changing only the uploaded image's encoding:

| query encoding | median | `<25 km` | vs shipping |
|---|---|---|---|
| 3 crops @224 (ships) | 13.2 km | 56.4% | — |
| 3 crops @336 | 15.1 km | 54.8% | **-1.60 [-2.9, -0.3]** |
| 3 crops @448 | 17.1 km | 53.2% | **-3.27 [-4.8, -1.7]** |
| crops+tiles blended | — | — | +0.2 (inert) |

Framing is identical at every size (each crop covers **56% of the width**), so
only sampling density changes. **The bank's encoding is a contract that cannot
be improved from one side.**

Also from this run, worth its own line: **top-1 retrieval alone scores 56.4%
`<25 km` where the full agent scores 57.5%.** Most of the agent's accuracy is
already in the nearest-neighbour lookup.

### 8f. The gain GROWS with bank density — my objection refuted

| bank rows | crops | crops+tiles | gain |
|---|---|---|---|
| 25,000 | 21.0% | 22.8% | +1.80 |
| 50,000 | 24.4% | 26.6% | +2.20 |
| 100,000 | 29.4% | 31.4% | +2.03 |
| 200,000 | 35.2% | 37.7% | +2.53 |
| 400,000 | 40.7% | 43.3% | **+2.60** |

I argued a gain measured on a sparse bank would **shrink** at 3.4M, because a
dense bank already holds a near-duplicate for most queries. It **grows**. The
reading that fits: a sparse bank is limited by *coverage* (no representation
invents a match); a dense bank is limited by *discrimination*, which is what
finer features supply. **This removed my main argument against the rebuild.**

Do not splice the earlier "+3.0 pp at 96k" into this series — different subset,
split and query set.

### 8g. A cost estimate I got badly wrong

I said native resolution would be **cheaper** than tiling ("3 forwards against
9", ~3-4 h against ~16 h). That counted forwards as if they were the same size.

| scheme | pixels/image | measured |
|---|---|---|
| crops + tiles @224 | 9 x 224^2 = 451,584 | 43.5 img/s |
| crops @518/512 | 3 x 518^2 = 804,492 | ~33.5 img/s |

Native is **1.8x the pixels** and attention is superlinear, so it is roughly
**3x more expensive than tiling, not 4x cheaper.** If resolution wins on
quality it is a genuine trade, not a free lunch.

### 8h. Per-encoder native sizes — a constraint I invented

I claimed a shared input size must be a multiple of lcm(14,16)=112, forcing
448. **Nothing requires a shared size**: the encoders run independently and
only their 768-d outputs are concatenated. Frames are **512 tall** (widths
682-1228), so:

* **SigLIP /16 at 512** = 32x16 — the frame's own pixels, *no resampling*
* **DINOv2 /14 at 518** = 37x14 — its **exact native** training resolution

DINOv2's default `img_size` really is 518; the pipeline has been forcing 224
since the beginning, i.e. **44% of linear resolution, ~80% of pixels
discarded.** At 448 DINOv2 sat 14% below its native grid, so the -3.27 pp was
measured at a compromise size neither encoder wanted.

### 8i. Throughput work (applies to the rebuild)

`18.2 -> 33.5 img/s`, **1.84x**, from two independent fixes:

* **prefetch** — preprocessing ran serially with the forward, so the card
  sawtoothed 100% -> 0%. A thread pool one batch ahead (8 workers; PIL and
  numpy release the GIL) gave 1.36x. Total CPU was 25% of 16 cores: one core
  doing all the JPEG decode, 921x518 resize and float32 normalise.
* **bf16 weights, not autocast** — `.to(torch.bfloat16)` on the model gave
  another 1.36x and VRAM 2.30 -> 1.48 GB. Autocast re-casts fp32 weights on
  every op; this pays once.

**Ruled out by benchmark:** batch size (saturated at 16; 48 is no faster and
2.5x the VRAM), `channels_last` (a no-op on a ViT), fused attention
(**already on** — timm 1.0.28 + torch 2.6 flash SDPA).

**Why it will not reach 300 W:** 100% util at 225 W with 50% memory-controller
load means SMs resident but stalling, and neither compute nor bandwidth is
saturated. The limiter is the memory-bound elementwise work between the
matmuls (layernorm, GELU, softmax, residuals), which does not scale with batch.
**`torch.compile` is the one untested lever** — it fuses exactly those chains.
Worth benchmarking **before** the rebuild, where 20% is 4 h of 19 h.

---

## 9. ANSWERED: build tiles, not resolution

`scripts/resmatch.py --rows 50000 --queries 3000` -- four **matched** banks
(query and bank built identically, because §8d showed nothing else is
informative), 39,831 bank rows, 3,000 held-out queries. Full write-up in
`runs/RESMATCH.md`.

| arm | median km | `<1km` | `<25km` | `<200km` |
|---|---|---|---|---|
| 1  crops @224 (incumbent) | 392.4 | 1.2% | 19.8% | 39.9% |
| **2  crops+tiles @224** | **342.1** | 1.3% | **21.1%** | **42.9%** |
| 3  crops @518/512 native | 418.5 | **1.5%** | 19.4% | 39.3% |
| 4  native crops + 224 tiles | 370.2 | **1.6%** | 20.8% | 41.8% |

```
2  crops+tiles @224      +0.17~  +1.27[+0.4,+2.2]  +3.00[+1.7,+4.3]
3  crops @518/512 native +0.37   -0.37[-1.4,+0.7]~ -0.57[-2.0,+0.9]~
4  native crops + 224 t. +0.40   +0.97[-0.1,+2.0]~ +1.90[+0.6,+3.3]
```

**Arm 3 closes two open questions at once.** It is matched, so "the query was
mismatched against a 224 bank" is out; and it has **no PCA in the path**, so
"the basis was fitted on 224-derived vectors" is out. Both were live
explanations for the -3.27 pp at 448 (§8e). With both removed, native
resolution is still flat-to-negative at every threshold from 25 km up and its
median is *worse* than the incumbent's. **More pixels of the same framing do not
carry more locatable signal.** Note this also rules out "the encoder was never
trained on more pixels" *for this arm*: 518 is exactly DINOv2's training
resolution, so it is the encoder at home, not extrapolating.

**The one exception points the other way, and is not closed.** `<1 km` is the
only threshold where resolution wins, and it wins in **both** arms that use it
(+0.37, +0.40, separated) while tiles do nothing there (+0.17 ~). Finer sampling
buys discrimination among near-duplicates. That is a 1.2% base on a 39,831-row
bank where near-duplicates barely exist -- but the shipping bank is 3.4M and
lives much more in that regime (top-1 alone is 56.4% `<25 km` there). **The one
result here a denser bank could amplify rather than dilute. Re-measure it after
the rebuild.**

**Arm 4 answers "why not both": no.** +0.97 pp is *below* arm 2's +1.27 and no
longer separated. Resolution does not complement tiles, it dilutes them, at ~3x
the cost (§8g). Caveat in its favour: it mixes native crops with *224* tiles, so
its ceiling is higher than measured -- but it does not currently clear arm 2 at
all.

**DECISION: tiles at 224, the ~19 h extension pass.** Read +1.27 pp as a
**floor**: §8f measured the matched gain *growing* with density (+1.80 at 25k ->
+2.60 at 400k) and this bank is 39,831 rows, so arm 2 is measured near its
weakest. Do not splice the two series -- different subset, split and query set.

---

## 9b. Next steps, in order

1. ~~Read `resmatch` and pick what to rebuild.~~ **Done: tiles at 224 (§9).**
2. ~~Write the id-to-shard addressing.~~ **Already built — a claim of mine
   corrected.** I looked at `bank_ext70_meta.npz` (which really does carry only
   `image_id`, `x16`, `y16`, `sequence` and a shard list) and concluded the
   per-image mapping had to be reconstructed by reading 60 zip directories. It
   does not: `build_bank_ext.py:113` writes a **parquet beside** the npz with
   `zip_name` already materialised as `<shard>/<image_id>.jpg`.

       data/processed/s10/bank_ext{,2,3,4}.parquet   750,000 rows each

   Concatenated in that order they are **3,000,000 rows in the exact row order
   of `bank_ext70_meta.npz`** (`np.array_equal` on `image_id`: True), covering
   all 60 shards. So the remaining work is not a build but a flag:
   `tile_cache.py:145` hardcodes `config.DATASET_PARQUET`, and wants a
   `--rows-parquet` that defaults to it and is recorded in the cache metadata.
   Spot-checked 200 random paths across 15 shards: **200/200 readable, 0
   missing.** **Same shape as §6: I checked the artefact that *names* the
   thing rather than every artefact the thing is written to.**
3. **Benchmark `torch.compile`** before committing (§8i) — the only untested
   throughput lever, and 20% is 4 h of 19 h.
4. **Rebuild** at the winner: release + extension, PCA refit **on the training
   split**, `stack_bank`, kNN rebuild (~1 h after the embedding pass).
5. **Retrain the ladder, ~80 min.** The only step that yields an agent number.
6. **Bootstrap + KartaView**, ~40 min, on the frozen cohort `14680b9ed911`.

### State of the caches

* **`tile6` is complete**: 500,000 x 6 x 1536, all release rows tiled
  (380,000 embedded this session at 23.0 ms/img, 0 unreadable).
* The **extension has no tiles at all** — 96,091 of the shipping bank's
  3,400,180 rows (2.8%) are tiled.
* `complete_rows()` in `osv_pyramid.py` reads the done mask and returns only
  finished rows. **Necessary during a pass**: `tile_cache` grows in place, so a
  cache caught mid-run is mostly zero fill, and a zero row L2-normalises to a
  unit-length nothing that ranks like a real vector. My first blocked loader
  omitted that check and would have built a bank 76% zeros.

### Branch and PR state

On **`osv-pyramid-blend`**, last commit `5ad689c`. Per the user's instruction:
**new branch and a new PR per change, all targeting `main`.**

* **#16 merged** — the blend weight
* **#17 merged** — per-step does not transfer
* **#18 open** — `pyramid-gem-pooling`, GeM is inert
* **#19 open** — `osv-pyramid-blend`, crops+tiles on OSV-5M, the demo
  `/compare` panel, the completion-mask guard, the 2x2 and density sweep

403 tests pass. `runs/` holds `PYR_BLEND.md`, `PYR_PERSTEP.md`, `PYR_GEM.md`,
`OSV_PYRAMID.md`, `XBANK.md`.

### The demo

`py scripts/serve.py --tag d768-b350-e6-drop70 --tiles-panel` — the agent plus
a `/compare` panel showing the same image under crops and crops+tiles over the
96,091 tiled bank rows, ~0.07 s per request. **Currently stopped** so the GPU
is free. Never stack GPU jobs: 2.6x measured cost.

---

## 9c. REVIEW4 — still 15 items open

7 of 22 fixed (#3, #4, #5, #11, #12, #17, #20). Open and untouched:
**#1, #2, #6, #7, #9, #10, #13, #14, #15, #16, #18, #19, #21, #22** plus the
second half of #8. Five are high severity. **#6 and #7 are the reviewer
correcting my round-three marker fix** — generic argv-based input/output
discovery covers almost no real stage (`build_knn` records no outputs at all),
and per-stage declaration is the fix I rejected as too large.

`docs/REVIEW4.md` carries a triage header with the measured reproductions.

---

## 10. Tools and tests written this session

`scripts/seqleak.py` (+`--verify`), `seqfix_eval.py`, `seqfix2.py`, `wd29.py`,
`backfill_prov.py`, `parity_report.py`, `pair_npz.py`, `rope_probe.py`,
`cond_probe.py`, `fuse_init_probe.py`; `src/provenance.py`, `src/runlog.py`,
`src/safeio.py`; `bootstrap.py --retr-off`, `train.py --seed/--wd-legacy`,
`multiquery --calib`.

Tests (315): `test_provenance.py`, `test_runlog.py`, `test_param_groups.py`,
`test_merge_candidates.py`, `test_written_row_guards.py`, `test_argparse_help.py`,
`test_split_hash_and_cache.py`, `test_checkpoint_contract.py`, `test_topk_bounds.py`,
`test_import_order.py`, `test_safeio.py`, `test_cache_completeness.py`.

---

## 11. Environment

* RTX 3070, 8 GB. Training is GPU-bound, 90–96% at 165–176 W. **Never stack
  GPU jobs** — 2.6× measured cost. kNN rebuild ~5–7 min; a training rung ~26 min.
* **Use `py`.** Three names, two interpreters: `py` is the Windows launcher
  (`C:\Windows\py.exe`) and resolves to Python 3.13 with numpy 2.3.5, torch
  2.6.0+cu126 and CUDA — the project stack. Both `python` and `python3` are
  MSYS2's `mingw64` 3.12 and have none of it. `py -m pytest tests/ -q` runs the
  suite; the hardcoded
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe` is the
  same binary and only needed where a launcher is not wanted. This is what the
  round-three reviewer hit when they reported the suite could not be run —
  `py` was there the whole time.
* Tile server `192.168.50.1:3000`. `OSV_RELEASE` is required, no default.
* `.gitignore` now covers `runs/*.npz`, `runs/errs_prebug/`, `.pytest_cache/`,
  `.claude/`, safeio `.tmp<pid>` and `.grow.npy` leftovers.

---

## 12. Rules learned the hard way

* **A cache key must name everything the value depends on**, not everything
  that looks like an identifier. Three instances today (§6.5).
* **A fix is not done until it is tested against the failure it prevents.**
* **Ask whether a measurement error is independent of the treatment.**
* **Never generalise a null from arms that are near-identical by construction**
  (§6.1). Pick contrasts that span the axes before concluding a metric is blind.
* **Never compare two numbers from different embedding spaces** (§6.2).
* **Do not chain an unverified edit into a background job.** A patch script
  asserted, wrote nothing, and the run that followed died on an unrecognised
  flag with the error nowhere visible. Verify the edit landed (`grep` for the
  new symbol, or `--help`) *before* launching anything that depends on it.
* **Large heredocs break.** Backslashes arrive mangled and long content trips
  the parser outright. Use the Write tool for anything over a few lines, and
  the Edit tool for precise source surgery.
* **Sample a GPU faster than you think you need to.** Two-second polling
  showed a steady 99-100% while the card was actually sawtoothing 100 -> 0;
  the serial data pipeline was invisible until sampled at 1 s. `utilization`
  means "a kernel is resident", not "the SMs are busy" — read it beside power
  draw and memory-controller load.
* **Never `git stash`** — it swept uncommitted work out from under a diagnostic.
  Use `git show <rev>:<path>` to a scratch file.
* **Never edit code that a running chain has not finished importing.** Each
  stage is a fresh subprocess, so an edit lands on every stage that starts
  after it and none that started before. This voided the clean-bank result:
  `train.py`'s negative seeding changed at 12:32:53, `train-clean-e4` launched
  at 12:33:38, and the ladder's three rungs were then not the same arm. Fixing
  a review item and running an experiment are both fine; doing them in the
  same hour on the same file is not. Check `runs/logs/*_runner.log` for an
  in-flight chain before touching `src/`, or stage the edit and commit after.
* **A guard defined and never called is not a guard.** Nor is one that fails open.
* **Never type a backslash inside a Bash heredoc** — it arrives as a newline.
  Build it as `chr(92)` or write the patch script with the Write tool.
* **Scope every log check to the newest `==== attempt`.**
* **Do not block on `WaitForExit` in the PowerShell tool**; detach with
  `Start-Process` and poll.
* The **2,000-image selection set is not evidence**; it picks checkpoints, and
  it produced a clean monotone trend out of pure noise today (§3).
* **A wide interval means get more data, not weaken the claim.**
