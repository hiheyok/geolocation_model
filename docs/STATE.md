# Live state - updated 2026-09-05 evening, before a compaction

Read this first. `docs/REVIEW.md` through `REVIEW7.md` hold seven Codex
reviews; `docs/BACKLOG.md` the code work; durable findings are in the memory
directory.

**521 tests pass. Branch `pyramid-derisk`, PR #26; PR #25 open for the mapping
service. #16-#23 merged.**

**THE HEADLINE (SS9b): crops+tiles beats crops at the AGENT level, +2.22 to
+4.64 pp on `<25 km`, separated, on a 400,180-row bank.** That is the first
agent-level confirmation of the whole pyramid line. Absolute numbers there
compare to nothing on record; only the paired contrast does.

**RUNNING (SS9d):** the native-resolution conditioning encode, SigLIP half,
done ~22:30. **It is not resumable** - that process predates the fix.

**OWED (SS9c):** `RESMATCH`'s "resolution is inert" used an out-of-distribution
SigLIP and needs re-measuring. The L2 weight sweep selected and reported on one
cohort. "+2.03 is a floor" is already retracted.

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

## 9b. THE HEADLINE: tiles reach the agent

`scripts/tiledrisk.py`, two ladders one flag apart on a **400,180-row** bank
every row of which is tiled in `tile6`. Paired bootstrap, 5,000 test images:

| contrast | median | `<25 km` |
|---|---|---|
| `pyrL0` vs **`pyrL0L1`** at e6 | [+9.7, +20.3] km | **[−4.64, −2.22] pp — separated** |
| `pyrL0` vs **`pyrL0L1`** at e4 | [+7.8, +17.3] km | **[−4.26, −2.00] pp — separated** |
| e6 vs e4 *within* each arm | inside noise | inside noise |

Sign convention: positive `<25 km` means the **first** arm is better, so
crops+tiles wins by **+2.22 to +4.64 pp**. Epoch contrasts flat inside each arm,
so it is the representation and not training length.

**Absolute numbers here compare to nothing on record.** The bank is 400,180
rows against the shipping 3,400,180, and corpus is the axis that has moved this
metric most. Only the paired contrast means anything.

Per-step val accuracy (5,000 images, teacher-forced) says the gain is at the
**coarse** steps: s0 +1.3, s1 +1.0, s2 +0.2, s3 −0.2 at e6, click MAE
unchanged. That fits the rescue finding below and contradicts a prediction I
made from the threshold profile.

### Retrieval-level, in the caches the ladders trained on (`runs/KNN_GAP.md`)

49,788 test queries. The production path applies a PCA to 768 the probes never
did, and it **preserves the gain**:

| window | `<1 km` | `<25 km` | `<200 km` |
|---|---|---|---|
| top-1 | +0.29 | **+2.76 [+2.5, +3.0]** | +3.10 |
| any-of-16 (the agent's) | +0.67 | **+2.03 [+1.8, +2.3]** | +0.71 |
| any-of-32 | +0.78 | +1.45 | +0.36 |

**It is churn, not a lift**: at top-1, 6.57% of queries won and 3.80% lost, a
1.73x ratio. **And the wins are rescues** — of the queries crossing 25 km, the
crops error they came from had median **379 km**, p90 **6,031 km**. Tiles fix
gross failures rather than refining near-misses.

**A per-query gate cannot capture the rest.** Oracle arm-selection is worth
+6.57 pp against the blend's +2.76, but all three observable rules score
*worse* than doing nothing, and the gain is positive in every confidence
bucket so no threshold rule exists either. Fourth failed attempt at a
conditional blend; a single global constant keeps winning.

---

## 9c. Two claims of mine that are weaker than I presented them

**`RESMATCH`'s "resolution is inert" needs re-measuring.** Arm 3 was DINOv2 at
its genuine native 518 paired with `vit_base_patch16_siglip_224.v2_webli` run
at `img_size=512` — a 224 checkpoint with interpolated position embeddings, not
a native model. That makes the null unreadable: "more pixels do not help" and
"this checkpoint cannot use them" look identical. `timm` ships a real
`vit_base_patch16_siglip_512.v2_webli`; `embed_native.py` now uses it.
**~45 min to re-run. It does not change the tiles decision** — arm 2 beat arm 1
on its own — but the resolution claim should not be quoted until it is redone.

**The L2 weight sweep selected and reported on one cohort** (REVIEW5 #8). The
direction saves the conclusion — an inflated maximum that still does not
separate makes a *null* conservative — but the interval is not pre-registered.

**Already corrected earlier**: "+2.03 is a floor" is retracted.
`runs/GAIN_DENSITY.md` tested the mechanism behind it (coverage giving way to
discrimination) by varying *local* density inside one fixed bank and found the
gain **flat across four orders of magnitude** — +2.71 [+2.1,+3.3] at 10–99 bank
rows per cell against +2.82 [+2.5,+3.2] at 100–999, over 47,107 queries. Local
and global density are different manipulations so it is not a refutation, but
it is the only independent check and it fails. **Whether the gain grows past
400k is open, which is exactly what `tilebig` would measure.**

---

## 9d. RUNNING NOW

`scripts/embed_native.py --out cond_native` — 500,000 release rows at each
encoder's own native size, **conditioning only, never a retrieval bank**.

* DINOv2 @518 **finished**, 16,001 s.
* SigLIP @512 running at ~37 img/s, **done ~22:30**.
* **NOT resumable.** The process was launched at 13:48; the two-column,
  per-shard-flushed mask landed at ~15:40. The mask on disk is `(500000,)
  sum 0`. If it dies, all of it is lost and the restart uses the fixed code.
* Both encoders log as `patch1` — `spec.split("_")[2]` collides. Cosmetic.

---

## 9e. The decoupled conditioning experiment (built, blocked three times, ready)

`runs/PYR_LEVELS.md` closed the obvious route: pushed **into** the retrieval
vector against a fixed `L0L1` bank, an extra level at weight 0.10 replaces 8.2%
of the top-16 and flips the top-1 for 12.7% of queries, and the hit rate does
not move at all (59.6% → 59.6%). It reorders without being about location.
Retrieval is offline, so a head cannot un-retrieve what the cosine chose.

So conditioning rides **beside** the retrieval vector, in one tensor:

    [ retrieval 768 | conditioning 1536 ] = 2304-d, StreetProj splits it

One tensor rather than a second batch field is deliberate — `fuse_flat` records
that beam search once inlined the fusion concatenation and drifted, so
threading a new argument through `fuse`/`policy_from`/`forward`/`beam.search`
would rebuild that hazard. A wider tensor changes no signature anywhere.

**Zero-gated**, so loading `pyrL0L1-e6` is bit-identical to what it was —
asserted with `torch.equal`, because a fine-tune must start *at* the trained
point. Not the zero-gate deadlock: the table is randomly initialised, so the
gate has gradient at step 0 (measured 3.8e-06) and `cond_proj` starts once it
moves.

### The plan, when the encode lands

1. `make_cond --retrieval pyr768_mix.f16.npy --cond cond_native.f16.npy --out
   pyr768_mix_cond.f16.npy` — minutes.
2. `pyrL0L1-e8` — +2 epochs, **no adapter**. The control is not optional or the
   comparison confounds conditioning with two more epochs. ~52 min.
3. `pyrL0L1-cond-e8` — +2 epochs with the adapter, `--init pyrL0L1-e6`. ~52 min.
4. Paired bootstrap over e6 / e8 / cond-e8. ~20 min.

**Prior: small.** With retrieval off the system collapses 75.6% → 2.7%, and
top-1 alone scores 56.4% where the agent scores 57.5%. The conditioning path
has historically carried very little. That is what makes it a real experiment.

### L2 is not the conditioning, and cannot be

A 6×4 grid of 224 tiles needs 1344×896 and **0% of OSV-5M frames have it**
(910×512 and 682×512 are 90% of them). 4×2 is native for only 51%, and a
half-native/half-upsampled corpus is the two-halves failure again. **3×2 is the
widest grid the corpus supports.** Native resolution is the honest
"more pixels" on this data.

---

## 9f. Reviews: three more rounds

| round | found | fixed |
|---|---|---|
| REVIEW5 | 10 (6 high) | 3 — #1, #6, #10 |
| REVIEW6 | 10 (7 high) | 6 — #1, #2, #5, #7, #8, #9 |
| REVIEW7 | structural, 11 | 3 taken: #5, #6, #9 |

**Three were blockers in the conditioning code**, i.e. the experiment could not
have started: REVIEW5 #6 (`--init` allowlist missing the adapter's five keys),
REVIEW6 #1 (`retr_prior` sliced the query's suffix but not the neighbours', so
`pos`/`dual` raised on the first batch), REVIEW6 #5 (default k-NN name derived
from the joined cache).

**Two of them passed my tests because the tests asserted *near* the thing
rather than *through* it** — `load_state_dict` instead of the production
filter, "some parameter is retrieval-width" instead of calling the prior. Both
now go through the real path and were checked non-vacuous by reproducing the
pre-fix failure. This is the recurring failure mode of the session.

**Still open.** REVIEW5 #2, #3, #4, #5, #7, #8, #9. REVIEW6 #3, #4, #6, #10.
Twelve from REVIEW4. REVIEW7 #1, #2, #3, #4, #7, #8, #10, #11.

**REVIEW6 #3 is the substantive one**: a k-NN cache is not bound to the *bytes*
of the street file that produced it. Rebuild a street cache under the same name
and the k-NN silently addresses the old embedding space. `safeio.content_digest`
now exists (1,463 MB/s, so 0.53 s for a 0.77 GB cache); wiring it into
`build_knn` and `knnmeta` is the fix. **On the rebuild path, so it matters
before `tilebig`.**

### An audit: none of it reached a published number

`scripts/audit_knn.py`, 16 caches. No street file newer than the table built
from it; bank rows in range, unique, non-negative; every neighbour id a member
of `bank_rows`; extension identity matching by ordered digest; both pyramid
caches hashing `exact` against the live split. `pyr47` complete, 0 zero rows.
All five pyramid sidecars match the release digest.
**The mtime test is one-sided** — it cannot see a rewrite that preserved mtime,
which is why #3 asks for a content digest rather than this.

---

## 9g. The mapping service now has an identity

PR **#25** (open, mergeable) brings the service into this repo as
`services/mapping/`, done on the MacBook that hosts it. Verified byte-identical
across the move on **both** deployments from **cold renders** — Linux via an
empty `OUTPUT_DIR`, Mac via forced cache eviction, so neither replayed a cache.

**`config.TILE_SERVERS`** is now a candidate list; `tiles.connect(bases)`
returns a client for the first answering `/health` and **announces a fallback**.
`10.0.0.84:3000` is the backup (4.1 ms/tile primary, 34.2 ms backup).

**The two deployments render differently, and it is now attributable.**
Identical `cacheNamespaces`, identical `sources.digest`, identical mbtiles
sha256 across 75 GB, identical version string `6.4.1` — but different
`maplibre_native_digest`, so `r-e42ea3b3bbf5` against `r-3a8cfb6e81c9`. Masks
differ on **0.03–0.28% of pixels** at every zoom, all on feature boundaries.
Benign where the model reads: **max 7.8e-3** on the 12-d class fractions.
`cacheNamespaces` fingerprints configuration, not rendering, and cannot be an
identity.

`TileClient.compare_with` reports magnitude at the token level for that reason
— a byte comparison, which is the obvious implementation, called all five masks
different and would have condemned an interchangeable backup.

**The existing 626,284-tile map cache is attributed empirically**, not assumed:
50 of 50 decisive tiles match `r-e42ea3b3bbf5` at exactly `0.00e+00`, 0 match
the other; 10 ties where the renderers agree are counted and excluded.
`cache/map/s10/renderer.json` records the evidence beside the id.
`tiles.check_renderer` binds a cache on first use and **refuses** a second
renderer after — verified live.

---

## 9h. Next steps, in order

1. **REVIEW6 #3 and #4**, plus REVIEW5 #3/#5 — CPU only, do while the GPU is
   busy. **#5 matters before `make_cond` runs.**
2. **The conditioning ladder** (§9e) when the encode lands, ~22:30.
3. **The `--w` sweep on L0/L1**, 15 min. The 0.5 in `l2((L0+L1)/2)` was never
   chosen, and the same unchosen constant just moved L2 by 2.3 pp.
   **`tilebig` would bake it into a 4 h pass and a 1.15M rebuild, so this runs
   first or not at all.**
4. **`tilebig`**, ~9 h. Opens with the **cell8 geographic holdout** (~15 min, on
   caches already on disk) before committing to the tile pass — §8c found the
   25 km gain 5.6x larger on `sequence` than `cell8`. Then 750k tiles → a
   complete 1,150,180-row bank → `knn_gap` → paired ladders. **The 1.15M
   question and the geographic holdout are both still unanswered.**
5. **Re-measure `resmatch`** with the native SigLIP-512, ~45 min. Corrects the
   record; gates nothing.

---

## 9i. State of things

**521 tests pass.** Branch `pyramid-derisk`, **PR #26** open (9 commits).
**PR #25** open for the mapping service. #16–#23 merged.

Caches: `tile6` complete for all 500,000 release rows. The 3M extension has no
tiles. `bank_ext{,2,3,4}.parquet` carry `zip_name` for all 3,000,000 extension
rows in `bank_ext70_meta.npz` order — the addressing is **already built**, and
`tile_cache --parquet` can reach it.

New this session: `resmatch`, `tilebig`, `tiledrisk`, `pool_pyramid`,
`knn_gap`, `gain_density`, `pyr_levels`, `embed_native`, `make_cond`,
`tilebench`, `audit_knn`, `which_renderer`; `src/knnmeta.py`,
`NeighborBatch`, `safeio.content_digest`, `tiles.connect`/`check_renderer`.

---

## 11. Environment

* RTX 3070, 8 GB. Training is GPU-bound, 90â€“96% at 165â€“176 W. **Never stack
  GPU jobs** â€” 2.6Ã— measured cost. kNN rebuild ~5â€“7 min; a training rung ~26 min.
* **Use `py`.** Three names, two interpreters: `py` is the Windows launcher
  (`C:\Windows\py.exe`) and resolves to Python 3.13 with numpy 2.3.5, torch
  2.6.0+cu126 and CUDA â€” the project stack. Both `python` and `python3` are
  MSYS2's `mingw64` 3.12 and have none of it. `py -m pytest tests/ -q` runs the
  suite; the hardcoded
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe` is the
  same binary and only needed where a launcher is not wanted. This is what the
  round-three reviewer hit when they reported the suite could not be run â€”
  `py` was there the whole time.
* Tile servers: `config.TILE_SERVERS` = `192.168.50.1:3000` (primary,
  4.1 ms/tile) then `10.0.0.84:3000` (backup, 34.2 ms). Use
  `tiles.connect(config.TILE_SERVERS)`, never a bare `TileClient`, so the
  fallback happens and announces itself. They are DIFFERENT RENDERERS --
  `r-e42ea3b3bbf5` and `r-3a8cfb6e81c9` -- and a map cache is bound to one
  by `cache/map/<rel>/renderer.json`. `OSV_RELEASE` is required, no default.
* `triton-windows` is installed (3.2.0.post21, pairs with torch 2.6), so
  `torch.compile` works -- it needs MSVC on PATH:
  `C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/
  14.44.35207/bin/Hostx64/x64`. Inductor logs "Not enough SMs to use
  max_autotune_gemm" on a 3070; compile is worth ~1.18x.
* `.gitignore` now covers `runs/*.npz`, `runs/errs_prebug/`, `.pytest_cache/`,
  `.claude/`, safeio `.tmp<pid>` and `.grow.npy` leftovers.

---

## 12. Rules learned the hard way

* **Assert through the thing, not near it.** Three times today a test of mine
  passed while the production path was broken: `load_state_dict` instead of the
  `--init` filter, "some parameter is retrieval-width" instead of calling the
  prior, and a union over call sites that the *val* dataset satisfied on its
  own. Call what `main` calls, and **prove the test fails against the pre-fix
  code** before believing it.
* **Never type a backslash inside a Bash heredoc.** Three times today: it
  arrives as a real newline and produces an unterminated string literal. Use
  `chr(10)`, or the Write/Edit tools. This rule was already written down.
* **A completion signal that only fires on the happy path is not one.**
  `O.SAMPLER.stop()` did not exist; it raised *after* every stage finished, so
  no result was lost — but the runner died on a traceback instead of logging
  the line the monitor watched for, nothing fired, and **6.5 h of GPU sat
  idle**. Monitors must match failure signatures too.
* **Verify the harness, not just the payload.** Every flag in the stage
  commands was `--help`-checked; the runner's own last two lines never were.
* **A cache key must name everything the value depends on**, not everything
  that looks like an identifier.
* **A fix is not done until it is tested against the failure it prevents.**
* **Ask whether a measurement error is independent of the treatment.**
* **Never generalise a null from arms near-identical by construction.**
* **Never compare two numbers from different embedding spaces.**
* **Do not chain an unverified edit into a background job.**
* **Sample a GPU faster than you think you need to.** 2 s polling showed a
  steady 99–100% while the card sawtoothed; at 100 ms the shape was a ~20 s
  producer/consumer oscillation. `utilization` means "a kernel is resident".
* **Never `git stash`** — use `git show <rev>:<path>` to a scratch file.
* **Never edit code a running chain has not finished importing.**
* **A guard defined and never called is not a guard. Nor is one that fails
  open.** Nor one whose sample has a fixed seed: 64 deterministic rows of
  500,000 skip the same 499,936 forever.
* **Scope every log check to the newest `==== attempt`.**
* **Do not block on `WaitForExit` in the PowerShell tool.**
* The **2,000-image selection set is not evidence.**
* **A wide interval means get more data, not weaken the claim.**
* **Claims must not outrun the measurement that produced them.** Eight
  corrections on record now come from this one shape.
