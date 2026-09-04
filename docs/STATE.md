# Live state — updated 2026-09-04

Read this first. `docs/REVIEW.md` and `docs/REVIEW2.md` hold the two Codex
reviews with every item marked; `docs/BACKLOG.md` the code and performance
work; `docs/ARCHITECTURE_NEXT.md` the architecture directions. Durable findings
are in the memory directory. Branch `retrieval-dropout`, PR #13, all pushed.
**277 tests pass. Nothing is running. Tile server is up.**

---

## 1. What to ship

**`d768-b350-e6-drop70`** — 768-d street vector, 3.40M bank, 6 epochs,
`--retr-drop 0.7`. Unchanged as the recommendation, but **choose arms on
KartaView, not on OSV-5M** — see §2.

| arm | OSV-5M `<25 km` | KartaView `<25 km` | KartaView median |
|---|---|---|---|
| `d768-b265-e6` | — | 12.1% | 474.8 km |
| `d768-b350-e6` | 57.3% | 11.6% | 505.6 km |
| **`d768-b350-e6-drop70`** | **57.5%** | **13.4%** | **435.7 km** |
| `d768-b350-e6-drop90` | — | 14.0% | 390.0 km |
| `d768-b350-e6-sub2` | — | 12.5% | 459.7 km |
| `d1536-b350-e6` | — | **11.2%** | **532.5 km** |
| `d1536-b350-e6-drop30` | — | 12.3% | 492.0 km |
| `d1536-b350-e6-drop70` | — | 12.8% | 467.4 km |

OSV-5M figures are **post-leak-fix** (2026-09-04) and only the two arms
re-measured so far are shown; the rest are stale by ~18 pp and are not
reproduced here. All KartaView numbers are at shipping parity and are
unaffected by the leak.

**The OSV-5M column can no longer separate these arms.** Every pairwise
contrast among the four re-measured arms is inside noise, and the spread across
all four is 0.2 pp against 0.7 pp before. KartaView still separates them, and
it always ranked `drop70` first — that ranking was the honest signal all along.

## 2. The results that matter

**The OSV-5M headline was inflated ~18 pp by a leaking retrieval bank.**
`build_knn` offset the extension's sequence ids into a separate namespace,
making the two corpora disjoint by construction rather than by fact — they are
the same OSV-5M shards with the same sequence naming, and 75.5% of test
sequences also appear in `bank_ext70`. So same-sequence exclusion never crossed
the boundary and the bank served near-duplicate frames of the query's own drive
as retrieval neighbours: **41.6% of test queries had a same-drive top-1, at a
median 0.31 km**. Fixed, cache rebuilt and verified clean (0 same-sequence
neighbours across all 32 ranks; 3.7M pairs excluded, 7.4 per query; mean top-1
cosine 0.8261 → 0.7987).

*A correction to my own first report of this:* I first wrote that drop as
0.9080 → 0.7987, which compared the 768-d bank's new value against the
**1536-d** bank's old one — two different embedding spaces. The real drops are
0.8261 → 0.7987 at 768-d and 0.9080 → 0.8948 at 1536-d. That makes the result
more striking, not less: a 0.027 change in mean top-1 similarity is worth
18 pp, because only 41.6% of queries had a leaked top-1 and the honest
replacement is usually still a reasonable match — just not a photograph taken
seconds away on the same drive.

Re-measured, 5,000 test images, paired:

| arm | `<25 km` before | after | median before | after |
|---|---|---|---|---|
| `d768-b350-e6-drop70` | 75.2% | **57.5%** | 2.2 km | **15.4 km** |
| `d768-b350-e6` | 75.6% | **57.3%** | 1.9 km | **14.7 km** |
| `wd29-fix-e6` | 74.9% | 57.3% | 2.5 km | 15.1 km |
| `wd29-legacy-e6` | 75.4% | 57.4% | 2.4 km | 15.0 km |

**Caveat, and it cuts both ways.** These arms were *trained* against the leaky
cache, so this is a lower bound on what a cleanly-trained arm scores: one that
learned to lean on near-duplicates may be worse at using honest neighbours. It
is a *sound* bound on how much of the published number was the leak.

**The benchmark has stopped discriminating.** Every pairwise contrast among the
four is inside noise; the spread is 0.2 pp against 0.7 pp before. Nearly
everything OSV-5M appeared to say about these arms was a statement about how
well each exploited the leak. **Select on KartaView.** It has no same-drive
frames in the bank, it separates the arms, and it always ranked `drop70` first.

**The model is overwhelmingly a retrieval system.** With the prior removed at
inference (`bootstrap.py --retr-off`), measured on the leaky cache:

| arm | retrieval on | off | loses |
|---|---|---|---|
| `d768-b350-e6` | 75.6% | **2.7%** | −72.9 pp |
| `drop70` | 75.2% | **10.7%** | −64.5 pp |
| `drop90` | 69.9% | **19.3%** | −50.6 pp |

Monotone in p, every contrast separated. Dropout buys standalone capability.
Against the clean cache the *on* column is ~18 pp lower, so the dependency is
smaller than this table says — but the ordering and the mechanism stand, and
they were predicted independently by `cond_probe` and `diag_beam`.

**`--retr-drop 0.7`.** The curve at parity on KartaView: 11.6 / 11.1 / 12.1 /
12.6 / **13.4** / 14.0% for p = 0 … 0.9. 0.7 vs 0.9 on `<25 km` is +0.58 pp
[−0.24, +1.38], inside noise; what 0.9 costs is `<1 km` and a −5.7 pp
benchmark regression. p ≥ 0.5 before anything is separable. **This axis was
measured on KartaView throughout, so the leak does not touch it.**

**Step 0 is the entire tail.** `error_profile` on the shipping arm: 7.9% of
images first go wrong at step 0 and carry **84.3% of the mean error**; 2.5%
carry 66.9%. 33.6% first go wrong at step 3 and contribute 0.0 km. Measured on
the leaky cache; the shape is a property of the search, but the numbers deserve
re-running.

---

## 3. Corrections to already-published claims

**Every OSV-5M corpus result is now suspect, and one mechanism explains all of
them.** A bigger bank holds more same-drive frames, so "the corpus pays" and
"the bank beats the training set" were, in unknown proportion, measurements of
leakage. The four claims below all have that shape and all need re-measuring
against the clean cache before they are repeated:

* corpus still pays at 2M (+6.50 pp, median halved to 3.6 km)
* the bank is worth ~2.6x the training set on sequence
* the corpus gain does not transfer externally
* the benchmark measures bank coverage (442 km external vs 2.7 km OSV-5M)

The last one is now largely *explained*: the gap was read as density and domain
shift, and a large part of it was the OSV-5M side being inflated. The external
side was always clean.


**The corpus axis.** Reported as −1.32 pp [−2.04, −0.60] separated on
photographs. At parity it is **−0.46 pp [−1.12, +0.18], inside noise**. The
median effect survives (+30.8 km, separated) and `<1 km` slightly favours the
bigger bank. The 2×2 blaming the model rather than the bank **survives**: bank
effect noise in both cells, model effect separated on the median at both banks.
And **with dropout the corpus transfers**: b265 → drop70 is +1.34 pp, +3.58 pp
at 200 km, −39.2 km, all separated.

**Why the caveat was wrong.** It said every arm was measured the same way so
paired directions would hold. Parity cost p=0 nothing and each dropout arm
~0.6 pp — the error was **correlated with the treatment**, because neighbour
count is what `--retr-drop` manipulates. The rule that replaces it: *ask whether
the error is independent of the treatment first.*

**The pyramid fusion head: retracted, then re-established.** Reported negative
from a confounded comparison. Retracted after the review, then survived three
attempts to break it. The head at **initialisation** scores 33.1% against the
1536-d mean's 33.7%, so the random projection is worth 0.63 pp of the 11.30.
Training is what destroys it:

| epochs / objective | train loss | `<25 km` |
|---|---|---|
| **1 epoch** | — | **32.2%** |
| 12, 5 km positives | 0.063 | 20.5% |
| 12, 0.5 km positives | 0.0145 | 12.9% |

Loss and retrieval move in opposite directions — overfitting, 2.5M parameters
on 38,009 images. Three explanations were tested and none survives: the random
projection (0.63 pp), the false negatives (50.3% of off-diagonal cells were
true positives; fixing them moved 20.9 → 20.5%), and the positive radius (my
hypothesis, wrong in the opposite direction). **The multi-level mean stands**:
L0+L1+L2 beats L0 alone by +1.6 pp and 40 km at equal width, with no learned
head. *Caveat:* REVIEW2 new#6 found the OneCycle schedule was truncated in
these runs, so the individual numbers were produced under a shortened schedule;
the epoch curve is robust to that, the point estimates less so.

**KartaView/OSV leakage, audited.** 0 image-id and 0 sequence-id overlaps.
Median nearest-OSV distance 231 m; 1.06% of evaluated queries within 10 m.
Dropping those 53 moves the dropout result from +1.80 to +1.80 pp and the
corpus result from +1.34 to +1.33. **Immaterial to every paired contrast.**

---

## 4. Reviews

`docs/REVIEW.md` — 86 items: **68 fixed, 8 refuted, 7 deferred, 2 open, 1
forced a retraction.** The two still open (24, 39) both need a re-baseline
decision and are in §5.
`docs/REVIEW2.md` — 22 items: **all 22 fixed.**

Round two found that **three of round one's forty were wrong**, two of them
defects I introduced while fixing something else, and one fixed in the wrong
file. Chief among them: `split_hash` hashed only each label's first character,
so train and test were indistinguishable and the bootstrap row-parity guarantee
was hollow. Now fixed, with the legacy digest kept so 122 checkpoints load.

**Still open, REVIEW.md:** 24 and 39 only, both awaiting a re-baseline call.

**The largest thing found closing them was not a review item as written.**
Item 61 was stated conditionally — "*if* one real drive crosses the corpus
boundary, same-sequence exclusion fails". It is the normal case, and it
inflates the OSV-5M benchmark. See §3.
**Still open, REVIEW2.md:** none.

The last eight of round two closed on 2026-09-04, and each was checked by
reverting its guard and watching its test go red -- the discipline whose
absence produced three wrong fixes in round one. Five of the eight are the same
shape: a consumer reading an artifact without asking which of its rows are
real, or a guard that stopped one file short of the last place it was needed.

**The provenance family closed on 2026-09-04.** #40 with 43, 44, 46, 47, 62,
63 and 66 were one issue — every artifact addressed by row position with
nothing proving two describe the same rows — and one design change retired all
eight: `src/provenance.py`, a digest of the ordered ids in a sidecar beside
each artifact, with `scripts/backfill_prov.py` for the 38 that already existed.
Absent warns, mismatched refuses. See the closing section of `docs/REVIEW.md`.

Doing it turned up two things reading would not have. The first resolver
stamped `pyr47_fuse_p05` as release rows: its `_rows.i64.npy` holds 0..47645,
a legal index into a 500,000-row release that actually indexes a KartaView
cache. And **a live regression**: the strong `split_hash` had landed without a
legacy fallback in `dataset.py`, so every kNN cache on disk was refused and the
shipping retrieval path could not build at all. All three sites comparing a
split digest now go through one `splits.hash_matches`.

**#70–73 closed the same day**, and it was the other structural group: a
stage identified by its name alone, and a log appended to across days. Both
made a file describe work a different run did — the pair that killed
`fuse-attn-pyr47` at epoch 11 of 12. `src/runlog.py` holds both conventions.
Measured rather than argued: 47 of 200 logs had been parsing to zero epochs,
so those report rows all carried a NaN seconds-per-epoch, and two training
logs were genuinely spliced across attempts.

---

## 5. Needs your decision — do not fix these silently

Eight items change what a trained arm *is*, so fixing any makes future arms
incomparable with all 122 checkpoints on file. Each needs a re-baseline.
**#29 is fixed AND measured — it changes nothing, so no re-baseline is needed
and the 122 checkpoints stay comparable (§6). #24 joined the list on
2026-09-04. Seven remain.**

* **#29 weight decay — CLOSED 2026-09-04. Fixed, measured, no effect (§6).**
  Kept here for the record because the enumeration is what made the fix safe.
  The rule was
  `"pos" in name`. Enumerated rather than assumed, and the line above was
  wrong: it caught **three** ndim-2 tensors, not two. Two are
  `retr.q_pos.weight` and `retr.k_pos.weight` — `nn.Linear` projections, the
  **learned retrieval keys**, 196,608 parameters, 3.7% of the model, worth
  +4.1 pp on record, and the branch `--retr-drop` exists to regularise. The
  third is `map.pos.weight`, a real `nn.Embedding` that *should* be exempt, so
  simply deleting the clause would have been a second bug. Now classified by
  module type: `nn.Embedding` covers `map.pos`, `state.step` and `GeoMem.emb`
  (retiring the `"geo."` clause), `ndim <= 1` covers the norms, biases and the
  `retr.w_pos` scalar. One other change: `geo.q_geo`, a dense Linear the geo
  clause also exempted, now decays — 8,192 parameters, geo arms only.
  `--wd-legacy` restores the old rule. Both sides were run fresh at one seed;
  every contrast is inside noise on both benchmarks (§6).
* **#28 `quality()`** sees the unmasked top-1 similarity even when
  `--retr-drop` hid that neighbour.
* **#27 `--save-opt`** writes optimizer state nothing loads; scheduler and RNG
  are not saved, so a 2+2+2 ladder is three optimizer restarts.
* **#2 `FuseHead.baseline`** omits a per-encoder renormalisation the printed
  baseline applies. The head is a residual on exactly that vector, so changing
  it invalidates today's fusion runs and the 33.1% init score.
* **#31 sink coefficient** unconstrained (dormant — all checkpoints `sink_k=1`).
* **#32 memory dropout** lacks inverted-dropout scaling.
* **#24 multi-photo scoring** — the learned neighbour score is
  `cos(q_pos(q_emb), k_pos(nbr_emb))` and one query embedding is supplied per
  group, so candidates a *secondary* photograph retrieved are scored against a
  photograph that did not retrieve them. Closing it means letting the
  retrieval prior take a query embedding per neighbour: a shipped model's
  interface, and it invalidates "several photos help a little" (+1.2 pp).
  `multiquery.py` prints the caveat whenever a pos/dual arm runs at N>1.
* **#39 RoPE** gives no relative position. Measured: with identical content at
  all 256 positions, the attention logit's within-offset spread is 0.845 of its
  overall spread as built, against 0.000 for the textbook arrangement. Live for
  108 of 122 checkpoints including the shipping arm.

---

## 6. #29, answered: the fix is right and changes nothing

Ran 2026-09-04, 00:45–03:50. Two full d768/3.50M/p=0.7 ladders (e2 → e4 → e6,
two epochs a rung with `--init`, exactly how the shipping arm was built), one
with the corrected weight-decay grouping and one with `--wd-legacy`, **both at
`--seed 0`**. All nine stages succeeded.

**Every contrast is inside noise, on both benchmarks, at every threshold.**

OSV-5M, paired bootstrap, 5,000 test images:

| contrast | median | | `<25 km` | |
|---|---|---|---|---|
| fix vs legacy | [−0.0, +0.2] km | inside noise | [−1.08, +0.10] pp | inside noise |

KartaView, paired, 5,000 images:

| | fix | legacy | diff |
|---|---|---|---|
| `<1 km` | 1.7% | 1.5% | −0.18 pp [−0.38, +0.02] |
| `<25 km` | 12.3% | 12.4% | +0.16 pp [−0.40, +0.72] |
| `<200 km` | 32.4% | 32.1% | −0.30 pp [−1.14, +0.52] |
| median | 462.0 km | 466.2 km | +4.1 km [−13.0, +20.4] |

The defect was real — the substring exempted the learned retrieval keys, which
are `nn.Linear` — but correcting it moves nothing. **#29 was the item flagged
as most likely to have moved a published result. It did not.** No re-baseline
is needed; the 122 checkpoints stay comparable.

**Two things the paired design caught that the obvious comparison would not.**

The selection set pointed the *wrong way*, monotonically: fix beat legacy by
+0.15 / +0.30 / +0.55 pp across the three rungs, and fix-e6 read 0.7475 against
the shipping arm's 0.7355. On the real 5,000-image test set fix-e6 is 74.9% and
the shipping arm 75.2%. The 2,000-image set picks checkpoints; it does not
measure them, and here it produced a clean monotone trend out of noise.

And **the seed is the larger effect**: both wd29 arms are separated *worse*
than the shipping arm on the OSV-5M median ([+0.2, +0.6] and [+0.1, +0.4] km),
differing from it only in the seed. Everything the naive comparison would have
credited to the weight-decay rule was the seed — which is exactly the confound
the two-fresh-ladder design existed to separate.

Artifacts: `runs/BOOTSTRAP_wd29.md`, `runs/wd29_fix.npz`, `runs/wd29_legacy.npz`,
`scripts/wd29.py`, logs `runs/logs/*wd29*`.

---

## 7. Best next experiment

**An auxiliary coarse street-only head.** Two independent measurements point at
it: step 0 carries 84.3% of the mean error, and with retrieval off the visual
branch scores 2.7%. It is Tier 1 in `ARCHITECTURE_NEXT.md`, needs tiles and a
training run.

Also promoted there: **capped or mixture retrieval combination**, because
`cond_probe` showed the prior's contribution outweighs the top-1 margin in 85%
of rows — it is deciding the answer often enough that how it is combined
matters. That measurement reversed three conclusions I had drawn from the
static gates alone, which the same document had flagged as a floor rather than
the whole story.

---

## 8. Tools written this session

`scripts/parity_report.py` — the whole external table from the exports.
`scripts/pair_npz.py` — paired bootstrap over two exported error arrays.
`scripts/rope_probe.py` — is rotary giving relative position (no).
`scripts/cond_probe.py` — the retrieval prior's real influence on a logit.
`scripts/fuse_init_probe.py` — the fusion head scored at initialisation.
`bootstrap.py --retr-off` — corpus dependency measured directly.
`eval_highres.py --match-bank` — reproduce the bank's embedding pipeline.
`src/safeio.py` — atomic checkpoint and report writes.

Tests: `test_checkpoint_contract.py`, `test_split_hash_and_cache.py`,
`test_topk_bounds.py`, `test_import_order.py`, `test_safeio.py`,
`test_cache_completeness.py`.

---

## 9. Hardware and environment

* RTX 3070, 8 GB. Training is **GPU-bound**, 92–96% at 165–176 W. The fusion
  head runs 97% at 236 W with 6.9/8 GB resident. **Never stack GPU jobs** —
  measured 2.6× cost.
* 31.7 GB RAM. Casting a bank to float32 whole is a 10.7 GB allocation; chunk.
* `python` on PATH is MSYS2's and lacks numpy. Always
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe`.
* Tile server `192.168.50.1:3000`. Needed by all training (beam rollout) and
  all evaluation. Not needed by pyramid/fusion work or any probe above.
* `OSV_RELEASE` is now **required** — no default. The runner sets it.

---

## 10. Rules learned the hard way

* **A fix is not done until it is tested against the failure it prevents.**
  Round two found three wrong fixes and every one had never been run against
  its own failure case: the split hash never fed a swapped assignment, the
  cache resolver never given a retrained checkpoint, the top-k never run at
  `K == bank size`. Two of those I introduced while fixing something else.
* **Ask whether a measurement error is independent of the treatment**, not
  merely whether it was applied uniformly.
* **A guard that is defined and never called is not a guard.** Happened twice
  (#65, #17). Nor is one that fails open (#42).
* **Never type a backslash inside a Bash heredoc** — it arrives as a real
  newline. Hit six times today despite being written down. Build it as
  `chr(92)` or write the patch script to a file with the Write tool.
* **Assert before writing** in patch scripts. Two anchors missed today and
  neither corrupted a file. But re-running a patch script double-applies —
  check for duplication after.
* **Scope every log check to the newest `==== attempt`.** An append-only log
  matched a `FAIL` from the previous day and killed `fuse-attn-pyr47` at epoch
  11 of 12.
* **Do not block on `WaitForExit` in the PowerShell tool**, and expect the Bash
  tool's background jobs to be reaped. Long runs survive as detached
  `Start-Process` with output redirected to a file, polled afterwards.
* **`git checkout <file>` discards uncommitted work.** Lost the `overnight.py`
  edits that way; commit before sabotage-testing.
* **Never `git stash`.** It swept a session's uncommitted work out from under
  a diagnostic on 2026-09-04. Recovered with `stash pop`, but the safe form of
  "what did this look like before my change?" is `git show <rev>:<path>` to a
  scratch file, never anything that touches the working tree.
* The **2,000-image selection set is not evidence**; it picks checkpoints.
* **A wide interval means get more data, not weaken the claim.**
