# Map-guided visual geolocation: what works, what does not, and what is open

**Written for an outside reader with no context, to solicit new ideas.**
Everything here is measured on this project's own data unless it says
otherwise. Where a number was later found wrong, the correction is kept rather
than the original, because the corrections are among the more useful content.

Date: 2026-09-05. Hardware: one RTX 3070, 8 GB. That constraint shapes
everything below.

---

## 1. The task and the architecture

Given a single street-level photograph, predict latitude and longitude.

The system is **not** a classifier over geocells. It is a **search agent** that
descends a quadtree of map tiles, and the key structural fact is that the
16x16 recursive grid used for the action space and standard XYZ web-map tiles
are the same object: 16 = 2^4, so one agent step is four XYZ zoom levels.

```
step 0: z0   whole world      cell width 2504 km
step 1: z4                    cell width  156 km
step 2: z8                    cell width  9.8 km
step 3: z12                   cell width  611 m
then:   a click head regresses (u, v) inside the final z16 tile
```

At each step the model sees the **map** of its current tile -- a 512x512
segmentation mask from a self-hosted MapLibre tile server, reduced to a 16x16
grid of 12-d class-fraction histograms (water, road, building, wood, ...) --
and picks one of 256 children. Descent is a base-16 digit append, so the
action path *is* the tile address.

### Components

| part | detail |
|---|---|
| street encoder | **frozen** DINOv2 ViT-B/14 + SigLIP ViT-B/16, 768-d each, concatenated to 1536-d |
| map tokens | mask -> 16x16 patches -> 12-d class fractions -> projected, with rotary position |
| policy head | cross-attention: logit_i = <q, k_i> / sqrt(d) over the 256 map tokens |
| retrieval prior | top-16 neighbours from an offline kNN bank, with their z16 addresses |
| sink class | a 257th "reject" action so beam search can decline a branch |
| trainable | ~6M parameters. Everything upstream is frozen and precomputed |

Training is **supervised imitation from GPS labels**, teacher-forced. Because
the teacher-forced prefix is a pure function of (lat, lon), all five rows of an
image are independent and are generated up front -- no rollout. One training
rung is ~26 minutes.

### The retrieval bank is the centre of gravity

An offline kNN pass embeds a **bank** of up to 3.4M images and, for each query,
stores the top-32 neighbours by cosine. The agent consumes 16 of them through a
trained prior that injects their z16 addresses into the policy logits.

**Retrieval is offline and therefore final.** The cosine picks the neighbours
before the model sees anything; no downstream head can un-retrieve what the
cosine chose. This single fact has killed several otherwise reasonable ideas
(section 4).

---

## 2. Where the numbers actually stand

Two benchmarks, and they disagree in an informative way.

**OSV-5M test split**, 5,000 paired images, sequence-based holdout:

| arm | `<25 km` | median |
|---|---|---|
| `d1536-b350-e6-drop30` (best) | 59.1% | 13.4 km |
| `d768-b350-e6-drop70` (ships) | 57.5% | 15.4 km |
| `d768-b265-e6` (smaller bank) | 55.0% | 17.8 km |
| `d768-b350-e6-drop90` | 52.7% | 21.1 km |

**KartaView**, a genuinely external corpus, same models:

| arm | `<1 km` | `<25 km` | median |
|---|---|---|---|
| `d768-b350-e6-drop70` (ships) | 2.1% | 13.4% | 435.7 km |
| `d1536-b350-e6-drop70` | -- | 12.8% | 467.4 km |

**57.5% versus 13.4% is the single most important number in this document.**

An earlier version of this brief said the gap "is bank coverage". That was
asserted from an aggregate difference between two benchmarks, which cannot
establish a mechanism, and it is **wrong for OSV-5M**. Decomposed over 49,788
test queries against the shipping bank (`runs/COVERAGE.md`):

| radius | coverage-limited | retrieval-limited | ranking-limited | solved |
|---|---|---|---|---|
| 1 km | 18.5% | 51.6% | 19.4% | 10.4% |
| **25 km** | **0.2%** | 16.9% | **33.2%** | 49.7% |
| 200 km | 0.0% | 4.7% | 25.8% | 69.4% |

*coverage* = no eligible bank row within r at all; *retrieval* = one exists but
is not in the cached top-32; *ranking* = a top-32 candidate is within r but
rank 1 is not.

**At 25 km the constraint is ranking, not coverage.** 99.8% of test queries
have a legal bank row within 25 km and the median distance to the nearest one
is 0.32 km. At 1 km coverage genuinely binds. Whether the *external* gap is
coverage is a separate question that has not been decomposed the same way.

### The uncomfortable ablation

| configuration | `<25 km` on OSV-5M |
|---|---|
| full agent | 57.5% |
| top-1 retrieval neighbour alone, no agent at all | **~49.7%** |
| agent with the retrieval prior disabled | **2.7%** |
| oracle over the cached top-32 | 82.9% |

**Correction.** An earlier version put top-1 retrieval at 56.4% and concluded
the agent adds about a point. That 56.4% was a cohort artifact: the script
selected `flatnonzero(labels == "test")[:3000]`, and `dataset.parquet` is
shard-ordered, so the head of the test split is BR/AR/ZA at a median nearest
legal bank row of 0.212 km against US/DE/RU at 0.337 km — a different
continent and 35% denser.

```
first  3,000 test rows (parquet order):  56.4%   median 13.30 km
random 3,000                          :  49.4%   median 26.29 km
all   49,788                          :  49.7%   median 25.68 km
```

**The agent adds roughly 7-8 pp over rank-1 retrieval, not 1 pp.** The two
figures still come from different cohorts, so that is a difference and not yet
a measured treatment effect; a paired run on identical ids is outstanding. It
collapses to near-nothing without retrieval. Whatever the map-descent
architecture is doing, it is currently a thin layer on top of a retrieval
system. Training with `--retr-drop 0.7` (randomly hiding the prior) lifts the
standalone number 2.7% -> 19.3%, which is the only lever found so far that
makes the non-retrieval path learn anything.

**This is the central strategic question and the reason for this document.**

---

## 3. What worked, in rough order of value

Every entry is a paired bootstrap with a 95% CI unless noted. "Separated" means
the interval excludes zero.

1. **Corpus size dominates everything.** Bank 1.15M -> 1.90M was +6.50 pp and
   halved the median to 3.6 km. Bank is worth roughly **2.6x the training set**
   on the sequence split. This is the strongest single axis found.
2. **The retrieval prior itself: +17.3 pp.** Adding position/dual keys on top:
   +4.1 pp more, monotone and separated.
3. **Dual encoder.** DINOv2 alone -> DINOv2 + SigLIP: median 356.8 -> 294.4 km.
   Caveat found later: only the joined vector was normalised, so DINOv2 was
   taking 81% of the cosine by raw-norm accident (norms 82.96 vs 20.58). Fixing
   the per-encoder scale mattered.
4. **Sink class (a 257th reject action).** Beam search *degraded* with width --
   the compute curve sloped downward past k=2, which is embarrassing for a
   test-time-compute thesis. A learned reject action fixed it: verifier AUC
   0.988, degradation 8.0% -> 1.6%.
5. **Scoring depth.** Ranking beams on the full path score was dominated by
   noise from the fine steps. Ranking on s0-s2 only: 294.4 -> 279.2 km, and the
   accuracy-versus-compute curve finally slopes upward.
6. **A retrieval-keys inference bug.** `beam.search` ignored the neighbour
   embeddings entirely, so pos/dual models decoded as if unconditioned:
   81.6 -> 55.8 km when fixed. Worth noting as a class of error: the training
   path and the inference path had drifted apart.
7. **Image pyramid in the embedding (`L0L1`).** L0 = 3 full-height crops at 224;
   L1 = a 3x2 grid of 224 tiles over the same frame; the representation is
   `l2((L0 + L1) / 2)`. Retrieval top-1 +2.76 pp `<25 km`; **agent-level
   +2.22 to +4.64 pp, separated** -- the first agent-level confirmation.
   Interesting texture: it is **churn, not lift** (6.57% of queries win, 3.80%
   lose), and the wins are **rescues** -- the errors it fixes had median 379 km
   and p90 6,031 km. Tiles repair gross failures, they do not refine near-misses.
8. **Retrieval dropout at p=0.7.** +1.80 pp, and more importantly it is what
   makes the corpus gain **transfer externally** (+1.34 pp separated on
   KartaView, where the untreated model was actually *worse* externally
   despite +2.8 pp on OSV-5M).
9. **Blend weight.** The fusion of a level-mean and a learned head was
   `concat([l2(a), l2(b)])`, which is exactly `0.5*cos_a + 0.5*cos_b`. Nobody
   chose 0.5; it fell out of concatenating two unit blocks and was ~10x too
   much head. The optimum is a flat plateau w = 0.02-0.15. At w=0.05:
   +0.79 / +1.18 / +2.17 / +2.89 / +2.50 pp at 1 / 25 / 200 / 750 / 2500 km.
10. **Attention pooling with rotary position over map tokens**, replacing mean
    pooling: 375 -> 351 km. Mean pooling had destroyed spatial layout.
11. **Bank pooling.** A 3x smaller bank and a 34% smaller model with hit rate
    change inside [-0.74, +0.94] pp. Free compression.
12. **Throughput, 1.84x** (18.2 -> 33.5 img/s): a prefetch thread pool (the card
    was sawtoothing 100% -> 0% waiting on JPEG decode) plus casting model
    weights to bf16 rather than using autocast. Ruled out by benchmark: larger
    batch (saturates at 16), `channels_last` (no-op on a ViT), fused attention
    (already on).

---

## 4. What did not work

This section is longer than the previous one and is probably more useful.

### Representation and encoders

* **Higher resolution does not help retrieval.** Matched banks at each
  encoder's true native size (DINOv2 518, SigLIP 512, no interpolation, no PCA
  in the path) are **flat to negative from 25 km up**, with a *worse* median
  than 224 crops. Tiles at 224 beat native resolution at about a third of the
  cost. The single exception: `<1 km` improves (+0.37, +0.40, separated) in
  both arms that use resolution, where tiles do nothing -- finer sampling buys
  discrimination among near-duplicates. That exception is unresolved and was
  measured on a bank too sparse to hold many near-duplicates.
  *(Caveat on the record: one arm of this used a 224 SigLIP interpolated to
  512, which is out of distribution; that arm needs re-measuring.)*
* **"Resolution does not complement tiles" no longer holds.** On the biased
  cohort, crops+tiles was +1.27 [+0.4, +2.2] separated and native-crops+tiles
  +0.97 [−0.1, +2.0] inside noise. On a random cohort the ordering reverses:
  +0.83 [−0.1, +1.7] against **+1.27 [+0.2, +2.3]**. The conclusion was
  cohort-dependent. It does not reinstate resolution either — the native arms
  used a 224 SigLIP at `img_size=512`, out of distribution, which is a
  confound pointing the wrong way. Re-running against the genuine 512
  checkpoint is in flight.
* **DINOv3 is not an upgrade.** Loses to DINOv2 at every threshold, both at 224
  and at its own native 256.
* **A third pyramid level (L2) is impossible on this corpus.** A 6x4 grid of
  224 tiles needs 1344x896; **0% of OSV-5M frames have it** (910x512 and
  682x512 are 90% of the data). Pushed in anyway at mismatched resolution it is
  harmful. 3x2 is the widest grid the corpus supports.
* **More map detail is not the bottleneck.** Replacing the 12-d class-fraction
  token with 2x2 sub-tokens carries strictly more information -- including
  orientation the 12-d token destroys -- and scores *worse*.
* **GeM pooling within a pyramid level is inert.** Every p inside noise against
  the plain mean; the one separated result is a loss. Equal averaging was
  already optimal.

### Conditioning, gating, and "learn the blend"

* **Four separate attempts to condition the retrieval blend all lose to one
  global constant**: per-step weights, per-encoder gating, positional gating,
  per-query gating. Per-step weights looked uniformly positive until scored on
  a held-out split, at which point three of five thresholds went zero or
  negative. That difference *is* the result.
* **A per-query gate cannot capture the remaining headroom either.** Oracle
  arm-selection between crops and crops+tiles is worth +6.57 pp against the
  blend's +2.76 -- so headroom exists -- but all three observable selection
  rules score *worse than doing nothing*, and the gain is positive in every
  confidence bucket, so no threshold rule exists.
* **Extra information pushed into the retrieval vector does not survive.** At
  blend weight 0.10 an extra level replaces 8.2% of the top-16 and flips the
  top-1 for 12.7% of queries -- and the hit rate does not move at all
  (59.6% -> 59.6%). It reorders neighbours in a way uncorrelated with whether
  they are correct. Because retrieval is offline, a downstream head cannot
  recover from this. Any new signal must ride *beside* the retrieval vector,
  not inside it.
* **A pyramid fusion head as a replacement is harmful**; it only adds at blend
  weight 0.05-0.10.
* **Width does not compose with dropout.** d1536 with retr-drop 0.7 loses both
  ways; p has to be re-tuned per width.

### Training and objectives

* **Soft tile labels: a clean negative.** Distance-weighted soft targets with
  KL instead of CE. One temperature cannot span cells that are 2504 km wide at
  step 0 and 611 m wide at step 3, and per-step temperatures did not rescue it.
* **Looped / weight-tied transformer: null.** Weight tying regularises but
  validation is flat. Depth 1 still ships.
* **Epochs do not substitute for data.** A 50k x 80-epoch budget peaks at epoch
  3. Small arms cannot absorb compute.
* **Per-tile memory keys learn an occupancy mask**, not a map match -- they
  reproduce a prior the model already has.
* **A zero-gated adapter times a zero-initialised table never trains** and
  looks exactly like an honest null. Worth knowing before designing one.

### Other

* **OCR on dashcam frames is at chance** for label matching. Separately, 1.45%
  of frames have their true GPS burned into the image, worth +1.3 pp if
  exploited -- which is a leak, not a feature.
* **Multiple photos of one location help a little and saturate**: +1.2 pp for a
  second angle, nothing after.
* **`cell8` step-1 accuracy is below chance** (0.0% against 0.39%). The policy
  is ranking cells by occupancy, not by map match. This is a live, unexplained
  failure.
* **Query-only re-encoding is a dose-response failure.** Improving only the
  uploaded image's encoding against a fixed bank is *harmful*, monotonically:
  -1.60 pp at 336, -3.27 pp at 448. The bank's encoding is a contract that
  cannot be improved from one side; both must be rebuilt together.

---

## 5b. Reranking: two eliminations

33.2% of queries at 25 km are ranking-limited, so a correct candidate is
already cached. But rank-1 maximises cosine by construction, so a reranker
must supply information the cosine lacks. Its two blind spots are set
structure (candidates scored independently) and pooling (spatial detail
averaged away). The cheap exploit of each was tested and both fail:

| geographic consensus (49,788 q) | `<25 km` | | alphaQE, no coordinates (10,000 q) | `<25 km` |
|---|---|---|---|---|
| rank-1 | 49.73% | | rank-1 | 49.39% |
| K=8 | 50.48% | | k=2 | 49.45% |
| K=16 | 48.90% | | k=3 | 46.62% |
| K=32 | **46.59%** | | k=10 | **40.16%** |

Both degrade monotonically with K, for one shared reason: **they aggregate
over a set that is mostly wrong.** Rank-1 is right 49.7% of the time and the
marginal candidate is far worse.

A separate argument kills positional features specifically. **A query has no
position at inference** — that is the thing being predicted — so a positional
feature is a function of the candidates' positions alone, and the query enters
only through which rows were retrieved, which is the cosine again. It can only
express how clustered the retrieved set is, i.e. where the bank is dense.

The distinction that survives: position as *the thing being verified* works
(the retrieval prior proposes neighbour addresses and lets image evidence
choose among them, +17.3 pp); position as *the evidence itself* does not.

**What is left is per-candidate verification** — does this one candidate
depict the same place as the query? No positions, no pooling, unaffected by
the other 31 being wrong. That is the pooling blind spot, and it needs a
14-56 GB local-token cache.

---

## 5. Methodological hazards found the hard way

An outside reader should weight the numbers above with these.

* **A retrieval bank leaked, and it was worth 18 pp of the headline.** OSV-5M
  captures run consecutively along a road, so frames from one drive are
  near-duplicates metres apart. Same-sequence exclusion existed but silently
  stopped applying across a corpus boundary because the extension's sequence
  ids had been offset into a separate namespace. Result: 41.6% of test queries
  had a same-drive top-1 at median 0.31 km. The headline fell 75.2% -> 57.5%.
* **Training against the leak *helped*.** The expectation was that a model
  trained on a leaky bank would be *worse* at using an honest one, making 57.5%
  a floor. Measured, it is the other way round: retraining against the clean
  bank scores 53.5-54.6%, separated. Hypothesised mechanism (untested):
  near-duplicate neighbours are a strong low-noise signal that teaches the
  model to *use* the prior, while a clean bank's noisier neighbours teach it to
  lean on retrieval less. The leak acted as a curriculum.
* **Select on hit rate, not the median.** The median swings 300 km between
  epochs; two seeds differ by 18 km. The median error CI is ~16 km, so most arm
  comparisons in this project are not separable and must be paired.
* **The seed is often a larger effect than the treatment.** Two ablation arms
  differing only in seed were separated from the reference while the treatment
  itself was inside noise.
* **A 2,000-image selection set pointed the wrong way monotonically.** It picks
  checkpoints; it does not measure them.
* **Evaluation took the first n rows** rather than a random sample, so absolute
  metrics were optimistic (paired ones were fine).
* **Never generalise a null from arms that are near-identical by
  construction.** "The benchmark no longer separates arms" was retracted after
  it turned out to have been measured on four arms sharing bank, width and
  dropout; the full eight-arm table spans 52.7% to 59.1%.
* **`[:n]` over an ordered artifact is a stratified sample.** Four probe
  scripts sliced the head of the test split, which is shard-ordered and
  therefore country-ordered and therefore density-ordered. Worth 7 pp, and it
  shaped a strategic conclusion for weeks. Agent-level evaluation was
  unaffected — it already sampled randomly.
* **Adjacent KartaView ids are one drive**, a median 2.0 s and 67 m apart. The
  harvester took three frames per drive, so 69.8% of 100 m spatial groups
  contain a single drive — which means "+1.2 pp for a second angle" was
  measured where 70% of second photographs are the same angle two seconds
  later. The number stands; its description does not.
* **A recurring bug class: a cache key that names an identifier rather than
  everything the value depends on** -- a tag, a filename, a length, a prefix, a
  substring. Six separate defects share this shape. Artifacts here are now
  bound by content digest.

---

## 6. Hard constraints on any proposal

* **One RTX 3070, 8 GB.** A training rung is ~26 min; a kNN rebuild 5-7 min; a
  full bank embedding pass ~19 h. Never stack GPU jobs (2.6x measured cost).
* **Encoders are frozen and their outputs are precomputed.** Anything that
  requires fine-tuning a ViT-B on 3.4M images is out of budget.
* **Retrieval is offline.** Neighbours are chosen by cosine before the model
  runs. Re-ranking a cached top-32 is free; changing *which* 32 requires a bank
  rebuild.
* **The bank is a contract.** Query and bank must be embedded identically.
  Partial re-embedding is not allowed: one cosine ranking blended rows against
  unblended rows silently demotes the unblended ones. It is all or nothing.
* **The frames are 512 tall**, widths 682-1228. That caps any tiling scheme at
  a 3x2 grid of 224 tiles.
* **The map service is a real dependency** -- a self-hosted MapLibre tile
  server. Masks are class-id images and must never be interpolated, only
  aggregated into per-patch histograms.

---

## 7. Open questions, and where fresh thinking would help most

Ordered by how much a good answer would be worth.

1. **The agent adds ~7-8 pp over its own top-1 retrieval — less pressing than
   this brief first claimed, but still the structural question.**
   Either find what the map-descent is uniquely able to do that a nearest
   neighbour cannot, or restructure so retrieval and reasoning are not
   competing for the same job. Note the one piece of evidence in the agent's
   favour: with retrieval dropout the standalone path reaches 19.3%, so the
   agent *can* learn to locate without the bank -- it just does not have to.
2. **Everything scales with corpus, which is a data-acquisition answer, not a
   research answer.** Is there a formulation where model capability substitutes
   for corpus density instead of complementing it? The learning curve and a
   direct probe both say data is the binding constraint today.
3. **`<1 km` behaves differently from every other threshold.** Resolution helps
   only there; tiles help everywhere else and not there. Two different regimes
   -- coverage versus discrimination -- may want two different mechanisms, and
   nothing currently exploits that split.
4. **`cell8` step-1 accuracy is below chance.** The policy ranks candidate
   cells by occupancy rather than by matching the map. This is a concrete,
   reproducible failure of the core mechanism and nobody has explained it.
5. **The external gap (57.5% vs 13.4%) is NOT explained by OSV-5M coverage**,
   which is 99.8% at 25 km. The external side has not been decomposed. Is there a way to
   generalise from a dense-corpus regime to a sparse one? Retrieval dropout is
   the only thing found so far that transfers.
6. **The wins from tiles are rescues of gross failures** (median 379 km errors),
   not refinements. That suggests a distinct failure mode -- images the
   representation places on the wrong continent -- which might be attackable
   directly rather than as a side effect of a better embedding.
7. **Conditional / adaptive computation has failed four times.** Oracle gating
   is worth +6.57 pp against a global constant's +2.76, so the headroom is
   real and no observable rule has ever captured any of it. Is there a reason
   to expect this is fundamentally not learnable from the available signals?

---

## 8. Things deliberately not tried yet

* An mRoPE transformer over the zoom axis instead of the Markov state (street
  patch tokens are ~20 GB at 50k images and ~1 TB at full scale, so the storage
  story blocks it).
* Reinforcement learning of any kind. The whole system is supervised imitation.
* Learning a stopping rule on the confidence radius (adaptive compute).
* Anything that fine-tunes the frozen encoders.
