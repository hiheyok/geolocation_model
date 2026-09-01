# Where to pick this up

State as of 2026-08-31. Written to be read cold: what ships, what is closed and
must not be retried blind, what is open and in what order.

## What ships

`s10_bal_bank25_c6` — **7.7 km median, 386.9 mean, 64.2% within 25 km** on the
`sequence` test split at n=5,000, and **247.5 km / 7.66%** on the `cell8`
geographic holdout. 9,215,444 trainable parameters in front of 178,609,152
frozen ones. Full end-to-end diagram in the roadmap artifact, §9.

The load-bearing part is not the network. The retrieval prior is worth
+17.3 pp on its own, the keyed variants +4.1 pp more, and growing the bank from
400k to 1.15M images was +20.4 pp. The trained model is 9.2M parameters; what
makes it work is 11.5 GB of other people's photographs.

## Validated but not yet default

**Pooled street vector, 4608 → 1536-d.** Mean-pool the three crop tokens
instead of concatenating them. Test split, paired: median [-0.5, +0.7] km, hit
rate [-0.74, +0.94] pp — indistinguishable, at a third of the bank bytes and
34% fewer parameters. `scripts/pool_street.py` converts an existing cache in
about a minute.

Two things still owed on it: the scripts all still default to 4608, and it has
only been validated on `sequence`. A `cell8` arm is the one gap.

## Next, in order

### 1. Spend the bank RAM that pooling freed

The bank is 3.53 GB at 1536-d for 1.15M entries, so the same 11.52 GB holds
**3.75M**. Corpus is the axis with the largest measured effect in this project
and it was RAM-capped.

**No download needed.** 98 OSV-5M shards are on disk and only 25 are used —
73 unused shards, ~3.65M images. Cost is ~2 h of embedding per 500k images at
3 crops through two encoders, so roughly 10 h to fill it, plus a kNN rebuild and
one training arm.

Do it in pooled form; the two changes compose, and the pooled bank is what makes
the larger corpus affordable in the first place.

### 2. Decompose the error by step before spending 10 hours

The median is **7.7 km** and the s2 cell is **9.8 km**. Those are not
independent: an image lost at step 2 lands somewhere inside a 9.8 km cell and
never recovers, so the median is essentially pinned to s2 accuracy (42.2%).
Step 3 at 11.7% only decides the last 611 m and barely moves it.

If that holds, **s2 is the binding step for the headline metric**, not s3, and
that should steer everything above. It is a ~20 minute analysis and it is the
cheapest thing on this list.

### 3. External datasets, for the two things OSV-5M cannot give

Not for volume — see item 1. OSV-5M is 211 countries but the top 10 are 58.2%
of the data, the US alone is 24.9%, and **150 of 211 countries have under 1,000
images** with 71 under 100. More shards inherit that skew.

* **Benchmark comparability.** `Im2GPS3k` (2,997 images), `YFCC4k` (4,536),
  `GWS15k` (15k, uniformly sampled globally and much harder). Test-only, tiny
  downloads, and they would situate 7.7 km against published numbers — which
  this project currently cannot do at all. Cheapest external win by far.
* **Geographic rebalancing**, which attacks the `cell8` gap directly (7.7 km
  against 247.5 km is a transfer failure, and the skew is part of it):
  * **MSLS** (Mapillary Street-Level Sequences), ~1.6M street-level images with
    sequence structure — closest in character to OSV-5M, and bank entries need
    only an image and a coordinate, no training labels.
  * **MP-16 / YFCC100M geotagged subset**, ~4.7M Flickr images. Far more
    diverse but much of it is indoor, close-up or landmark photography, so it is
    a distribution shift rather than more of the same. Would need a
    does-it-help test, not an assumption.

E: has 538 GB free.

### 4. The 18-token cross-attention head

The only live architecture idea. +5.00 pp [+4.23, +5.80] of oracle headroom from
tile/crop complementarity — 150 of 2,686 queries that every crop missed were
retrievable from a tile. Needs contrastive training infrastructure that does not
exist yet, and two 2026-08-31 results lowered its prior. Design note in
`docs/tile-fusion.md`, including the decomposition that matters: one token per
(region, encoder), 18 not 9, because with the encoders concatenated into one
token attention can mix them but cannot select between them.

## Closed — do not retry without new information

| direction | why it died | do not cite the old reason |
|---|---|---|
| Tiling instead of crops | -1.3 to -1.6 pp any32; each tile is individually weaker than each crop (sky 2.7-3.3% against a crop's 7.4%) | the first claim was never statistically separated; the balanced baseline is what makes it true |
| Per-step encoder gate | gates flat: moved 0.026 from zero while other scalars in the same run moved 0.11-0.47 | — |
| DINOv3 ViT-B | loses to DINOv2 at every threshold, at 224 **and** at its native 256 | — |
| Per-tile key table | learns an occupancy mask the model already has; `corr(log images in cell, |key|) = -0.92` | the old "leakage detector" reason **inverted** — the count table went from beating the model to 5.3x below it |
| OCR + map labels | at chance; readable text is dashcam chrome, not signage | — |

## Hazards

**1.45% of OSV-5M frames have their true GPS burned in** as a dashcam overlay,
median 0.05 km from the label, 90% within 1 km. Exploiting it is worth about
+1.3 pp on `<25 km` — the same magnitude as every effect measured here, so it
would present as a real result. The shipping pipeline is safe only because
`preprocess` scales the short side to 224 and 8-pixel overlay text is
unresolvable there. **Any higher-resolution direction walks straight into it.**

**Noise floor:** on this recipe at n=5,000, nothing under about 1 pp of hit rate
is resolvable. Two arms that were functionally identical kept different epochs
because the 2,000-image selector split them.

## Discipline that earned its keep

**Read the parameters, not the metric.** It caught a `GeoMem` table that was
bit-for-bit zero while producing metrics indistinguishable from an honest null —
and which matched the prediction that had already been made, which is what made
it dangerous. It also made the encoder-gate null *interpretable* rather than
ambiguous. `tests/test_geomem.py` encodes the resulting invariant: a module that
starts inert must still receive gradient on the first backward pass.

**Check whether inherited evidence still holds.** Three separate objections to
tile memory were all answerable — the mechanism was a different thing, the
occupancy was measured at a tenth of the data, and the leakage argument had
inverted outright. The idea still failed, but for a reason measured today.

**Pick the threshold to match the decision.** Reading encoder complementarity at
25 km alone hid a 5 pp effect at 2500 km. For a step-0 action the right
threshold is 2500 km, not 25.
