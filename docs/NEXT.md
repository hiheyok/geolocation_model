> **Start with `docs/STATE.md`** — current results, what is unfinished
> and resumable, the measured hardware constraints, and the mistakes
> that recur. `docs/NAMING.md` decodes arm names.

# Where to pick this up

> **See `docs/STATE.md` first** for what is running right now and the
> results from the most recent session. This file is the forward plan.

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

* **Benchmark comparability.** `Im2GPS3k` (2,997 globally distributed images,
  a standard out-of-domain generalisation test), `YFCC4k` (4,536), and
  `GWS15k` (15k, predominantly *unseen* locations and much harder — the closest
  external analogue to our `cell8` holdout). Test-only, tiny downloads, and they
  would situate 7.7 km against published numbers, which this project currently
  cannot do at all. **Cheapest external win by far**, and `GWS15k` in particular
  would tell us whether the 247.5 km transfer gap is normal or ours.
* **Geographic rebalancing**, which attacks the `cell8` gap directly (7.7 km
  against 247.5 km is a transfer failure, and the skew is part of it).

  * **MP-16 / EMP-16** (Extended MediaEval Placing Tasks 2016), ~4.6M geotagged
    images taken by ordinary people, drawn from YFCC100M. **This is the one to
    try.** A 2025 paper builds a "hybrid gallery" of EMP-16 **plus** OSV-5M for
    retrieval-augmented geolocation — which is exactly the bank-extension use
    here, published and working. It is also the standard training corpus for the
    whole Im2GPS lineage, so it is well characterised. The catch is distribution:
    much of it is indoor, close-up or landmark photography rather than
    roadside, so it is a shift rather than more of the same, and it needs a
    does-it-help test rather than an assumption. Bank entries need only an image
    and a coordinate — no training labels, no sequence structure.
    <https://arxiv.org/html/2509.01341>

  * **MSLS** (Mapillary Street-Level Sequences), 1.6M street-level images with
    sequence structure, GPS plus compass angle — closest in *character* to
    OSV-5M. But it is **30 cities across 6 continents**, i.e. dense urban depth,
    not global breadth, so it does not attack the skew and would not help on
    held-out regions. Useful only if the goal is urban density in those specific
    cities. <https://www.mapillary.com/dataset/places> ·
    <https://github.com/mapillary/mapillary_sls>

  * **StreetLearn** — 113,767 panoramas, Manhattan and Pittsburgh only. Too
    narrow to matter here.

  Note OSV-5M itself is 5.1M images over 225 countries and we already hold all
  98 shards, so it remains the best breadth-per-byte on disk; the skew is
  inherent to where dashcam and street-view coverage exists, not to our
  sampling.

E: has 538 GB free.

### 3b. Higher-resolution imagery — Mapillary's API, not MSLS

OSV-5M frames are 682x512 (0.35 MP) and that cap is load-bearing: it is what
made OCR read dashcam chrome instead of signage, and it is why tiles carried
barely more real information than crops.

**Mapillary's API serves the originals.** 98% are available at 2048x1536, i.e.
**9x the pixels**, globally and community-uploaded rather than the 30 curated
cities MSLS ships. Per image it exposes `thumb_256/512/1024/2048_url` and
`thumb_original_url`, plus `computed_geometry` (GPS after processing),
`computed_compass_angle`, `altitude` and `captured_at` -- the compass angle in
particular is metadata OSV-5M does not carry.

Access needs a client token (OAuth 2). Rate limits are 60,000 entity requests a
minute and 10,000 search, but the **tile API is 50,000 per *day***, and the tile
API is how image IDs are discovered inside a bounding box -- so **discovery is
the bottleneck, not download**. Plan harvesting around that: pick target regions
first (the 150 countries with under 1,000 OSV-5M images), enumerate once, then
pull.

<https://www.mapillary.com/developer/api-documentation>

Storage: 500k images at 2048px is roughly 250-500 GB against 538 GB free on E:.
Better to stream, embed and discard, the way `embed_street.py` already reads
from a zip without extracting -- only the embeddings persist, and 500k pooled is
1.5 GB. The cost of that is re-downloading if the encoder changes; a 1024px
cache (~75 GB for 500k) is the middle path.

**Two things to settle before spending the bandwidth.**

*Today's pipeline discards the resolution.* `preprocess` scales the short side
to 224 first, so a 2048px Mapillary image and a 682px OSV-5M frame become the
same tensor. Higher-resolution data is worth nothing on its own -- it only pays
**in combination with a pipeline change that consumes it**, which is the tiling
direction in `docs/tile-fusion.md`. That direction lost at 682px, where six
tiles carried 4.5x the pixels but each tile was individually weaker than a crop.
At 2048px the tiles would be genuinely sharper rather than merely more numerous,
so the experiment is different and worth redoing -- but redo it as an
experiment, not as an assumption, and re-read why it failed first.

*High resolution is where the GPS leakage becomes readable.* Mapillary hosts
plenty of dashcam uploads. At 682px burned-in coordinates are unresolvable and
the 1.45% of affected frames leak harmlessly; at 2048px they are crisp. The
coordinate regex in `scripts/ocr_probe.py` should be run as a **screening pass**
over any high-resolution corpus before it enters a bank or a training set.
Otherwise the first "more pixels helped" result will be the model reading the
answer off the image.


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
