# Tile fusion by self-attention — design note

Status: **designed, not built.** Written 2026-08-31, after the concatenation
version was measured and lost. No code exists for any of this.

## What this is trying to fix

`embed_street.preprocess` scales the short side to 224 *before* cropping. OSV-5M
frames are 682×512 or 910×512, so **2.3× of linear resolution — over 80% of the
pixels — is discarded before either encoder sees the image.** Three 224×224
crops are then taken across the width of what remains.

The obvious response is to cut the image into a grid of 224 tiles at native
resolution and encode each one, giving the model more of the photograph than a
single pooled vector can hold. That was measured. It lost:

| scheme | dims | top-1 `<25 km` | any-of-32 `<25 km` |
|---|---:|---:|---:|
| crop3 | 4608 | 8.0% | 21.4% |
| tile6 | 9216 | 7.8% | 21.8% |
| crop3, PCA | 512 | **7.7%** | **20.9%** |
| tile6, PCA | 512 | 7.3% | 20.1% |

118,500-image bank, 3,000 queries, same-sequence masked. At equal bank bytes
tiling loses at every compressed width; at full width it is a wash for twice the
storage.

**But the tiles were combined by concatenation, and that is very likely the
defect rather than the tiling.** Cosine over concatenated blocks compares tile
*i* of the query only against tile *i* of the bank image. Two photographs of the
same street from ten metres apart put the same building in different tiles, so
every per-tile term misses at once. Three full-height crops are coarser and far
more tolerant of that shift.

Two further confounds in that measurement, both fixable:

- The tile arm carried the **81/19 encoder imbalance** — it was measured before
  `--scale-b` existed. Worse, the same defect operates *between tiles*: each
  tile's block enters the cosine at its own raw activation norm, so a
  high-contrast road tile silently outweighs a flat sky one.
- **The model never saw tiles at all.** That experiment measured k-NN retrieval
  quality only. `StreetProj` is `Linear(4608 → 512)`; a tiled representation
  could not reach the policy without retraining.

So the defensible claim is *six concatenated tiles, unbalanced, scored by
cosine, lose to three crops at equal bytes* — not *tiling does not work*.

## The proposal

Replace concatenation with a small self-attention encoder over the tile set.
Attention removes the rigid index correspondence, can route context between
tiles (a sky tile informing a road tile that this is a desert), and can
**downweight an uninformative tile rather than averaging it in** — which is
precisely the failure mode hypothesised above.

```
image
  └─ 3×2 grid of 224×224 tiles at ~native resolution
       └─ DINOv2 (frozen) ─┐
       └─ SigLIP  (frozen) ─┴─ 6 tokens × 1536-d
            └─ per-block normalise, project 1536 → 512
            └─ + learned 2D position for the 3×2 grid
            └─ 1–2 multi-head self-attention layers, d=512
            └─ attention pooling, ~4 learned queries
            └─ 512-d output vector
```

Six tiles is a ceiling rather than a guess: 512/224 ≈ 2.3, so a third row is
interpolation of pixels that are not there.

### Precedent in this codebase, in both directions

The map side already does exactly this. `--map-layers 1` runs self-attention
over the 256 map tokens, and `--pool attn --pool-q 4` exists because **mean
pooling the map was spatially blind and attention pooling fixed it — 375 → 351
km**. The machinery is proven here and can be reused rather than invented.

The warning attached to that same result is the counter-argument: pooling
destroys layout. For retrieval you *want* some translation tolerance, but not so
much that spatial arrangement stops mattering. The two pressures point in
opposite directions, which is the reason to measure rather than argue.

## Where it sits, and why the loop is cheap

The bank needs one vector per image, computed offline, so the fusion head lives
between the frozen encoders and the bank. Two consumers:

| consumer | what it takes | count |
|---|---|---|
| retrieval bank | the pooled 512-d vector | 1.15M entries |
| policy (`StreetProj`) | the same vector, or the 6 tokens | 1 per query |

**This is the property that makes it tractable, and it is what separates this
idea from LoRA on the encoders.** The fusion head operates on *cached tile
vectors*, not pixels. Re-embedding all 1.25M bank entries after a weight change
is a small matmul — minutes — where re-running a ViT is ~2 h per encoder. The
kNN rebuild is 3.9 min at 1.15M. So even refreshing the bank every epoch costs
roughly 30% overhead, which makes end-to-end training viable rather than
theoretical.

### The memory arithmetic is the strongest argument

Bank RAM caps corpus size, and corpus size is the axis that moved this metric
most (+20.4 pp for 400k → 1.15M).

| representation | dims | bank at 1.15M |
|---|---:|---:|
| today (3 crops, 2 encoders) | 4,608 | 10.60 GB |
| tiles concatenated | 9,216 | 21.20 GB |
| **fused** | **512** | **1.18 GB** |
| fused + int8 | 512 | 0.59 GB |

Tiling by concatenation *doubles* the thing that constrains the system. Fusion
**shrinks it 9×**, or 18× with int8 — and int8 is already measured as
near-lossless here (top-1 neighbour unchanged in 512/512 queries, recall@32
0.9945, reconstruction cosine 0.99981, no outlier-channel problem).

So the fused head does not merely make tiling affordable; it makes the corpus
substantially larger at the same RAM, which is independently worth more than any
architecture change measured so far.

## Higher-resolution inputs, and why only the fused version can use them

Today a better photograph buys the user nothing. `preprocess` scales the short
side to 224 **first**, so a 4032×3024 phone photo and a 682×512 dataset frame
are reduced to the same thing before either encoder runs. The demo advertises
"any size, JPEG or PNG" and then discards everything above 224 on the short
edge. That is a real capability gap, not just wasted bytes: the system cannot
reward a user for taking a better picture.

Tiling changes that, because the amount of the photograph reaching the encoders
scales with the tile count:

| source | grid | pixels reaching the encoders | downsample |
|---|---|---:|---:|
| 682×512 dataset frame | 3×2 | 672×448 | ~1.0× |
| 4032×3024 phone photo | 3×2 | 672×448 | 6.0× |
| 4032×3024 phone photo | 6×4 | 1344×896 | 3.0× |
| 4032×3024 phone photo | 9×6 | 2016×1344 | 2.0× |

A *fixed* grid only makes each tile sharper; it does not use more of the frame.
To actually exploit a 12-megapixel upload you need **more tiles for the query
than the bank has** — and that is the point where the two designs diverge:

- **Concatenation cannot do it.** The output width is `n_tiles × 1536`, so a
  query with 24 tiles produces a vector of a different length than a bank entry
  with 6. There is nothing to take a cosine against.
- **Attention pooling can.** It consumes a *set* and emits a fixed 512-d vector
  regardless of how many tokens went in. Query and bank stay comparable while
  the query uses four times the tiles.

So resolution-adaptive inference is not an extra feature bolted onto this
design — it is a property that falls out of choosing set-based fusion over
concatenation, and it is unavailable in the version that was already measured
and rejected.

Two honest limits on it:

- **The gain is query-side only.** The bank is built from 512px OSV-5M frames
  and stays capped by them. What improves is the *query* embedding, hence which
  neighbours come back, hence the retrieval prior — which is worth having, since
  the prior is most of the system, but it is one-sided.
- **Variable `n` is a distribution shift.** Pooling over 24 tokens has different
  statistics than pooling over 6, so a head trained only at `n=6` may not be
  calibrated at `n=24`. The mitigation is cheap and should be built in from the
  start: **sample the tile count during training** so the head sees a range of
  `n`, and verify the similarity distributions of a `n=6` and `n=24` embedding of
  the *same* image match before trusting a mixed-`n` bank query.

## What trains it

The objective is the hard choice; the adapter is not.

1. **Geographic contrastive (recommended).** Positives are images within a few
   km, negatives far. Directly optimises what the bank is used for — cosine
   ranking — and is decoupled from the agent, so the bank is built once and the
   comparison against today is clean. Closest published analogue is GeoCLIP's
   image/GPS alignment.
2. **Through the agent's policy loss.** Tempting, because it optimises the
   actual task, but circular: the bank would need re-embedding every epoch and
   the retrieval targets move under the model. Viable given the cheap refresh
   above, but not the first experiment.
3. **Distillation onto the concatenated vector.** Cheap sanity check that the
   head can preserve information before asking it to improve on it.

Start with (1). 400k labelled training images is ample for a head of this size;
it would be hopeless for a from-scratch encoder.

## Risks, stated in advance

- **It can overfit training geography.** A learned metric has far more capacity
  to memorise than a fixed concatenation. `cell8` must be in the acceptance
  criteria from the start, not added afterwards — bank scaling is worth +20.4 pp
  under `sequence` and +2.0 pp under `cell8`, so the primary split will flatter
  this by roughly an order of magnitude.
- **Pooling may destroy the layout that makes the map policy work.** The map-side
  result says attention pooling preserves more than mean pooling, not that it
  preserves everything.
- **It breaks `k_pos`.** The shipping `dual` head feeds neighbour embeddings to
  `Linear(4608 → 128)`. A 512-d bank changes that input space, so the retrieval
  keys must be retrained alongside. Not a blocker — it is one retrain — but it
  means the fused bank and the old checkpoints are not interchangeable.
- **Out-of-distribution tiles.** A tile of bare road or empty sky is unlike
  anything these encoders were trained on. Overlapping tiles, or keeping one
  whole-image crop as a seventh token, is cheap insurance.

## Ablation ladder, cheapest first

Each step is a decision point, not a formality.

1. **Save the tile embeddings.** `tile_probe.py` computed 120k × 6 tile vectors
   and threw them away — every experiment below wants that cache, and
   regenerating it is ~2 h. Do this first regardless of anything else.
2. **Mean-pool instead of concatenate**, on that cache, numpy only.
   - recovers the deficit → rigid correspondence was the problem, attention is
     the upgrade path, continue
   - still worse than crop3 → the tiles genuinely carry less than full-height
     crops, and attention will not rescue that; stop here
3. **Per-block normalisation** before pooling, isolating the 81/19 effect at
   token level.
4. **Attention pooling with no self-attention layer** — separates "learned
   weighting of tiles" from "tiles talking to each other".
5. **Full self-attention head**, trained contrastively, evaluated on retrieval
   quality at equal bank bytes against crop3. Sample the tile count during
   training from the start — retrofitting variable `n` to a head trained only at
   `n=6` means retraining it.
6. **Variable-`n` check**: embed the same image at `n=6` and `n=24` and confirm
   the two agree well enough to share a bank. Cheap, and it is what licenses the
   high-resolution upload path.
7. **Only then**, rebuild the kNN cache and retrain the agent end to end.

Steps 2–4 are minutes each once step 1 exists. Step 5 is the first that needs
training. Step 6 is the first that needs the tile server.

## What would falsify it

- Mean-pooling does not close the gap to crop3 (step 2). The premise dies there.
- The fused 512-d head loses to a plain PCA of crop3 to 512-d — that is the null
  hypothesis and it is a real one, since PCA already preserves geographic
  retrieval quality while dropping recall@32 to 0.82.
- `sequence` improves while `cell8` degrades. That is memorisation, and it is the
  specific failure this whole direction risks.

## Measurement discipline this needs

The first tiling probe used a 30,000-image bank, where the hit rate is 0.67% —
about ten successful queries out of 1,500 — and it showed tiling **winning** at
every width. The same cached embeddings score 0.67% and 5.47% at bank sizes of
28,500 and 40,000, because each draw changes the queries as well as the corpus.
The sign flipped once the bank was large enough to resolve it.

So: **check that the metric has resolution at the sample size before reading a
direction off it**, use a bank of ≥100k for any retrieval comparison, and prefer
a control that can be checked against something already trusted — the
re-implemented preprocessing was verified against the cached embeddings at
cosine 0.9937 before its numbers were believed.
