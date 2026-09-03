# Architecture directions, after the pyramid

External review of `ebda2e4`, with my analysis. **Sequenced after the pyramid
work** (`pyrcache-hr47k` + the deconfounded attention arm), which is the last
outstanding item from the original plan.

The review's framing is right and worth keeping: raw capacity is not the
bottleneck. Retrieval is +17.3 pp, corpus is +12.1 pp, and 1536-d beats 768-d by
an amount that is no longer separable at a 3.40M bank. The gains left are in how
retrieval is combined, how the inputs are structured, and how the steps differ —
not in width.

I have reordered their list by *evidence support × cost*, and flagged three
places where the premise does not survive contact with the checkpoints.

---

## Measured before writing this

The review's first and strongest item argues the retrieval prior is a product of
experts that can annihilate a good visual prediction. That is checkable, so I
checked it. Learned values across five trained arms:

| arm | eps | log eps | `g_cell` per step | `g_sink` per step |
|---|---|---|---|---|
| `d768-b350-e6` | 0.0059 | −5.14 | −0.01, 0.11, 0.17, −0.04 | 0.22, 0.12, 0.36, **2.29** |
| `drop30` | 0.0062 | −5.09 | −0.03, 0.08, 0.14, −0.06 | 0.31, 0.05, 0.45, **2.37** |
| `drop70` | 0.0059 | −5.13 | −0.11, 0.07, 0.11, −0.07 | 0.32, −0.07, 0.50, **2.36** |
| `drop90` | 0.0038 | −5.58 | −0.06, 0.03, 0.12, −0.09 | **1.86**, 0.17, 0.15, 2.19 |
| `d1536-drop30` | 0.0063 | −5.07 | −0.02, 0.12, 0.11, −0.05 | 0.28, −0.07, 0.55, **2.57** |

**Three things follow.**

1. **The annihilation risk is already bounded.** `lp = log(p + eps)` with a
   *learned* eps that converges to ~0.006 in every arm, so the log-prior floors
   at −5.1 rather than diverging. Worst-case static contribution to a cell logit
   is `g_cell * log(eps)`, which is between −0.85 and +0.57 across every arm and
   step. That cannot erase a confident visual prediction. The review's mechanism
   is not what is happening.
2. **The static cell gates are near zero, and some are negative.** The prior is
   barely touching the cell decision through this path.
3. **The prior is overwhelmingly a sink signal at the finest step.** `g_sink` at
   step 3 is 2.2–2.6 in every arm — an order of magnitude above any cell gate.
   The retrieval prior's learned job is mostly *"the answer is not in this
   tile"*, not *"the answer is this cell"*.

**Caveat, and it matters:** `cond` adds a per-row `delta` to both gates, so the
effective gate is input-dependent and these statics are a floor, not the whole
story. Measuring the delta distribution needs a forward pass over real data.
**Do that before acting on items 1 or 10** — it decides whether the prior's
influence is small everywhere or merely small on average.

Note also `drop90`, the arm that broke `<1 km` precision: its step-0 `g_sink`
jumped to 1.86 from ~0.3. Whatever went wrong at p=0.9 shows up in the gates.

---

## Tier 1 — strong, and complementary to what we just learned

**Train on the states beam search actually visits** (their #8). The best item on
the list. Teacher forcing plus random sibling negatives is a partial
approximation of the rollout distribution, and today's audit found the gap
concretely: `dataset._negatives` samples `t ∈ {1,2}`, so **step 3 gets no sink
positives at all** — and step 3 is exactly where a wrong descendant needs
rejecting. This supplies a missing distribution rather than adding parameters,
and it is in the original plan (Phase I, scheduled sampling) but never done.
Their claim that it beats sink capacity is almost certainly right, and today's
supervision gap is the reason.

**Supervise retrieval reliability directly** (their #10). At training time we
know whether the prior's top cell is correct and how much mass sits on the true
child. The `cond` gate currently learns reliability only through downstream
cross-entropy, which is a very weak signal — and note that the conditional gate
was recorded as adding "essentially nothing", which is exactly what an
under-supervised head looks like. A direct reliability target is cheap and would
convert an existing null into a test of the *idea* rather than of the
*supervision*.

**Auxiliary street-only losses** (their #9). Directly motivated by today's
finding: the model over-relies on retrieval because nothing forces the visual
branch to stand alone. `--retr-drop 0.7` is a training-time patch for this;
a coarse street-only z4/z8 or country head is a structural one. Cheap, and the
two should compose.

## Tier 2 — credible, moderate cost

**Retrieve 128, learned-rerank to 16** (their #2). The learned keys can only
reweight the raw-cosine top 16, so a better neighbour ranked 17th is
unreachable. Keyed retrieval is worth +4.1 pp, which makes widening the pool
credible. Cost is modest: `build_knn` already does a full top-k pass, and k=128
takes the kNN cache from 187 MB to ~1.5 GB.

**Per-step policy adapters** (their #4). One `q_proj` and one fusion MLP serve
decisions from 2504 km to 611 m. Their dismissal of the per-encoder-gate null is
correct — that changed input scaling, not decision geometry. Supporting
evidence they did not cite: the retrieval prior *already* has per-step gates,
and the sink now has per-step keys and gates, so per-step specialisation is
established elsewhere in the model. Four `q_proj` copies is ~524k parameters.

**A conv stem over the 16×16 token grid** (their #5, step 2). Worth separating
from the raw-pixel conv, which is what I had been costing. A conv over *tokens*
gives adjacency as an inductive bias with **no cache change, no mask storage,
and negligible compute** — it is a different and much cheaper proposal than the
89.2% raw-pixel probe. Do it regardless of how sub2 lands; if sub2 is positive
it is the natural follow-up, and if sub2 is flat it still tests whether the
model lacks adjacency rather than within-patch detail.

**RoPE is applied to the wrong tensors** (their #6). Worth verifying: standard
RoPE rotates queries and keys inside attention; rotating the input embeddings
before arbitrary Q/K/V projections does not give the relative-position property.
If confirmed this is a latent correctness issue, not just a design choice, and
it would explain why `pos=both` was kept — the rotary half may be contributing
little. Cheap to check, cheap to fix, and it should be ablated against additive
learned positions rather than stacked with them.

## Tier 3 — worth doing, but the evidence is thinner

**Capped or mixture retrieval combination** (their #1). Demoted from their #1 on
the measurement above: the eps floor and the near-zero cell gates mean the
stated failure mode is already bounded. A convex mixture may still be better
parameterised than a floored product, and it composes with `--retr-drop`, but it
should be motivated by the measured `delta` distribution rather than by the
annihilation argument. **Measure first.**

**Six street tokens instead of one pooled vector** (their #3). Partly
contradicted by existing evidence they did not weigh: the 4608-d *concatenated*
arm, which preserves crop and encoder identity, tied with the 1536-d pooled arm.
If concatenation carries no advantage over pooling, crop identity is not obviously
what is missing. Cross-attention with a step-specific query is a stronger test
than concatenation, so this is not settled — but it is also largely what the
pyramid fusion head already explores, so run it *after* the pyramid arm rather
than beside it.

**Probabilistic click head** (their #11) and **hierarchical consistency losses**
(their #12). Both reasonable, both aimed at the tail and calibration, neither
addressing anything currently known to be broken.

## Tier 4 — high risk, or contradicts a locked decision

**Recurrent belief through the hierarchy** (their #7). The Markov property is a
locked decision in the plan, made deliberately: the state is the tile address
and nothing else. A recurrent state must branch correctly per beam, which is the
kind of plumbing that produced this week's parity bugs. The looped-transformer
arm was also a null. Not before Tier 1 and 2 are exhausted.

## Their "would not prioritise" list

Consistent with the record, and I would add nothing to it: wider street vectors
(now inside noise at 3.40M), deeper map transformer, wider beam (saturates at
~4, and the sink class was the fix), another absolute geographic table (GeoMem
failed and risks occupancy memorisation), a larger handcrafted conditional gate
(the existing one added nothing — though see Tier 1, it may be under-supervised
rather than under-sized), more low-resolution crop tiling (lost at equal bytes).

## Evaluation protocol for every arm

Their closing point is the most important sentence in the review and matches
today's result exactly:

> optimizing only the benchmark median will preferentially reward architectures
> that lean hardest on corpus coverage.

So every architecture arm is judged on **benchmark hit rate, KartaView n=5,000,
`cell8`, mean and P95 tail, and retrieval-disabled fallback**. The last one is
new and worth building: an arm's score with the prior switched off at inference
measures the dependency directly, rather than inferring it from a domain gap.
