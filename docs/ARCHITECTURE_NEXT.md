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

**Three things follow.** *(All three were wrong. See the measurement below,
which is the one this section itself said to do before acting.)*

1. ~~The annihilation risk is already bounded.~~ `lp = log(p + eps)` with a
   *learned* eps converging to ~0.006 does floor the log-prior at -5.1, but the
   worst-case contribution is `g_cell * log(eps)` and `g_cell` is not the static
   value.
2. ~~The static cell gates are near zero, and some are negative.~~ They are, and
   it does not matter: `cond` moves them.
3. ~~The prior is overwhelmingly a sink signal at the finest step.~~ It is a
   strong sink signal at *every* step.

## Measured properly, 2026-09-03 — and it reverses the conclusion

The caveat above said the statics are a floor and that measuring the `delta`
distribution needs a forward pass over real data. That pass is
`scripts/cond_probe.py`, and it changes the answer. Effective gates, static plus
`cond` delta, over 10,240 teacher-forced val rows:

| gate | static, as recorded above | effective, measured |
|---|---|---|
| `g_cell` | -0.11 .. +0.17 | **median 1.0-2.6 by step, p99 2.97, negative in 0.0% of rows** |
| `g_sink` | 0.2 .. 2.6 | **median 4.9-7.3, p99 9.5, at every step not just step 3** |

The conditional gate is doing essentially all of the work, so reading the
statics said almost nothing about the model's behaviour.

**What the prior is actually worth on a decision.** Comparing its additive
contribution against the policy logits it lands on, same rows:

| | `d768-b350-e6` | `d768-b350-e6-drop70` |
|---|---|---|
| final logit spread (max-min) | 15.65 | 16.67 |
| top1 - top2 margin | 2.71 | 2.19 |
| **prior's own spread** | **6.74** | **5.07** |
| prior / final spread | 0.42 | 0.28 |
| **prior spread exceeds the top-1 margin** | **85.4% of rows** | **76.9% of rows** |
| prior spread exceeds the whole logit spread | 0.1% | 0.0% |

**The prior is a first-class term in the decision, not a tiebreaker.** It
outweighs the margin between the best and second-best cell in most rows. So the
review's mechanism is real and item 1 is **re-promoted**: how the prior is
combined is worth testing, because it is deciding the answer often enough to
matter. Its strongest framing is still wrong, though -- the prior dominates the
*entire* logit range in 0.1% of rows, so "annihilates a good visual prediction"
is not the typical case.

**And this is independent evidence for why `--retr-drop` works.** Dropout cuts
the prior's influence from 42% to 28% of the logit spread and from 85.4% to
76.9% of rows where it outweighs the margin. The p-curve showed the knob helps
on real photographs; this shows the mechanism in the model's own logits rather
than inferring it from a domain gap. See [[benchmark-is-blind-to-retr-drop]].

**The lesson worth keeping.** The original three conclusions were drawn from
parameters that were easy to read, in place of a measurement that was known to
be the right one and was recorded as "do this first". Easy-to-read stood in for
correct, which is the same shape as this project's other silent failures.

---

## Tier 1 — strong, and complementary to what we just learned

**Capped or mixture retrieval combination** (their #1). **Promoted here on
2026-09-03**, having been demoted to Tier 3 on a reading of the static gates.
The measurement that reading deferred says the prior's additive contribution
outweighs the top-1 margin in 85% of rows, so how it is combined is deciding the
answer often enough to matter. A convex mixture is a different parameterisation
of a quantity that is already doing most of the work, not a guard against a rare
failure -- and it composes with `--retr-drop`, which reduces the same influence
from 42% to 28% of the logit spread by a different route. Worth testing both
separately and together, since they may be substitutes.

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

*(Their #1, capped or mixture retrieval combination, has moved to Tier 1. It was
demoted here on a reading of the static gates; the measurement that reading
deferred says the opposite.)*

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
