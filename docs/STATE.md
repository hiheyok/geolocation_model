# Live state — updated 2026-09-03 11:55, queue drained

Read this first after a compaction. `docs/BACKLOG.md` holds deferred code and
performance work, `docs/ARCHITECTURE_NEXT.md` the reviewed architecture
directions, `docs/NAMING.md` the arm and stage naming. Durable findings are in
the memory directory; `git log --oneline -20` on branch `retrieval-dropout`
reconstructs the reasoning. Open PR: #13.

**Something is running.** `scripts/tonight_0902.py` drives a marker-gated
queue; restarting it is always safe and resumes from markers.
**The tile server goes offline at 12:30**, and the runner's cutoff is 12:20.
Every training and evaluation stage needs it, because beam rollout fetches
z12/z16 live. The pyramid and fusion stages do not, and are ordered last.

## The headline result, as it stands after parity

**The corpus axis, which produced nearly every gain in this project, does not
transfer to real photographs -- but the claim narrowed on 2026-09-03.**

| 2.65M -> 3.40M bank | as first reported | at shipping parity |
|---|---|---|
| OSV-5M `<25 km` | **+2.8 pp [+1.96, +3.66]** | unaffected (bootstraps were never wrong) |
| KartaView `<25 km` | **-1.32 pp [-2.04, -0.60]** separated | **-0.46 pp [-1.12, +0.18]** inside noise |
| KartaView `<200 km` | -1.84 pp separated | -0.92 pp [-1.90, +0.02] inside noise |
| KartaView median | +35.5 km separated | **+30.8 km [+11.5, +47.8]** separated |
| KartaView `<1 km` | +0.08 pp noise | **+0.30 pp [+0.02, +0.60]** separated, *favouring* the bigger bank |

Parity cost the small-bank arm 0.86 pp and the large-bank arm nothing, which is
why the gap closed. **What survives: the bigger corpus makes the typical
off-domain error ~31 km worse. What does not: that it costs hit rate.**

The 2x2 attributing this to the **model** rather than the bank (model effect
-0.76 and -1.20 pp, both separated; bank effect -0.12 pp noise and -0.56 pp)
is **still on the old measurement** and is re-running as `hrfix-x-*`. Do not
quote it until then -- and note `--bank` had its own bug, fixed 2026-09-03: it
took the row restriction from the checkpoint's k-NN cache, which indexes a
different bank file.

**`--retr-drop` works, and p = 0.7 survives parity.** The full curve,
re-measured at each checkpoint's own bank and its own `retr_k`, n=5,000 paired:

| p | ext `<1 km` | ext `<25 km` | ext median | vs p=0 on `<25 km` | vs p=0 on `<1 km` |
|---|---|---|---|---|---|
| 0.0 | 2.3% | 11.6% | 505.6 | — | — |
| 0.1 | 2.2% | 11.1% | 515.9 | -0.46 noise | -0.14 noise |
| 0.3 | 2.1% | 12.1% | 503.1 | +0.50 noise | -0.20 noise |
| 0.5 | 2.2% | 12.6% | 468.1 | **+1.00 [+0.36, +1.62]** | -0.12 noise |
| **0.7** | 2.1% | 13.4% | **435.7** | **+1.80 [+1.08, +2.50]** | -0.20 noise |
| 0.9 | **1.7%** | 14.0% | 390.0 | **+2.38 [+1.50, +3.22]** | **-0.58 [-0.92, -0.24]** |

**Still use p = 0.7, but for a different reason than before.** The old story was
that 0.9 overshoots on the external hit rate. It does not: 0.7 vs 0.9 on
`<25 km` is +0.58 pp [-0.24, +1.38], inside noise, and 0.9 is separated *better*
on `<200 km` (+2.02 pp) and on the median (-45.7 km). What 0.9 actually costs is
`<1 km` -- separated worse than 0.7 by 0.38 pp and worse than no dropout at all
by 0.58 pp -- plus a **-5.7 pp separated benchmark regression**, which is an
OSV-5M bootstrap and so untouched by the parity bug. So 0.9 buys coarse
accuracy with fine precision and with the dense-coverage case.

**Two claims did not survive.** The effect at p=0.3 is now inside noise
(+0.50 pp, was +1.10 pp separated), so p >= 0.5 is needed before anything is
separable -- the original 0.3 guess is no longer supported by this data. And
the headline size shrinks: +1.80 pp rather than the +2.48 pp on file.

**Why parity moved the dropout arms and not p=0.** Every arm in the curve shares
one bank and one restriction, so the spread is about the arm: p=0 lost 0.00 pp,
while p=0.1 to 0.7 lost 0.56 to 0.68. K=32 is twice the neighbours these arms
train with, and a model trained on a randomly thinned neighbour set has more to
gain from the extras than one tuned to exactly 16 -- so **the confound was
correlated with the treatment**, and the old measurement flattered the very
thing under test. p=0.9 breaks the pattern (-0.16 pp), which fits: at 0.9 the
policy barely uses retrieval, so extra neighbours do little for it either. The
relationship is an inverted U in p, not monotone.

**And with dropout the corpus does transfer.** b265 (no dropout, 2.65M) against
drop70 (3.40M), at parity: `<25 km` **+1.34 pp [+0.60, +2.06]**, `<200 km`
**+3.58 pp [+2.52, +4.64]**, median **-39.2 km [-61.1, -20.4]**, all separated.
So "the corpus axis does not transfer" is a statement about the *untreated*
model. Scale the corpus **and** train with `--retr-drop 0.7`.

`scripts/parity_report.py` builds the whole comparison from the exports as they
land; run it rather than reading numbers out of logs.

## Arms, at shipping parity

| arm | OSV-5M `<25 km` | KartaView `<25 km` | KartaView median |
|---|---|---|---|
| `d768-b265-e6` | 72.8% | 12.1% | 474.8 km |
| `d768-b350-e6` | 75.6% | 11.6% | 505.6 km |
| **`d768-b350-e6-drop70`** | ~75.6% | **13.4%** | **435.7 km** |
| `d768-b350-e6-drop90` | **-5.7 pp** | 14.0% | 390.0 km |
| `d768-b350-e6-sub2` | — | 12.5% | 459.7 km |
| `d1536-b350-e6` | 76.2% | **11.2%** | **532.5 km** |
| `d1536-b350-e6-drop30` | **76.9%** | 12.3% | 492.0 km |
| `d1536-b350-e6-drop70` | 75.9% | 12.8% | 467.4 km |

**Ship `d768-b350-e6-drop70`.** The rank inversion is sharper at parity than it
was before: `d1536-b350-e6-drop30` has the best benchmark number of any arm and
loses externally to the half-width `drop70` by **-1.06 pp [-1.78, -0.32]** and
**-56.3 km [-80.0, -35.6]**, both separated. Plain `d1536-b350-e6` is the best
non-dropout benchmark arm and the worst arm on photographs, on both measures.

## The parity caveat is resolved

Every external number above is measured against each checkpoint's own bank at
its own `retr_k`. All ten arms, the 2x2 and the multi-photograph curve were
re-run; nothing outstanding.

**Keep the lesson, not the numbers.** The caveat written when the bug was found
said every arm was measured the same way, so paired *directions* would hold. It
was wrong. Parity cost `p=0` nothing and each dropout arm 0.6 pp — same bank,
same restriction, so the spread is about the arm. K=32 is twice the neighbours
these arms train with, and a model trained on a thinned neighbour set gains more
from the extras than one tuned to 16. **The confound was correlated with the
treatment**, and it flattered exactly the thing under test. One separated result
became noise and one effect size fell by a quarter.

So the rule is not "a uniform error preserves paired directions" — it is **ask
whether the error is independent of the treatment first**, and re-measure before
quoting anything when it touches the same quantity the experiment manipulates.

## Corpus dependency, measured directly

| arm | retrieval **on** | retrieval **off** | it loses |
|---|---|---|---|
| `d768-b350-e6` (p=0) | 75.6% | **2.7%** | **-72.9 pp** |
| `d768-b350-e6-drop70` | 75.2% | **10.7%** | **-64.5 pp** |
| `d768-b350-e6-drop90` | 69.9% | **19.3%** | **-50.6 pp** |

Medians without retrieval: 888.9, 340.7, 177.8 km. Every pairwise contrast
separated, and the ordering is monotone in p.

**Dropout buys standalone capability.** The three arms are within 6 pp of each
other with the corpus and differ by 7x without it, which is the dependency the
`--retr-drop` experiment was about, measured directly instead of inferred from
a domain gap.

This was a *prediction*, not a post-hoc reading: `scripts/cond_probe.py` put the
prior's share of the logit spread at 42% for p=0 against 28% for drop70, and
`scripts/diag_beam.py` showed dropout trading top-1 for in-beam recall. Three
independent routes, one mechanism.

**The first column is the uncomfortable one.** With retrieval removed the
shipping arm scores **2.7%** and a median of 889 km. Even the most regularised
arm reaches only 19.3%. So the 75.6% headline is overwhelmingly corpus
coverage, and the visual-plus-map policy is weak standing alone -- consistent
with [[bank-beats-training-data]] and [[data-is-the-binding-constraint]], but
far starker than either records.

**How to apply:** report `--retr-off` alongside every architecture arm. An
arm that improves the headline while losing standalone capability has bought
coverage, not geolocation, and only this column can tell the difference.
Verified before believing: `retr_prior` returns None for `nbrs=None`, so the
prior is genuinely absent rather than contributing a degenerate term.

## Other results

* **Multi-photograph query** — a second angle is +1.2 pp [+0.12, +2.32] and
  −43 km [−74, −14], saturating there; photos 3–4 add +0.1 pp between them. No
  retraining: the prior is permutation-invariant over neighbours. Needs the
  round-robin merge with dedupe, and ~2,500 groups to resolve at all.
* **768-d is free at 3.40M** — +0.6 pp [−0.18, +1.32], no longer separable,
  against +1.05 pp at 2.65M. Predicted 75.2% / 2.0 km, observed 75.6% / 1.9 km.
* **2×2 map tokens: null, slightly negative.** Road orientation is recoverable
  at 74.8% from 2×2 against 52.7% from the current token (52.6% majority), so
  the tokenizer genuinely destroys real structure — and the model is *worse*
  with it: −0.50 pp `<1 km`, −1.58 pp `<200 km`, +26 km median, all separated.
  **This also settles the raw-pixel conv**, which was gated on sub2 helping.
  Untested and much cheaper: a conv over the 16×16 *token* grid, giving
  adjacency rather than detail at no storage cost.
* **Sink capacity retired, not deferred.** `dataset._negatives` draws
  `t ∈ {1,2}`, so step 3 gets no sink positives. With per-step extra keys the
  unsupervised step drove its gate to −1.58: it learned "never reject at step
  3", which every training row rewards and beam search punishes. Re-run only
  after negatives cover step 3.
* **Retrieval gates, read from five checkpoints.** `log_eps` converges to about
  −5.1 everywhere, so the log-prior is floored, not divergent; `g_cell` is
  −0.11..+0.17 with several negative; `g_sink` at step 3 is 2.2–2.6. The
  prior's learned job is mostly "the answer is not in this tile" rather than
  "the answer is this cell".

## Queue: empty. Everything ran.

The runner finished at 06:24 and the chained relaunch drained the stages added
after it started, finishing 06:50. No stage was dropped and nothing needs the
tile server any more. `scripts/parity_report.py` rebuilds the external table
from the exports; `scripts/pair_npz.py` does any single contrast.

**Results that landed after 02:45, all of them settling something:**

**The gap in the matrix is not worth filling.** `d1536-b350-e6-drop70` — best
width times best p — loses on both measures. Against `d768-b350-e6-drop70`
externally: `<1 km` **-0.60 pp [-0.90, -0.30]**, `<200 km` **-1.92 pp
[-3.02, -0.84]**, median **+31.7 km [+8.2, +53.2]**, all separated. On the
benchmark it loses to `d1536-b350-e6-drop30` by +0.20 to +1.62 pp on `<25 km`
and 0.4-0.8 km of median, also separated. **Ship `d768-b350-e6-drop70`. Width
and dropout do not compose.**

**And the benchmark cost of dropout is width-dependent** — a claim that would
have been easy to carry over from the 768-d curve without checking. At 768-d the
benchmark is flat from p=0 to p=0.7. At 1536-d, `drop30 -> drop70` trades
`<1 km` precision (-0.64 pp, separated) for coarse accuracy (+1.50 pp at 200 km,
-24.6 km median, both separated) — which is exactly the signature p=0.9 shows at
768-d. **The wider model reaches the overshoot regime at a lower p.**

**The 2x2 survives parity: it is the model, not the bank.**

| `<25 km`, at parity | bank 2.65M | bank 3.40M |
|---|---|---|
| model b265 | 12.06% | 12.00% |
| model b350 | 11.34% | 11.60% |

Bank effect at fixed model: -0.06 pp and +0.26 pp, both noise; +12.1 km and
-5.5 km, both noise. Model effect at fixed bank: **-0.72 pp [-1.32, -0.12]**
separated on the 2.65M bank and -0.40 pp (noise) on the 3.40M, with the median
separated at both (**+36.2 km** and **+18.7 km**). The bank is innocent in all
four cells; the mechanism claim holds.

**Several photographs: unchanged by parity.** +1.36 pp [+0.28, +2.48] for a
second angle against +1.24 pp before, still separated, still saturating at two
(2->4 adds +0.16 pp). 2,496 groups.

**The pyramid fusion head: retracted, then re-established on better evidence
(2026-09-03).** It was first reported negative from a comparison a Codex review
correctly showed was confounded. The conclusion was withdrawn, then survived
three attempts to break it.

| configuration | train loss | `<25 km` | median km |
|---|---|---|---|
| **head at INIT, no training** | -- | **33.1%** | -- |
| 5.0 km positives, unmasked (original) | 3.82 | 20.9% | 383.4 |
| 5.0 km positives, false negatives masked | 0.063 | 20.5% | 384.1 |
| 0.5 km positives, masked | 0.0145 | **12.9%** | 670.5 |
| *L0+L1+L2 level (mean), 1536-d* | -- | *33.8%* | *179.1* |

**Every trained configuration is far worse than not training at all**, and the
ordering is monotone in the wrong direction: the better the training loss, the
worse the retrieval. That is overfitting, not a tuning problem -- 2.5M
parameters against 38,009 images on a contrastive task the head can nearly
solve exactly. A tighter positive radius makes the positives near-identical
viewpoints, which is easier to memorise and generalises worse.

Three candidate explanations were tested and none survived:

* **the random projection** -- worth 0.63 pp of the 11.30 (head at init 33.1%
  against the 1536-d mean's 33.7%);
* **the false negatives** -- 50.3% of off-diagonal cells were true positives,
  which is a real defect and was fixed, and it moved the result 20.9% -> 20.5%;
* **the positive radius** -- predicted to be the binding constraint because
  `--pos-km 5.0` declares any two images within 5 km identical. Tightening it
  to 0.5 km made things **much worse**, 20.5% -> 12.9%. That prediction was
  wrong.

So the conclusion returns to where it started, on far better evidence: **the
learned fusion head is harmful, and the multi-level mean is the win.** What
changed is that it is now a measured claim about training rather than an
artefact of a confounded comparison. If it is ever reopened, the thing to
attack is capacity and data volume, not the objective's details.

**What still stands, untouched by all of this:** the multi-level mean. L0+L1+L2
beats L0 crops alone by +1.6 pp at 25 km and 40 km of median at equal width,
with no learned head anywhere in it.

## Fixes: what got done on 2026-09-03, and what is left

**Done tonight**, each verified by making it fire, not by reading it:

* `config.RELEASE` no longer defaults to `s01`. It was the root cause of five
  silent failures; 39 scripts were relying on the default. Tests name a release
  in `tests/conftest.py`. The live queue is unaffected -- the runner sets the
  variable before importing config.
* `dataset.py` reads `done.u8.npy` and refuses a half-fetched tile cache; an
  unfetched tile is an all-zero histogram, which is a legal input. Checks the
  rows a split can reach, plus all z4/z8 once negatives are on. `fetch_tiles`
  exits non-zero when the mask is short, so the runner stops marking it done.
  Both live s10 caches are complete, so nothing in flight is affected.
* Beam width can no longer go ragged under sink pruning (it would have paired a
  beam with another image's street row, and an unused score slot read as 0.0 --
  better than any real path). Old and new code give **bit-identical** per-image
  errors at k=1, 2, 16.
* The kNN contract: split hash, street file, query count and neighbour count
  are checked, not just `split_mode`. All 14 distinct combinations load clean.
* `eval_highres --bank` took its row restriction from the checkpoint's own kNN
  cache, which indexes a different file -- silently, since the indices are in
  range. It now uses the cache built over the bank in use, and its summary line
  names that bank.
* bootstrap's per-image error cache is keyed on the checkpoint file, not the
  tag. Of 66 existing caches, 46 were provably written after their checkpoint,
  20 belong to deleted arms, and **none were stale** -- so the bug never fired,
  and the migration recomputes nothing.
* `diag_beam` now gathers `nbr_emb` (it was diagnosing `pos` arms on raw cosine
  and `dual` arms with the negative branch off); `error_profile` takes
  `evaluate()`'s seeded sample instead of the geographically biased first N;
  all three diagnostics call `check_split`.

**Still open**

* `quality()` in the retrieval prior sees the **unmasked** top-1 similarity even
  when `--retr-drop` hid that neighbour. **Deliberately unfixed:** changing it
  alters training semantics and breaks comparability with the p-curve. Fix with
  a re-baseline, not silently.
* `--save-opt` saves optimizer state that nothing loads; scheduler and RNG are
  not saved. A 2+2+2 ladder is therefore **three optimizer restarts**, not six
  continuous epochs -- which bears on reading "e4 vs e6 is inside noise" as
  convergence. Deferred for the same reason as `quality()`: fixing it changes
  training, and every arm on file was trained the current way.
* **Structural** -- one checkpoint runtime factory; a versioned checkpoint
  schema; artifact provenance manifests (subsumes filename inference and cache
  keys, and would have prevented four of this week's guards); one public
  `score_policy` so a caller cannot drop an argument.
* **Highest-value test** -- a checkpoint-contract integration test through
  training, beam, bootstrap and serve. Component tests could not have caught the
  sink parity bug: the defect was in how a caller assembled the pieces.

## Hardware, measured

* **Training is GPU-bound**, 92–96% at 165–172 W. The old "1% GPU" reading was
  a broken sensor. **Never stack GPU jobs**, and treat wall-clock timings as
  unreliable when the machine is also running a game.
* Encoder passes draw ~297 W at 99%; the fusion head ~205 W at 85%.
* 31.7 GB RAM. The 15.4 GB sub2 map cache runs at ~31 GB total: tight, not
  thrashing. 768-d is the only width whose 4.90M bank could be resident.
* C: NVMe; E: HDD holding the 47,646-image harvest.

## Recurring mistakes

* **A value derivable from an artifact, restated in code or taken from an
  ambient default.** Ten of fourteen bugs this week. None raised; all returned
  plausible numbers. Hence provenance manifests leading the backlog.
* **Conditions that match stale data.** `fuse-attn-pyr47` was killed at epoch
  11/12 because a wait loop grepped an append-only log for `FAIL` and matched
  failures from the previous day. Same shape as `check_args` testing the string
  `"0"` for truthiness. Scope every log check to the newest `==== attempt`.
* **The 2,000-image selection set is not evidence.** It pointed the wrong way
  three times in one day: `d1536` dropout, the `near` ordering, and sub2. It
  picks checkpoints; it does not measure them.
* **A wide interval means get more data, not weaken the claim.** n=1,000 put
  the corpus contrast inside noise; n=5,000 separated it.
* **Capacity without supervision gives confident wrong behaviour, not a null**
  — the step-3 sink gate at −1.58.
* **Do not theorise about performance; measure it.** One slowdown was blamed on
  token width, then cold cache, before the real cause (a game on the GPU); and
  the JPEG decode was assumed expensive when the 33-way crop was.
* **A backslash inside a Bash heredoc does not survive.** `"\n"` written in a
  patch script inside `python - <<'PY'` arrives as a real newline, producing an
  unterminated string literal. This happened **four times on 2026-09-03 alone**,
  despite already being on this list -- the old wording said "prefer line-based
  edits", which was not a rule anyone could follow. The rule is:
  **never type a backslash inside a Bash heredoc.** Either build it as
  `N = chr(92) + "n"` and concatenate, or write the patch script to a file with
  the Write tool and run it. Same for an apostrophe inside `$(printf ...)`,
  which once broke a commit; use `git commit -F -`.
* `python` on PATH is MSYS2's and lacks numpy. Always
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe`.
