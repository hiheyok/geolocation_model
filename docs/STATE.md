# Live state — rewritten 2026-09-03 02:10, before a context compaction

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

## The headline result of 2026-09-02

**The corpus axis, which produced nearly every gain in this project, is
negative on real photographs.**

| step | OSV-5M `<25 km` | KartaView `<25 km`, n=5,000 |
|---|---|---|
| 2.65M → 3.40M bank | **+2.8 pp [+1.96, +3.66]** | **−1.32 pp [−2.04, −0.60]** |

Both separated. A 2×2 (each model against both banks) blames the **model**, not
the bank: model effect at fixed bank −0.76 and −1.20 pp, both separated; bank
effect at fixed model −0.12 pp (noise) and −0.56 pp. Training against dense
retrieval teaches a dependency that fails where top-1 similarity is 0.71
rather than 0.90.

**`--retr-drop` fixes it and costs nothing.** Hide each retrieved neighbour
with probability p during training:

| p | 0.0 | 0.1 | 0.3 | 0.5 | **0.7** | 0.9 |
|---|---|---|---|---|---|---|
| external `<25 km` | 11.6% | 11.7% | 12.7% | 13.3% | **14.1%** | 14.1% |
| external `<1 km` | 2.2% | 2.0% | 2.2% | 2.3% | **2.3%** | 1.7% |
| external median | 486 | 497 | 475 | 436 | **410 km** | 382 |
| benchmark vs p=0 | — | borderline | noise | noise | noise | **−5.7 pp** |

**Use p = 0.7.** The benchmark is flat across the whole useful range and only
reacts to the overshoot at 0.9, so tuning this on OSV-5M alone sets it to zero.
p=0.9 costs the `<1 km` bucket, which is the bucket retrieval delivers: dropout
trades fine precision for coarse robustness, and past 0.7 the trade turns.

## Arms

| arm | OSV-5M `<25 km` | KartaView `<25 km` | KartaView median |
|---|---|---|---|
| `d768-b265-e6` | 72.8% | 12.9% | 450.6 km |
| `d768-b350-e6` | 75.6% | 11.6% | 486.2 km |
| `d1536-b350-e6` | 76.2% | 11.7% | 518.2 km |
| `d768-b350-e6-drop70` | ~75.6% | **14.1%** | **409.8 km** |
| `d1536-b350-e6-drop30` | **~76.9%** | 12.5% | 458.1 km |

Best benchmark and best real-world are **not the same arm**.
`d1536-b350-e6-drop70` is queued and is the obvious gap in the matrix.

## READ BEFORE QUOTING ANY EXTERNAL NUMBER

Every KartaView figure above was measured with `eval_highres` **searching all
3,500,000 rows of the embedding file rather than the 3,400,180 the checkpoint's
k-NN was built over, and with K=32 where these arms train with `retr_k=16`.**
Fixed — it now reads `bank_rows` and `retr_k` from the k-NN metadata — and
eleven `hrfix-*` re-runs are queued.

Every arm was measured identically, so the paired *directions* should hold. But
**neighbour count is exactly what `--retr-drop` manipulates**, so the p=0.7
optimum must be reconfirmed at K=16 before it is called tuned. The four
affected memory files carry this caveat. OSV-5M bootstraps are unaffected.

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

## Queue

1. Eleven `hrfix-*` shipping-parity external re-runs (~2.5 h, needs tiles).
2. `d1536-b350-e6-drop70` ladder + bootstrap + external (~2 h, needs tiles).
3. `fuse-attn-pyr47` — the deconfounded pyramid attention arm. **Must re-run:
   it reached epoch 11 of 12 and was killed by accident (see below). Needs no
   tiles.** 38,009 training images against the 14,938 that produced the
   −15.75 pp collapse, which was recorded as confounded and is still unsettled.

`cache/street/s10/pyr47.f16.npy` is complete and verified: 47,646 images,
33 × 2 × 768, 4.83 GB, both encoder slices non-zero, zero decode failures.

## Fixes still to do

Priority order; detail in `docs/BACKLOG.md`.

**Behavioural**

* `config.RELEASE` defaults to `s01` — **root cause of four silent failures**.
  The fix is no default at all; touches every script and runner.
* `dataset.py` never reads `done.u8.npy`, so a partially-fetched tile cache
  gives zero-filled token rows that look valid. `fetch_tiles` also exits 0 with
  failures outstanding, so the runner writes a success marker.
* `quality()` in the retrieval prior sees the **unmasked** top-1 similarity even
  when `--retr-drop` hid that neighbour. **Deliberately unfixed:** changing it
  alters training semantics and breaks comparability with the p-curve. Fix with
  a re-baseline, not silently.
* k-NN caches validated on `split_mode` only — not split hash, street file,
  query count or `knn_k`.
* Bootstrap error cache keyed by tag with no checkpoint hash, so retraining
  under one tag returns stale numbers.
* `diag_beam` builds the right dataset but never gathers `nbr_emb`, so `pos`
  falls back to raw cosine and `dual` loses its negative branch.
* Beam sink pruning assumes equal live beams per image; unequal counts can pair
  a beam with the wrong street image.
* `error_profile` samples the first N test rows, which are geographically
  biased; the main evaluator uses a seeded random sample.
* `summary_table`, `error_profile`, `diag_beam` never call `check_split()`.
* `--save-opt` saves optimizer state that nothing loads; scheduler and RNG are
  not saved. A 2+2+2 ladder is therefore **three optimizer restarts**, not six
  continuous epochs — which bears on reading "e4 vs e6 is inside noise" as
  convergence.

**Structural** — one checkpoint runtime factory; a versioned checkpoint schema;
artifact provenance manifests (subsumes filename inference and cache keys, and
would have prevented three of this week's guards); one public `score_policy` so
a caller cannot drop an argument.

**Highest-value test** — a checkpoint-contract integration test through
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
* `\n` inside heredoc patch strings repeatedly produced broken files, and an
  apostrophe inside a `$(printf ...)` in a heredoc broke a commit. Prefer
  line-based edits and `-F -` for commit messages.
* `python` on PATH is MSYS2's and lacks numpy. Always
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe`.
