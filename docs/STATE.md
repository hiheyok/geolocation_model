# Live state — updated 2026-09-03 02:45

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

**`--retr-drop` fixes it and costs nothing** -- also on the old measurement,
and the one most exposed, because neighbour count is exactly what the knob
manipulates:

| p | 0.0 | 0.1 | 0.3 | 0.5 | **0.7** | 0.9 |
|---|---|---|---|---|---|---|
| external `<25 km` | 11.6% | 11.7% | 12.7% | 13.3% | **14.1%** | 14.1% |
| external `<1 km` | 2.2% | 2.0% | 2.2% | 2.3% | **2.3%** | 1.7% |
| external median | 486 | 497 | 475 | 436 | **410 km** | 382 |
| benchmark vs p=0 | -- | borderline | noise | noise | noise | **-5.7 pp** |

**p = 0.7 is not tuned until the parity curve confirms it.** The benchmark is
flat across the whole useful range and only reacts to the overshoot at 0.9, so
tuning this on OSV-5M alone sets it to zero.

`scripts/parity_report.py` builds the whole comparison from the exports as they
land; run it rather than reading numbers out of logs.

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

Every KartaView figure in the arms table was measured with `eval_highres`
**searching all 3,500,000 rows of the embedding file rather than the 3,400,180
the checkpoint's k-NN was built over, and with K=32 where these arms train with
`retr_k=16`.** Fixed; the `hrfix-*` stages re-measure the same 5,000 images.

**Two arms are re-measured so far, and parity is not a uniform shift.** It cost
`d768-b265-e6` 0.86 pp on `<25 km` (separated) and `d768-b350-e6` nothing
(+0.00 pp). So the assumption that every arm was measured identically and the
paired directions would hold is **not safe**: the size of the correction
depends on the arm, and it already turned one separated contrast into noise.

Treat every un-re-run external number as provisional, including the whole
p-curve. Run `scripts/parity_report.py` rather than reading logs; it marks what
has landed. OSV-5M bootstraps were never affected.

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

The runner (pid 23964 as of 02:40) holds the list it was started with. A second
process is chained to its exit -- it polls that **pid**, not a log, and has no
"proceed anyway" fallback, because two jobs on one GPU is the thing the
measurements say never to do. That relaunch picks up the two stages added after
the runner started.

1. Eight remaining `hrfix-*` shipping-parity external re-runs (~13 min each,
   needs tiles). `b265` and `b350` are done.
2. `hrfix-x-m265-b350` and `hrfix-x-m350-b265` -- the 2x2 at parity. **Added
   after the runner started, so they need the chained relaunch.**
3. `d1536-b350-e6-drop70` ladder + bootstrap + external (~2 h, needs tiles).
4. `fuse-attn-pyr47` -- the deconfounded pyramid attention arm. **Must re-run:
   it reached epoch 11 of 12 and was killed by accident. Needs no tiles.**
   38,009 training images against the 14,938 that produced the -15.75 pp
   collapse, which was recorded as confounded and is still unsettled.

`cache/street/s10/pyr47.f16.npy` is complete and verified: 47,646 images,
33 x 2 x 768, 4.83 GB, both encoder slices non-zero, zero decode failures.

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
* `\n` inside heredoc patch strings repeatedly produced broken files, and an
  apostrophe inside a `$(printf ...)` in a heredoc broke a commit. Prefer
  line-based edits and `-F -` for commit messages.
* `python` on PATH is MSYS2's and lacks numpy. Always
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe`.
