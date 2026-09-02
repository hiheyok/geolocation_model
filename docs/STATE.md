# Live state — rewritten 2026-09-02 02:00, before a context compaction

Read this first after a compaction. `docs/NEXT.md` is the forward plan;
`docs/NAMING.md` explains arm names. Durable findings also live in the memory
directory and in the git log, which is unusually detailed —
`git log --oneline -40` reconstructs most of the reasoning.

**Nothing is running.** The demo server, all training, and the harvest are
stopped. `git status` is clean apart from two probe `.npz` files in `runs/`.

## What ships

**`d1536-b350-e6` — 1.8 km median, 311.0 mean, 76.2% within 25 km** on the
`sequence` test split at n=5,000 (`runs/BOOTSTRAP_bank70.md`). Pooled 1536-d
street vector, 3.50M-image retrieval bank, ~6.07M trainable parameters in front
of 178.6M frozen.

Read that number with the coverage caveat below. It is a benchmark figure, not
accuracy on an arbitrary photograph.

## The corpus curve, four points

| bank | median km | mean | `<25 km` | step |
|---|---|---|---|---|
| 1.15M `d1536-b115-e6` | 7.7 | 391.4 | 64.1% | — |
| 1.90M `d1536-b190-e6` | 3.6 | 367.5 | 70.6% | +6.50 [+5.50, +7.64] |
| 2.65M `d1536-b265-e6` | 2.5 | 356.7 | 73.8% | +3.21 [+2.40, +4.02] |
| **3.50M `d1536-b350-e6`** | **1.8** | **311.0** | **76.2%** | **+2.34 [+1.56, +3.12]** |

**4.3x median reduction, +12.1 pp, architecture untouched throughout.** Every
ladder converges at six epochs (e4 vs e6 inside noise), so the gains are not
undertrained baselines flattering newer arms.

**Predict the next step from retrieval, not from the metric's history.**
Extrapolating the previous *metric* gain overshot once (+9 predicted, +6.5
observed). Extrapolating the **measured top-1 similarity increments** was right
three times running, most recently predicting +2.0 to +2.6 pp and observing
+2.34. The curve: 0.8926 → 0.9000 → 0.9046 → 0.9080, near log-linear, converting
at 8.8 then 7.0 pp per 0.01 of similarity. Build the kNN, read the delta,
multiply by the most recent conversion rate.

**The mean barely moves while the median collapses** — 391.4 → 311.0 against a
4.3x median improvement. Corpus scaling fixes the typical case and leaves the
tail alone. On `cell8` the mean sits near 1,020 km for every arm because there
almost everything is tail. The mean is close to useless for selection here,
which is why `--select hit` exists.

## The benchmark measures bank coverage as much as model quality

1,000 held-out KartaView photographs through the shipping pipeline and the real
beam rollout, identical settings (beam 2, ranked s0–s2):

| | OSV-5M test | KartaView |
|---|---|---|
| top-1 retrieval sim | ~0.90 | **0.7182** |
| median | 2.7 km | **442.6 km** |
| `<25 km` | 72.8% | **12.8%** |

**Two explanations were tested and both are dead.** Aspect ratio: top-1 is flat
at 0.70–0.73 across portrait, square, 4:3 and 16:9 — including the 16:9 bucket
that matches OSV-5M's uniform 910×512 exactly. Encoder domain shift: killed by
the self-retrieval control, where KartaView scores **0.7103 against the OSV-5M
bank** and only **0.6830 against other KartaView images**. It retrieves better
from OSV-5M than from itself, which is what a 146x larger bank should do.

What remains is corpus density. OSV-5M's 0.90 comes from queries whose own
streets the bank covers densely — the `sequence` split holds out a drive but not
the road. That is legitimate geolocation and it is what the retrieval prior is
for, but it means the headline is partly a statement about coverage.

Every paired comparison here is unaffected, since all arms are measured
identically. The absolute number is optimistic for arbitrary photographs.
`scripts/eval_highres.py` runs this end to end and verifies its own embedding
against stored bank vectors first (projection cosine 1.000000, embedding
0.992–0.996).

## Multi-photograph queries — built, demoed, not yet measured

If the benchmark is easy because the *bank* is dense around the query, then
several photographs of one spot manufacture that density on the **query** side,
which is free. The retrieval prior takes k neighbours as a similarity-weighted
set and cannot tell which image produced them, so **this needed no retraining**:
extra photographs contribute candidates, one image still drives the policy.

`scripts/serve.py` does it live and the page accumulates pasted photographs with
a thumbnail strip and a Start over button. Ten hand-checked groups gave median
820 → 578 → 409 km for 1 → 2 → 4 photographs, with one case going 10,824 → 319
and one getting *worse*, 68 → 820. **n=10 is not evidence.**
`scripts/multiquery.py` is written and unrun: it builds spatial groups from the
harvest and measures the 1/2/4/8 curve properly.

Two design points already learned: a global top-k lets the strongest-matching
photograph fill every slot (four photographs changed the answer in one case of
five), so each photograph gets its own share round-robin by rank; and dedupe, or
one bank image found by two photographs votes twice. If the noisy behaviour
persists, similarity-weighted allocation instead of equal shares is the next
refinement.

## 768-d: costs about a point, worth paying

| rung | 1536-d | 768-d | `<25 km` deficit |
|---|---|---|---|
| e2 | 2.7 km / 72.8% | 3.2 / 70.6% | +3.23 [+2.46, +4.00] |
| e4 | 2.6 / 73.6% | 2.8 / 72.3% | +1.49 [+0.76, +2.22] |
| e6 | 2.5 / 73.8% | 2.7 / 72.8% | **+1.05 [+0.28, +1.82]** |

The gap closes (3.23 → 1.49 → 1.05) and both ladders converge, but unlike
pooling it never reaches parity. 768-d is *better* on the mean at e6 (330.8 vs
356.7) while worse on median and hit rate — it trades the head of the
distribution for the tail.

**Adopt it anyway.** Doubling the corpus is worth ~+3.9 pp against a 1.05 pp
cost, and at 1536-d a 4.90M table is 15.05 GB, which can never clear
`0.4 × free` on a 31.7 GB machine. 768-d is the only configuration in which all
of OSV-5M is resident (7.53 GB).

Retrieval said it was free (−0.17 pp [−1.00, +0.63], while 384 was separated at
−0.87). The mechanism for the disagreement: PCA keeps 97.89% of *variance* and a
cosine is dominated by exactly those high-variance directions, while the agent
additionally learns `StreetProj` and the retrieval keys, which can exploit the
low-variance directions that were discarded. Half the pooled vector is a shared
component — mean norm 46.44 against a row norm of 65.87 — which is why raw
cosines sit at 0.90.

## The pyramid: mean pooling cannot use the third level; attention untested

18,812 global high-resolution images cached as 33 tokens each (3 crops, 6 tiles
at 2x linear, 24 at 4x) in `cache/street/s10/pyr33.f16.npy`.

| paired against `L0` | `<1 km` | `<25 km` |
|---|---|---|
| L0+L1 [level] | −0.43~ | +0.27~ |
| L0+L1+L2 [**token**] | −4.37 | −2.90 |
| L0+L1+L2 [**level**] | −1.40 | −0.97~ |
| L2 alone | −6.73 | −5.13 |

Changing only *how the union is averaged* recovers two thirds of the three-level
penalty — 24 of 33 tokens are L2, so a token mean hands the deepest level 73% of
the vector. But level weighting still does not make L2 pay. **Mean pooling
cannot extract it, and most of its apparent harm is an averaging artifact.**

The attention arm collapsed: −15.75 pp [−17.8, −13.8] alone, −2.15 combined.
**Recorded as confounded, not as a result** — it trained on 14,938 images
against the 96,091 that produced +3.67 pp on OSV-5M, with 33 tokens instead of
9. The harvest below is what disambiguates data volume from architecture.

Levels are not redundant: L0 vs L2 cosine is 0.615. Leak screen on the harvest:
**0 of 2,000** images carry a parseable burned-in coordinate, against 1.45% of
OSV-5M.

## Assets on disk

* **47,646 KartaView images, 84 GB**, `E:/data/kartaview_hr`, median 6.0 MP,
  17,378 sequences, global (US 14,944 / DE 4,115 / CA 2,909 / FR 1,926 / GB
  1,892 / TH 1,554 / AU 1,539 …). The harvest completed on its own. 2.5x the set
  that produced the confounded pyramid null.
* `pool_bal_bank70.f16.npy` 10.75 GB (3.50M × 1536), `pca768_bank70.f16.npy`
  5.38 GB, `pyr33.f16.npy` 1.91 GB. Street cache totals 155 GB.
* 240 GB free on C:, 415 GB on E:. Shards 70–97 unused — 1.4M more images,
  taking the corpus to its 4.90M ceiling.

## Unfinished, resumable

**`scripts/w768b70.py`** — 768-d on the 3.50M bank. `w70_proj` is marked done
(the projection exists); it was stopped during `w70_knn`. Re-running resumes
there: index, then the 2+2+2 ladder, then the eval.
**Prediction recorded in the script**: the width cost should be roughly constant
in bank size, so **~1 pp, about 75.2% and a ~2.0 km median** against
`d1536-b350-e6`. If much worse, the cost grows with corpus and 768-d stops being
the route to 4.90M.

Also queued and unrun: `scripts/multiquery.py` (the multi-photograph curve), and
re-running the pyramid attention arm on 47k images rather than 18k.

## Hardware constraints, measured

* **31.7 GB RAM.** The host-RAM tier needs `0.4 × free` to exceed the table, so
  anything above ~2.9M images at 1536-d falls to the memmap tier. That tier
  costs a measured ~4% (805 s vs 738 s cold), and it is *cheaper* than a paged
  private copy, because a memmapped page is clean and a dirty private page must
  be written to the pagefile before eviction.
* **Large pages work only as the first big allocation after boot.** Confirmed
  in both directions in one evening: 10.75 GB succeeds on a freshly booted idle
  machine; four hours later with three jobs running, 4.22 GB succeeds and
  10.75 GB fails with ERROR_NO_SYSTEM_RESOURCES. `dataset.py` tries and falls
  back cleanly, saying why. `NO_LARGE_PAGES=1` disables it.
* **Training is data-bound, not GPU-bound.** Every training run logs 1% GPU;
  the only 99% readings came from a concurrent encoder pass. Consequently
  stacking jobs costs far more than memory arithmetic suggests — three at once
  made a rung 2.6x slower, where I predicted 30%.
* **C: is NVMe (0.20 ms), E: is a 2 TB HDD.** Shard zips live on E:, which is
  why sequential `slurp()` took `embed_street` from 3 to 83 MB/s.

## Recurring mistakes worth not repeating

* **`OSV_RELEASE` defaults to `s01`.** Three silent wrong answers in one day:
  test fixtures in the wrong cache, `after.py` skipping the entire corpus block
  while logging that block 3 had not finished, and the harvester seeding from a
  50k pool while reporting a plausible number. Each failed by quietly doing
  less. The scripts that hit it are pinned individually; the real fix is for
  `config.py` to have no default.
* **`ThreadPoolExecutor.map` buffers everything.** It OOM'd `pyramid_cache.py`
  after `tile_cache.py` already carried a comment warning about it. Use the
  bounded sliding window.
* **Unquoted heredocs let backticks expand**, mangling several patches
  including one that silently no-matched and left a memory file inconsistent
  with its own description.
* **Mechanical substitutions applied too broadly** — the arm rename rewrote
  `NAMING.md`'s own counter-example so the doc argued against itself.
* **Markers record that a stage ran, not that its inputs are unchanged.**
  `w768_eval` was skipped as done when it had only ever evaluated one rung.

## The habit that keeps earning its keep

**Verify the reproduction before reading the number.** The high-resolution
evaluation looked like catastrophic domain shift; checking the pipeline against
stored vectors first is what made it safe to trust the result, and the two
controls that followed killed both of my explanations for it. Two predictions
were falsified this way in one day — the undertraining story for `fh_pos1`, and
aspect ratio for KartaView — and both were cheaper to test than to argue about.
