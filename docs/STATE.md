# Live state — rewritten 2026-09-02 02:00, before a context compaction

Read this first after a compaction. `docs/NEXT.md` is the forward plan;
`docs/NAMING.md` explains arm names. Durable findings also live in the memory
directory and in the git log, which is unusually detailed —
`git log --oneline -40` reconstructs most of the reasoning.

**Nothing is running.** The demo server, all training, and the harvest are
stopped. `git status` is clean apart from two probe `.npz` files in `runs/`.

## What ships

**`d1536-b350-e6-drop30`** for quality; **`d768-b350-e6-drop30`** if the corpus
is to grow past 3.40M. The two are indistinguishable on held-out photographs;
the 1536-d arm wins the benchmark, the 768-d arm is the only one whose bank can
reach 4.90M on a 31.7 GB machine.

| arm | OSV-5M `<25 km` | KartaView `<25 km` (n=5,000) | KartaView median |
|---|---|---|---|
| `d768-b265-e6` | 72.8% | 12.9% | 450.6 km |
| `d768-b350-e6` | 75.6% | 11.6% | 486.2 km |
| `d768-b350-e6-drop30` | ~75.6% | 12.7% | 475.3 km |
| `d1536-b350-e6` | 76.2% | 11.7% | 518.2 km |
| **`d1536-b350-e6-drop30`** | **~76.9%** | **12.5%** | **458.1 km** |

`d1536-b350-e6-drop30` beats `d1536-b350-e6` by +0.10 to +1.30 pp on the
benchmark and by +0.82 pp [+0.26, +1.38] and −60.1 km [−76.0, −38.8] externally,
all separated. Against `d768-b350-e6-drop30` it is separated-better on the
benchmark and inside noise on every external metric.

**Neighbour dropout is a plain regulariser, not a trade.** Across both widths it
is separated-better in five of six benchmark/external cells and **never
measurably worse anywhere**. The retrieval prior was under-regularised; the
transfer gap was the symptom that made it visible.

**The selection set is not evidence.** Its 2,000 images said dropout was 0.6 pp
*behind* on `d1536` at e4; the 5,000-image bootstrap says it is ahead,
separated. It picks checkpoints, it does not measure them.

## The corpus curve, four points

| bank | median km | mean | `<25 km` | step |
|---|---|---|---|---|
| 1.15M `d1536-b115-e6` | 7.7 | 391.4 | 64.1% | — |
| 1.90M `d1536-b190-e6` | 3.6 | 367.5 | 70.6% | +6.50 [+5.50, +7.64] |
| 2.65M `d1536-b265-e6` | 2.5 | 356.7 | 73.8% | +3.21 [+2.40, +4.02] |
| **3.40M `d1536-b350-e6`** | **1.8** | **311.0** | **76.2%** | **+2.34 [+1.56, +3.12]** |

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

## The benchmark and the world disagree in sign

Measured 2026-09-02 on 5,000 held-out KartaView photographs, paired, each model
against its own bank:

| step | OSV-5M `<25 km` | KartaView `<25 km` |
|---|---|---|
| `d768-b265-e6` → `d768-b350-e6` | **+2.8 pp [+1.96, +3.66]** | **−1.32 pp [−2.04, −0.60]** |

Both separated. **The corpus step that is this project's main axis buys
benchmark points and loses real ones.** `<1 km` is unchanged (+0.08 pp): the
damage is at 25–200 km, exactly where the retrieval prior operates.

**A 2×2 blames the model, not the bank.** Each model run against both banks
(`eval_highres --bank` overrides the checkpoint's own):

| `<25 km` | bank 2.65M | bank 3.50M |
|---|---|---|
| model b265 | 12.92% | 12.80% |
| model b350 | 12.16% | 11.60% |

* bank effect at fixed model: −0.12 pp [−0.66, +0.40] and −0.56 pp [−1.04, −0.08]
* model effect at fixed bank: **−0.76 pp [−1.44, −0.06]** and **−1.20 pp [−1.86, −0.54]**

The model trained against denser retrieval is worse off-domain *whichever bank
it is given at inference*. It is a training-time dependency on dense coverage,
not a property of the bank. Off-domain top-1 similarity is 0.71 against 0.90 on
the benchmark, and a policy that only ever saw 0.90 over-trusts the prior.

**`--retr-drop` fixes it, for free.** Hiding each retrieved neighbour with
probability 0.3 during training recovers the external loss
(**+1.10 pp [+0.44, +1.78]** over `d768-b350-e6`, back to level with
`d768-b265-e6` at −0.22 pp inside noise) while costing nothing on the benchmark
([−0.98, +0.30] pp). The prediction that it would trade benchmark for
generalisation was **falsified** — the selection curve suggests the earlier arms
were simply under-regularised: no-drop drifts 74.4% → 74.2% from e4 to e6 while
drop30 holds 75.0%. The median against `b265` is still worse
(+24.6 km [+3.5, +44.9]), so the recovery is in hit rate, not everywhere.
p was not tuned; the p-curve is open.

**Two measurement lessons, both of which nearly produced wrong answers.**
At n=1,000 the corpus contrast reads −1.40 pp [−3.00, +0.20] — inside noise. I
briefly reported it as "no transfer", and only n=5,000 separated it. A wide
interval is a reason to get more data, not to weaken the claim. And
`--n 1000 --seed 0` was *not* a fixed sample: it permuted the manifest, which
grew 18,812 → 47,646 overnight, so two runs with identical flags scored
different images. Selection is now `crc32(seed + image_id)`, stable under
growth. Numbers from before that fix (the 442.6 km figure) are not comparable.

## Multi-photograph queries — measured, real, small

If the benchmark is easy because the *bank* is dense around the query, several
photographs of one spot manufacture that density on the **query** side, which is
free. The retrieval prior takes k neighbours as a similarity-weighted set and
cannot tell which image produced them, so **this needed no retraining**.

2,496 spatial groups within 100 m, paired on the same anchors:

| photos | median km | `<25 km` | hit rate vs 1 | median vs 1 |
|---|---|---|---|---|
| 1 | 557.0 | 10.7% | — | — |
| 2 | 513.6 | 11.9% | **+1.2 pp [+0.12, +2.32]** | **−43 km [−74, −14]** |
| 4 | 498.1 | 12.0% | +1.3 pp [+0.12, +2.56] | −59 km [−94, −27] |

**It saturates at two.** The second photograph buys +1.2 pp; the third and
fourth add +0.1 pp between them. Ask for one more angle, not eight. Worth about
half a corpus doubling, for free.

**Three things it needs to be visible at all**, each of which produced a wrong
answer first:

* **Groups chosen before embedding, and full.** Sampling images first and
  grouping second gave 200 groups of which *one* had four members, so
  `take[:N]` returned the same two photographs for N=2, 4 and 8 and the curve
  read as saturating at two when there was never a third to add.
* **Round-robin the merge, and dedupe.** Ranking all N×32 candidates together
  lets the strongest-matching photograph fill every slot. On the same groups the
  naive merge reads +0.5/+0.8/+1.3 and round-robin reads +2.1/+2.3/+1.8.
  `serve.py` already did this; the experiment did not.
* **~2,500 groups.** At 390 every step is inside noise and the `near` control
  came out non-monotone — 8.0 → 7.5 → 6.4 → 9.5 — which is what noise looks
  like when only one ordering is run and it happens to rise.

Never compare absolute numbers across group populations: requiring 8
photographs within 100 m selects a harder set of places (1221 km median) than
requiring 4 (557 km). Per-group errors are in `runs/multiq_pow4.npz`.

## 768-d: the width cost vanishes as the bank grows

**Updated 2026-09-02: at the bank size that matters the cost is gone.** On the
3.40M bank, `d768-b350-e6` is **1.9 km / 320.4 mean / 75.6%** against
`d1536-b350-e6` at 1.8 / 311.0 / 76.2% — a gap of **+0.6 pp, CI
[−0.18, +1.32], inside noise**. At 2.65M the same contrast was +1.05 pp
[+0.28, +1.82] and separated.

| bank | 1536-d | 768-d | deficit |
|---|---|---|---|
| 2.65M | 2.5 km / 73.8% | 2.7 / 72.8% | +1.05 [+0.28, +1.82] separated |
| **3.40M** | 1.8 / 76.2% | **1.9 / 75.6%** | **+0.6 [−0.18, +1.32]** inside noise |

**The width cost shrinks as the corpus grows**, which was the branch written
down before the run ("if much better, the projection is cheaper on a denser
bank"). Treat +1.05 pp as a decaying upper bound, not a fixed toll. The ladder
converges as every other one does — e4 (75.5%) vs e6 (75.6%) is inside noise.

768-d is still worse on the *mean* (320.4 vs 311.0) while level on median and
hit rate: it trades the head of the distribution for the tail.

**So 768-d is the route to 4.90M.** At 1536-d that table is 15.05 GB and can
never clear `0.4 × free` on a 31.7 GB machine; at 768-d it is 7.53 GB and
resident. The only argument against the projection was its accuracy cost, and
at 3.40M that cost is not measurable.

**The prediction was recorded before the run and held:** ~75.2% and ~2.0 km
against an observed 75.6% and 1.9 km.

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

**`pyrcache-hr47k`** — the pyramid cache over the full 47,646-image harvest,
running since 07:24 with a 900-minute window for a ~620-minute job (2.6 img/s
per encoder, two encoders). Needs no tile server. Resumable through its own
`pyr47_done.u8.npy` mask, so killing it costs one image. `fuse-attn-pyr47`
follows and is guarded by `Stage(needs=...)`, so it skips rather than failing
three times in three seconds when the cache is absent.

Open questions in rough order of value:

* **The `--retr-drop` p-curve.** 0.3 was a guess that happened to work. 0.1 and
  0.5 would say whether the effect is a plateau or a peak.
* **Does `d1536` carry the same dependency?** It holds the best benchmark
  number and has never been measured externally. If it behaves like `d768`,
  the shipping recommendation changes again.
* **A second external domain.** Every transfer claim here rests on one test set
  of KartaView dashcam frames. "Does not transfer to KartaView" is not "does not
  transfer".
* **Seeds.** Every arm here is a single seed, and seeds have differed by 18 km
  on the median in this project.
* Shards 70–97 remain unused — 1.4M more images to the 4.90M ceiling. Worth
  doing *with* `--retr-drop`, given what the corpus step did without it.

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
* **Training is GPU-bound.** Corrected 2026-09-02: measured directly at
  **94-96% utilisation and 172 W** with nothing else on the card. The earlier
  "every run logs 1% GPU" reading is from before the reboot and cannot be
  reconciled with the same ~27 min per rung; two other measurements agree with
  the new one and not the old, which is what settles it. The memmap tier costs
  only ~4%, which a genuinely data-bound run could not manage, and stacking
  three jobs cost 2.6x, which a data-bound run would not — they would interleave
  their I/O rather than queue for one saturated device. Treat 1% as a broken
  sensor, not as evidence. The practical consequence: **never stack GPU jobs**,
  and faster iteration comes from less work per step, not more I/O throughput.
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
