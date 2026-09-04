# Live state — rewritten 2026-09-03, before a compaction

Read this first. `docs/REVIEW.md` and `docs/REVIEW2.md` hold the two Codex
reviews with every item marked; `docs/BACKLOG.md` the code and performance
work; `docs/ARCHITECTURE_NEXT.md` the architecture directions. Durable findings
are in the memory directory. Branch `retrieval-dropout`, PR #13, 50 commits
this session, all pushed. **133 tests pass. Nothing is running. Tile server is
up.**

---

## 1. What to ship

**`d768-b350-e6-drop70`** — 768-d street vector, 3.40M bank, 6 epochs,
`--retr-drop 0.7`.

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

All KartaView numbers are at shipping parity (each checkpoint's own bank, its
own `retr_k`). **Best benchmark and best real-world are different arms**, and
the gap widened at parity: `d1536-b350-e6-drop30` has the best benchmark of any
arm and loses externally to the half-width `drop70` by −1.06 pp and −56.3 km,
both separated. Width and dropout do not compose — `d1536-b350-e6-drop70` loses
on both measures.

---

## 2. The three results that matter

**The model is overwhelmingly a retrieval system.** With the prior removed at
inference (`bootstrap.py --retr-off`):

| arm | retrieval on | off | loses |
|---|---|---|---|
| `d768-b350-e6` | 75.6% | **2.7%** | −72.9 pp |
| `drop70` | 75.2% | **10.7%** | −64.5 pp |
| `drop90` | 69.9% | **19.3%** | −50.6 pp |

Monotone in p, every contrast separated. Dropout buys standalone capability.
The 75.6% headline is mostly corpus coverage. This was **predicted** from
`cond_probe` (the prior's share of the logit spread, 42% → 28%) and agrees with
`diag_beam` (headroom up, top-1 down) — three independent routes, one mechanism.

**`--retr-drop 0.7`, and the reason is not the one first recorded.** The curve
at parity: 11.6 / 11.1 / 12.1 / 12.6 / **13.4** / 14.0% for p = 0 … 0.9.
0.7 vs 0.9 on `<25 km` is +0.58 pp [−0.24, +1.38], **inside noise** — 0.9 does
not overshoot the hit rate, and is separated *better* on `<200 km` and the
median. What 0.9 costs is `<1 km` (separated worse than both 0.7 and no
dropout) plus a −5.7 pp benchmark regression. p=0.3 is now inside noise, so
**p ≥ 0.5 before anything is separable** — the original 0.3 guess is not
supported.

**Step 0 is the entire tail.** `error_profile` on the shipping arm: 7.9% of
images first go wrong at step 0 and carry **84.3% of the mean error**; 2.5% of
images carry 66.9%. Meanwhile 33.6% first go wrong at step 3 and contribute
0.0 km. The median and the tail are different problems.

---

## 3. Corrections to already-published claims

**The corpus axis.** Reported as −1.32 pp [−2.04, −0.60] separated on
photographs. At parity it is **−0.46 pp [−1.12, +0.18], inside noise**. The
median effect survives (+30.8 km, separated) and `<1 km` slightly favours the
bigger bank. The 2×2 blaming the model rather than the bank **survives**: bank
effect noise in both cells, model effect separated on the median at both banks.
And **with dropout the corpus transfers**: b265 → drop70 is +1.34 pp, +3.58 pp
at 200 km, −39.2 km, all separated.

**Why the caveat was wrong.** It said every arm was measured the same way so
paired directions would hold. Parity cost p=0 nothing and each dropout arm
~0.6 pp — the error was **correlated with the treatment**, because neighbour
count is what `--retr-drop` manipulates. The rule that replaces it: *ask whether
the error is independent of the treatment first.*

**The pyramid fusion head: retracted, then re-established.** Reported negative
from a confounded comparison. Retracted after the review, then survived three
attempts to break it. The head at **initialisation** scores 33.1% against the
1536-d mean's 33.7%, so the random projection is worth 0.63 pp of the 11.30.
Training is what destroys it:

| epochs / objective | train loss | `<25 km` |
|---|---|---|
| **1 epoch** | — | **32.2%** |
| 12, 5 km positives | 0.063 | 20.5% |
| 12, 0.5 km positives | 0.0145 | 12.9% |

Loss and retrieval move in opposite directions — overfitting, 2.5M parameters
on 38,009 images. Three explanations were tested and none survives: the random
projection (0.63 pp), the false negatives (50.3% of off-diagonal cells were
true positives; fixing them moved 20.9 → 20.5%), and the positive radius (my
hypothesis, wrong in the opposite direction). **The multi-level mean stands**:
L0+L1+L2 beats L0 alone by +1.6 pp and 40 km at equal width, with no learned
head. *Caveat:* REVIEW2 new#6 found the OneCycle schedule was truncated in
these runs, so the individual numbers were produced under a shortened schedule;
the epoch curve is robust to that, the point estimates less so.

**KartaView/OSV leakage, audited.** 0 image-id and 0 sequence-id overlaps.
Median nearest-OSV distance 231 m; 1.06% of evaluated queries within 10 m.
Dropping those 53 moves the dropout result from +1.80 to +1.80 pp and the
corpus result from +1.34 to +1.33. **Immaterial to every paired contrast.**

---

## 4. Reviews

`docs/REVIEW.md` — 86 items: **40 fixed, 8 refuted, 7 deferred, 30 open, 1
forced a retraction.**
`docs/REVIEW2.md` — 22 items: **14 fixed, 8 open.**

Round two found that **three of round one's forty were wrong**, two of them
defects I introduced while fixing something else, and one fixed in the wrong
file. Chief among them: `split_hash` hashed only each label's first character,
so train and test were indistinguishable and the bootstrap row-parity guarantee
was hollow. Now fixed, with the legacy digest kept so 122 checkpoints load.

**Still open, REVIEW.md:** 5, 6, 7, 9, 10, 14, 18, 20, 23, 24, 39, 40, 43, 44,
46, 47, 48, 49, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 72, 73.
**Still open, REVIEW2.md:** reopened 8, 10, 11; new 1, 2, 7, 8, 9.

Two of those thirty are really two structural problems: **#40 with 43, 44, 46,
47, 62, 63, 66** is one issue — every artifact addressed by row position with
nothing proving two describe the same rows (the provenance-manifest item; one
design change retires eight). **#70–73** is the runner's marker and log design.

---

## 5. Needs your decision — do not fix these silently

Seven items change what a trained arm *is*, so fixing any makes future arms
incomparable with all 122 checkpoints on file. Each needs a re-baseline.

* **#29 weight decay** — the rule is `"pos" in name`, which exempts exactly
  `retr.q_pos.weight` and `retr.k_pos.weight`, 196,608 parameters, 3.7% of the
  model. Those are the **learned retrieval keys**, worth +4.1 pp on record. So
  the branch this whole session regularises has been training with no decay,
  unintentionally. **Most likely of the seven to have moved a result.**
* **#28 `quality()`** sees the unmasked top-1 similarity even when
  `--retr-drop` hid that neighbour.
* **#27 `--save-opt`** writes optimizer state nothing loads; scheduler and RNG
  are not saved, so a 2+2+2 ladder is three optimizer restarts.
* **#2 `FuseHead.baseline`** omits a per-encoder renormalisation the printed
  baseline applies. The head is a residual on exactly that vector, so changing
  it invalidates today's fusion runs and the 33.1% init score.
* **#31 sink coefficient** unconstrained (dormant — all checkpoints `sink_k=1`).
* **#32 memory dropout** lacks inverted-dropout scaling.
* **#39 RoPE** gives no relative position. Measured: with identical content at
  all 256 positions, the attention logit's within-offset spread is 0.845 of its
  overall spread as built, against 0.000 for the textbook arrangement. Live for
  108 of 122 checkpoints including the shipping arm.

---

## 6. Best next experiment

**An auxiliary coarse street-only head.** Two independent measurements point at
it: step 0 carries 84.3% of the mean error, and with retrieval off the visual
branch scores 2.7%. It is Tier 1 in `ARCHITECTURE_NEXT.md`, needs tiles and a
training run.

Also promoted there: **capped or mixture retrieval combination**, because
`cond_probe` showed the prior's contribution outweighs the top-1 margin in 85%
of rows — it is deciding the answer often enough that how it is combined
matters. That measurement reversed three conclusions I had drawn from the
static gates alone, which the same document had flagged as a floor rather than
the whole story.

---

## 7. Tools written this session

`scripts/parity_report.py` — the whole external table from the exports.
`scripts/pair_npz.py` — paired bootstrap over two exported error arrays.
`scripts/rope_probe.py` — is rotary giving relative position (no).
`scripts/cond_probe.py` — the retrieval prior's real influence on a logit.
`scripts/fuse_init_probe.py` — the fusion head scored at initialisation.
`bootstrap.py --retr-off` — corpus dependency measured directly.
`eval_highres.py --match-bank` — reproduce the bank's embedding pipeline.
`src/safeio.py` — atomic checkpoint and report writes.

Tests: `test_checkpoint_contract.py`, `test_split_hash_and_cache.py`,
`test_topk_bounds.py`, `test_import_order.py`, `test_safeio.py`,
`test_cache_completeness.py`.

---

## 8. Hardware and environment

* RTX 3070, 8 GB. Training is **GPU-bound**, 92–96% at 165–176 W. The fusion
  head runs 97% at 236 W with 6.9/8 GB resident. **Never stack GPU jobs** —
  measured 2.6× cost.
* 31.7 GB RAM. Casting a bank to float32 whole is a 10.7 GB allocation; chunk.
* `python` on PATH is MSYS2's and lacks numpy. Always
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe`.
* Tile server `192.168.50.1:3000`. Needed by all training (beam rollout) and
  all evaluation. Not needed by pyramid/fusion work or any probe above.
* `OSV_RELEASE` is now **required** — no default. The runner sets it.

---

## 9. Rules learned the hard way

* **A fix is not done until it is tested against the failure it prevents.**
  Round two found three wrong fixes and every one had never been run against
  its own failure case: the split hash never fed a swapped assignment, the
  cache resolver never given a retrained checkpoint, the top-k never run at
  `K == bank size`. Two of those I introduced while fixing something else.
* **Ask whether a measurement error is independent of the treatment**, not
  merely whether it was applied uniformly.
* **A guard that is defined and never called is not a guard.** Happened twice
  (#65, #17). Nor is one that fails open (#42).
* **Never type a backslash inside a Bash heredoc** — it arrives as a real
  newline. Hit six times today despite being written down. Build it as
  `chr(92)` or write the patch script to a file with the Write tool.
* **Assert before writing** in patch scripts. Two anchors missed today and
  neither corrupted a file. But re-running a patch script double-applies —
  check for duplication after.
* **Scope every log check to the newest `==== attempt`.** An append-only log
  matched a `FAIL` from the previous day and killed `fuse-attn-pyr47` at epoch
  11 of 12.
* **Do not block on `WaitForExit` in the PowerShell tool**, and expect the Bash
  tool's background jobs to be reaped. Long runs survive as detached
  `Start-Process` with output redirected to a file, polled afterwards.
* **`git checkout <file>` discards uncommitted work.** Lost the `overnight.py`
  edits that way; commit before sabotage-testing.
* The **2,000-image selection set is not evidence**; it picks checkpoints.
* **A wide interval means get more data, not weaken the claim.**
