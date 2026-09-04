# Live state — updated 2026-09-04 13:35

Read this first. `docs/REVIEW.md`, `REVIEW2.md` and `REVIEW3.md` hold three
Codex reviews; `docs/BACKLOG.md` the code and performance work;
`docs/ARCHITECTURE_NEXT.md` the architecture directions. Durable findings are in
the memory directory. Branch `retrieval-dropout`, PR #13, all pushed.
**355 tests pass. REVIEW3 is closed (§4). The clean-bank result from
`seqfix2.py` is VOID and re-running as `seqfix3.py` — read §7 before citing
anything about a clean-trained arm. Tile server up.**

---

## 1. The headline finding: the retrieval bank was leaking

`build_knn` offset the bank extension's sequence ids into a separate namespace,
making the release and the extension disjoint **by construction**. They are not
— the extension is the same OSV-5M shards with the same sequence naming, and
**75.5% of test sequences also appear in `bank_ext70`**. Same-sequence exclusion
therefore never crossed the corpus boundary, and the bank served near-duplicate
frames of the query's own drive as retrieval neighbours.

* **41.6% of test queries had a same-drive top-1**, at a median **0.31 km**
* legitimate top-1 sat at a median 28.4 km

Fixed in `build_knn` (factorise both corpora together). **All three caches
rebuilt and verified clean** — 0 same-sequence neighbours across all 32 ranks:

| cache | pairs excluded | per query | top-1 cosine |
|---|---|---|---|
| `pca768_bank70` | 3,722,574 | 7.4 | 0.8261 → 0.7987 |
| `pool_bal_bank70` | 3,722,574 | 7.4 | 0.9080 → 0.8948 |
| `pca768_bank55` | 2,988,188 | 6.0 | (no comparable prior) |

The two bank70 counts being identical is the check that matters: same corpus,
same sequences, so the exclusion must not depend on embedding width.

Old caches kept as `knn_*.LEAKY-preseqfix.npz`, so every published number stays
reproducible. `scripts/seqleak.py` measures; `--verify` asserts a rebuild is
clean and exits non-zero if not.

---

## 2. The corrected benchmark — and a claim of mine to retract

**~18 pp of the OSV-5M headline was the leak.** 5,000 paired test images:

| arm | `<25 km` | median | (was) |
|---|---|---|---|
| `d1536-b350-e6-drop30` | **59.1%** | 13.4 km | 76.9% |
| `d1536-b350-e6-drop70` | 58.3% | 14.6 km | 75.9% |
| `d1536-b350-e6` | 58.0% | 14.1 km | 76.2% |
| `d768-b350-e6-drop30` | 57.8% | 14.4 km | — |
| **`d768-b350-e6-drop70`** (shipping) | 57.5% | 15.4 km | 75.2% |
| `d768-b350-e6` | 57.3% | 14.7 km | 75.6% |
| `d768-b265-e6` | 55.0% | 17.8 km | 72.8% |
| `d768-b350-e6-drop90` | 52.7% | 21.1 km | 69.9% |

**RETRACTION.** I wrote that "the OSV-5M benchmark no longer separates arms."
That was measured on four arms which are near-identical by construction
(same bank, same width, similar dropout) and it does not generalise. The full
eight-arm table spans **52.7% to 59.1%**, and the axes still separate:

* **corpus still pays** — b265 55.0% vs b350 57.3%, +2.3 pp. This survives the
  fix, which I had flagged as needing re-measurement. It does.
* **width still pays** — d1536 58.0% vs d768 57.3%
* **drop90 is clearly bad** — 52.7%, worst of the eight

What is true: among arms differing only slightly, the benchmark cannot separate
them, and it never could. Best on OSV-5M is `d1536-b350-e6-drop30`; best
externally is `d768-b350-e6-drop70`. That split is unchanged by the fix.

**Caveat, both directions.** These arms were *trained* against the leaky cache.
57.5% is a **lower bound** on a cleanly-trained arm — one that learned to lean
on near-duplicates may be worse at using honest neighbours. It is a **sound**
bound on how much of the published number was the leak. §7 is measuring the
difference right now.

**KartaView is unaffected** (no same-drive frames in the bank) and is the
selection benchmark: drop70 13.4% / 435.7 km, drop90 14.0% / 390.0 km,
d1536-drop30 12.3% / 492.0 km, d1536 11.2% / 532.5 km.

---

## 3. #29 weight decay: answered, no effect

The rule `"pos" in name` exempted `retr.q_pos.weight` and `retr.k_pos.weight`
— `nn.Linear` projections, the **learned retrieval keys**, 196,608 params,
3.7% of the model — from weight decay. Now classified by module type.

Two full ladders at `--seed 0`, one flag apart. **Every contrast inside noise**:
OSV-5M `<25 km` [−1.08, +0.10] pp; KartaView `<1 km` −0.18 pp, `<25 km`
+0.16 pp, `<200 km` −0.30 pp, median +4.1 km. No re-baseline needed.

Two things the paired design caught:
* the 2,000-image selection set pointed the **wrong way, monotonically**
  (+0.15 / +0.30 / +0.55 pp) — it picks checkpoints, it does not measure them
* **the seed is the larger effect** — both wd29 arms are separated worse than
  the shipping arm while differing from it only in the seed

`--wd-legacy` restores the old rule. Also corrected: STATE.md had said the
substring caught "exactly two" tensors; it caught **three** — the third is
`map.pos.weight`, a real `nn.Embedding` that must stay exempt, so deleting the
clause would have been a second bug.

---

## 4. Reviews

`REVIEW.md` 86 items: **68 fixed, 8 refuted, 7 deferred, 2 open** (24, 39 —
both need a re-baseline decision, §5).
`REVIEW2.md` 22 items: **all 22 fixed.**
`REVIEW3.md` 10 items: **all 10 fixed**, commit `4fbe408`. Tests 321 → 355.

### REVIEW3 — closed 2026-09-04, two of them defects in my own work that day

Each was reproduced against the old implementation before being fixed, so the
new tests are known to fail without the fix. The measured reproductions:

| # | the bug, measured | |
|---|---|---|
| 7 | `file_stamp` size discarded — 1 byte and 999,999,999 bytes stamp identically at equal mtime | confirmed |
| 1 | old rule on a 750k extension-only cache | `ACCEPTED UNCHECKED` |
| 9 | old mask rule calls a `2` fetched | `True` |
| 8 | old `marker_matches("{not json}")` → `None`, which the caller reads as satisfied | confirmed |
| 5 | old identity unchanged by a rebuilt bank | `True` |
| 3 | old PCA fit sample: **39,903 of 200,000 rows (20.0%) were val or test** | confirmed |

**#7 and #2 were mine, from earlier the same day.** #7 weakened the error-cache
key I added that morning to fix a *different* staleness bug. #2 contradicts a
claim I wrote into `REVIEW2.md` **and** into a source comment at `train.py:448`;
the *train* set took `neg_random=True` and drew off-path tiles from OS entropy,
so every `--neg 4` run was unseeded. The wd29 conclusion survives — both ladders
were equally unseeded, so it adds noise to a result already inside noise — but
the recorded seed overstated what it covered. Checkpoints now record
`neg_random`, and `bootstrap.py` prints it whenever arms disagree.

**The pattern, which is the useful part:** every one of the ten sits at a
*seam*. One round fixed a producer, the next fixed a consumer, and the hole was
in whichever of the two nobody looked at that day — the completion mask is
validated on write and read back as "nonzero means done"; row provenance is
checked on the one cache length that happens to be the release's.

**What the fixes deliberately do not do:**

* The `pca768_*_pca.npz` bases on disk are **still the transductive ones** (#3).
  Refitting changes the coordinate system, so every 768-d arm would need
  re-measuring against a rebuilt bank before its number could be compared to
  the others. Script fixed; artifacts not rebuilt. **This is a re-baseline
  decision — see §5.**
* Marker input stamps cover the entry script, not its imports (#5). Stamping
  all of `src/` would invalidate every marker on any edit, which in a research
  runner means re-running finished work several times a day.
* `runs/FINAL.md` was regenerated at 00:18, **before** the leak was found, so it
  carries the inflated headline. Marked stale in place; needs regenerating once
  every arm is re-measured.

---

## 5. Needs your decision — do not fix silently

Seven items change what a trained arm *is*. #29 is closed (§3); #24 joined.

* **#24 multi-photo scoring** — one query embedding per group, so candidates a
  *secondary* photo retrieved are scored against a photo that did not retrieve
  them. Fixing changes the prior's interface and invalidates "+1.2 pp for a
  second angle". `multiquery` prints the caveat at N>1.
* **#39 RoPE** gives no relative position (measured: within-offset spread
  0.845 of overall, vs 0.000 for the textbook arrangement). Live for 108 of 122
  checkpoints including the shipping arm.
* **#28** `quality()` sees the unmasked top-1 even when `--retr-drop` hid it.
* **#27** `--save-opt` writes optimizer state nothing loads.
* **#2** `FuseHead.baseline` omits a per-encoder renormalisation.
* **#31** sink coefficient unconstrained (dormant, all `sink_k=1`).
* **#32** memory dropout lacks inverted-dropout scaling.

Added by REVIEW3 (§4):

* **REVIEW3 #3 — the PCA bases on disk are transductive.** 20.0% of the
  200,000-row fit sample was val/test (39,903 rows, measured). The *script* now
  fits on the training split; the `pca768_*_pca.npz` files are unchanged.
  Refitting changes the coordinate system, so it is not a drop-in: the bank
  must be re-projected and **every 768-d arm re-measured** before its number is
  comparable to anything on record. That is the entire d768 family, which is
  most of the recent work. PCA is unsupervised so the effect is likely small —
  but "likely small" is a claim, not a measurement, and the cheap version of
  the measurement does not exist. Your call whether to spend the rebuild.

---

## 6. Corrections I made to my own claims today

Recorded because the pattern matters more than any one of them.

1. **"the benchmark no longer separates arms"** — retracted, §2. Generalised
   from four near-identical arms.
2. **top-1 cosine "0.9080 → 0.7987"** — compared the 768-d bank's new value to
   the **1536-d** bank's old one. Real drop is 0.8261 → 0.7987 (−0.027). The
   18 pp result is unaffected; it was measured, not inferred.
3. **"#29 exempts exactly two tensors"** — it was three (§3).
4. **"training passes `neg_random=False`"** — that is val, not train (§4 #2).
5. **the error cache did not depend on the retrieval cache** — the first
   re-measurement returned in 6 s with pre-rebuild numbers to 4 s.f. Third
   instance today of one shape: a cache key that names an identifier rather
   than everything the value depends on.
6. **`eval_highres` printed "a gap here is domain shift, not overfitting"** —
   half wrong; much of the gap was the OSV-5M side inflated.
7. **`bootstrap` labelled every pre-flag checkpoint "by module type"** — a
   missing field read as falsy, so the table added to prevent misreading
   mislabelled all 122 arms.
8. **`file_stamp` "includes size and mtime"** — it never included the size.
   The `[-12:]` slice keeps only the low bits of mtime, so the field I relied
   on that morning to fix a staleness bug was doing half of what its docstring
   said. Same day, same file, one function apart.

Six of the eight are one shape: **a check that names an identifier rather than
everything the value depends on** — a tag, a filename, a length, a prefix, a
substring, a size that gets sliced off. When something here is wrong, that is
the first thing to test for.

---

## 7. The clean-bank experiment — first attempt VOID, re-running

`scripts/seqfix2.py` (11:53) finished stages 1–3 correctly and they stand:

1. ✅ rebuild `pool_bal_bank70` kNN + verify — 0 same-sequence over 32 ranks
2. ✅ rebuild `pca768_bank55` kNN + verify — 0 over 32 ranks
3. ✅ `seqfix2-boot-all` → `runs/BOOTSTRAP_seqfix_all.md` (the §2 eight-arm table)

Stages 4–6 produced `clean-drop70-e2/e4/e6` and
`runs/BOOTSTRAP_cleantrain.md`, which reads −2.5 to −4.8 pp against both
reference arms, separated. **Do not cite it.**

### Why it is void — my own edit, mid-ladder

| rung | written | `neg_random` | trained with |
|---|---|---|---|
| `clean-drop70-e2` | 12:33:37 | `None` | old code, OS-entropy negatives |
| `clean-drop70-e4` | 12:47:25 | `False` | **new code, seeded negatives** |
| `clean-drop70-e6` | 13:28:15 | `False` | new code |

I committed REVIEW3 #2 (seed the sink negatives) to `train.py` at **12:32:53**.
`train-clean-e4` launched at **12:33:38**. Each stage is a fresh subprocess, so
the edit landed on rungs 2 and 3 and not on rung 1.

This is not cosmetic. Seeded negatives are a pure function of
(split, seed, index), so an image sees the **same four off-path tiles every
epoch** instead of fresh ones — the tradeoff named in `GeoStepDataset.neg_rng`.
Over e4 and e6 the sink class saw 1.6M distinct negatives repeated four times
where every reference arm saw 6.4M distinct. That plausibly explains both the
ladder's decline (54.65% at e2/e4 → 54.10% at e6) and the deficit. So
`clean-drop70-e6` differs from `wd29-fix-e6` in **two** ways, and the question
needed one.

### The re-run — `scripts/seqfix3.py`

Passes `--neg-random`, which is what every arm on record used, making the bank
the only difference again. Trains `clean2-drop70-e2/e4/e6`, then bootstraps
**all three rungs** against `wd29-fix-e6` and `d768-b350-e6-drop70`, then
KartaView on e6.

Bootstrapping the whole ladder is deliberate: the first ladder peaked at e2 and
declined by e6, so reporting only e6 would understate a clean-trained arm even
with the confound gone — and if that shape survives, it is itself the result,
since every reference arm is an e6.

    OSV_RELEASE=s10 py scripts/seqfix3.py 4

~81 min of training + ~45 min of evaluation. Markers make it resumable.
**Wait for `seqfix2-hr-clean` to finish before starting it** — never stack GPU
jobs (§10). `clean-drop70-*` are kept, not deleted; they are a real measurement
of a differently-trained arm, just not an answer to this question.

---

## 8. What to do next

1. **Run `scripts/seqfix3.py` and read `runs/BOOTSTRAP_cleantrain2.md`** (§7),
   once `seqfix2-hr-clean` has released the GPU. If `clean2-drop70-*` beats the
   references, training against a leaking bank was itself harmful and every arm
   on record is understated; if it matches, **57.5% is the honest number**;
   if it loses *with the negative draw held fixed*, a model that learned to
   lean on near-duplicates really is worse at using honest neighbours, and the
   whole retrieval prior needs re-tuning against the clean bank.
2. **Regenerate `runs/FINAL.md`** — it still carries the pre-leak headline and
   is marked stale in place. Needs every arm re-measured first, not just the
   eight in §2.
3. **#3 (transductive PCA)** is the one most likely to move a published number
   after the leak: every 768-d arm uses a basis fitted partly on val/test. This
   is a decision, not a task — see §5.
4. The auxiliary coarse street-only head (Tier 1 in ARCHITECTURE_NEXT) — step 0
   carries 84.3% of the mean error and the visual branch alone scores 2.7%.

REVIEW3 is done (§4) and is no longer on this list.

---

## 9. Tools and tests written this session

`scripts/seqleak.py` (+`--verify`), `seqfix_eval.py`, `seqfix2.py`, `wd29.py`,
`backfill_prov.py`, `parity_report.py`, `pair_npz.py`, `rope_probe.py`,
`cond_probe.py`, `fuse_init_probe.py`; `src/provenance.py`, `src/runlog.py`,
`src/safeio.py`; `bootstrap.py --retr-off`, `train.py --seed/--wd-legacy`,
`multiquery --calib`.

Tests (315): `test_provenance.py`, `test_runlog.py`, `test_param_groups.py`,
`test_merge_candidates.py`, `test_written_row_guards.py`, `test_argparse_help.py`,
`test_split_hash_and_cache.py`, `test_checkpoint_contract.py`, `test_topk_bounds.py`,
`test_import_order.py`, `test_safeio.py`, `test_cache_completeness.py`.

---

## 10. Environment

* RTX 3070, 8 GB. Training is GPU-bound, 90–96% at 165–176 W. **Never stack
  GPU jobs** — 2.6× measured cost. kNN rebuild ~5–7 min; a training rung ~26 min.
* **Use `py`.** Three names, two interpreters: `py` is the Windows launcher
  (`C:\Windows\py.exe`) and resolves to Python 3.13 with numpy 2.3.5, torch
  2.6.0+cu126 and CUDA — the project stack. Both `python` and `python3` are
  MSYS2's `mingw64` 3.12 and have none of it. `py -m pytest tests/ -q` runs the
  suite; the hardcoded
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe` is the
  same binary and only needed where a launcher is not wanted. This is what the
  round-three reviewer hit when they reported the suite could not be run —
  `py` was there the whole time.
* Tile server `192.168.50.1:3000`. `OSV_RELEASE` is required, no default.
* `.gitignore` now covers `runs/*.npz`, `runs/errs_prebug/`, `.pytest_cache/`,
  `.claude/`, safeio `.tmp<pid>` and `.grow.npy` leftovers.

---

## 11. Rules learned the hard way

* **A cache key must name everything the value depends on**, not everything
  that looks like an identifier. Three instances today (§6.5).
* **A fix is not done until it is tested against the failure it prevents.**
* **Ask whether a measurement error is independent of the treatment.**
* **Never generalise a null from arms that are near-identical by construction**
  (§6.1). Pick contrasts that span the axes before concluding a metric is blind.
* **Never compare two numbers from different embedding spaces** (§6.2).
* **Never `git stash`** — it swept uncommitted work out from under a diagnostic.
  Use `git show <rev>:<path>` to a scratch file.
* **Never edit code that a running chain has not finished importing.** Each
  stage is a fresh subprocess, so an edit lands on every stage that starts
  after it and none that started before. This voided the clean-bank result:
  `train.py`'s negative seeding changed at 12:32:53, `train-clean-e4` launched
  at 12:33:38, and the ladder's three rungs were then not the same arm. Fixing
  a review item and running an experiment are both fine; doing them in the
  same hour on the same file is not. Check `runs/logs/*_runner.log` for an
  in-flight chain before touching `src/`, or stage the edit and commit after.
* **A guard defined and never called is not a guard.** Nor is one that fails open.
* **Never type a backslash inside a Bash heredoc** — it arrives as a newline.
  Build it as `chr(92)` or write the patch script with the Write tool.
* **Scope every log check to the newest `==== attempt`.**
* **Do not block on `WaitForExit` in the PowerShell tool**; detach with
  `Start-Process` and poll.
* The **2,000-image selection set is not evidence**; it picks checkpoints, and
  it produced a clean monotone trend out of pure noise today (§3).
* **A wide interval means get more data, not weaken the claim.**
