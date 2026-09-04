# Live state — rewritten 2026-09-04 12:10, before a compaction

Read this first. `docs/REVIEW.md`, `REVIEW2.md` and `REVIEW3.md` hold three
Codex reviews; `docs/BACKLOG.md` the code and performance work;
`docs/ARCHITECTURE_NEXT.md` the architecture directions. Durable findings are in
the memory directory. Branch `retrieval-dropout`, PR #13, all pushed.
**315 tests pass. `scripts/seqfix2.py` is RUNNING (see §7). Tile server up.**

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
`REVIEW3.md` 10 items: **NONE fixed — arrived 2026-09-04, unread until now.**

### REVIEW3 — verified, and two of them are my own work from today

**#7 `file_stamp` discards the size. CONFIRMED by direct test.**
`"{:x}{:x}".format(size, mtime_ns)[-12:]` — hex(mtime_ns) is 16 chars, so the
slice keeps mtime only. Sizes 1, 999,999,999 and 5 GB all produce the identical
stamp. This weakens the bootstrap error-cache key and the multiquery query-cache
key **that I added today**. Fix: encode the fields separately.

**#2 `--seed` does not make sink training reproducible. CONFIRMED.**
`train.py:438` builds the *train* dataset with the default `neg_random=True`;
only *val* (line 442) passes `neg_random=False`. My REVIEW2 note said training
passes it, and the comment I wrote at `train.py:448` repeats that error. Every
`--neg 4` run draws off-path tiles from OS entropy. **The wd29 conclusion
survives** — both ladders were equally unseeded here, so it adds noise to a
result that was already inside noise — but the recorded seed overstates
reproducibility.

Not yet verified, in the reviewer's severity order:
* **#1** any non-release-length street cache bypasses row provenance; an
  extension-only 750k cache would pair every photo with another's embedding
* **#3** the production PCA basis (`pca768_bank55_pca.npz`) is fitted on the
  first 500k rows = the whole release, so ~20% is val/test. Same transductive
  leak fixed in `fuse_head` in round one, still live in the pipeline that built
  every 768-d arm
* **#4** `stack_bank` cannot rebuild the documented multi-extension banks
* **#5** marker identity covers argv only, not inputs/outputs — reopens #70,
  which I closed today. Dangerous here: a clean kNN build can be replaced under
  the same name while the stage still reports satisfied
* **#6** `concat_street`/`pool_street` write no sidecars; `stack_bank` writes
  `basis="built"` without checking its inputs first
* **#8** a corrupt JSON marker reads as a trusted legacy success — and
  `tests/test_runlog.py` asserts that behaviour. Marker writes are non-atomic
* **#9** `_check_fetched` treats any nonzero mask value as complete
* **#10** `serve.py` and `occupancy_probe.py` bypass `provenance.bank_ext`

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

---

## 7. RUNNING NOW — `scripts/seqfix2.py` (launched 11:53, 5 h deadline)

Stages in order. First three are **done**; the ladder is training.

1. ✅ rebuild `pool_bal_bank70` kNN + verify clean
2. ✅ rebuild `pca768_bank55` kNN + verify clean
3. ✅ `seqfix2-boot-all` → `runs/BOOTSTRAP_seqfix_all.md` (the §2 table)
4. ⏳ `train-clean-e2/e4/e6` → **`clean-drop70-e6`**, the shipping ladder
   retrained against the **clean** cache at `--seed 0`. ~26 min a rung.
5. ⏳ `seqfix2-boot-clean` → `runs/BOOTSTRAP_cleantrain.md`, comparing
   `clean-drop70-e6` vs `wd29-fix-e6` vs the shipping arm
6. ⏳ `seqfix2-hr-clean` → `runs/hr_cleandrop70.npz`, KartaView at 5,000

**The point of step 4:** `clean-drop70-e6` differs from `wd29-fix-e6` in
exactly one thing — whether the bank it trained against leaked. That converts
§2's lower bound into an estimate of what the system is actually worth.

Monitor: `runs/logs/seqfix2_runner.log`. If it died, re-run
`OSV_RELEASE=s10 python scripts/seqfix2.py 5` — markers make it resume.

---

## 8. What to do next

1. **Read the clean-train result** when §7 finishes. If `clean-drop70-e6` beats
   `wd29-fix-e6`, training against a leaking bank was itself harmful and every
   arm on record is understated; if it matches, 57.5% is the honest number.
2. **REVIEW3**, in the reviewer's order. #7 and #2 first — they are mine, from
   today, and #7 undermines a cache key I added to fix a different bug.
3. **#3 (transductive PCA)** is the one most likely to move a published number
   after the leak: every 768-d arm uses a basis fitted partly on val/test.
4. The auxiliary coarse street-only head (Tier 1 in ARCHITECTURE_NEXT) — step 0
   carries 84.3% of the mean error and the visual branch alone scores 2.7%.

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
* `python` on PATH is MSYS2's and lacks numpy. Always
  `/c/Users/longd/AppData/Local/Programs/Python/Python313/python.exe`.
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
* **A guard defined and never called is not a guard.** Nor is one that fails open.
* **Never type a backslash inside a Bash heredoc** — it arrives as a newline.
  Build it as `chr(92)` or write the patch script with the Write tool.
* **Scope every log check to the newest `==== attempt`.**
* **Do not block on `WaitForExit` in the PowerShell tool**; detach with
  `Start-Process` and poll.
* The **2,000-image selection set is not evidence**; it picks checkpoints, and
  it produced a clean monotone trend out of pure noise today (§3).
* **A wide interval means get more data, not weaken the claim.**
