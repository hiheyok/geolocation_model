# The four retrieval probes, re-measured on random cohorts

Every probe selected its queries as `flatnonzero(labels == "test")[:n]`.
`dataset.parquet` is shard-ordered, so that head is BR/AR/ZA at a median
nearest legal bank row of 0.212 km, against US/DE/RU and 0.337 km for a random
draw — a different continent and 35% denser. Re-run 2026-09-06 with a seeded
random cohort and **every other argument identical to the original**.

Absolute numbers fall everywhere, as expected: the old cohort was easier. Only
the paired contrasts carry meaning, and three of the four conclusions survive.

---

## 1. `query_only` — query-side re-encoding (§8e). **Confirmed, ~1.7x stronger.**

| query encoding | old `<25 km` | new `<25 km` | old vs ships | new vs ships |
|---|---|---|---|---|
| 3 crops @224 (ships) | 56.4% | 51.0% | — | — |
| 3 crops @336 | 54.8% | 48.3% | −1.60 [−2.9, −0.3] | **−2.73 [−4.0, −1.4]** |
| 3 crops @448 | 53.2% | 46.6% | −3.27 [−4.8, −1.7] | **−4.47 [−5.8, −3.0]** |

The dose-response is real and larger than reported. **The bank's encoding is a
contract that cannot be improved from one side** — unchanged.

---

## 2. `xbank` — the matched 2x2 (§8d). **One sub-claim retracted.**

| arm | old `<25 km` | new `<25 km` |
|---|---|---|
| query only | **−1.43 [−2.6, −0.3]** separated | **−0.53 [−1.6, +0.6] inside noise** |
| bank only | −0.57 [−1.8, +0.6] ~ | +0.27 [−0.9, +1.4] ~ |
| both matched | +2.63 [+1.3, +4.0] | **+3.33 [+2.2, +4.5]** |

**RETRACTED: "query-only is separated harmful."** On a random cohort it spans
zero. The matched rebuild is still the only arm that works, and the *reason*
stands — one side cannot be improved alone — but it now rests on `query_only`
above, which measured it directly and at a real dose, rather than on this
contrast.

The matched gain is **larger**, not smaller: +2.63 → +3.33.

---

## 3. `xbank` — gain against bank density (§8f). **Confirmed, steeper.**

| bank rows | old gain | new gain |
|---|---|---|
| 25,000 | +1.80 | **+1.10** |
| 50,000 | +2.20 | +1.90 |
| 100,000 | +2.03 | +2.23 |
| 200,000 | +2.53 | **+3.50** |
| 400,000 | +2.60 | **+3.33** |

The slope roughly triples. That is the expected direction: the biased cohort
was 35% denser, which lifted the sparse end and flattened the curve. **My
original objection — that the gain would shrink at 3.4M — is refuted more
firmly than before.**

---

## 4. `resmatch` — tiles versus resolution (§9). **The decision moves.**

| arm | old `<25 km` vs incumbent | new `<25 km` vs incumbent |
|---|---|---|
| 2 crops+tiles @224 | **+1.27 [+0.4, +2.2]** separated | +0.83 [−0.1, +1.7] **inside noise** |
| 3 crops @518/512 native | −0.37 [−1.4, +0.7] ~ | −0.30 [−1.3, +0.7] ~ |
| 4 native crops + 224 tiles | +0.97 [−0.1, +2.0] ~ | **+1.27 [+0.2, +2.3]** separated |

**The ordering reverses.** Arm 2 beat arm 4 on the biased cohort; arm 4 beats
arm 2 on a random one. `RESMATCH.md` concluded "resolution does not complement
tiles, it dilutes them" — that conclusion was cohort-dependent and does not
survive.

At `<1 km` the picture is unchanged and consistent: resolution wins in both
arms that use it (arm 3 +0.50 [+0.2, +0.8], arm 4 +0.47 [+0.2, +0.8]) while
tiles alone do not (arm 2 +0.23 ~).

**This does not yet reinstate resolution.** Arms 3 and 4 still use
`vit_base_patch16_siglip_224.v2_webli` at `img_size=512` — a 224 checkpoint
with interpolated position embeddings, which REVIEW5 #10 flagged as out of
distribution. `cond_native.f16.npy` was built with the genuine
`siglip_512.v2_webli`; re-running `resmatch` against it is the outstanding
work, and until then arm 4's edge is confounded in the direction that would
*understate* it.

---

## 5. `pyr_levels` — the L2 null (§9e). **Confirmed, slightly stronger.**

`L0L1L2 / L0L1` at `<25 km`: −0.74 [−1.4, −0.1] → **−0.94 [−1.6, −0.3]**.
A third pyramid level pushed into the retrieval vector remains separated
*harmful* against the matched two-level arm.

---

## What was invalid in the first attempt, and why

The 00:06 chain ran `query_only` and `resmatch` with the fix and `xbank` and
`pyr_levels` without it — those two reproduced the old numbers to three
significant figures. **I switched git branches while the chain was running**,
which rewrote the working tree under stages that had not yet started. A stage
imports its code when it starts, so the two that began after the switch loaded
the pre-fix files.

That is the same hazard as editing a file mid-chain, arriving by a mechanism
the rule did not name. `AGENTS.md` now names it.
