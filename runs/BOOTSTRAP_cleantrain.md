
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |

**Sink negatives are not drawn the same way across these arms.** A seeded arm sees the same off-path tiles on a re-run; an OS-entropy arm does not, whatever seed it records.

| arm | sink negatives |
|---|---|
| `clean-drop70-e6` | seeded |
| `wd29-fix-e6` | OS entropy (predates the flag) |
| `d768-b350-e6-drop70` | OS entropy (predates the flag) |

**Weight-decay grouping differs between these arms.** The substring rule exempted the learned retrieval keys (196,608 parameters, 3.7% of the model) from decay.

| arm | weight-decay grouping |
|---|---|
| `clean-drop70-e6` | by module type |
| `wd29-fix-e6` | by module type |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| clean-drop70-e6 vs wd29-fix-e6 | [+3.3, +5.9] km | separated | [-4.58, -2.52] pp | separated |
| clean-drop70-e6 vs d768-b350-e6-drop70 | [+3.1, +5.7] km | separated | [-4.84, -2.74] pp | separated |
| wd29-fix-e6 vs d768-b350-e6-drop70 | [-1.2, +0.9] km | inside noise | [-1.26, +0.74] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
