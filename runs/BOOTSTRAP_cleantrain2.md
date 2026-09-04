
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |

**Sink negatives are not drawn the same way across these arms.** A seeded arm sees the same off-path tiles on a re-run; an OS-entropy arm does not, whatever seed it records.

| arm | sink negatives |
|---|---|
| `clean2-drop70-e2` | OS entropy |
| `clean2-drop70-e4` | OS entropy |
| `clean2-drop70-e6` | OS entropy |
| `wd29-fix-e6` | OS entropy (predates the flag) |
| `d768-b350-e6-drop70` | OS entropy (predates the flag) |

**Weight-decay grouping differs between these arms.** The substring rule exempted the learned retrieval keys (196,608 parameters, 3.7% of the model) from decay.

| arm | weight-decay grouping |
|---|---|
| `clean2-drop70-e2` | by module type |
| `clean2-drop70-e4` | by module type |
| `clean2-drop70-e6` | by module type |
| `wd29-fix-e6` | by module type |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| clean2-drop70-e2 vs clean2-drop70-e4 | [-0.9, +0.8] km | inside noise | [-0.84, +0.52] pp | inside noise |
| clean2-drop70-e2 vs clean2-drop70-e6 | [-2.3, +0.1] km | inside noise | [+0.04, +1.78] pp | separated |
| clean2-drop70-e2 vs wd29-fix-e6 | [+2.4, +4.9] km | separated | [-3.82, -1.96] pp | separated |
| clean2-drop70-e2 vs d768-b350-e6-drop70 | [+2.2, +4.7] km | separated | [-4.08, -2.12] pp | separated |
| clean2-drop70-e4 vs clean2-drop70-e6 | [-1.9, -0.1] km | separated | [+0.46, +1.70] pp | separated |
| clean2-drop70-e4 vs wd29-fix-e6 | [+2.5, +5.0] km | separated | [-3.74, -1.84] pp | separated |
| clean2-drop70-e4 vs d768-b350-e6-drop70 | [+2.2, +4.7] km | separated | [-4.00, -2.00] pp | separated |
| clean2-drop70-e6 vs wd29-fix-e6 | [+3.3, +6.0] km | separated | [-4.86, -2.82] pp | separated |
| clean2-drop70-e6 vs d768-b350-e6-drop70 | [+3.0, +5.9] km | separated | [-5.12, -3.00] pp | separated |
| wd29-fix-e6 vs d768-b350-e6-drop70 | [-1.3, +0.9] km | inside noise | [-1.22, +0.76] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
