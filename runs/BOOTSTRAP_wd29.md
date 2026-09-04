
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |

**Weight-decay grouping differs between these arms.** The substring rule exempted the learned retrieval keys (196,608 parameters, 3.7% of the model) from decay.

| arm | weight-decay grouping |
|---|---|
| `wd29-fix-e6` | by module type |
| `wd29-legacy-e6` | legacy substring |
| `d768-b350-e6-drop70` | by module type |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| wd29-fix-e6 vs wd29-legacy-e6 | [-0.0, +0.2] km | inside noise | [-1.08, +0.10] pp | inside noise |
| wd29-fix-e6 vs d768-b350-e6-drop70 | [+0.2, +0.6] km | separated | [-1.16, +0.54] pp | inside noise |
| wd29-legacy-e6 vs d768-b350-e6-drop70 | [+0.1, +0.4] km | separated | [-0.66, +1.00] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
