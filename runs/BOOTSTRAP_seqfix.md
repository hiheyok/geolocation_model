
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |

**Weight-decay grouping differs between these arms.** The substring rule exempted the learned retrieval keys (196,608 parameters, 3.7% of the model) from decay.

| arm | weight-decay grouping |
|---|---|
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |
| `d768-b350-e6` | legacy substring (predates the flag) |
| `wd29-fix-e6` | by module type |
| `wd29-legacy-e6` | legacy substring |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b350-e6-drop70 vs d768-b350-e6 | [-0.5, +1.6] km | inside noise | [-0.72, +1.06] pp | inside noise |
| d768-b350-e6-drop70 vs wd29-fix-e6 | [-0.9, +1.3] km | inside noise | [-0.74, +1.22] pp | inside noise |
| d768-b350-e6-drop70 vs wd29-legacy-e6 | [-0.6, +1.4] km | inside noise | [-0.76, +1.14] pp | inside noise |
| d768-b350-e6 vs wd29-fix-e6 | [-1.4, +0.8] km | inside noise | [-0.92, +1.02] pp | inside noise |
| d768-b350-e6 vs wd29-legacy-e6 | [-1.3, +0.9] km | inside noise | [-1.02, +0.92] pp | inside noise |
| wd29-fix-e6 vs wd29-legacy-e6 | [-0.6, +0.9] km | inside noise | [-0.68, +0.62] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
