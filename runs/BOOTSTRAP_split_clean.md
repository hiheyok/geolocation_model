
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |

**Sink negatives are not drawn the same way across these arms.** A seeded arm sees the same off-path tiles on a re-run; an OS-entropy arm does not, whatever seed it records.

| arm | sink negatives |
|---|---|
| `pyrL0L1-b340-e4` | OS entropy |
| `pyrL0-b340-e4` | OS entropy |
| `wd29-fix-e4` | OS entropy (predates the flag) |
| `wd29-legacy-e4` | OS entropy (predates the flag) |
| `d768-b350-e6-drop70` | OS entropy (predates the flag) |

**Weight-decay grouping differs between these arms.** The substring rule exempted the learned retrieval keys (196,608 parameters, 3.7% of the model) from decay.

| arm | weight-decay grouping |
|---|---|
| `pyrL0L1-b340-e4` | by module type |
| `pyrL0-b340-e4` | by module type |
| `wd29-fix-e4` | by module type |
| `wd29-legacy-e4` | legacy substring |
| `d768-b350-e6-drop70` | legacy substring (predates the flag) |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrL0L1-b340-e4 vs pyrL0-b340-e4 | [-5.4, -2.6] km | separated | [+2.54, +4.70] pp | separated |
| pyrL0L1-b340-e4 vs wd29-fix-e4 | [-1.1, +1.1] km | inside noise | [-0.54, +1.68] pp | inside noise |
| pyrL0L1-b340-e4 vs wd29-legacy-e4 | [-0.9, +1.3] km | inside noise | [-0.34, +1.76] pp | inside noise |
| pyrL0L1-b340-e4 vs d768-b350-e6-drop70 | [-1.4, +1.0] km | inside noise | [-0.54, +1.66] pp | inside noise |
| pyrL0-b340-e4 vs wd29-fix-e4 | [+2.7, +5.2] km | separated | [-3.98, -2.08] pp | separated |
| pyrL0-b340-e4 vs wd29-legacy-e4 | [+2.9, +5.4] km | separated | [-3.80, -1.96] pp | separated |
| pyrL0-b340-e4 vs d768-b350-e6-drop70 | [+2.5, +5.0] km | separated | [-4.00, -2.08] pp | separated |
| wd29-fix-e4 vs wd29-legacy-e4 | [-0.4, +0.8] km | inside noise | [-0.40, +0.68] pp | inside noise |
| wd29-fix-e4 vs d768-b350-e6-drop70 | [-1.2, +0.8] km | inside noise | [-1.02, +0.92] pp | inside noise |
| wd29-legacy-e4 vs d768-b350-e6-drop70 | [-1.4, +0.6] km | inside noise | [-1.08, +0.80] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
