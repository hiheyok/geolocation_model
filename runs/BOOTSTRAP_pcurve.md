
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop10` | 768-d / 3.50M bank / 6 ep / drop10 |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d768-b350-e6-drop50` | 768-d / 3.50M bank / 6 ep / drop50 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b350-e6 vs d768-b350-e6-drop10 | [-0.0, +0.2] km | inside noise | [-1.30, -0.02] pp | separated |
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.0, +0.3] km | inside noise | [-0.96, +0.32] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop50 | [-0.1, +0.2] km | inside noise | [-1.04, +0.44] pp | inside noise |
| d768-b350-e6-drop10 vs d768-b350-e6-drop30 | [-0.1, +0.1] km | inside noise | [-0.30, +1.00] pp | inside noise |
| d768-b350-e6-drop10 vs d768-b350-e6-drop50 | [-0.2, +0.0] km | inside noise | [-0.36, +1.08] pp | inside noise |
| d768-b350-e6-drop30 vs d768-b350-e6-drop50 | [-0.2, +0.0] km | inside noise | [-0.64, +0.72] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
