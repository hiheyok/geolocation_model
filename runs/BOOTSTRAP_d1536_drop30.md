
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d1536-b350-e6 vs d768-b350-e6-drop30 | [-0.2, +0.1] km | inside noise | [-0.46, +1.02] pp | inside noise |
| d1536-b350-e6 vs d1536-b350-e6-drop30 | [-0.1, +0.1] km | inside noise | [-1.30, -0.10] pp | separated |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop30 | [-0.1, +0.2] km | inside noise | [-1.68, -0.24] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
