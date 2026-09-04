
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |
| `d1536-b350-e6-drop70` | 1536-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d1536-b350-e6 vs d1536-b350-e6-drop30 | [-0.1, +0.1] km | inside noise | [-1.28, -0.10] pp | separated |
| d1536-b350-e6 vs d1536-b350-e6-drop70 | [-0.8, -0.4] km | separated | [-0.52, +0.98] pp | inside noise |
| d1536-b350-e6 vs d768-b350-e6-drop70 | [-0.5, -0.2] km | separated | [+0.20, +1.70] pp | separated |
| d1536-b350-e6-drop30 vs d1536-b350-e6-drop70 | [-0.8, -0.4] km | separated | [+0.20, +1.62] pp | separated |
| d1536-b350-e6-drop30 vs d768-b350-e6-drop70 | [-0.5, -0.2] km | separated | [+0.90, +2.40] pp | separated |
| d1536-b350-e6-drop70 vs d768-b350-e6-drop70 | [+0.0, +0.4] km | separated | [-0.08, +1.52] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
