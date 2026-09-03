
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop50` | 768-d / 3.50M bank / 6 ep / drop50 |
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6-drop90` | 768-d / 3.50M bank / 6 ep / drop90 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b350-e6 vs d768-b350-e6-drop50 | [-0.1, +0.2] km | inside noise | [-1.00, +0.42] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.42, +1.12] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop90 | [-2.4, -1.7] km | separated | [+4.70, +6.72] pp | separated |
| d768-b350-e6-drop50 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.04, +1.36] pp | inside noise |
| d768-b350-e6-drop50 vs d768-b350-e6-drop90 | [-2.4, -1.7] km | separated | [+5.02, +6.98] pp | separated |
| d768-b350-e6-drop70 vs d768-b350-e6-drop90 | [-2.1, -1.5] km | separated | [+4.40, +6.30] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
