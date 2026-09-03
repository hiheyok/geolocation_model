
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d768-b350-e6-drop50` | 768-d / 3.50M bank / 6 ep / drop50 |
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.0, +0.2] km | inside noise | [-0.98, +0.34] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop50 | [-0.1, +0.2] km | inside noise | [-1.02, +0.40] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.40, +1.12] pp | inside noise |
| d768-b350-e6 vs d1536-b350-e6-drop30 | [+0.0, +0.3] km | separated | [-2.04, -0.54] pp | separated |
| d768-b350-e6-drop30 vs d768-b350-e6-drop50 | [-0.2, +0.0] km | inside noise | [-0.64, +0.70] pp | inside noise |
| d768-b350-e6-drop30 vs d768-b350-e6-drop70 | [-0.5, -0.2] km | separated | [-0.04, +1.40] pp | inside noise |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop30 | [-0.1, +0.2] km | inside noise | [-1.68, -0.26] pp | separated |
| d768-b350-e6-drop50 vs d768-b350-e6-drop70 | [-0.4, -0.1] km | separated | [-0.02, +1.36] pp | inside noise |
| d768-b350-e6-drop50 vs d1536-b350-e6-drop30 | [+0.0, +0.2] km | separated | [-1.68, -0.26] pp | separated |
| d768-b350-e6-drop70 vs d1536-b350-e6-drop30 | [+0.2, +0.5] km | separated | [-2.38, -0.88] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
