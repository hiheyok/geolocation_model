
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b265-e6 vs d768-b350-e6 | [+0.6, +1.0] km | separated | [-3.66, -1.98] pp | separated |
| d768-b265-e6 vs d768-b350-e6-drop30 | [+0.7, +1.1] km | separated | [-4.04, -2.26] pp | separated |
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.0, +0.2] km | inside noise | [-0.98, +0.30] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
