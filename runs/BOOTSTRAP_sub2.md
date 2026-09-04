
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b350-e6-drop70 vs d768-b350-e6-drop70-sub2 | [-0.4, -0.0] km | separated | [-1.18, +0.32] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
