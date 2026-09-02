
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b265-e4` | 768-d / 2.65M bank / 4 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b265-e4 vs d768-b265-e6 | [-0.0, +0.2] km | inside noise | [-0.96, +0.06] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
