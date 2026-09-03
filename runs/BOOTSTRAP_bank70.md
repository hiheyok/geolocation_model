
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d1536-b265-e6` | 1536-d / 2.65M bank / 6 ep |
| `d1536-b350-e2` | 1536-d / 3.50M bank / 2 ep |
| `d1536-b350-e4` | 1536-d / 3.50M bank / 4 ep |
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d1536-b265-e6 vs d1536-b350-e2 | [+0.4, +0.8] km | separated | [-2.24, -0.62] pp | separated |
| d1536-b265-e6 vs d1536-b350-e4 | [+0.6, +0.9] km | separated | [-3.00, -1.50] pp | separated |
| d1536-b265-e6 vs d1536-b350-e6 | [+0.6, +0.9] km | separated | [-3.12, -1.56] pp | separated |
| d1536-b350-e2 vs d1536-b350-e4 | [+0.1, +0.3] km | separated | [-1.30, -0.30] pp | separated |
| d1536-b350-e2 vs d1536-b350-e6 | [+0.1, +0.3] km | separated | [-1.50, -0.38] pp | separated |
| d1536-b350-e4 vs d1536-b350-e6 | [-0.1, +0.1] km | inside noise | [-0.54, +0.32] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
