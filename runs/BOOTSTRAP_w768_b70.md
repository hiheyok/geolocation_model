
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| `d768-b350-e2` | 768-d / 3.50M bank / 2 ep |
| `d768-b350-e4` | 768-d / 3.50M bank / 4 ep |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d1536-b350-e6 vs d768-b265-e6 | [-1.2, -0.7] km | separated | [+2.56, +4.30] pp | separated |
| d1536-b350-e6 vs d768-b350-e2 | [-0.5, -0.2] km | separated | [+1.20, +2.62] pp | separated |
| d1536-b350-e6 vs d768-b350-e4 | [-0.3, -0.0] km | separated | [-0.08, +1.38] pp | inside noise |
| d1536-b350-e6 vs d768-b350-e6 | [-0.3, -0.0] km | separated | [-0.18, +1.32] pp | inside noise |
| d768-b265-e6 vs d768-b350-e2 | [+0.4, +0.9] km | separated | [-2.38, -0.64] pp | separated |
| d768-b265-e6 vs d768-b350-e4 | [+0.6, +1.0] km | separated | [-3.60, -1.88] pp | separated |
| d768-b265-e6 vs d768-b350-e6 | [+0.6, +1.0] km | separated | [-3.66, -1.96] pp | separated |
| d768-b350-e2 vs d768-b350-e4 | [+0.0, +0.3] km | separated | [-1.78, -0.76] pp | separated |
| d768-b350-e2 vs d768-b350-e6 | [+0.0, +0.3] km | separated | [-1.94, -0.72] pp | separated |
| d768-b350-e4 vs d768-b350-e6 | [-0.1, +0.1] km | inside noise | [-0.52, +0.42] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
