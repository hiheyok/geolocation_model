
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| arm | what it is |
|---|---|
| `d768-b350-e6-drop70` | 768-d / 3.50M bank / 6 ep / drop70 |
| `d768-b350-e6` | 768-d / 3.50M bank / 6 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |
| `d768-b350-e6-drop30` | 768-d / 3.50M bank / 6 ep / drop30 |
| `d768-b350-e6-drop90` | 768-d / 3.50M bank / 6 ep / drop90 |
| `d1536-b350-e6` | 1536-d / 3.50M bank / 6 ep |
| `d1536-b350-e6-drop30` | 1536-d / 3.50M bank / 6 ep / drop30 |
| `d1536-b350-e6-drop70` | 1536-d / 3.50M bank / 6 ep / drop70 |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| d768-b350-e6-drop70 vs d768-b350-e6 | [-0.5, +1.6] km | inside noise | [-0.72, +1.06] pp | inside noise |
| d768-b350-e6-drop70 vs d768-b265-e6 | [-3.7, -1.3] km | separated | [+1.60, +3.56] pp | separated |
| d768-b350-e6-drop70 vs d768-b350-e6-drop30 | [-0.0, +1.8] km | inside noise | [-1.12, +0.54] pp | inside noise |
| d768-b350-e6-drop70 vs d768-b350-e6-drop90 | [-7.0, -4.3] km | separated | [+3.76, +5.90] pp | separated |
| d768-b350-e6-drop70 vs d1536-b350-e6 | [+0.3, +2.2] km | separated | [-1.34, +0.44] pp | inside noise |
| d768-b350-e6-drop70 vs d1536-b350-e6-drop30 | [+1.1, +3.0] km | separated | [-2.50, -0.72] pp | separated |
| d768-b350-e6-drop70 vs d1536-b350-e6-drop70 | [-0.3, +1.7] km | inside noise | [-1.70, +0.14] pp | inside noise |
| d768-b350-e6 vs d768-b265-e6 | [-4.3, -1.9] km | separated | [+1.44, +3.24] pp | separated |
| d768-b350-e6 vs d768-b350-e6-drop30 | [-0.6, +1.3] km | inside noise | [-1.28, +0.32] pp | inside noise |
| d768-b350-e6 vs d768-b350-e6-drop90 | [-7.7, -4.8] km | separated | [+3.56, +5.74] pp | separated |
| d768-b350-e6 vs d1536-b350-e6 | [-0.4, +1.6] km | inside noise | [-1.50, +0.22] pp | inside noise |
| d768-b350-e6 vs d1536-b350-e6-drop30 | [+0.5, +2.5] km | separated | [-2.72, -0.94] pp | separated |
| d768-b350-e6 vs d1536-b350-e6-drop70 | [-1.0, +1.3] km | inside noise | [-1.92, +0.00] pp | inside noise |
| d768-b265-e6 vs d768-b350-e6-drop30 | [+2.2, +4.5] km | separated | [-3.80, -1.88] pp | separated |
| d768-b265-e6 vs d768-b350-e6-drop90 | [-4.7, -1.8] km | separated | [+1.18, +3.44] pp | separated |
| d768-b265-e6 vs d1536-b350-e6 | [+2.5, +4.9] km | separated | [-3.98, -2.00] pp | separated |
| d768-b265-e6 vs d1536-b350-e6-drop30 | [+3.3, +5.7] km | separated | [-5.16, -3.16] pp | separated |
| d768-b265-e6 vs d1536-b350-e6-drop70 | [+1.9, +4.5] km | separated | [-4.34, -2.26] pp | separated |
| d768-b350-e6-drop30 vs d768-b350-e6-drop90 | [-8.0, -5.2] km | separated | [+4.12, +6.26] pp | separated |
| d768-b350-e6-drop30 vs d1536-b350-e6 | [-0.6, +1.2] km | inside noise | [-0.98, +0.72] pp | inside noise |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop30 | [+0.2, +2.1] km | separated | [-2.16, -0.44] pp | separated |
| d768-b350-e6-drop30 vs d1536-b350-e6-drop70 | [-1.3, +0.9] km | inside noise | [-1.44, +0.50] pp | inside noise |
| d768-b350-e6-drop90 vs d1536-b350-e6 | [+5.4, +8.4] km | separated | [-6.42, -4.16] pp | separated |
| d768-b350-e6-drop90 vs d1536-b350-e6-drop30 | [+6.3, +9.1] km | separated | [-7.54, -5.40] pp | separated |
| d768-b350-e6-drop90 vs d1536-b350-e6-drop70 | [+4.9, +7.8] km | separated | [-6.68, -4.56] pp | separated |
| d1536-b350-e6 vs d1536-b350-e6-drop30 | [+0.1, +1.6] km | separated | [-1.82, -0.48] pp | separated |
| d1536-b350-e6 vs d1536-b350-e6-drop70 | [-1.5, +0.4] km | inside noise | [-1.16, +0.54] pp | inside noise |
| d1536-b350-e6-drop30 vs d1536-b350-e6-drop70 | [-2.3, -0.5] km | separated | [-0.02, +1.70] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
