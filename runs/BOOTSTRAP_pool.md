
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_pool | [-1.2, -0.1] km | separated | [+0.70, +2.36] pp | separated |
| s10_bal_bank25_c6 vs s10_pool_c4 | [-0.6, +0.5] km | inside noise | [-0.58, +1.04] pp | inside noise |
| s10_bal_bank25_c6 vs s10_pool_c6 | [-0.5, +0.7] km | inside noise | [-0.74, +0.94] pp | inside noise |
| s10_pool vs s10_pool_c4 | [+0.2, +1.1] km | separated | [-1.86, -0.74] pp | separated |
| s10_pool vs s10_pool_c6 | [+0.3, +1.3] km | separated | [-2.14, -0.76] pp | separated |
| s10_pool_c4 vs s10_pool_c6 | [-0.2, +0.5] km | inside noise | [-0.64, +0.34] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
