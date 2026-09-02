
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8 | [-23.5, -3.2] km | separated | [+0.04, +1.08] pp | separated |
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8_c4 | [-7.9, +10.3] km | inside noise | [-0.40, +0.56] pp | inside noise |
| s10_cell8_bank25_lr1e4_c vs s10_pool_c8_c6 | [-5.1, +12.6] km | inside noise | [-0.44, +0.56] pp | inside noise |
| s10_pool_c8 vs s10_pool_c8_c4 | [+6.9, +21.7] km | separated | [-0.86, -0.14] pp | separated |
| s10_pool_c8 vs s10_pool_c8_c6 | [+9.0, +25.2] km | separated | [-0.92, -0.08] pp | separated |
| s10_pool_c8_c4 vs s10_pool_c8_c6 | [-2.4, +8.5] km | inside noise | [-0.34, +0.32] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
