
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_pool_c6 | [-0.5, +0.6] km | inside noise | [-0.70, +0.90] pp | inside noise |
| s10_bal_bank25_c6 vs s10_b40 | [+3.0, +4.3] km | separated | [-5.88, -3.76] pp | separated |
| s10_bal_bank25_c6 vs s10_b40_c4 | [+3.3, +4.6] km | separated | [-7.50, -5.30] pp | separated |
| s10_bal_bank25_c6 vs s10_b40_c6 | [+3.6, +4.9] km | separated | [-7.60, -5.40] pp | separated |
| s10_pool_c6 vs s10_b40 | [+2.9, +4.2] km | separated | [-6.02, -3.90] pp | separated |
| s10_pool_c6 vs s10_b40_c4 | [+3.1, +4.5] km | separated | [-7.48, -5.46] pp | separated |
| s10_pool_c6 vs s10_b40_c6 | [+3.4, +4.8] km | separated | [-7.64, -5.50] pp | separated |
| s10_b40 vs s10_b40_c4 | [+0.1, +0.5] km | separated | [-2.10, -0.98] pp | separated |
| s10_b40 vs s10_b40_c6 | [+0.3, +0.8] km | separated | [-2.28, -1.00] pp | separated |
| s10_b40_c4 vs s10_b40_c6 | [+0.1, +0.4] km | separated | [-0.58, +0.38] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
