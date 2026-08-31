
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25 vs s10_bal_bank25_c4 | [+0.2, +1.0] km | separated | [-1.08, -0.02] pp | separated |
| s10_bal_bank25 vs s10_n400k_bank25_c4 | [-0.5, +0.6] km | inside noise | [-1.22, +0.42] pp | inside noise |
| s10_bal_bank25_c4 vs s10_n400k_bank25_c4 | [-1.0, +0.0] km | inside noise | [-0.68, +0.92] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
