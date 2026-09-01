
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_bal_bank25_c6 vs s10_bal_c8 | [-0.3, +0.4] km | inside noise | [-0.90, +0.10] pp | inside noise |
| s10_bal_bank25_c6 vs s10_geo_bias | [-0.2, +0.6] km | inside noise | [-0.94, +0.12] pp | inside noise |
| s10_bal_bank25_c6 vs s10_geo_key | [-0.2, +0.5] km | inside noise | [-0.76, +0.24] pp | inside noise |
| s10_bal_c8 vs s10_geo_bias | [-0.2, +0.5] km | inside noise | [-0.52, +0.52] pp | inside noise |
| s10_bal_c8 vs s10_geo_key | [-0.2, +0.4] km | inside noise | [-0.26, +0.54] pp | inside noise |
| s10_geo_bias vs s10_geo_key | [-0.5, +0.3] km | inside noise | [-0.42, +0.68] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
