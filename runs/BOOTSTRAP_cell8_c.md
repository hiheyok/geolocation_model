
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_bank25_lr1e4 vs s10_cell8_bank25_lr1e4_c | [-3.4, +8.9] km | inside noise | [-0.46, +0.18] pp | inside noise |
| s10_cell8_bank25_lr1e4 vs s10_cell8_bank25_cont | [-20.4, -1.5] km | separated | [-0.94, +0.00] pp | separated |
| s10_cell8_bank25_lr1e4_c vs s10_cell8_bank25_cont | [-23.7, -4.3] km | separated | [-0.82, +0.14] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
