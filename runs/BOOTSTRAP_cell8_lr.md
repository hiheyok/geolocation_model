
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_cell8_bank25_cont vs s10_cell8_bank25_lr3e4 | [-43.7, -22.3] km | separated | [+0.02, +0.82] pp | separated |
| s10_cell8_bank25_cont vs s10_cell8_bank25_lr1e4 | [+1.5, +20.4] km | separated | [+0.00, +0.94] pp | inside noise |
| s10_cell8_bank25_lr3e4 vs s10_cell8_bank25_lr1e4 | [+32.8, +56.5] km | separated | [-0.40, +0.54] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
