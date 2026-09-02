
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_b55_c4 vs s10_b55_c6 | [-0.0, +0.2] km | inside noise | [-0.66, +0.18] pp | inside noise |
| s10_b55_c4 vs s10_w768 | [-0.8, -0.4] km | separated | [+2.26, +3.76] pp | separated |
| s10_b55_c4 vs s10_w768_c4 | [-0.4, -0.0] km | separated | [+0.52, +1.98] pp | separated |
| s10_b55_c4 vs s10_w768_c6 | [-0.3, +0.1] km | inside noise | [+0.06, +1.56] pp | separated |
| s10_b55_c6 vs s10_w768 | [-0.9, -0.5] km | separated | [+2.46, +4.00] pp | separated |
| s10_b55_c6 vs s10_w768_c4 | [-0.5, -0.1] km | separated | [+0.76, +2.22] pp | separated |
| s10_b55_c6 vs s10_w768_c6 | [-0.4, -0.0] km | separated | [+0.28, +1.82] pp | separated |
| s10_w768 vs s10_w768_c4 | [+0.2, +0.5] km | separated | [-2.32, -1.18] pp | separated |
| s10_w768 vs s10_w768_c6 | [+0.2, +0.6] km | separated | [-2.88, -1.48] pp | separated |
| s10_w768_c4 vs s10_w768_c6 | [-0.0, +0.2] km | inside noise | [-0.94, +0.06] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
