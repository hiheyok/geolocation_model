
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6 | [-1.3, +0.1] km | inside noise | [+0.36, +1.60] pp | separated |
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6-lr5e-5 | [-0.9, +0.1] km | inside noise | [-0.10, +0.86] pp | inside noise |
| pyrL0L1-b340-e4 vs pyrL0L1-b340-e6-lr3e-5 | [-1.1, -0.0] km | separated | [+0.18, +1.08] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0L1-b340-e6-lr5e-5 | [-0.5, +0.8] km | inside noise | [-1.18, -0.02] pp | separated |
| pyrL0L1-b340-e6 vs pyrL0L1-b340-e6-lr3e-5 | [-0.6, +0.7] km | inside noise | [-0.94, +0.24] pp | inside noise |
| pyrL0L1-b340-e6-lr5e-5 vs pyrL0L1-b340-e6-lr3e-5 | [-0.5, +0.2] km | inside noise | [-0.08, +0.56] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
