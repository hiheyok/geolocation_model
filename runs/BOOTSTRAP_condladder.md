
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| pyrMIX-e6 vs pyrMIX-e8 | [-3.5, +0.8] km | inside noise | [+0.16, +1.36] pp | separated |
| pyrMIX-e6 vs pyrMIX-cond-e8 | [-3.7, +1.3] km | inside noise | [-0.56, +1.00] pp | inside noise |
| pyrMIX-e8 vs pyrMIX-cond-e8 | [-2.4, +2.6] km | inside noise | [-1.30, +0.24] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
