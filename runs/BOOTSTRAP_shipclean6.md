
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| shipclean-e4 vs shipclean-e6 | [-1.8, -0.2] km | separated | [+0.20, +1.40] pp | separated |
| shipclean-e4 vs pyrL0-b340-e4 | [-1.7, +0.6] km | inside noise | [-0.78, +1.04] pp | inside noise |
| shipclean-e4 vs pyrL0L1-b340-e4 | [+2.2, +4.7] km | separated | [-4.54, -2.38] pp | separated |
| shipclean-e6 vs pyrL0-b340-e4 | [-0.8, +1.7] km | inside noise | [-1.66, +0.32] pp | inside noise |
| shipclean-e6 vs pyrL0L1-b340-e4 | [+3.0, +5.7] km | separated | [-5.36, -3.16] pp | separated |
| pyrL0-b340-e4 vs pyrL0L1-b340-e4 | [+2.6, +5.3] km | separated | [-4.64, -2.60] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
