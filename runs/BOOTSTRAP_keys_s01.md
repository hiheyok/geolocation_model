
paired bootstrap, test split, 4,907 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| retr_seq vs retr_gate | [-23.1, +10.6] km | inside noise | [-1.51, +0.53] pp | inside noise |
| retr_seq vs retr_lkey | [-18.6, +17.0] km | inside noise | [-1.77, +0.18] pp | inside noise |
| retr_seq vs retr_dual | [-22.7, +12.0] km | inside noise | [-1.51, +0.55] pp | inside noise |
| retr_seq vs retr_dual32 | [-17.3, +17.3] km | inside noise | [-0.49, +1.51] pp | inside noise |
| retr_gate vs retr_lkey | [-12.2, +23.8] km | inside noise | [-1.24, +0.65] pp | inside noise |
| retr_gate vs retr_dual | [-15.7, +17.3] km | inside noise | [-1.00, +1.00] pp | inside noise |
| retr_gate vs retr_dual32 | [-10.3, +21.4] km | inside noise | [+0.02, +2.00] pp | separated |
| retr_lkey vs retr_dual | [-21.5, +11.7] km | inside noise | [-0.63, +1.24] pp | inside noise |
| retr_lkey vs retr_dual32 | [-16.7, +17.0] km | inside noise | [+0.33, +2.28] pp | separated |
| retr_dual vs retr_dual32 | [-10.5, +19.8] km | inside noise | [+0.06, +1.94] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
