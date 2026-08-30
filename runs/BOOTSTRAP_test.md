
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n25k_e38 vs s10_n50k_e19 | [-10.2, +2.4] km | inside noise | [-0.46, +1.44] pp | inside noise |
| s10_n25k_e38 vs s10_n100k_e10 | [+12.5, +25.0] km | separated | [-3.56, -1.46] pp | separated |
| s10_n25k_e38 vs s10_n200k_e5 | [+12.7, +24.9] km | separated | [-3.36, -1.44] pp | separated |
| s10_n25k_e38 vs s10_n400k_e2 | [+17.7, +29.8] km | separated | [-4.56, -2.66] pp | separated |
| s10_n25k_e38 vs s10_n50k_e80 | [+1.9, +13.5] km | separated | [-1.92, -0.08] pp | separated |
| s10_n50k_e19 vs s10_n100k_e10 | [+15.7, +29.4] km | separated | [-3.94, -2.00] pp | separated |
| s10_n50k_e19 vs s10_n200k_e5 | [+15.8, +28.8] km | separated | [-3.80, -2.00] pp | separated |
| s10_n50k_e19 vs s10_n400k_e2 | [+20.6, +34.5] km | separated | [-5.04, -3.20] pp | separated |
| s10_n50k_e19 vs s10_n50k_e80 | [+5.3, +18.1] km | separated | [-2.40, -0.58] pp | separated |
| s10_n100k_e10 vs s10_n200k_e5 | [-4.5, +3.9] km | inside noise | [-0.74, +0.98] pp | inside noise |
| s10_n100k_e10 vs s10_n400k_e2 | [+0.8, +9.0] km | separated | [-1.94, -0.26] pp | separated |
| s10_n100k_e10 vs s10_n50k_e80 | [-17.4, -5.3] km | separated | [+0.52, +2.50] pp | separated |
| s10_n200k_e5 vs s10_n400k_e2 | [+1.6, +8.8] km | separated | [-1.98, -0.52] pp | separated |
| s10_n200k_e5 vs s10_n50k_e80 | [-16.9, -5.5] km | separated | [+0.50, +2.28] pp | separated |
| s10_n400k_e2 vs s10_n50k_e80 | [-21.9, -10.4] km | separated | [+1.72, +3.54] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
