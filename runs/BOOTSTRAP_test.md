
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_n25k_e38 vs s10_n50k_e19 | [-3.1, +17.9] km | inside noise | [-1.56, +0.18] pp | inside noise |
| s10_n25k_e38 vs s10_n100k_e10 | [+16.9, +40.8] km | separated | [-2.38, -0.38] pp | separated |
| s10_n25k_e38 vs s10_n200k_e5 | [+17.4, +41.0] km | separated | [-3.56, -1.70] pp | separated |
| s10_n25k_e38 vs s10_n400k_e2 | [+24.0, +47.7] km | separated | [-3.96, -2.12] pp | separated |
| s10_n25k_e38 vs s10_n50k_e80 | [+2.9, +23.6] km | separated | [-1.62, +0.12] pp | inside noise |
| s10_n25k_e38 vs s10_n25k_bank25k | [-371.8, -327.3] km | separated | [+18.08, +20.72] pp | separated |
| s10_n50k_e19 vs s10_n100k_e10 | [+13.6, +31.1] km | separated | [-1.72, +0.26] pp | inside noise |
| s10_n50k_e19 vs s10_n200k_e5 | [+13.7, +29.6] km | separated | [-2.82, -1.10] pp | separated |
| s10_n50k_e19 vs s10_n400k_e2 | [+21.2, +37.0] km | separated | [-3.30, -1.50] pp | separated |
| s10_n50k_e19 vs s10_n50k_e80 | [-1.0, +14.8] km | inside noise | [-0.92, +0.76] pp | inside noise |
| s10_n50k_e19 vs s10_n25k_bank25k | [-379.4, -333.7] km | separated | [+18.72, +21.40] pp | separated |
| s10_n100k_e10 vs s10_n200k_e5 | [-6.5, +6.0] km | inside noise | [-2.08, -0.36] pp | separated |
| s10_n100k_e10 vs s10_n400k_e2 | [+1.4, +12.5] km | separated | [-2.44, -0.80] pp | separated |
| s10_n100k_e10 vs s10_n50k_e80 | [-23.8, -6.9] km | separated | [-0.32, +1.56] pp | inside noise |
| s10_n100k_e10 vs s10_n25k_bank25k | [-402.3, -356.8] km | separated | [+19.48, +22.14] pp | separated |
| s10_n200k_e5 vs s10_n400k_e2 | [+2.0, +12.8] km | separated | [-1.06, +0.26] pp | inside noise |
| s10_n200k_e5 vs s10_n50k_e80 | [-22.6, -7.1] km | separated | [+1.02, +2.72] pp | separated |
| s10_n200k_e5 vs s10_n25k_bank25k | [-399.3, -357.0] km | separated | [+20.70, +23.40] pp | separated |
| s10_n400k_e2 vs s10_n50k_e80 | [-30.1, -14.1] km | separated | [+1.42, +3.16] pp | separated |
| s10_n400k_e2 vs s10_n25k_bank25k | [-407.4, -364.2] km | separated | [+21.08, +23.82] pp | separated |
| s10_n50k_e80 vs s10_n25k_bank25k | [-386.1, -341.0] km | separated | [+18.86, +21.52] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
