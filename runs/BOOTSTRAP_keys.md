
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| s10_key_none vs s10_key_scalar | [+84.0, +111.6] km | separated | [-18.62, -15.90] pp | separated |
| s10_key_none vs s10_key_cond | [+78.5, +105.9] km | separated | [-18.58, -15.74] pp | separated |
| s10_key_none vs s10_key_pos | [+94.6, +121.6] km | separated | [-20.58, -17.82] pp | separated |
| s10_key_none vs s10_key_dual | [+103.9, +131.1] km | separated | [-22.78, -19.92] pp | separated |
| s10_key_scalar vs s10_key_cond | [-10.5, -0.1] km | separated | [-0.60, +0.90] pp | inside noise |
| s10_key_scalar vs s10_key_pos | [+5.2, +16.1] km | separated | [-2.78, -1.10] pp | separated |
| s10_key_scalar vs s10_key_dual | [+13.8, +25.9] km | separated | [-5.04, -3.08] pp | separated |
| s10_key_cond vs s10_key_pos | [+10.1, +21.5] km | separated | [-2.90, -1.30] pp | separated |
| s10_key_cond vs s10_key_dual | [+18.5, +31.4] km | separated | [-5.16, -3.26] pp | separated |
| s10_key_pos vs s10_key_dual | [+5.1, +13.9] km | separated | [-2.98, -1.26] pp | separated |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
