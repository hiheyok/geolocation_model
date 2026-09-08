
paired bootstrap, test split, 5,000 images, 3,000 resamples, k=2, ranked on s0-s2

**Sink negatives are not drawn the same way across these arms.** A seeded arm sees the same off-path tiles on a re-run; an OS-entropy arm does not, whatever seed it records.

| arm | sink negatives |
|---|---|
| `shipclean-e4` | OS entropy |
| `pyrL0-b340-e4` | OS entropy |
| `pyrL0L1-b340-e4` | OS entropy |
| `wd29-fix-e4` | OS entropy (predates the flag) |

| contrast | median diff, 95% CI | | <25km diff, 95% CI | |
|---|---|---|---|---|
| shipclean-e4 vs pyrL0-b340-e4 | [-1.8, +0.6] km | inside noise | [-0.78, +1.06] pp | inside noise |
| shipclean-e4 vs pyrL0L1-b340-e4 | [+2.2, +4.7] km | separated | [-4.56, -2.36] pp | separated |
| shipclean-e4 vs wd29-fix-e4 | [+2.4, +4.5] km | separated | [-3.80, -1.96] pp | separated |
| pyrL0-b340-e4 vs pyrL0L1-b340-e4 | [+2.6, +5.3] km | separated | [-4.66, -2.54] pp | separated |
| pyrL0-b340-e4 vs wd29-fix-e4 | [+2.7, +5.2] km | separated | [-3.98, -2.08] pp | separated |
| pyrL0L1-b340-e4 vs wd29-fix-e4 | [-1.2, +1.1] km | inside noise | [-0.50, +1.64] pp | inside noise |

A positive median difference means the first arm is worse (more km); a positive <25km difference means it is better.
