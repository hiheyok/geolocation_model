# Within-level pooling: the mean is already right

`scripts/pyr_gem.py`, pyr47, 3,000 queries against a 38,009 bank, reported on
the 1,522 held-out half.

Third unswept constant of the same shape, after the head/mean blend (fixed at
0.5 by concatenating two unit blocks) and the level weights (fixed at 1:1:1 by
`stack(per).mean(0)`). Both of those moved when swept. **This one does not.**

`FuseHead.baseline` pools a level with `mean(1)` over its tiles -- 24 of them
at L2, equal weight. The hypothesis was dilution: a featureless frame has 24
tiles of which two carry signal, and a mean divides those by twelve. GeM tests
it with no parameters at all, `p=1` being exactly the mean and larger `p`
approaching a max.

**Every contrast against the mean spans zero except one, and that one is a
loss** (`p=2` at 200 km, -1.18 pp [-2.2, -0.3]). The two halves disagree about
the best `p` -- selection says 2, reporting says 3 -- which is what a flat
landscape looks like. Equal averaging within a level is not a lazy default; it
is the right choice, and the remaining headroom is not here.

The `+head` rows are the `w=0.04` blend from `pyr_blend.py` and reproduce its
gain (+2.23 pp at 200 km, +2.43 at 750 km, separated). That is the head doing
the work, not the pooling.

**A note on comparability.** `p=1` here reads 32.9% at 25 km against
`pyr_blend`'s 32.72% for the same arm on the same half. The gap is fp16: this
script holds the normalised tokens as float16 so each `p` is a re-pool rather
than another pass, where `pyr_blend` normalises in float32 per block. Every
comparison *within* this table shares that representation, so the ranking is
unaffected.

**Performance.** The first version ran the pooling as numpy `|x|**p` -- a
ufunc, not a BLAS call, so it never threads: 15 of 16 cores idle, the card at
21%, and it had not finished `p=1` after 90 s. Moved to the GPU and checked
against the numpy definition (identical to 3e-5 at every `p`), the whole sweep
is 37 s.

---

```text
pyr47  47,646 rows, levels [ 3  6 24]
3,000 queries  38,009 bank   1,478 choose / 1,522 report

tokens + head in 16s
pooling                median km     <1km    <25km   <200km   <750km  <2500km
-----------------------------------------------------------------------------
p = 1       <- the mean     214.5    17.5%    32.9%    49.5%    70.0%    86.7%
p = 1    +head             174.1    18.3%    34.0%    51.5%    72.7%    88.8%
p = 1.5                    219.1    17.4%    32.7%    49.1%    70.0%    86.5%
p = 1.5  +head             174.7    18.2%    33.9%    51.6%    72.7%    88.5%
p = 2                      233.1    17.5%    32.6%    48.4%    69.6%    86.3%
p = 2    +head             173.7    18.1%    33.6%    51.8%    72.4%    88.7%
p = 3                      212.5    17.3%    33.0%    49.5%    70.6%    87.1%
p = 3    +head             174.7    18.4%    33.9%    51.8%    72.2%    89.0%
p = 4                      213.6    17.2%    32.9%    49.5%    71.1%    87.4%
p = 4    +head             176.2    18.0%    34.0%    51.5%    72.3%    89.2%
p = 6                      188.7    17.0%    33.0%    50.2%    71.0%    87.1%
p = 6    +head             174.1    17.8%    33.9%    51.7%    73.0%    89.1%
p = 10                     205.2    17.1%    32.8%    49.9%    71.2%    87.2%
p = 10   +head             174.7    17.8%    34.0%    51.6%    73.0%    88.8%

best p on <25 km: selection half 2, reporting half 3

--- against the mean (p=1), on the 1,522 held-out queries ---
p = 1.5        -0.13[-0.5,+0.2]~ -0.20[-0.9,+0.4]~ -0.46[-1.2,+0.3]~ +0.07[-0.6,+0.7]~ -0.13[-0.7,+0.3]~
p = 2          -0.07[-0.5,+0.3]~ -0.26[-1.0,+0.5]~ -1.18[-2.2,-0.3]  -0.39[-1.2,+0.5]~ -0.33[-0.9,+0.3]~
p = 3          -0.20[-0.7,+0.3]~ +0.20[-0.8,+1.2]~ +0.00[-1.2,+1.2]~ +0.59[-0.5,+1.7]~ +0.39[-0.4,+1.2]~
p = 4          -0.33[-1.1,+0.3]~ +0.07[-1.1,+1.2]~ +0.00[-1.4,+1.3]~ +1.12[-0.1,+2.4]~ +0.72[-0.1,+1.6]~
p = 6          -0.53[-1.3,+0.3]~ +0.20[-1.1,+1.4]~ +0.66[-0.9,+2.2]~ +0.99[-0.4,+2.4]~ +0.39[-0.7,+1.5]~
p = 10         -0.46[-1.2,+0.3]~ -0.07[-1.3,+1.2]~ +0.39[-1.3,+2.0]~ +1.25[-0.2,+2.7]~ +0.53[-0.5,+1.5]~
p = 2+head     +0.53[-0.2,+1.2]~ +0.72[-0.4,+1.8]~ +2.23[+0.9,+3.7]  +2.43[+1.1,+3.8]  +2.04[+0.9,+3.2] 

~ marks an interval spanning zero. 37s total
```
