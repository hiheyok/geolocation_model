# Arm names

## The problem

Tags accreted one experiment at a time and stopped carrying their own meaning:

| old tag | what it actually was |
|---|---|
| `s10_bal_bank25_c6` | 4608-d, 1.15M bank, 6 ep |
| `s10_pool_c6` | 1536-d, 1.15M bank, 6 ep |
| `s10_b55_c6` | 1536-d, 2.65M bank, 6 ep |
| `s10_w768_c4` | 768-d, 2.65M bank, 4 ep |

`s10_b55_c6` and `s10_w768_c4` differ in **three** dimensions at once and the
names said none of them. Worse, the bank was not recoverable from `s10_pool_c6`
at all — a pattern-matching decoder confidently labelled it a 0.40M bank when it
was trained on 1.15M, which is how a wrong number gets into a table.

## The scheme

```
d<width>-b<bank>-e<epochs>[-<split>]
```

* `width` — street vector dimension: `4608` dual, `1536` pooled, `768` projected
* `bank` — retrieval bank in hundredths of a million, zero-padded to three
  digits so it sorts: `115`, `190`, `265`, `350`
* `epochs` — cumulative, so a 2+2+2 ladder is `e2`, `e4`, `e6`
* `split` — omitted for `sequence`, the benchmark; `-cell8` otherwise

```
d1536-b265-e6          1536-d, 2.65M bank, 6 epochs, sequence
d768-b265-e6           the 768-d arm it is compared against
d1536-b350-e6-cell8    same width and epochs on the geographic holdout
```

Build one with `names.new_tag(width, bank_millions, epochs, split)`.

Two properties worth having: arms differing in one dimension differ in one
field, and an alphabetical listing groups by width, then bank, then epochs.

## The rename, and why old names still resolve

`scripts/rename_arms.py` renamed the 20 width/bank ladder checkpoints in place.
Checkpoints are gitignored, so no history was touched. The other ~70 on disk are
architecture ablations on a different axis — `ab_attn`, `mem_z4_seq`,
`retr_dual` — and a width/bank name would assert things about them that are not
true, so they keep their names.

The old names appear in every `BOOTSTRAP_*.md` and commit message already
written. `names.ALIASES` maps each onto its new name, so those stay readable.

`src/names.py` holds an **explicit** table, read off the runner that produced
each arm and never inferred from the string, and `bootstrap.py` prints a legend
above every results table:

| arm | what it is |
|---|---|
| `d1536-b265-e6` | 1536-d / 2.65M bank / 6 ep |
| `d768-b265-e6` | 768-d / 2.65M bank / 6 ep |

The separator is `/` rather than `|`, because these strings land inside markdown
cells and a pipe silently splits one column into four. An arm that cannot be
named honestly is left out of the legend rather than guessed at.

## Adding an arm

When a runner introduces a tag, add it to `KNOWN` in `src/names.py` in the same
commit. New-scheme tags decode without an entry; a missing entry shows up as an
arm absent from the legend rather than as a wrong label.
