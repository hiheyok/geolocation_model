# What the 16 h tile pass should run as

`tile_cache.py` is about to embed 3,000,000 extension rows, so its per-image
cost was worth an hour of measurement first. Three levers, two of them dead.

    OSV_RELEASE=s10 py scripts/tilebench.py --n 2000 --max-shards 2

2,000 release images, 3x2 tiles, batch 32 = 192 tile forwards per batch. Tiles
are decoded once up front, so these numbers are the GPU alone.

| mode | pass | img/s | tile/s | VRAM |
|---|---|---|---|---|
| autocast (ships) | 35.3 s | 56.6 | 339.7 | 2.35 GB |
| bf16 weights | 30.9 s | 64.7 | 388.0 | **1.59 GB** |
| `torch.compile` | -- | -- | -- | -- |

## torch.compile is not available here

`RuntimeError: Cannot find a working triton installation` -- inductor's GPU
backend needs Triton, which has no Windows build. This was recorded in
`docs/STATE.md` §8i as "the one untested lever" for getting the card past
220 W of its 300 W limit. It is not a lever on this machine; it is untestable.
The benchmark now treats a mode that cannot run as a result rather than a
crash, because the first version took the whole run down with it and lost the
agreement measurement, which is the half that decides anything.

## bf16 weights: 1.14x, not the 1.36x I carried over

I measured 1.36x for `.to(torch.bfloat16)` against autocast earlier this
session, on `resmatch`'s native-resolution pass, and assumed it would carry.
**It does not.** At 224 with 192 tile forwards per batch the card is much
closer to compute-saturated, so the per-op weight recast that autocast pays is
a smaller share of the time. Same fix, same hardware, 1.14x instead of 1.36x --
the speedup is a property of the workload, not of the flag.

## And it is rejected anyway, on agreement

`tile6`'s 500,000 release rows were embedded under autocast. If the extension
is embedded any other way, one bank holds two numerically different halves and
the retrieval that consumes it is a cosine across the boundary.

| arm | min cos | mean cos | top-1 changed |
|---|---|---|---|
| autocast | 1.000000 | 1.000000 | baseline |
| bf16 weights | 0.999738 | 0.999946 | **0.80%** |

The cosine is reassuring and not sufficient. Retrieval only cares whether the
*ranking* survives, and embedding the bank one way and the query the other
moves the nearest neighbour for **0.80% of queries** -- above the 0.5%
tripwire, which was written into the script before the number was known.

Two honest caveats, in opposite directions. A crossing is not automatically an
error: near-tied neighbours are often near each other, so some fraction of
those 0.8% cost nothing in km. And 0.80% is measured against a 1,000-row bank,
which is not obviously the rate at 3.4M -- a denser bank has both more decisive
winners and far more chances for a near-tie to flip, and I do not know which
dominates. Claiming a 3.4M rate from a 1,000-row sample is the exact error
shape §6 keeps catching, so it is not claimed.

That ambiguity is the argument, not against it. The saving is **~2 h out of
~16**; the cost is a permanent, undetectable numeric seam inside a 3.4M bank
that would sit under every number measured afterwards. **Embed the extension
the way `tile6` was: autocast.**

## The decode path is not starving the GPU either

Worth checking, since the pass averaged 43.5 img/s while the card alone does
56.6. It is not the decode threads. From `tile_cache_full.log`, the marginal
time between consecutive 2,000-row progress lines is 36-37 s per 1,984 images:

    steady state  ~54 img/s      against a 56.6 img/s GPU ceiling -- 95%

The gap between 54 and the 43.5 average is **shard preloads**, not starvation:
each 2.5 GB shard is slurped whole at ~85 MB/s before its images are read. So
`--workers` is not a lever either; four threads already keep the card fed.

## Estimate for the extension

    3,000,000 / 54 img/s          15.4 h
    + 60 shard preloads @ ~30 s    0.5 h
    ------------------------------------
                                  ~16 h

Storage is 4 x 13.8 GB = 55.3 GB, run as one cache per extension parquet
(`--parquet bank_ext2.parquet --out tile6_ext2 --n 750000`) rather than one
growing file. `tile_cache`'s resume path grows a cache by writing a full-size
temporary copy beside it, which at one 55 GB file would peak at 110 GB; four
fixed-size files never grow and never pay that.
