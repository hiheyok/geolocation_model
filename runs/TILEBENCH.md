# What the extension tile pass should run as

`tile_cache.py` is about to embed extension rows in bulk, so its per-image cost
was worth an hour of measurement first. Three levers: one faster than I
expected, one slower, and all three rejected for the same reason.

    OSV_RELEASE=s10 py scripts/tilebench.py --n 2000 --max-shards 2

2,000 release images, 3x2 tiles, batch 32 = 192 tile forwards per batch. Tiles
are decoded once up front, so these numbers are the GPU alone.

| mode | pass | img/s | tile/s | VRAM | vs autocast |
|---|---|---|---|---|---|
| autocast (ships) | 35.3 s | 56.7 | 340.2 | 2.35 GB | -- |
| bf16 weights | 30.9 s | 64.7 | 388.0 | **1.59 GB** | 1.141x |
| autocast + compile | 29.8 s | 67.0 | 402.3 | 2.10 GB | 1.183x |
| bf16 + compile | 28.9 s | **69.1** | 414.5 | 1.76 GB | **1.218x** |

## torch.compile does run here -- a claim of mine corrected within the hour

My first pass reported `RuntimeError: Cannot find a working triton
installation` and I wrote that inductor's GPU backend "has no Windows build",
so the lever `docs/STATE.md` §8i had been holding open was untestable. **The
user pointed out the `triton-windows` fork.** It installs cleanly and works:

    py -m pip install "triton-windows>=3.2,<3.3"    # 3.2.x pairs with torch 2.6

It needs MSVC on `PATH` at compile time (VS 2022 Community 14.44 and CUDA 12.6
are both already on this machine). What I actually verified was that the
default `pip`-installed environment had no Triton; what I wrote was that none
exists for the platform. Those are different claims and only the first was
measured.

Inductor also logs `Not enough SMs to use max_autotune_gemm mode` -- the 3070
is below the threshold for its best GEMM path, which is why compile returns
1.18x rather than the larger numbers reported on bigger cards.

`autocast+compile` is in the table because the switches compose and that is the
cell that matters: it is the only compiled variant whose arithmetic stays close
to the autocast `tile6` already on disk. Measuring `compile` only on top of
bf16 weights would have confounded the two.

The benchmark treats a mode that cannot run as a result rather than a crash.
The first version let the Triton failure take the whole process down and lose
the agreement section, which is the half that decides anything.

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
| autocast + compile | 0.999840 | 0.999931 | **1.00%** |
| bf16 + compile | 0.999698 | 0.999947 | **1.00%** |

The cosine is reassuring and not sufficient. Retrieval only cares whether the
*ranking* survives, and embedding the bank one way and the query the other
moves the nearest neighbour for **0.8-1.0% of queries** -- above the 0.5%
tripwire, which was written into the script before any of these numbers were
known.

**Every faster variant fails it, and `compile` fails it worse than bf16** --
including `autocast+compile`, which was added precisely because it keeps the
arithmetic closest to the incumbent. So the seam argument does not rest on one
measurement, and there is no fast-and-faithful cell to reach for.

Two honest caveats, in opposite directions. A crossing is not automatically an
error: near-tied neighbours are often near each other, so some fraction of
those 0.8% cost nothing in km. And 0.80% is measured against a 1,000-row bank,
which is not obviously the rate at 3.4M -- a denser bank has both more decisive
winners and far more chances for a near-tie to flip, and I do not know which
dominates. Claiming a 3.4M rate from a 1,000-row sample is the exact error
shape §6 keeps catching, so it is not claimed.

That ambiguity is the argument, not against it. The whole speed ceiling is
**1.218x** -- about 3 h off a 16 h pass, less off the 4 h one now planned --
against a permanent, undetectable numeric seam that would sit under every
number measured afterwards. **Extend `tile6` the way it was built: autocast.**

**This is a property of *extending*, not of the flags.** For a fresh full
rebuild, where release and extension are embedded together and no seam exists,
`bf16+compile` is legitimate and free: 1.218x, and 1.76 GB against 2.35 GB.
Worth taking whenever the whole cache is rewritten at once.

## The decode path is not starving the GPU either

Worth checking, since the pass averaged 43.5 img/s while the card alone does
56.6. It is not the decode threads. From `tile_cache_full.log`, the marginal
time between consecutive 2,000-row progress lines is 36-37 s per 1,984 images:

    steady state  ~54 img/s      against a 56.6 img/s GPU ceiling -- 95%

The gap between 54 and the 43.5 average is **shard preloads**, not starvation:
each 2.5 GB shard is slurped whole at ~85 MB/s before its images are read. So
`--workers` is not a lever either; four threads already keep the card fed.

## Estimate

At the ~54 img/s steady rate, autocast:

| target | rows | pass | tile cache |
|---|---|---|---|
| **`bank_ext.parquet` only** (planned) | 750,000 | **~4.0 h** | 13.8 GB |
| all four extensions | 3,000,000 | ~16 h | 55.3 GB |

Plus ~30 s per shard preloaded. The planned target gives a **complete**
1,150,180-row bank: 400,180 release train rows, already tiled in `tile6`, plus
750,000 extension rows.

**Completeness is the constraint, not size.** A partly-tiled bank is not a
smaller version of a tiled one -- it is a broken one. Tiled rows are
`nrm((L0+L1)/2)` and untiled rows are `L0`, and one cosine ranks both, so for
an untiled row the query's `L1` half scores against the bank's `L0`:
cross-space, and systematically lower. Untiled rows are silently demoted. That
is why the extension is taken one whole parquet at a time and why "tile 25% and
measure" is not an option.

Run one fixed-size cache per extension parquet
(`--parquet bank_ext2.parquet --out tile6_ext2 --n 750000`) rather than one
growing file. `tile_cache`'s resume path grows a cache by writing a full-size
temporary copy beside it, which at one 55 GB file would peak at 110 GB; four
fixed-size files never grow and never pay that.
