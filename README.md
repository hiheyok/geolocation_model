# Map-guided visual geolocation agent

Locate a street photograph on Earth by navigating a map, not by classifying it.

The agent starts at zoom 0 — the whole world in one tile — and takes four steps.
At each step it sees the street photo and a 16×16 grid of map tokens for its
current tile, picks one of the 256 children, and descends. Four steps reach
zoom 16, a 611 m cell, where a click head regresses the final lat/lon.

The recursion is the whole idea: **a 16×16 grid and XYZ map tiles are the same
object.** 16 = 2⁴, so one step of the agent is exactly four zoom levels, the path
is a base-16 address, and descending is a digit append:

```
x_{t+1} = x_t·16 + (a_t % 16)
y_{t+1} = y_t·16 + (a_t // 16)
```

That factorisation is what makes 1 km global resolution tractable: a flat
classifier would need a 2³² -way softmax, while the agent addresses the same
leaves with 4 × 256 = 1,024 logits and regresses inside a 611 m cell instead of
a 156 km one.

## Results

Trained on OSV-5M. Test split, 5,000 images, beam k=2, ranked on steps 0–2.

| release | training images | retrieval bank | median km | mean km | `<25 km` |
|---|---:|---:|---:|---:|---:|
| s01 — 1 shard | 40,079 | 40,079 | 242.7 | — | 20.1% |
| s10 — 10 shards | 400,000 | 400,180 | **81.6** | 779 | **35.6%** |

**The headline finding is that almost none of that came from the training set.**
Holding the training images fixed at 25,000 and varying only the retrieval bank:

| training images | bank | median km | `<25 km` |
|---:|---:|---:|---:|
| 25,000 | 400,180 | 115.8 | 32.5% |
| 25,000 | 25,000 | 465.1 | 13.1% |

Scaling the bank 25k → 400k is worth **+19.4 pp** (paired 95% CI [+18.1, +20.7]);
scaling the training set over the same range at a fixed bank is worth **+3.1 pp**.
The non-parametric memory is worth roughly six times the parametric data — so
data binds as *retrieval corpus*, not as gradient signal, and the corpus is the
cheaper axis: a bank entry costs one embedding forward pass, a training example
costs optimizer steps.

Two further results:

- **Epochs do not substitute for data.** 50k images × 80 epochs peaks at
  **epoch 3 of 80** and decays to 29.4% by the end, val loss 12.7 → 39.5. It ties
  50k × 19 inside noise: 61 extra epochs bought nothing measurable.
- **The mean is 30× the median because 95% of it is step 0.** A z4 cell is
  2,504 km across, so one wrong first digit costs thousands of kilometres by
  construction. 24.5% of images miss step 0 and average 6,386 km; 6.3% land over
  10,000 km away and carry half of all error mass.

Full write-up, including the seven scale-dependent defects the 10× release
exposed, is in [`runs/REPORT.md`](runs/REPORT.md).

### These numbers are not the OSV-5M leaderboard

The [OSV-5M benchmark](https://osv5m.github.io/) holds test points out by 1 km of
physical distance and keeps one image per capture sequence, so that a model
"cannot simply rely on memorizing places". **The split here only guarantees that
a sequence never spans train/test** — 10.2% of val sits within 1 km of a training
image. Given the control above, that difference is likely to matter a lot here
specifically, because this system is built to exploit exactly what their split
removes.

For an external comparison use the `cell8` stress split, which holds out whole z8
cells; at 50k scale it cost 1.6× the median error. It has not been run at s10.

## Layout

```
src/
  tile_math.py   addressing: tile_for, path_to_xyz, target_actions, descend.
                 Pure integer math, no I/O, the one source of truth.
  tiles.py       the only module that speaks HTTP to the tile server
  splits.py      the one owner of what train/val/test mean
  dataset.py     memmap rows -> tensors; never touches the network
  encoders.py    street projection, map tokenizer, state encoder
  model.py       GeoAgent: fusion, cross-attention policy, click head
  retrieval.py   content-keyed retrieval prior over child cells
  beam.py        rollout with live tile fetches
  train.py       teacher-forced training
  evaluate.py    metrics, split-contamination guards
  baselines.py   centroid, kNN, flat classifier, oracle path
scripts/         dataset build, embedding, tile fetch, kNN bank, experiment runners
tests/           property tests for the addressing round-trip
```

Three data paths, deliberately separate: offline prep hits the tile server once
and fills `cache/`; **training reads only memmaps and parquet**; inference fetches
arbitrary z12/z16 tiles live, since no cache can cover them.

## Setup

```bash
pip install -r requirements.txt
export TILE_SERVER=http://192.168.50.1:3000     # XYZ tile server
export OSV_ROOT=/path/to/osv5m                  # OSV-5M with images/train/*.zip
```

Python 3.13, one CUDA GPU. Everything here was measured on an RTX 3070 (8 GB),
which is what several of the design choices are for.

### Build a release

A **release** is the set of shards everything downstream derives from. The
parquet, the embeddings, the token cache and the kNN bank are all indexed by row
order in `dataset.parquet`, so adding shards invalidates all of them at once —
hence `OSV_RELEASE`, which keeps generations side by side. `s01` is 1 shard
(50k images), `s10` is 10 (500k). Checkpoints record theirs and `src/evaluate.py`
refuses a cross-release comparison.

```bash
export OSV_RELEASE=s10
python scripts/build_dataset.py --shard 00,01,02,03,04,05,06,07,08,09
python scripts/embed_street.py --model vit_base_patch14_dinov2.lvd142m --crops 3 --out embeddings_c3
python scripts/embed_street.py --model vit_base_patch16_siglip_224.v2_webli --crops 3 --out siglip_c3
python scripts/concat_street.py --a embeddings_c3 --b siglip_c3 --out dual_c3
python scripts/fetch_tiles.py --exhaustive-z8 --seed-from cache/map/s01
python scripts/build_knn.py --street-file dual_c3.f16.npy --split-mode sequence
```

About two hours on the reference machine; the two embedding passes are ~100 min
of it.

### Train and evaluate

```bash
python src/train.py --tag myrun --epochs 2 \
  --street-file dual_c3.f16.npy --pool attn --pool-q 4 --pos both \
  --map-layers 1 --neg 4 --retr --retr-k 16 --retr-mode dual --d-key 128

python src/evaluate.py --tag myrun --split test --n 5000 --ks 2 --score-steps 3
python scripts/bootstrap.py --tags myrun,other --split test
```

`--n` draws a **seeded random** subset, not the first n rows. It used to take the
first n, and split order is the order shards were joined, so that sample was
geographic rather than random — worth 34 km of median error on the s10 test
split. The seed is fixed, so every arm still sees the same images and paired
comparisons stay valid.

Two defaults are load-bearing and were both learned the hard way:

- **`--select hit`** keeps the epoch with the best `<25 km` rate. The median of a
  heavy-tailed great-circle error swings by hundreds of km between adjacent
  epochs, so selecting on it selects on noise.
- **`--score-steps 3`** ranks beams on steps 0–2 only. Steps 2 and 3 supply most
  of a path's score from distributions that are 5.9% and 2.6% accurate, and
  their spread swamps the signal that decides where the answer actually is.

**Always report a paired bootstrap interval beside a median difference.** The
median carries a ~16 km 95% interval at n ≈ 5,000, and two runs of the same
configuration have landed 18 km apart — so single-seed median differences under
~20 km are not claims. `scripts/bootstrap.py` caches per-image errors, so the
test costs seconds.

### Scale the retrieval bank

Given the result above, the cheap axis is the corpus. Bank-only shards never
enter `dataset.parquet`, so the split hash and every existing checkpoint are
untouched — and because a bank entry votes with its z16 address, which is
arithmetic on lat/lon, **no tiles are fetched for them**.

```bash
python scripts/build_bank_ext.py --shards 10,11,12,13,14 --out bank_ext
python scripts/embed_street.py --parquet bank_ext.parquet --model vit_base_patch14_dinov2.lvd142m --crops 3 --out bank_ext_dino
python scripts/embed_street.py --parquet bank_ext.parquet --model vit_base_patch16_siglip_224.v2_webli --crops 3 --out bank_ext_siglip
python scripts/concat_street.py --a bank_ext_dino --b bank_ext_siglip --out bank_ext_dual
python scripts/stack_bank.py --base dual_c3 --ext bank_ext_dual --out dual_c3_bank
python scripts/build_knn.py --street-file dual_c3_bank.f16.npy --split-mode sequence --bank-ext bank_ext
```

The stacked file keeps release rows first, so row *i* is still image *i* and a
neighbour index addresses both corpora. Train against it by passing
`--street-file dual_c3_bank.f16.npy`; the kNN filename is derived from that, so
it does not need to be given.

To go the other way and *shrink* the bank — the control that separates corpus
from training set — use `--bank-limit N`, which restricts the bank to exactly
the images `train.py --limit N` trains on.

### Analysis

| script | answers |
|---|---|
| `scripts/bootstrap.py` | is this difference real? paired CIs on median and `<25 km` |
| `scripts/summary_table.py` | per-step accuracy and loss, train vs test |
| `scripts/error_profile.py` | which step's failures the mean is made of |
| `scripts/standard_metrics.py` | GeoScore and the 1/25/200/750/2500 km recalls |

### Experiment runners

`scripts/overnight.py` and `scripts/bank25.py` run multi-stage experiments under
a deadline: per-stage done-markers, retries, a watchdog that restarts after a
crash, and system-utilisation sampling that records shared GPU memory alongside
utilisation — on Windows an over-committed process does not OOM, it spills to
system DRAM and keeps reporting 100%.

```bash
powershell -ExecutionPolicy Bypass -File scripts/overnight.ps1 -Script scripts/bank25.py -Hours 6
```

## Web demo

`scripts/serve.py` puts the agent behind a page: drop in a photograph of any
size — drag it, pick it, or just paste it — and it answers with a lat/lon — and, more usefully, with **the descent**.
The four map tiles the agent actually looked at are rendered in order, each with
the cell it chose, so the answer arrives with its own explanation rather than as
a pin.

```bash
export OSV_RELEASE=s10
python scripts/serve.py --tag s10_n400k_bank25     # 10.6 GB bank, best
python scripts/serve.py --tag s10_n400k_e2         # 3.7 GB bank, starts faster
```

Then open <http://127.0.0.1:8000>. `--host 0.0.0.0` exposes it on the LAN and
`--port` moves it; `--bank-gpu` holds the bank in VRAM, which is only sane for
the smaller one on an 8 GB card.

Startup loads both encoders, the policy, and **the bank the checkpoint was
actually trained against** — read from the kNN cache the checkpoint names, so a
`bank25` tag brings up all 1,150,180 images and a plain one brings up 400,180.
That takes a couple of minutes and roughly 11 GB of host RAM for the large bank.
A request then costs one encoder pass, one matmul against the bank, and four
tile fetches: **~0.7 s end to end** on the reference machine.

| endpoint | does |
|---|---|
| `GET /` | the page |
| `POST /locate` | raw image bytes in, JSON out: lat/lon, confidence radius, path, the four tiles, the nearest bank images |
| `GET /map/{z}/{x}/{y}.png` | proxies the tile server, so the browser never needs to reach it |

Two things worth knowing when reading its output. The **confidence radius** is
the spread of the beam's candidates, and it ranks error at ρ ≈ 0.55 — useful for
sorting, not a calibrated distance. And a neighbour listed as *bank-only shard*
has no lat/lon to show: bank-extension images carry a z16 address and nothing
else, which is exactly why they are cheap.

## Documents

`project_plan.pdf` is the original 29-page plan, written before the map backend
existed. It specifies a FastAPI + Playwright screenshot service; what got built
is an XYZ tile server returning segmentation masks, which is better and collapsed
a quarter of the plan. Where the two disagree, the code and `runs/REPORT.md` are
current.

`tile-server-documentation.pdf` documents the live server that `src/tiles.py`
talks to.
