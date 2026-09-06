# Tile Server

On-demand single-channel raster tiles for ML pipelines, rendered from a local
OpenMapTiles `.mbtiles` planet file.

Each tile is available three ways:

| Endpoint | What it is |
|---|---|
| `/tile/:z/:x/:y.png` | 1-channel photometric greyscale map image |
| `/tile/:z/:x/:y/mask.png` | 1-channel class-index segmentation mask (`?mode=class\|ambiguous\|raw`) |
| `/tile/:z/:x/:y/labels.json` | named features with pixel coordinates |

Rendering issues **zero network requests** — the bundled styles reference no
glyphs and no sprites, so output is deterministic and the service runs fully
offline.

---

## Wire contract with `src/tiles.py`

**This service lives in this repository because it has exactly one consumer and
that consumer hardcodes its wire format.** `src/tiles.py` declares:

```python
CLASS_STEP = 23          # ids are spaced 23 apart: 0, 23, ..., 253
N_CLASSES  = 12
TILE_PX    = 512
LEGAL_IDS  = frozenset(range(0, CLASS_STEP * N_CLASSES, CLASS_STEP))
```

Those three numbers are this renderer's output format, restated in Python. If
the renderer changes any of them and the client does not change in the same
commit, nothing raises: `check_legal()` still passes, `to_tokens()` still
returns a `(256, 12)` float array, and **every map token silently becomes a
different quantity**. Cached tokens and published results carry the old
meaning with no marker. Keeping both halves in one repository is what makes a
change to either visibly a change to both.

The service address is *not* part of this contract — that is deployment
wiring, and it lives in `config.TILE_SERVER` (default
`http://192.168.50.1:3000`), passed into `TileClient` as an argument.

### Endpoints the client actually calls

All are served by `server.js`; the full reference is under [API](#api) below.

| Endpoint | Used by | Contract the client relies on |
| --- | --- | --- |
| `GET /health` | `TileClient.health()` | JSON object. Fields are **additive-only** — the client reads them by name. |
| `GET /classes` | `TileClient.classes()` | `{name: id}` for all 12 classes. `?mode=ambiguous` adds `ambiguous: 255`. |
| `GET /tile/{z}/{x}/{y}/mask.png` | `TileClient.mask()` | **The load-bearing one.** Default `mode=class`. 512×512, single channel, PNG colour type 0. |
| `GET /tile/{z}/{x}/{y}.png` | `TileClient.image()`, `.png()` | Photometric greyscale. Ablation only — not part of the token path. |
| `GET /tile/{z}/{x}/{y}/labels.json` | `TileClient.labels()` | Named features with pixel coordinates. |

### The mask format

- **512 × 512, exactly.** `to_tokens()` raises unless `mask.shape == (512, 512)`.
- **Single channel, PNG colour type 0.** `TileClient.mask()` rejects anything
  Pillow does not open as mode `L`. This is why `lib/renderer.js` calls
  `.toColourspace('b-w')` before encoding: without it sharp promotes the single
  band to sRGB and writes colour type 2, which still *decodes* to the right
  values but hands the client `(H, W, 3)` instead of `(H, W)`. That call is
  contract, not cosmetics.
- **Every pixel is a legal class id**, i.e. a multiple of 23 in `0..253`.
  `check_legal()` rejects any pixel where `value % 23 != 0`.

| id | class | | id | class |
| --- | --- | --- | --- | --- |
| 0 | `background` | | 138 | `landuse` |
| 23 | `water` | | 161 | `building` |
| 46 | `ice` | | 184 | `railway` |
| 69 | `wood` | | 207 | `road_minor` |
| 92 | `grass` | | 230 | `road_major` |
| 115 | `farmland` | | 253 | `aeroway` |

`255` is reserved as `ambiguous` (the segmentation-convention `ignore_index`)
and is emitted **only** under `?mode=ambiguous`. It is not a class, it is not a
multiple of 23, and it will fail `check_legal()` — which is correct, because a
caller asking for ambiguity must handle it explicitly.

### A class-id mask must never be interpolated, only aggregated

This is the rule the whole format is built around, and it binds both sides.

The ids are *labels*, not intensities. The numeric midpoint of `water` (23) and
`wood` (69) is 46, which is `ice` — a class that appears nowhere near either.
So any operation that averages, blends, or resamples mask pixels can invent a
class that was never rendered. Concretely, this forbids bilinear/bicubic
resizing, JPEG or any lossy encoding, alpha blending, and mipmapping.

Each side upholds it separately:

- **Server.** MapLibre antialiases line geometry regardless of
  `fill-antialias: false`, so roughly one pixel in six along a road edge lands
  between two class values. `mode=class` therefore renders at
  `MASK_SUPERSAMPLE`× and reduces by **majority vote over legal values**
  (`downsampleMode` in `lib/renderer.js`) — a vote, never a mean. PNG is
  lossless, so the bytes the client decodes are the bytes that were voted.
- **Client.** `to_tokens()` aggregates into per-patch class *fractions* — a
  histogram, of which majority vote is the argmax. It never resizes the mask.

If you need a mask at another resolution, re-request it at the zoom you want.
Do not scale one you already have.

### Changing any of this

A change to the class table, tile size, class spacing, colour type, or the
`mode=class` resolution rule is a breaking change to `src/tiles.py` and to
every cached map token already derived from it. It must land as one commit
touching both sides, and it must move the `renderer.id` reported by `/health`
(see [Renderer identity](#renderer-identity)) so caches built before and after
are distinguishable.

## Renderer identity

`GET /health` reports a `renderer` object alongside the existing fields:

```jsonc
"renderer": {
  "id": "r-e42ea3b3bbf5",          // record this next to any cached tokens
  "commit": "a1b2c3d", "dirty": false,
  "sources": { "digest": "449bba4ade94be8f", "files": [ ... ] },
  "libs": { "maplibre_gl_native": "6.4.1",
            "maplibre_native_digest": "203c45b2c215",
            "vips": "8.17.3", "spng": "0.7.4", ... }
}
```

**Why this exists, given `cacheNamespaces` already looked like an identity.**
It is not one. `cacheNamespaces` hashes the *configuration* — style objects,
tile size, PNG level, supersample factor, and the `.mbtiles` basename and byte
count. Two deployments of this service (`192.168.50.1` and `10.0.0.84`) report
byte-identical `/health` metadata and byte-identical `cacheNamespaces`, off a
`.mbtiles` verified byte-identical by sha256 ([DATA.md](DATA.md)) — and their
masks still disagree on **0.03%–0.28% of pixels at every zoom from 0 to 16**.
Class tables match, value sets match, and every differing pixel sits on a
feature boundary (`23<->138`, `0<->230`): rasterisation drift between two
`maplibre-gl-native` prebuilds, not a data or configuration difference. At the
12-d per-patch fractions the model consumes it is max `7.8e-3`, mean `3.3e-4`
— benign in magnitude, but until now invisible.

So `renderer` fingerprints what `cacheNamespaces` structurally cannot: the
rendering *code* (a digest over the four source files on the mask path) and the
native libraries that rasterise and encode (versions **plus a hash of the
loaded `.node` binaries**, which is what actually separates those two servers —
same version string, different compiled rasteriser).

`id` is a function of the sources digest and the libraries only. `commit` and
`dirty` are reported for provenance but deliberately excluded from it: a commit
touching this README does not change a pixel, and an id that moved on such a
commit would invalidate caches for no reason. Both fields are `null` when the
service runs from a plain directory rather than a checkout — as both current
deployments do. See `lib/identity.js` for the full reasoning.

## Requirements

- Node.js 18+
- An OpenMapTiles-schema `.mbtiles` file (vector, `format=pbf`)

```bash
npm install
```

## Running

```bash
MBTILES_PATH=./maptiler-osm-2020-02-10-v3.11-planet.mbtiles node server.js
```

| Env var | Default | Description |
|---|---|---|
| `MBTILES_PATH` | `./planet.mbtiles` | Path to the .mbtiles file |
| `OUTPUT_DIR` | `./tiles` | Disk cache root |
| `PORT` | `3000` | HTTP port |
| `WORKERS` | `cpus - 2` | Render worker processes |
| `TILE_SIZE` | `512` | Output edge length in pixels |
| `PNG_LEVEL` | `1` | PNG compression level (0-9) |
| `MASK_SUPERSAMPLE` | `2` | Masks render at this multiple of the pixel ratio, then downsample by majority vote. `1` disables it |
| `MASK_AMBIGUOUS_THRESHOLD` | `0.5` | In `mode=ambiguous`, flag a pixel when its winning class holds no more than this share of the samples |
| `MAX_ZOOM` | `20` | Highest z accepted; above source maxzoom the renderer overzooms |
| `CACHE_TTL_MS` | `0` | Re-render after this many ms (0 = cache forever) |
| `CACHE_MAX_SIZE` | unset | Disk cache cap, e.g. `20GB`, `500MB`. Unset means unbounded growth |
| `CACHE_SWEEP_MS` | `30000` | How often the primary checks the cap and evicts |

## API

### `GET /tile/:z/:x/:y.png`

Single-channel greyscale PNG. The style paints every class as a pure grey
(`r == g == b`) spread across 0-255, so channel 0 *is* the value — extracted
with no luma conversion and no rounding.

```bash
curl -o tile.png http://localhost:3000/tile/14/8187/5448.png
```

Response headers include `X-Tile-Cached: 0|1` and `X-Tile-Channels: 1`.

### `GET /tile/:z/:x/:y/mask.png`

Single-channel mask where each pixel value is a semantic class id. `GET
/classes` returns the mapping:

```json
{"background":0,"water":23,"ice":46,"wood":69,"grass":92,
 "farmland":115,"landuse":138,"building":161,"railway":184,
 "road_minor":207,"road_major":230,"aeroway":253}
```

Note one asymmetry against the image palette: `waterway` is painted in
`image.png` but is **not** a mask class. Linear waterways therefore take
whatever polygon class lies beneath them in the mask, usually `background` or
`landuse`. Water *polygons* are class `water` in both.

MapLibre always antialiases line geometry, so roughly 17% of raw pixels land
between two class values. Snapping each to the nearest legal class is not
sufficient on its own: a blend of two classes either side of a third lands
exactly on that third class. Mixing `water` (23) and `wood` (69) yields 46,
which reads as `ice`.

Masks are therefore rendered at `MASK_SUPERSAMPLE`x the pixel ratio and reduced
by majority vote, so each output pixel is decided by several samples that are
mostly interior. Measured on `z14/8187/5448`, whose vector data contains only
`grass` and `sand` land cover — no ice whatsoever:

| Supersample | Render | Spurious `ice` | `railway` | `road_minor` |
|---|---|---|---|---|
| 1 (snap only) | 4.6 ms | 1326 (0.506%) | 6.58% | 13.81% |
| 2 (default) | 10.7 ms | 9 (0.003%) | 2.45% | 18.23% |
| 3 | 19.2 ms | 0 (0.000%) | 2.40% | 18.40% |

The error is systematic rather than cosmetic: at supersample 1 this tile
over-reports `railway` by a factor of nearly three and under-reports
`road_minor` by 4.4 points. Supersample 3 moves both by under 0.2 points against
2, so the default is close to converged. Output contains only legal class ids at
every setting — the difference is whether they are the right ones.

#### Mask modes

`?mode=` selects how boundary pixels are resolved. Each mode is cached
separately and reported back in an `X-Mask-Mode` response header.

| Mode | Every pixel labelled? | Render | Use |
|---|---|---|---|
| `class` (default) | yes | 10.9 ms | inference, evaluation, anything needing a complete labelling |
| `ambiguous` | no — split votes become `255` | 10.5 ms | training with `ignore_index=255` |
| `raw` | no — blends as rendered | 2.4 ms | doing your own edge handling |

`ambiguous` costs nothing extra: the vote already computes the sample counts, so
emitting `255` on a split instead of the winner is a branch, not a second
render. It is a strict refinement of `class` — every pixel it does not flag
carries exactly the value `class` would have given it.

```bash
curl -o mask.png "localhost:3000/tile/14/8187/5448/mask.png?mode=ambiguous"
curl -s "localhost:3000/classes?mode=ambiguous"   # includes "ambiguous": 255
```

255 is the conventional `ignore_index` (Cityscapes and PASCAL VOC both use it)
and sits clear of the class ids, which stop at 253. `?snap=0` is the original
spelling of `mode=raw` and still works.

Flagging by "not a legal value" would be both incomplete and far too broad. It
misses blends that land *exactly* on a legal id — the `ice` case above — and on
a dense urban tile it discards 17.3% of pixels when only 2.4% of those are
genuinely undecidable. Voting first cuts that to 2.65%, and spreads it evenly
rather than gutting the thinnest classes:

| Flagged share of each class | by "illegal value" | by split vote |
|---|---|---|
| `railway` | 41.0% | 5.0% |
| `road_minor` | 40.2% | 5.0% |
| `building` | 4.5% | 1.0% |
| `water` | 3.5% | 0.8% |

`road_minor` is only 2.3 px wide at z14 and `railway` 1.8 px, so for linear
classes the antialiased edge *is* most of the feature — discarding it removes
the boundary supervision a segmentation model most needs.

Residual boundary error is roughly half a pixel along class edges; for exact
boundaries, rasterise the polygons directly from the vector tile, where there is
no antialiasing at all.

### `GET /tile/:z/:x/:y/labels.json`

Named features from the same vector tile the image was rendered from, in pixel
coordinates relative to the tile's top-left corner.

```bash
curl "http://localhost:3000/tile/14/8187/5448/labels.json?layers=place&limit=3"
```

```json
{
  "z": 14, "x": 8187, "y": 5448, "size": 512, "count": 3,
  "labels": [
    {"layer":"place","text":"Borough","class":"suburb","rank":11,
     "geometry":"point","x":374.3,"y":242.3}
  ]
}
```

| Query param | Description |
|---|---|
| `layers` | Comma-separated subset of `place`, `water_name`, `transportation_name`, `poi`, `mountain_peak`, `park`, `aerodrome_label` |
| `lang` | Language for `name:<lang>`, falling back to `name` (default `latin`) |
| `limit` | Keep the N most prominent features by rank |

Point features report their anchor; lines and polygons report the midpoint of
their geometry. A renderer places road labels repeatedly along a line, so
treat line anchors as approximate. Above the source maxzoom the parent tile is
scaled and offset so coordinates stay relative to the requested tile.

### `GET /classes`, `GET /health`

Class mapping and worker/source status.

## Performance

Measured on a 10-core machine, 8 workers, dense z14 tiles over London:

| | |
|---|---|
| Image render, per worker | ~7 ms/tile median (4-17 ms, varies with feature density) |
| Mask render, per worker | ~14 ms/tile median at supersample 2 |
| Image throughput, 8 workers | ~1,350 tiles/s cold |
| Mask throughput, 8 workers | ~610 tiles/s cold |
| Cache hits | 12,000-18,000 tiles/s |
| Payloads | image 15-131 KB, mask 16-81 KB, labels 1-127 KB |

Three choices account for most of that. Dropping symbol layers is worth ~65% of
render time, since text layout and collision detection dominate a styled
render. Extracting channel 0 rather than converting to greyscale is both faster
and smaller. PNG level 1 costs ~1.3 ms against ~2.7 ms at level 6 for ~50% more
bytes.

Output is PNG colour type 0 (true greyscale), so it decodes to `(H, W)` rather
than `(H, W, 3)`. This needs `toColourspace('b-w')` at encode time — sharp
otherwise promotes a single band to sRGB and writes RGB with `r == g == b`,
which carries the same values in triple the channels and ~40% more bytes.

Renders are serialised per process because maplibre-gl-native is not
thread-safe; parallelism comes from the worker pool. Concurrent requests for the
same cold tile are coalesced *within a worker*, so N simultaneous callers cost at
most one render per worker rather than N. Cache writes are atomic (temp file plus
rename), so a reader never sees a partial tile even when two workers render the
same address.

## Output layout

```
tiles/
  image-a54b3a89/14/8187/5448.png
  mask-87addb24/14/8187/5448.png
  mask_ambiguous-61a2e49b/14/8187/5448.png
  mask_raw-fca59587/14/8187/5448.png
```

Each mask mode is a separate namespace, so they never collide. Labels are not
cached to disk — they are re-decoded from the vector tile each request, which is
~5 ms.

### Configuration is part of the cache key

A tile address alone does not identify an artefact: the same `z/x/y` renders
differently under a different style, tile size, PNG level, or supersample
setting. Each namespace carries a short hash of exactly the inputs that affect
it, so a config change lands in a fresh namespace instead of serving output from
a configuration you have since changed.

The hash covers the built style object, so editing a colour or adding a layer in
`styles.js` rotates it automatically. Inputs are per artefact, so a change only
invalidates what it actually affects:

| Change | image | mask | mask_ambiguous | mask_raw |
|---|---|---|---|---|
| `TILE_SIZE` | new | new | new | new |
| `PNG_LEVEL` | new | new | new | new |
| `MASK_SUPERSAMPLE` | kept | new | new | kept |
| `MASK_AMBIGUOUS_THRESHOLD` | kept | kept | new | kept |
| source `.mbtiles` | new | new | new | new |
| mask palette in `styles.js` | kept | new | new | new |

Superseded namespaces are **reported at startup, not deleted** — the output in
them is still valid, and flipping a setting back should find its cache intact:

```
Cache ns: image-a54b3a89  mask-e2c989f6  mask_ambiguous-3722bd13  mask_raw-fca59587
cache: 1 namespace(s) from a previous configuration will never be read again: mask-87addb24
```

With `CACHE_MAX_SIZE` set they age out on their own, since nothing reads them
and eviction is LRU.

### Size cap and eviction

Set `CACHE_MAX_SIZE` to bound the cache. Without it the cache grows forever: at
~62 KB per artefact that is ~100 GB per million tiles.

Eviction is **LRU on reads, not writes** — a tile written long ago but requested
constantly is kept, while one written recently and never re-read is evicted
first. When the cap is exceeded the reaper sweeps down to 90% of it, so a full
cache does not evict on every write.

All eviction happens in the cluster primary. Workers share one directory, so if
each evicted independently they would double-count the total and delete entries
the others had just written. Workers instead report writes and hits to the
primary, batched once a second — cache hits run at over 12,000/s, so one IPC
message per hit would cost more than the read it describes.

Recency is tracked from those reports rather than from `atime`. `relatime`, the
Linux default, only advances `atime` once a day, and `noatime` never does; either
would silently turn LRU into "least recently written". `atime` is used only to
seed files already on disk when the server starts.

`GET /health` reports the current state, alongside `cacheNamespaces`:

```json
{"files":81,"bytes":5657143,"size":"5.4 MB","maxSize":"6.0 MB",
 "usedPercent":89.9,"evictedTotal":263,"scanned":true}
```

In cluster mode this is the primary's last broadcast, so it lags by up to one
sweep interval. With no cap configured, no tracking is installed at all and the
cache behaves exactly as it did before.

## Notes

- The MBTiles Y coordinate is converted from XYZ to TMS before lookup; tile
  data is gunzipped automatically.
- Absent tiles (ocean, outside coverage) render as empty rather than failing.
- Low-zoom tiles (z0-z6) are far slower than z14 because they cover enormous
  areas; if your sampling spans them, seed them once rather than on demand.
