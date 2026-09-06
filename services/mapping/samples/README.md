# Samples

Six tiles chosen to cover different terrain, zoom levels, and edge cases. Each
folder holds exactly what the API returns for one tile, plus a `meta.json`
describing it.

| Folder | Tile | What it shows |
|---|---|---|
| `00-world-z0` | z0/0/0 | The entire world in one tile. 62% water, 20% ice, 46 labels |
| `01-urban-dense` | z14/8187/5448 | Southwark, London. Dense buildings, full road hierarchy, 1122 labels |
| `02-water-coast` | z13/4103/2725 | Thames estuary. 33% water, coastline edges |
| `03-park-green` | z14/8184/5448 | Hyde Park. Large green polygons against built-up landuse |
| `04-rural` | z13/4061/2698 | Warwickshire. 69% farmland, sparse buildings, only 8 labels |
| `05-low-zoom` | z10/511/340 | Greater London. Whole-city context, little building detail |
| `06-overzoom-z16` | z16/32750/21793 | Above the source maxzoom of 14 — renderer overzooms a z14 parent |

## What's in each folder

```
01-urban-dense/
  image.png            512x512, 1 channel  — photometric greyscale
  mask.png             512x512, 1 channel  — class-index segmentation mask
  mask_ambiguous.png   512x512, 1 channel  — same, with contested edges left as 255
  labels.json                              — named features with pixel coords
  meta.json                                — tile address, URLs, class breakdown
```

### `image.png` — the model input

Single channel. Pixel values are cartographic brightness, evenly spaced across
0–255 so no two classes collapse into each other:

```
background 255   grass      163   road_major  71
ice        232   wood       140   railway     48
farmland   209   road_minor 117   waterway    25
landuse    186   building    94   water        0
```

These are the values a feature is *painted*, not class ids — overlapping
geometry means a pixel's value is whatever was drawn last. Use `mask.png` when
you need the class.

### `mask.png` — the label map

Single channel, where every pixel value **is** a class id:

```
background   0   grass     92   railway    184
water       23   farmland 115   road_minor 207
ice         46   landuse  138   road_major 230
wood        69   building 161   aeroway    253
```

Guaranteed to contain only these values. Masks render at 2x pixel ratio and
downsample by majority vote, which matters more than it sounds: plain snapping
turns a blend of `water` (23) and `wood` (69) into 46, which reads as `ice`.
On `01-urban-dense` — a tile whose land cover is only grass and sand — that
produced 1326 spurious ice pixels and inflated `railway` from 2.45% to 6.58%
while deflating `road_minor` from 18.23% to 13.81%. Supersampling cuts the
spurious ice to 9 pixels (0.003%).

`GET /classes` returns the same mapping at runtime, and
`GET /classes?mode=ambiguous` adds the reserved value `255`, which
`mask.png?mode=ambiguous` emits where the vote is split. `?mode=raw` returns
unresolved values instead, which include blends between classes (~17% of pixels
on a dense urban tile, 2.6% on a rural one).

### `mask_ambiguous.png` — the same map, minus the guesses

Identical to `mask.png` except where the majority vote was split, which carries
**255** instead of a winning class. That is the conventional `ignore_index`, so
it drops straight into `CrossEntropyLoss(ignore_index=255)`.

It is a *strict refinement*: every pixel it does not flag holds exactly the
value `mask.png` holds. Verified across all seven samples — zero unflagged
pixels differ.

How much gets flagged varies with how much geometry a tile carries:

```
00-world-z0    0.00%      03-park-green    1.91%
04-rural       0.37%      01-urban-dense   2.65%
06-overzoom    0.68%      05-low-zoom      3.11%
02-water-coast 1.09%
```

The flagged share of a class tracks how *present* that class is. Anything
holding a real share of the tile loses very little; classes that appear only as
scattered slivers are flagged almost entirely:

```
01-urban-dense        share of tile   flagged
  building               30.51%          1.0%
  road_minor             18.23%          5.0%
  railway                 2.45%          5.0%
  farmland                0.01%         94.6%
  ice                     0.00%         88.9%
```

That is the intended behaviour, and it is the same mechanism that suppresses
the spurious `ice`: a class the renderer barely commits to gets withheld rather
than asserted. The flip side is that at low zoom, where buildings and minor
roads survive only as slivers, `05-low-zoom` flags 94.8% of `building` and 91%
of `road_minor` — if you train on low-zoom tiles, use `mask.png` or expect
almost no supervision for those classes.

`meta.json` records all of this per sample under `ambiguous`.

### `labels.json` — vector annotations

```json
{
  "z": 14, "x": 8187, "y": 5448, "size": 512, "count": 1122,
  "labels": [
    {
      "layer": "place",
      "text": "Borough",
      "class": "suburb",
      "rank": 11,
      "geometry": "point",
      "x": 374.3,
      "y": 242.3
    }
  ]
}
```

`x` and `y` are **pixels from the tile's top-left corner**, in the same frame
as `image.png` and `mask.png`. Sorted by `rank` ascending, so the most
prominent features come first — `?limit=N` keeps the top N.

Point features report their own anchor. Lines and polygons report the midpoint
of their geometry, which is approximate: a renderer repeats road labels along a
line rather than placing one.

Note that these are *all* named features present in the data, not just the ones
a cartographic style would draw. `01-urban-dense` has 1122 of them; a
conventional style would render around 25.

## Requests that produced these

```bash
curl -o image.png   http://localhost:3000/tile/14/8187/5448.png
curl -o mask.png    http://localhost:3000/tile/14/8187/5448/mask.png
curl -o labels.json http://localhost:3000/tile/14/8187/5448/labels.json
```

Useful query params:

```bash
# only place names, top 10 by prominence
".../labels.json?layers=place&limit=10"

# Japanese names where available
".../labels.json?lang=ja"

# unsnapped mask, for your own edge handling
".../mask.png?mode=raw"

# boundary pixels flagged 255 instead of guessed, for ignore_index training
".../mask.png?mode=ambiguous"
```

`mode=class` is the default and labels every pixel. `mode=ambiguous` is a strict
refinement of it: pixels it does not flag carry exactly the same value. On
`01-urban-dense` it flags 2.65% of the tile.

## Loading them

`load_sample.py` prints the shapes, class distribution, and label positions for
a sample folder:

```bash
python3 samples/load_sample.py samples/01-urban-dense
```

## Regenerating

```bash
node server.js &
node samples/generate.js http://localhost:3000
```

Edit the `SAMPLES` array in `generate.js` to add locations — it takes lat/lng
and zoom and works out the tile address.
