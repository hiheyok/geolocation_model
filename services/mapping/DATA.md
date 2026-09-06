# Source data

The renderer reads exactly one input: a planet-scale OpenMapTiles vector
extract. It is not in this repository and must never be committed — it is
~70 GiB, and it is an upstream artefact we did not produce. `.gitignore`
excludes `*.mbtiles` for that reason.

Its identity is recorded here instead, so a truncated download, a partially
written copy, or a different extract dropped in under the same filename is
detectable without the file being in the tree.

## maptiler-osm-2020-02-10-v3.11-planet.mbtiles

| field  | value |
| ------ | ----- |
| bytes  | `75031430656` (69.88 GiB) |
| sha256 | `b36dccf05bf61d5295822609cc515ab58f1dde92d519030516a5f165e6342368` |
| format | MBTiles 1.3, gzipped Mapbox Vector Tiles, OpenMapTiles schema v3.11 |
| zooms  | 0–14 (`/health` reports this as `sourceZoom`) |
| source | MapTiler OSM planet extract, 2020-02-10 |

Verify a copy with:

```sh
shasum -a 256 maptiler-osm-2020-02-10-v3.11-planet.mbtiles   # macOS
sha256sum  maptiler-osm-2020-02-10-v3.11-planet.mbtiles      # Linux
```

## Where it lives

The service locates it via `MBTILES_PATH`; there is no default worth relying
on (the built-in fallback is `./planet.mbtiles`, which matches neither
deployment).

| deployment | path |
| ---------- | ---- |
| MacBook, `192.168.50.1:3000` — the address `config.TILE_SERVER` defaults to | `/Users/longd/projects/mapping_service/maptiler-osm-2020-02-10-v3.11-planet.mbtiles` |
| headless Linux, `10.0.0.84:3000` | `/mnt/storage/mapping_service/maptiler-osm-2020-02-10-v3.11-planet.mbtiles` |

## Why the byte identity matters here

Both deployments render from a byte-identical copy of this file, and their
masks still disagree on 0.03%–0.28% of pixels at every zoom from 0 to 16.
Recording the sha256 is what lets that statement be made: with the source
bytes proven equal, the divergence is attributable to the renderer — a
different `maplibre-gl-native` prebuild rasterising a boundary half a pixel
differently — and not to the data.

That is the reasoning `lib/identity.js` and the `renderer` field of `/health`
exist to make checkable going forward. `cacheNamespaces` folds this file into
its hash as *basename plus byte count only* (`sourceIdentity` in
`lib/variants.js`), which is enough to catch a different extract at the same
path but is not a content hash — another reason it cannot serve as a
provenance record on its own.
