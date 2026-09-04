"""Harvest higher-resolution street imagery where OSV-5M is thin.

Why this exists.  OSV-5M frames are 682x512, and that cap is load-bearing: it is
why OCR reads dashcam chrome instead of signage, and why six tiles carried 4.5x
the pixels of three crops while each tile was individually weaker.  Mapillary
serves the originals -- 98% at 2048x1536, nine times the pixels -- and does it
globally rather than in the 30 curated cities MSLS ships.

It also targets the other gap.  OSV-5M is 211 countries but the top ten are
58.2% of the data and 150 countries have under 1,000 images, so simply adding
more OSV-5M shards inherits the same skew.

**Seeding.**  The search endpoint caps a bbox at 0.01 degrees square, about
1.1 km, so blanket sweeps are impossible and random probes into a sparse country
mostly return nothing.  The trick is that we already know where imagery exists
in thin countries -- OSV-5M has *some* images there, and each one is a
coordinate where a camera demonstrably drove.  So seed the search from our own
thin-country images and harvest Mapillary around them.  Coverage is guaranteed
by construction and the sample lands exactly where the corpus needs it.

**Rate limits shape the design.**  Entity requests are 60,000 a minute and
search 10,000, but coverage *tiles* are 50,000 a **day** -- so discovery is the
scarce resource and this script avoids the tile API entirely by seeding from
coordinates it already has.

**What is kept.**  Nothing but the image bytes, streamed.  A 2048px corpus of
500k images is 250-500 GB against 538 GB free, so the intended use is
harvest -> embed -> discard, the way `embed_street.py` already reads from a zip
without extracting.  `--save-dir` exists for a small cache when you want to
re-embed later without re-downloading.

**Screen before you trust it.**  1.45% of OSV-5M frames have their true GPS
burned in as a dashcam overlay, harmless at 682px because 8-pixel text is
unresolvable and *crisp at 2048px*.  Mapillary hosts plenty of dashcam uploads.
Run the coordinate detector in `ocr_probe.py` over anything harvested here
before it enters a bank or a training set, or the first "more pixels helped"
result will be the model reading the answer off the image.

Setup:  create a Mapillary account, go to the developer dashboard, register an
application, and export the client token:

    export MAPILLARY_TOKEN=MLY|...|...        # bash
    $env:MAPILLARY_TOKEN = 'MLY|...|...'      # powershell

    python scripts/mapillary_harvest.py --plan          # no token needed
    python scripts/mapillary_harvest.py --max-images 2000
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config

SEARCH = "https://graph.mapillary.com/images"
BBOX_MAX = 0.01                      # degrees; the endpoint's hard cap
FIELDS = ("id,computed_geometry,geometry,compass_angle,computed_compass_angle,"
          "captured_at,is_pano,quality_score,thumb_2048_url,thumb_1024_url")


def get_json(url, token, tries=3):
    req = urllib.request.Request(
        url, headers={"Authorization": "OAuth {}".format(token)})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and k < tries - 1:
                time.sleep(2 ** k)
                continue
            raise
        except Exception:
            if k < tries - 1:
                time.sleep(2 ** k)
                continue
            raise


def thin_countries(min_n, max_n):
    """Countries where OSV-5M is sparse, with one seed coordinate per image."""
    import pyarrow.parquet as pq
    ds = pq.read_table(config.DATASET_PARQUET,
                       columns=["country", "lat", "lon"])
    cc = np.asarray(ds["country"]).astype("U8")
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    counts = Counter(cc.tolist())
    keep = {k for k, v in counts.items() if min_n <= v <= max_n}
    m = np.isin(cc, list(keep))
    return cc[m], lat[m], lon[m], counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="print the harvest plan and exit; needs no token")
    ap.add_argument("--min-country", type=int, default=20)
    ap.add_argument("--max-country", type=int, default=1000,
                    help="only harvest around countries with at most this many "
                         "OSV-5M images -- the thin tail is the point")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--per-seed", type=int, default=50)
    ap.add_argument("--max-images", type=int, default=5000)
    ap.add_argument("--min-quality", type=float, default=0.2)
    ap.add_argument("--keep-pano", action="store_true",
                    help="panoramas are equirectangular and would need a "
                         "different crop scheme, so they are dropped by default")
    ap.add_argument("--out", default=str(ROOT / "data" / "mapillary"))
    ap.add_argument("--save-dir", default=None,
                    help="also write the JPEGs here; omit to keep metadata only")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cc, lat, lon, counts = thin_countries(a.min_country, a.max_country)
    rng = np.random.default_rng(a.seed)
    if len(cc) == 0:
        sys.exit("no countries in that band")
    pick = rng.choice(len(cc), min(a.seeds, len(cc)), replace=False)

    tally = Counter(cc[pick].tolist())
    print("OSV-5M has {:,} images across {} countries in the "
          "{}-{} band".format(len(cc), len({*cc.tolist()}),
                              a.min_country, a.max_country))
    print("{} seed coordinates over {} countries; top targets:".format(
        len(pick), len(tally)))
    for k, v in tally.most_common(12):
        print("   {:<4} {:>4} seeds   (OSV-5M has {:,})".format(k, v, counts[k]))
    print("\nbbox {} deg per seed (endpoint cap), up to {} images each, "
          "{} total".format(BBOX_MAX, a.per_seed, a.max_images))

    token = os.environ.get("MAPILLARY_TOKEN")
    if a.plan or not token:
        if not token and not a.plan:
            print("\nMAPILLARY_TOKEN is not set -- printing the plan only.")
            print("Create one at https://www.mapillary.com/dashboard/developers")
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
        # The flag promised a small image cache and only ever made the folder.
        # Saying so is better than a directory that fills with nothing.
        print("note: --save-dir writes metadata and URLs only; image bytes "
              "are not downloaded by this script", flush=True)

    rows, seen = [], set()
    t0 = time.time()
    for n, i in enumerate(pick):
        if len(rows) >= a.max_images:
            break
        h = BBOX_MAX / 2
        bbox = "{:.6f},{:.6f},{:.6f},{:.6f}".format(
            lon[i] - h, lat[i] - h, lon[i] + h, lat[i] + h)
        url = "{}?{}".format(SEARCH, urllib.parse.urlencode(
            {"fields": FIELDS, "bbox": bbox, "limit": a.per_seed}))
        try:
            js = get_json(url, token)
        except Exception as e:
            print("  seed {} failed: {}".format(n, str(e)[:80]), flush=True)
            continue
        for d in js.get("data", []):
            if d["id"] in seen:
                continue
            if d.get("is_pano") and not a.keep_pano:
                continue
            if float(d.get("quality_score") or 0) < a.min_quality:
                continue
            g = d.get("computed_geometry") or d.get("geometry")
            if not g:
                continue
            glon, glat = g["coordinates"]
            seen.add(d["id"])
            rows.append({"id": d["id"], "lat": glat, "lon": glon,
                         "country": str(cc[i]),
                         "compass": d.get("computed_compass_angle",
                                          d.get("compass_angle")),
                         "captured_at": d.get("captured_at"),
                         "quality": d.get("quality_score"),
                         "url": d.get("thumb_2048_url") or d.get("thumb_1024_url")})
        if (n + 1) % 25 == 0:
            print("  {}/{} seeds   {:,} images   {:.0f}s".format(
                n + 1, len(pick), len(rows), time.time() - t0), flush=True)

    man = out / "manifest.jsonl"
    with open(man, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    got = Counter(r["country"] for r in rows)
    print("\n{:,} images from {} countries in {:.0f}s".format(
        len(rows), len(got), time.time() - t0))
    for k, v in got.most_common(12):
        print("   {:<4} {:>5}  (OSV-5M had {:,})".format(k, v, counts[k]))
    print("manifest {}".format(man))
    print("\nNext: screen for burned-in coordinates before use --")
    print("  the regex in scripts/ocr_probe.py, over the 2048px versions.")


if __name__ == "__main__":
    main()
