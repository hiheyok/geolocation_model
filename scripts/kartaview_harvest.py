"""Harvest full-resolution street imagery from KartaView.

KartaView (formerly OpenStreetCam) is one of the two sources OSV-5M was built
from, so this is not new imagery -- it is *the same imagery at native
resolution* instead of downsampled to 682x512. That makes it the clean way to
ask whether resolution matters, because pixels are the only variable changed.

Measured on live records: 1920x1080 to 3840x2160, i.e. 2.1 to 8.3 MP against
OSV-5M's 0.35 MP, up to 24x the pixels. CC BY-SA. **No token, no account** --
`/1.0/list/nearby-photos/` is open, which is why this is preferred over
Mapillary despite Mapillary's better global coverage.

Two API calls, both unauthenticated:

    POST /1.0/list/nearby-photos/  {lat, lng, radius}
        -> lat/lng, match_lat/match_lng (snapped to an OSM way, more accurate
           than raw GPS), gps_accuracy, projection, sequence_id, and `name`,
           a storage path
    GET  /2.0/photo/?id=<id>
        -> fileurlProc, the full-resolution image

The storage URL is usually derivable from `name` (`storage13/files/...` ->
`https://storage13.openstreetcam.org/files/...`), so the detail call is only a
fallback -- it halves the request count when the derivation works.

**Seeding.** Same trick as the Mapillary version: seed from OSV-5M coordinates,
because every one is a place a camera demonstrably drove, so coverage is likely
by construction rather than by luck. Note the honest limit found while probing:
KartaView is dense in Europe and absent in Ethiopia and Uruguay within 600 m of
an OSV-5M seed. It solves *resolution*, not the geographic skew. Those are
separate problems and this fixes one.

**Sequence discipline.** Consecutive frames in a drive are near-duplicates and
this project splits on sequence for exactly that reason, so `--per-sequence`
caps how many frames one drive can contribute. Without it a single dense city
drive would dominate the sample and every retrieval metric computed on it would
be measuring near-duplicate matching.

**Screen before use.** 1.45% of OSV-5M frames carry their true GPS burned in as
a dashcam overlay -- harmless at 682px where 8-pixel text is unresolvable, and
crisp at 3264px. Run the coordinate regex in `ocr_probe.py` over anything
harvested here before it enters a bank or a training set.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config

NEARBY = "https://api.openstreetcam.org/1.0/list/nearby-photos/"
DETAIL = "https://api.openstreetcam.org/2.0/photo/?id="
UA = {"User-Agent": "geolocation-research/1.0"}


def post(url, data, tries=3):
    body = urllib.parse.urlencode(data).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=body, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(1.5 ** k)


def fetch(url, tries=2):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == tries - 1:
                return None
            time.sleep(1.0)
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(1.0)
    return None


def proc_url(name):
    """'storage13/files/x.jpg' -> 'https://storage13.openstreetcam.org/files/x.jpg'"""
    host, _, tail = name.partition("/")
    return "https://{}.openstreetcam.org/{}".format(host, tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/data/kartaview")
    ap.add_argument("--n", type=int, default=20000, help="images to keep")
    ap.add_argument("--seeds", type=int, default=4000)
    ap.add_argument("--radius", type=int, default=400, help="metres")
    ap.add_argument("--per-seed", type=int, default=12)
    ap.add_argument("--per-sequence", type=int, default=3,
                    help="cap per drive; consecutive frames are near-duplicates")
    ap.add_argument("--max-gps-err", type=float, default=10.0)
    ap.add_argument("--min-side", type=int, default=1280,
                    help="reject anything not actually high resolution")
    ap.add_argument("--workers", type=int, default=24,
                    help="parallel API calls and downloads. Serial harvesting "
                         "ran at 17 s an image -- 98 hours for 20k -- because "
                         "it is latency-bound, not bandwidth-bound.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image
    import io as _io

    ds = pq.read_table(config.DATASET_PARQUET, columns=["lat", "lon", "country"])
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    cc = np.asarray(ds["country"]).astype("U8")

    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    man_p = out / "manifest.jsonl"
    have = {}
    if man_p.exists():
        for line in man_p.open(encoding="utf-8"):
            r = json.loads(line)
            have[r["id"]] = r
    print("KartaView harvest -> {}   {:,} already held".format(out, len(have)),
          flush=True)

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(lat))[:a.seeds]
    t0 = time.time()

    # ---- phase 1: discover, in parallel.  The JSON is small and the calls are
    # independent, so this is latency-bound and threads fix it entirely.
    def discover(i):
        try:
            js = post(NEARBY, {"lat": float(lat[i]), "lng": float(lon[i]),
                               "radius": a.radius})
        except Exception:
            return i, []
        return i, (js.get("currentPageItems") or [])

    cand, empty, done = [], 0, 0
    with ThreadPoolExecutor(a.workers) as pool:
        for i, items in pool.map(discover, order):
            done += 1
            if not items:
                empty += 1
            for d in items[:a.per_seed * 4]:
                cand.append((i, d))
            if done % 500 == 0:
                print("  discovered {}/{} seeds   {:,} candidates   {:.0f}s"
                      .format(done, len(order), len(cand), time.time() - t0),
                      flush=True)
    print("phase 1: {:,} candidates from {:,} seeds ({} empty) in {:.0f}s"
          .format(len(cand), len(order), empty, time.time() - t0), flush=True)

    # ---- filter and apply the per-sequence cap BEFORE downloading, so a single
    # dense city drive cannot dominate the sample and no bytes are wasted on
    # frames that would be dropped afterwards.
    seq_count = Counter(r["sequence_id"] for r in have.values())
    per_seed = Counter()
    picked = []
    for i, d in cand:
        if len(picked) >= a.n - len(have):
            break
        pid = str(d.get("id"))
        if pid in have:
            continue
        if (d.get("projection") or "PLANE") != "PLANE":
            continue
        try:
            if float(d.get("gps_accuracy") or 99) > a.max_gps_err:
                continue
        except Exception:
            continue
        sid = str(d.get("sequence_id"))
        if seq_count[sid] >= a.per_sequence or per_seed[i] >= a.per_seed:
            continue
        if not d.get("name"):
            continue
        seq_count[sid] += 1
        per_seed[i] += 1
        have[pid] = None
        picked.append((i, d))
    print("phase 2: downloading {:,} of them across {:,} sequences"
          .format(len(picked), len({str(d.get("sequence_id")) for _, d in picked})),
          flush=True)

    lock = threading.Lock()
    fh = man_p.open("a", encoding="utf-8")
    kept = [r for r in have.values() if r]
    small = [0]

    def grab(item):
        i, d = item
        pid = str(d["id"])
        blob = fetch(proc_url(d["name"]))
        if blob is None:
            try:
                det = json.loads(urllib.request.urlopen(
                    urllib.request.Request(DETAIL + pid, headers=UA),
                    timeout=30).read())["result"]["data"][0]
                blob = fetch(det.get("fileurlProc") or det.get("fileurl", ""))
            except Exception:
                blob = None
        if not blob:
            return None
        try:
            w, h = Image.open(_io.BytesIO(blob)).size
        except Exception:
            return None
        if min(w, h) < a.min_side:
            with lock:
                small[0] += 1
            return None
        p = out / "img" / (pid + ".jpg")
        p.write_bytes(blob)
        return {"id": pid, "sequence_id": str(d.get("sequence_id")),
                "lat": float(d.get("match_lat") or d["lat"]),
                "lon": float(d.get("match_lng") or d["lng"]),
                "country": str(cc[i]), "w": w, "h": h,
                "gps_accuracy": float(d.get("gps_accuracy") or -1),
                "shot_date": d.get("shot_date"), "file": p.name}

    n_done = 0
    with ThreadPoolExecutor(a.workers) as pool:
        for rec in pool.map(grab, picked):
            n_done += 1
            if rec:
                with lock:
                    fh.write(json.dumps(rec) + "\n")
                    kept.append(rec)
            if n_done % 500 == 0:
                el = time.time() - t0
                fh.flush()
                print("  {:,}/{:,} downloaded   {:,} kept   {:.0f}s   "
                      "{:.1f} img/s".format(n_done, len(picked), len(kept), el,
                                            n_done / max(el, 1)), flush=True)
    fh.close()
    small = small[0]

    if not kept:
        print("nothing harvested")
        return
    mp = np.array([r["w"] * r["h"] for r in kept]) / 1e6
    gb = sum((out / "img" / r["file"]).stat().st_size for r in kept) / 1e9
    print("\n{:,} images  {:.1f} GB  in {:.0f}s".format(
        len(kept), gb, time.time() - t0))
    print("  megapixels  median {:.1f}  min {:.1f}  max {:.1f}   "
          "(OSV-5M is 0.35)".format(np.median(mp), mp.min(), mp.max()))
    print("  sequences   {:,}   empty seeds {}   rejected as small {}".format(
        len({r["sequence_id"] for r in kept}), empty, small))
    top = Counter(r["country"] for r in kept).most_common(10)
    print("  countries   " + ", ".join("{} {}".format(k, v) for k, v in top))
    print("\nmanifest {}".format(man_p))


if __name__ == "__main__":
    main()
