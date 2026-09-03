"""Screen a high-resolution corpus for frames that print their own coordinates.

1.45% of OSV-5M frames carry the true GPS burned in as a dashcam overlay --
`W121.990885 N46.53855` and similar -- landing a median 0.05 km from the label,
90% within 1 km. At 682x512 that text is 8 pixels tall and unresolvable, so the
shipping pipeline is safe by accident. **At 3264x1836 it is crisp.**

That matters because the leak is worth roughly +1.3 pp on `<25 km` if exploited,
which is the same magnitude as every effect measured in this project. A
higher-resolution corpus that improves the metric by reading coordinates off the
image would look exactly like a real result.

So: run this over any high-resolution corpus **before** it enters a bank, a
training set, or a resolution comparison. It reports the rate and, with
`--write-blocklist`, the ids to exclude.

Two design choices worth stating. Detection runs at `--ocr-side` rather than
native, because overlays are large relative to the frame and full-resolution OCR
costs several times more for no extra recall on this particular text. And a hit
is only counted when the parsed coordinate lands within `--match-km` of the
label -- a frame that happens to contain a number shaped like a latitude is not
leaking anything, and counting it would inflate the rate.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import safeio

# OCR confuses 0/o, 1/l/I, 5/S and 8/B in the small fixed-width fonts overlays use
FIX = str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1", "S": "5", "B": "8"})
PAT = re.compile(r"([NSEW])\s*([0-9oOlIB]{1,3})[.,\s]{1,3}([0-9oOlIB]{3,8})", re.I)


def great_circle(a1, o1, a2, o2):
    p = np.pi / 180
    d1, d2 = (a2 - a1) * p, (o2 - o1) * p
    h = (np.sin(d1 / 2) ** 2 +
         np.cos(a1 * p) * np.cos(a2 * p) * np.sin(d2 / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(min(1.0, np.sqrt(h)))


def parse_coords(text):
    got = {}
    for hemi, deg, frac in PAT.findall(text):
        try:
            v = float(deg.translate(FIX)) + float("0." + frac.translate(FIX))
        except Exception:
            continue
        h = hemi.upper()
        if h in "NS" and "lat" not in got:
            got["lat"] = v * (1 if h == "N" else -1)
        if h in "EW" and "lon" not in got:
            got["lon"] = v * (1 if h == "E" else -1)
    return got if "lat" in got and "lon" in got else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/data/kartaview")
    ap.add_argument("--n", type=int, default=0, help="0 = every image")
    ap.add_argument("--ocr-side", type=int, default=1600,
                    help="long side fed to the detector; overlays are large "
                         "relative to the frame so native costs more for "
                         "nothing")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--match-km", type=float, default=25.0,
                    help="a parsed coordinate only counts as a leak if it "
                         "actually lands near the label")
    ap.add_argument("--write-blocklist", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import easyocr
    from PIL import Image

    root = Path(a.data)
    recs, seen = [], set()
    for line in (root / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["id"] not in seen:
            seen.add(r["id"])
            recs.append(r)
    rng = np.random.default_rng(a.seed)
    rng.shuffle(recs)
    if a.n:
        recs = recs[:a.n]
    print("screening {:,} images from {}  (OCR at long side {})".format(
        len(recs), root, a.ocr_side), flush=True)

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    leaks, parsed, dists, t0 = [], 0, [], time.time()
    for n, r in enumerate(recs):
        try:
            im = Image.open(root / "img" / r["file"]).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        s = a.ocr_side / max(w, h)
        if s < 1.0:
            im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
        txt = " ".join(t for _, t, c in reader.readtext(np.asarray(im))
                       if c >= a.conf)
        got = parse_coords(txt)
        if got:
            parsed += 1
            d = great_circle(r["lat"], r["lon"], got["lat"], got["lon"])
            dists.append(d)
            if d <= a.match_km:
                leaks.append({"id": r["id"], "km": round(d, 4), "text": txt[:120]})
        if (n + 1) % 200 == 0:
            el = time.time() - t0
            print("  {:,}/{:,}  {:.0f}s  {:.0f} ms/img  {} parsed  {} leaking"
                  .format(n + 1, len(recs), el, 1000 * el / (n + 1), parsed,
                          len(leaks)), flush=True)

    n = len(recs)
    print("\n{:,} screened in {:.0f}s".format(n, time.time() - t0))
    print("  frames with a parseable lat AND lon   {:5.2f}%  ({})".format(
        100 * parsed / max(n, 1), parsed))
    print("  of those, landing within {:.0f} km      {:5.2f}% of all  ({})".format(
        a.match_km, 100 * len(leaks) / max(n, 1), len(leaks)))
    if dists:
        d = np.array(dists)
        print("  parsed-coordinate error: median {:.2f} km   <1 km {:.0f}%   "
              "<25 km {:.0f}%".format(np.median(d), 100 * (d < 1).mean(),
                                      100 * (d < 25).mean()))
    if leaks:
        print("\n  examples:")
        for e in leaks[:5]:
            print("    {}  {:.2f} km   {}".format(e["id"], e["km"], e["text"][:80]))
    if a.write_blocklist:
        p = root / "leak_blocklist.json"
        safeio.write_text(p, json.dumps({"screened": n, "match_km": a.match_km,
                                 "ids": [e["id"] for e in leaks],
                                 "detail": leaks}, indent=1), encoding="utf-8")
        print("\nblocklist -> {}  ({} ids)".format(p, len(leaks)))
    elif leaks:
        print("\nrerun with --write-blocklist to record the ids to exclude")


if __name__ == "__main__":
    main()
