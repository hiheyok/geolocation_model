"""Is there readable text in these frames, and does it point at the right tile?

Two questions, in the order that matters. The second is only worth asking if the
first survives.

**Is there legible text at all?** OSV-5M frames are 682x512 or 910x512.
Signage at that size is often a few pixels tall, and no amount of label quality
or language coverage compensates for text that cannot be resolved. EasyOCR's
CRAFT detector is script-agnostic -- it finds text regions before anything is
recognised -- so detection rate answers this globally even though recognition
only covers the loaded script group.

**Does the text discriminate?** The tile server returns labels with their pixel
position inside the tile, e.g.

    {"layer": "poi", "text": "Clapham Junction", "class": "railway",
     "rank": 1, "x": 30.6, "y": 414.3}

so a match is a *per-action* signal, not a global one. The test here is
oracle-style and needs no training: score the OCR tokens against the true
tile's labels and against random tiles' labels, and see whether the true tile
separates. If it does not separate with the answer handed to it, no learned
head will find it.

Scoring is IDF-weighted fuzzy matching. Raw overlap is useless because "street",
"rue" and "station" appear everywhere while "Ballycastle" is nearly unique, and
OCR output is misspelt often enough that exact match throws away most of the
signal.

Reported split by whether the image has text, because a signal that helps a
quarter of images a great deal is a real result that a pooled average hides.
"""

import argparse
import io
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import tile_math as tm

STOP = {"street", "st", "road", "rd", "avenue", "ave", "lane", "way", "drive",
        "rue", "strasse", "strase", "platz", "via", "calle", "plaza", "the",
        "de", "la", "le", "el", "du", "des", "and", "of", "station", "park",
        "school", "church", "hotel", "bar", "cafe", "restaurant", "shop"}


def norm(s):
    """Casefold, strip accents, keep letters and digits."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def tokens(s, min_len=3):
    return [t for t in norm(s).split() if len(t) >= min_len and t not in STOP]


def best_ratio(a, cands):
    """Fuzzy similarity of a against its closest candidate."""
    best = 0.0
    for c in cands:
        if abs(len(a) - len(c)) > 4:
            continue
        r = SequenceMatcher(None, a, c).ratio()
        if r > best:
            best = r
            if best == 1.0:
                break
    return best


def score(ocr_toks, label_toks, idf, thresh=0.82):
    """IDF-weighted count of OCR tokens that fuzzily match a label token."""
    if not ocr_toks or not label_toks:
        return 0.0
    tot = 0.0
    for t in ocr_toks:
        r = best_ratio(t, label_toks)
        if r >= thresh:
            tot += idf.get(t, max(idf.values()) if idf else 1.0) * r
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--zoom", type=int, default=12)
    ap.add_argument("--decoys", type=int, default=20)
    ap.add_argument("--langs", default="en",
                    help="EasyOCR script group; the detector is script-agnostic "
                         "so detection rate holds regardless")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--mag", type=float, default=1.0,
                    help="EasyOCR magnification before detection. These frames "
                         "are 682x512 and signage is often a few pixels tall, "
                         "so whether upscaling recovers detections is the "
                         "question that decides the whole direction.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import easyocr
    import pyarrow.parquet as pq
    from tiles import TileClient

    ds = pq.read_table(config.DATASET_PARQUET,
                       columns=["zip_name", "lat", "lon", "country"])
    zn = np.asarray(ds["zip_name"]).astype("U40")
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)
    country = np.asarray(ds["country"]).astype("U8")

    rng = np.random.default_rng(a.seed)
    sel = rng.choice(len(zn), a.n, replace=False)
    client = TileClient(config.TILE_SERVER)
    reader = easyocr.Reader(a.langs.split(","), gpu=True, verbose=False)
    print("release {}  {:,} images  z{}  langs {}  mag {}".format(
        config.RELEASE, a.n, a.zoom, a.langs, a.mag), flush=True)

    by_zip = defaultdict(list)
    for i in sel:
        by_zip[zn[i].split("/")[0]].append(i)

    recs = []
    t0 = time.time()
    for shard, items in sorted(by_zip.items()):
        zp = Path(config.OSV_ROOT) / "images" / "train" / (shard + ".zip")
        with zipfile.ZipFile(zp) as zf:
            for i in items:
                try:
                    blob = zf.read(zn[i])
                    from PIL import Image
                    img = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
                except Exception:
                    continue
                out = reader.readtext(img, mag_ratio=a.mag)
                dets = [(t, float(c)) for _, t, c in out]
                recs.append({"i": int(i), "dets": dets,
                             "hi": [t for t, c in dets if c >= a.conf]})
                if len(recs) % 50 == 0:
                    el = time.time() - t0
                    print("  {}/{}  {:.0f}s  {:.0f} ms/img".format(
                        len(recs), a.n, el, 1000 * el / len(recs)), flush=True)

    n = len(recs)
    any_det = sum(1 for r in recs if r["dets"])
    any_hi = sum(1 for r in recs if r["hi"])
    toks = [tokens(" ".join(r["hi"])) for r in recs]
    with_tok = sum(1 for t in toks if t)
    print("\n{:,} images in {:.0f}s ({:.0f} ms each)".format(
        n, time.time() - t0, 1000 * (time.time() - t0) / max(n, 1)))
    print("  any text region detected        {:5.1f}%".format(100 * any_det / n))
    print("  a region read at conf >= {:.2f}   {:5.1f}%".format(
        a.conf, 100 * any_hi / n))
    print("  yields a usable token (>=3 chars, not generic)  {:5.1f}%".format(
        100 * with_tok / n))
    if with_tok:
        print("  mean usable tokens, over images that have any: {:.2f}".format(
            sum(len(t) for t in toks) / with_tok))

    # ---- does the text point at the right tile? -------------------------
    live = [(r, t) for r, t in zip(recs, toks) if t]
    if not live:
        print("\nno usable text; the discrimination test is moot")
        return
    print("\nfetching labels for {:,} true tiles and {} decoys each".format(
        len(live), a.decoys), flush=True)

    def label_toks(z, x, y):
        try:
            L = client.labels(z, x, y)
        except Exception:
            return []
        out = []
        for e in L.get("labels", []):
            out.extend(tokens(e.get("text", "")))
        return out

    pool = [label_toks(*tm.tile_for(lat[r["i"]], lon[r["i"]], a.zoom))
            for r, _ in live]
    df = Counter()
    for p in pool:
        df.update(set(p))
    N = max(len(pool), 1)
    idf = {t: float(np.log(N / (1 + c)) + 1.0) for t, c in df.items()}

    others = rng.choice(len(zn), a.decoys, replace=False)
    decoy = [label_toks(*tm.tile_for(lat[j], lon[j], a.zoom)) for j in others]

    true_s, decoy_s, wins = [], [], 0
    for (r, t), p in zip(live, pool):
        st = score(t, p, idf)
        sd = [score(t, d, idf) for d in decoy]
        true_s.append(st)
        decoy_s.append(float(np.mean(sd)))
        if st > max(sd):
            wins += 1
    true_s, decoy_s = np.array(true_s), np.array(decoy_s)
    print("\ndiscrimination on the {:,} images with usable text".format(len(live)))
    print("  true-tile score   mean {:.3f}   nonzero {:5.1f}%".format(
        true_s.mean(), 100 * (true_s > 0).mean()))
    print("  decoy-tile score  mean {:.3f}   nonzero {:5.1f}%".format(
        decoy_s.mean(), 100 * (decoy_s > 0).mean()))
    print("  true beats every one of {} decoys: {:5.1f}%  (chance {:.1f}%)".format(
        a.decoys, 100 * wins / len(live), 100.0 / (a.decoys + 1)))
    hits = 100 * ((true_s > 0).mean())
    print("\n  as a share of ALL images, not just those with text: "
          "{:.1f}% get a nonzero true-tile match".format(
              hits * len(live) / n))


if __name__ == "__main__":
    main()
