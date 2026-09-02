"""Do the two encoders want different framings of the same photograph?

`res_probe.py` found that tiling helps SigLIP and hurts DINOv2, separated in
both directions on the same images and the same bank:

    tile6 against the shipping crop scheme, `<25 km`
        DINOv2   -5.50 pp [-8.8, -2.3]
        SigLIP   +3.75 pp [+0.3, +7.5]

which fits what the two are: DINOv2 is self-supervised on structure and global
layout, so a fragment is out of distribution, while SigLIP is language-supervised
on nameable content and a storefront that fills a tile is more nameable than one
occupying 5% of a wide crop.

If that holds, every tiling result this project has recorded is suspect, because
they were all measured on the *dual* vector, which DINOv2 dominates -- 81% of
the cosine before the equal-norm fix, and still the stronger encoder at fine
scale after it. "Tiling loses" may have been "DINOv2 loses at tiling, drowning
out a SigLIP gain".

So score the mixtures directly. Each half is L2-normalised before joining, so
the pairing is the only thing that changes and no arm wins by carrying a larger
raw activation norm -- the defect that silently gave DINOv2 81% of the cosine in
the first place.

    dino_crop3 + siglip_crop3    what ships today
    dino_crop3 + siglip_tile6    each encoder given the framing it prefers
    dino_tile6 + siglip_tile6    tiles for both, the scheme already rejected
    dino_tile6 + siglip_crop3    deliberately backwards, as a control
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tile_pool import l2, paired
from tile_match import dense_sim, topk_stats

THRESH = (1, 25, 200, 750, 2500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="dir holding emb_<model>.npz")
    ap.add_argument("--queries", type=int, default=400)
    a = ap.parse_args()

    d = Path(a.emb)
    D = np.load(d / "emb_dinov2.npz", allow_pickle=True)
    S = np.load(d / "emb_siglip.npz", allow_pickle=True)
    assert (D["ids"] == S["ids"]).all(), "the two runs used different images"
    lat, lon, seq = D["lat"], D["lon"], D["seq"]
    arms = [k for k in D.files if k not in ("ids", "lat", "lon", "seq")]
    print("{:,} images, arms per encoder: {}".format(len(lat), ", ".join(arms)))

    nq = a.queries
    qi, bi = np.arange(nq), np.arange(nq, len(lat))
    same = seq[qi][:, None] == seq[bi][None, :]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("{:,} queries  {:,} bank  {:,} same-sequence pairs masked\n".format(
        nq, len(bi), int(same.sum())))

    combos = [("dino_crop3 + siglip_crop3", "crop3_224", "crop3_224"),
              ("dino_crop3 + siglip_tile6", "crop3_224", "tile6"),
              ("dino_tile6  + siglip_tile6", "tile6", "tile6"),
              ("dino_tile6  + siglip_crop3", "tile6", "crop3_224")]
    combos = [c for c in combos if c[1] in arms and c[2] in arms]

    err = {}
    print("%-28s %11s %s" % ("pairing", "median km",
                             " ".join("%9s" % ("<%gkm" % t) for t in THRESH)))
    print("-" * 90)
    for name, dk, sk in combos:
        V = np.concatenate([l2(D[dk]), l2(S[sk])], axis=1)
        e, _ = topk_stats(dense_sim(V[qi], V[bi], dev), lat, lon, qi, bi, same)
        err[name] = e
        print("%-28s %11.1f %s" % (name, np.median(e), " ".join(
            "%8.1f%%" % (100 * (e < t).mean()) for t in THRESH)), flush=True)

    rng = np.random.default_rng(0)
    base = combos[0][0]
    print("\npaired against what ships today, percentage points, ~ spans zero")
    print("%-28s %s" % ("pairing", " ".join("%18s" % ("<%gkm" % t)
                                            for t in THRESH)))
    print("-" * 122)
    for name in err:
        if name == base:
            continue
        cells = []
        for t in THRESH:
            lo, hi = paired(err[base] < t, err[name] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[name] < t).mean() - (err[base] < t).mean()), lo, hi,
                " " if lo * hi > 0 else "~"))
        print("%-28s %s" % (name, " ".join(cells)))


if __name__ == "__main__":
    main()
