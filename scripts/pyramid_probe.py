"""A scale pyramid: whole image, then tiles, then tiles of tiles.

The union result showed that crops *plus* tiles beat crops alone, because a tile
is a fragment and needs the scene to anchor it. The obvious extension is to keep
subdividing -- whole image, 3x2, 6x4, and on down until a tile maps to 224
native pixels -- keeping every level rather than replacing one with another.

Where the recursion stops is set by the source. At 224px tiles a 682x512 OSV-5M
frame bottoms out at 3x2, so **no pyramid exists there at all**; a 5.3 MP
KartaView frame supports roughly 11x9. This idea only has room on
high-resolution data.

**The weighting is the real question, and it is easy to get wrong.** These
embeddings are means of L2-normalised tokens, so pooling the union of levels
weights *tokens* equally -- and a 3+6+24 pyramid then gives the deepest level
73% of the vector, which is the arm that loses on its own. Two schemes are
therefore scored separately:

    token   mean over all tokens of all levels; deep levels dominate by count
    level   mean of the per-level means; every scale contributes equally

Both are compared against crops alone at **identical pooled width**, so no arm
wins by carrying more bytes -- only by carrying better ones.
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
NTOK = {"crop3_224": 3, "tile6": 6, "tile24": 24}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--queries", type=int, default=1200)
    a = ap.parse_args()

    d = Path(a.emb)
    E = {m: np.load(d / ("emb_%s.npz" % m), allow_pickle=True)
         for m in ("dinov2", "siglip")}
    assert (E["dinov2"]["ids"] == E["siglip"]["ids"]).all()
    ref = E["dinov2"]
    lat, lon, seq = ref["lat"], ref["lon"], ref["seq"]
    nq = a.queries
    qi, bi = np.arange(nq), np.arange(nq, len(lat))
    same = seq[qi][:, None] == seq[bi][None, :]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("{:,} images  {:,} queries  {:,} bank  {:,} same-sequence masked\n"
          .format(len(lat), nq, len(bi), int(same.sum())))

    def by_token(F, levels):
        """mean over every token of every level"""
        n = sum(NTOK[k] for k in levels)
        return sum(NTOK[k] * F[k] for k in levels) / n

    def by_level(F, levels):
        """mean of the per-level means -- every scale weighted the same"""
        return sum(F[k] for k in levels) / len(levels)

    LEVELS = [("L0", ["crop3_224"]),
              ("L0+L1", ["crop3_224", "tile6"]),
              ("L0+L1+L2", ["crop3_224", "tile6", "tile24"]),
              ("L1+L2", ["tile6", "tile24"])]

    arms = {}
    for tag, ls in LEVELS:
        for how, fn in (("token", by_token), ("level", by_level)):
            if len(ls) == 1 and how == "level":
                continue
            name = "{} [{}]".format(tag, how) if len(ls) > 1 else tag
            arms[name] = np.concatenate(
                [l2(fn(E["dinov2"], ls)), l2(fn(E["siglip"], ls))], axis=1)

    err = {}
    print("%-22s %10s %s" % ("arm", "median km",
                            " ".join("%9s" % ("<%gkm" % t) for t in THRESH)))
    print("-" * 84)
    for k, V in arms.items():
        e, _ = topk_stats(dense_sim(V[qi], V[bi], dev), lat, lon, qi, bi, same)
        err[k] = e
        print("%-22s %10.1f %s" % (k, np.median(e), " ".join(
            "%8.1f%%" % (100 * (e < t).mean()) for t in THRESH)), flush=True)

    rng = np.random.default_rng(0)
    base = "L0"
    print("\nagainst L0 (crops alone), equal pooled width, ~ spans zero")
    print("%-22s %s" % ("arm", " ".join("%18s" % ("<%gkm" % t)
                                        for t in THRESH)))
    print("-" * 116)
    for k in arms:
        if k == base:
            continue
        cells = []
        for t in THRESH:
            lo, hi = paired(err[base] < t, err[k] < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((err[k] < t).mean() - (err[base] < t).mean()), lo, hi,
                " " if lo * hi > 0 else "~"))
        print("%-22s %s" % (k, " ".join(cells)))

    print("\nencoder forwards per image: L0 3, L0+L1 9, L0+L1+L2 33 -- an 11x "
          "embedding cost for the deepest arm, on a 1.15M bank.")


if __name__ == "__main__":
    main()
