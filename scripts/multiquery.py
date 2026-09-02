"""Several photos of one place, pooled at the retrieval step only.

`benchmark-measures-bank-coverage`: the model scores 2.7 km on OSV-5M test and
442 km on held-out KartaView photos, and the cause is not domain shift -- it is
that OSV-5M's bank densely covers the streets its own test images came from,
while a novel photo has thin coverage. Top-1 similarity falls 0.90 -> 0.71 and
the whole system degrades with it.

Density on the bank side is expensive: it took three shard blocks and six hours
of encoding to move 1.15M -> 3.50M. **Density on the query side is free.** A
person standing in one place can take four photographs in four directions, and
the true location is the one all four agree on while their spurious matches
disagree.

**No retraining.** The retrieval prior consumes K neighbours as a similarity-
weighted set over their z16 addresses; it is permutation-invariant and has no
idea which image produced them. So N images contribute N x 32 candidates, the
top K by similarity go to the prior, and one image -- the primary -- still
drives the policy through its own street vector. That is the whole change.

What this measures is the shape of the curve: does the second photo help as much
as the tenth, and does it saturate before the bank-side gains do. If two photos
recover a large fraction of the single-photo deficit, the practical answer to
thin coverage is to ask for more photographs rather than to harvest more bank.

Grouping is spatial, not by sequence: images within `--radius` metres of the
anchor, which is what "photos taken around you" means. Same-sequence frames are
allowed, since a person walking ten metres and turning round is exactly the case
this is for.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config
import names
import tile_math as tm
from dataset import street_table
from evaluate import evaluate, load_model
from beam import TokenSource
from eval_highres import ExternalSet, embed, tile_for_vec

R_EARTH = 6371.0088


def unit3(lat, lon):
    p = np.pi / 180
    return np.stack([np.cos(lat * p) * np.cos(lon * p),
                     np.cos(lat * p) * np.sin(lon * p), np.sin(lat * p)], 1)


def cached_queries(data, n, seed, dev, stem):
    """Embed once, reuse. Three separate runs re-embedded the same images."""
    p = config.STREET_CACHE / (stem + ".f16.npy")
    q = config.STREET_CACHE / (stem + "_meta.npz")
    recs = {}
    for line in (Path(data) / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        recs[r["id"]] = r
    ids = sorted(recs)
    pick = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))[:n]]
    if p.exists() and q.exists():
        m = np.load(q, allow_pickle=True)
        if len(m["image_id"]) == len(pick) and (m["image_id"] == np.array(pick)).all():
            print("reusing cached query embeddings, {:,}".format(len(pick)),
                  flush=True)
            return (np.load(p), m["lat"], m["lon"],
                    m["sequence"].astype("U40"))
    paths = [str(Path(data) / "img" / (i + ".jpg")) for i in pick]
    E = embed(paths, dev)
    tok = np.concatenate([E["dinov2"], 4.03 * E["siglip"]], axis=2)
    pooled = tok.mean(1).astype(np.float32)
    z = np.load(config.STREET_CACHE / "pca768_bank55_pca.npz")
    V = ((pooled - z["mu"]) @ z["P"]).astype(np.float16)
    lat = np.array([recs[i]["lat"] for i in pick], np.float64)
    lon = np.array([recs[i]["lon"] for i in pick], np.float64)
    seq = np.array([str(recs[i].get("sequence_id", i)) for i in pick])
    np.save(p, V)
    np.savez(q, image_id=np.array(pick), lat=lat, lon=lon, sequence=seq)
    return V, lat, lon, seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d1536-b350-e6")
    ap.add_argument("--bank", default=None,
                    help="defaults to the checkpoint's own street file")
    ap.add_argument("--data", default="E:/data/kartaview_hr")
    ap.add_argument("--n", type=int, default=4000,
                    help="images embedded; groups are drawn from these")
    ap.add_argument("--radius", type=float, default=30.0, help="metres")
    ap.add_argument("--sizes", default="1,2,4,8")
    ap.add_argument("--groups", type=int, default=400)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--score-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, _ = load_model(a.tag, dev)
    bank_file = a.bank or ck.get("street_file")
    dim = 768 if "768" in bank_file else 1536
    V, lat, lon, seq = cached_queries(a.data, a.n, a.seed, dev,
                                      "kvq{}".format(dim))
    if V.shape[1] != dim:
        sys.exit("cached queries are {}-d but {} wants {}-d; the 1536-d path "
                 "needs its own cache".format(V.shape[1], a.tag, dim))

    # ---- neighbours for every image, once -------------------------------
    bank = np.load(config.STREET_CACHE / bank_file, mmap_mode="r")
    Q = torch.nn.functional.normalize(
        torch.from_numpy(V.astype(np.float32)).to(dev), dim=1)
    K = 32
    bv = torch.full((len(V), K), -2.0, device=dev)
    bi = torch.zeros((len(V), K), dtype=torch.long, device=dev)
    t0 = time.time()
    for s in range(0, bank.shape[0], 200000):
        B = torch.nn.functional.normalize(
            torch.from_numpy(np.asarray(bank[s:s + 200000], np.float32)
                             ).to(dev), dim=1)
        v, i = torch.topk(Q @ B.T, K, dim=1)
        cv = torch.cat([bv, v], 1)
        ci = torch.cat([bi, i + s], 1)
        bv, sel = torch.topk(cv, K, dim=1)
        bi = torch.gather(ci, 1, sel)
        del B
    sim32, idx32 = bv.cpu().numpy(), bi.cpu().numpy()
    print("kNN over {:,} bank rows in {:.0f}s   top-1 {:.4f}".format(
        bank.shape[0], time.time() - t0, sim32[:, 0].mean()), flush=True)

    # ---- bank z16 addresses ---------------------------------------------
    import pyarrow.parquet as pq
    ds = pq.read_table(config.DATASET_PARQUET, columns=["lat", "lon"])
    bx, by = tile_for_vec(np.asarray(ds["lat"], np.float64),
                          np.asarray(ds["lon"], np.float64), 4 * tm.STEPS)
    stems = ["bank_ext", "bank_ext2", "bank_ext3"]
    if "bank70" in bank_file:
        stems.append("bank_ext4")
    for stem in stems:
        m = np.load(config.bank_meta(stem), allow_pickle=True)
        bx = np.concatenate([bx, m["x16"].astype(bx.dtype)])
        by = np.concatenate([by, m["y16"].astype(by.dtype)])
    assert len(bx) == bank.shape[0], (len(bx), bank.shape[0])

    # ---- spatial groups: everything within `radius` of an anchor ---------
    from scipy.spatial import cKDTree
    tree = cKDTree(unit3(lat, lon))
    rad = 2 * np.sin(a.radius / 1000.0 / (2 * R_EARTH))
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(lat))
    groups, used = [], set()
    for i in order:
        if i in used:
            continue
        nb = [j for j in tree.query_ball_point(unit3(lat[i:i + 1],
                                                     lon[i:i + 1])[0], rad)
              if j not in used]
        if len(nb) >= 2:
            groups.append((int(i), nb))
            used.update(nb)
        if len(groups) >= a.groups:
            break
    sizes = [int(x) for x in a.sizes.split(",")]
    avail = np.array([len(g[1]) for g in groups])
    print("\n{:,} groups within {:.0f} m   members: median {:.0f}, max {}"
          .format(len(groups), a.radius, np.median(avail), avail.max()),
          flush=True)
    if not len(groups):
        sys.exit("no groups; raise --radius or --n")

    tbl = street_table(config.STREET_CACHE / bank_file, dev) \
        if ck.get("retr_mode") in ("pos", "dual") else None
    src = TokenSource(tm.G)
    K_use = ck.get("retr_k", 16)

    print("\n{}  ({})".format(a.tag, names.describe(a.tag)))
    print("photos   n     median km      mean     <25km    <200km")
    print("-" * 58)
    base = None
    for N in sizes:
        anchors, nidx, nsim = [], [], []
        for anchor, members in groups:
            take = [anchor] + [j for j in members if j != anchor]
            take = take[:N]
            # every member's 32 candidates compete; the best K by similarity
            # reach the prior, exactly as they would from a single image
            cand_i = idx32[take].reshape(-1)
            cand_s = sim32[take].reshape(-1)
            o = np.argsort(-cand_s)[:K_use]
            anchors.append(anchor)
            nidx.append(cand_i[o])
            nsim.append(cand_s[o])
        anchors = np.array(anchors)
        ext = ExternalSet(V[anchors], lat[anchors], lon[anchors],
                          np.stack(nidx), np.stack(nsim).astype(np.float32),
                          bx, by)
        m = evaluate(model, ext, src, dev, None, beam_k=a.beam,
                     top_m=max(4, a.beam), greedy=(a.beam == 1),
                     score_steps=a.score_steps, street_gpu=tbl)
        e = m["err"]
        hit = float((e < 25).mean())
        if base is None:
            base = hit
        print("{:>4}  {:>5,} {:>10.1f} {:>10.1f} {:>9.1%} {:>9.1%}{}".format(
            N, len(e), float(np.median(e)), float(e.mean()), hit,
            float((e < 200).mean()),
            "" if N == sizes[0] else "   {:+.1f} pp".format(
                100 * (hit - base))), flush=True)

    print("\nOnly the retrieval prior sees the extra photographs; the policy "
          "still\nreads one street vector. No weights changed.")


if __name__ == "__main__":
    main()
