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

**Groups are chosen from the whole manifest before anything is embedded, and
every group must be full.** Sampling images at random first and grouping second
does not work here and fails in the direction that invents a result: a random
4,000 of the harvest yields 200 groups of which exactly *one* has four members,
so `take[:N]` returns the same two photographs for N=2, 4 and 8 and the curve
reads as saturating at two when there was never a third photograph to add. So
the dense spots are found first, groups smaller than the largest requested size
are discarded, and only the images actually used get embedded -- which is also
much cheaper: 2,712 images instead of 4,000, for 339 genuinely full groups.

Extra photographs are ordered to **round-robin across distinct sequences**, with
the anchor's own sequence last. Consecutive frames of one drive are near
duplicates and retrieve nearly the same neighbours, so taking them first would
measure the wrong thing; a person photographing their surroundings turns around.
`--order near` sorts by distance instead, as a control.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from collections import deque

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
from beam import TokenSource, source_for
from eval_highres import ExternalSet, embed, tile_for_vec

R_EARTH = 6371.0088


def unit3(lat, lon):
    p = np.pi / 180
    return np.stack([np.cos(lat * p) * np.cos(lon * p),
                     np.cos(lat * p) * np.sin(lon * p), np.sin(lat * p)], 1)


def merge_candidates(take, idx32, sim32, K, how):
    """Which neighbours the extra photographs actually get to contribute.

    `global` ranks every photograph's 32 candidates together and keeps the best
    K. That sounds neutral and is not: similarity scale varies per photograph,
    so the single strongest match fills most of the slots and the others are
    silently dropped -- adding photographs then changes little, and the curve
    reads as saturation. This was already found and fixed in serve.py.

    `rr` gives each photograph an equal share by rank and deduplicates, so a
    bank row two photographs both found does not vote twice. At N=1 the two are
    identical, so the baseline is untouched either way.
    """
    if how == "global":
        ci = idx32[take].reshape(-1)
        cs = sim32[take].reshape(-1)
        o = np.argsort(-cs)[:K]
        return ci[o], cs[o]
    seen, oi, os_ = set(), [], []
    for r in range(idx32.shape[1]):
        for t in take:
            if len(oi) >= K:
                break
            b = int(idx32[t, r])
            if b in seen:
                continue
            seen.add(b)
            oi.append(b)
            os_.append(float(sim32[t, r]))
        if len(oi) >= K:
            break
    if len(oi) < K:                     # heavy overlap: top up from the pool
        ci = idx32[take].reshape(-1)
        cs = sim32[take].reshape(-1)
        for j in np.argsort(-cs):
            if len(oi) >= K:
                break
            b = int(ci[j])
            if b in seen:
                continue
            seen.add(b)
            oi.append(b)
            os_.append(float(cs[j]))
    while len(oi) < K:                  # fewer than K distinct rows exist
        oi.append(oi[-1])
        os_.append(os_[-1])
    return np.array(oi), np.array(os_)


def order_members(anchor, nb, seq, P, how):
    """Anchor first, then the other photographs in the order a person takes
    them. `diverse` round-robins across sequences -- the anchor's own drive
    last -- because consecutive frames of one drive retrieve nearly the same
    neighbours and would understate what a second viewpoint is worth."""
    rest = [j for j in nb if j != anchor]
    if how == "near":
        d = [float(np.dot(P[anchor], P[j])) for j in rest]
        return [anchor] + [j for _, j in sorted(zip(d, rest), reverse=True)]
    by = {}
    for j in rest:
        by.setdefault(seq[j], []).append(j)
    own = seq[anchor]
    keys = [k for k in by if k != own] + ([own] if own in by else [])
    qs = [deque(by[k]) for k in keys]
    out = [anchor]
    while any(qs):
        for q in qs:
            if q:
                out.append(q.popleft())
    return out


def build_groups(lat, lon, seq, radius, need, want, seed, how):
    """Dense spots in the *whole* harvest, found before anything is embedded."""
    from scipy.spatial import cKDTree
    P = unit3(lat, lon)
    tree = cKDTree(P)
    rad = 2 * np.sin(radius / 1000.0 / (2 * R_EARTH))
    order = np.random.default_rng(seed).permutation(len(lat))
    groups, used = [], set()
    for i in order:
        if i in used:
            continue
        nb = [j for j in tree.query_ball_point(P[i], rad) if j not in used]
        if len(nb) < need:
            continue
        m = order_members(int(i), nb, seq, P, how)[:need]
        groups.append(m)
        used.update(m)
        if len(groups) >= want:
            break
    return groups


def cached_queries(data, pick, dev, stem):
    """Embed once, reuse. Three separate runs re-embedded the same images."""
    p = config.STREET_CACHE / (stem + ".f16.npy")
    q = config.STREET_CACHE / (stem + "_meta.npz")
    recs = {}
    for line in (Path(data) / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        recs[r["id"]] = r
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
    ap.add_argument("--radius", type=float, default=100.0, help="metres")
    ap.add_argument("--sizes", default="1,2,4,8")
    ap.add_argument("--groups", type=int, default=400,
                    help="cap; the harvest supplies 339 full groups at 100 m")
    ap.add_argument("--order", default="diverse", choices=("diverse", "near"),
                    help="how the extra photographs are ordered")
    ap.add_argument("--export", default="",
                    help="npz of per-group errors for later re-analysis")
    ap.add_argument("--merge", default="rr", choices=("rr", "global"),
                    help="rr matches what serve.py does: equal share per "
                         "photograph, deduplicated. global is the naive "
                         "top-k, kept as the control it needs to beat")
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--score-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, _ = load_model(a.tag, dev)
    bank_file = a.bank or ck.get("street_file")
    dim = 768 if "768" in bank_file else 1536
    sizes = [int(x) for x in a.sizes.split(",")]
    need = max(sizes)

    # Groups first, from the whole manifest -- see the module docstring for why
    # sampling images first and grouping second manufactures a false plateau.
    recs = [json.loads(l) for l in
            (Path(a.data) / "manifest.jsonl").open(encoding="utf-8")]
    mlat = np.array([r["lat"] for r in recs], np.float64)
    mlon = np.array([r["lon"] for r in recs], np.float64)
    mseq = np.array([str(r.get("sequence_id", r["id"])) for r in recs])
    gidx = build_groups(mlat, mlon, mseq, a.radius, need, a.groups,
                        a.seed, a.order)
    if not gidx:
        sys.exit("no full groups; raise --radius or lower --sizes")
    flat = sorted({int(j) for g in gidx for j in g})
    pick = [recs[j]["id"] for j in flat]
    row = {j: r for r, j in enumerate(flat)}
    nseq = [len(set(mseq[np.array(g)])) for g in gidx]
    print("{:,} full groups of {} within {:.0f} m   {:,} images   "
          "{:.0f} distinct sequences per group (median)".format(
              len(gidx), need, a.radius, len(pick), np.median(nseq)),
          flush=True)

    # Key the cache by image count as well as width. One shared stem made the
    # configurations evict each other: the 9,984-image run overwrote the
    # 3,120-image one, so alternating between them re-embedded every time.
    V, lat, lon, seq = cached_queries(
        a.data, pick, dev, "kvq{}_{}".format(dim, len(pick)))
    groups = [[row[int(j)] for j in g] for g in gidx]
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
    # Read the *same merged meta* that build_knn was given, rather than listing
    # the parts by hand. Naming the parts here duplicates merge_bank_meta.py's
    # part-order contract, which is silent when broken: a wrong order still
    # concatenates to the right length and every address is then attached to
    # the wrong bank row.
    stem = "bank_ext"
    for n in ("70", "55", "40"):
        if "bank" + n in bank_file:
            stem = "bank_ext" + n
            break
    m = np.load(config.bank_meta(stem), allow_pickle=True)
    bx = np.concatenate([bx, m["x16"].astype(bx.dtype)])
    by = np.concatenate([by, m["y16"].astype(by.dtype)])
    assert len(bx) == bank.shape[0], (stem, len(bx), bank.shape[0])

    tbl = street_table(config.STREET_CACHE / bank_file, dev) \
        if ck.get("retr_mode") in ("pos", "dual") else None
    src = source_for(ck)
    K_use = ck.get("retr_k", 16)

    print("\n{}  ({})   {:,} groups, order={}, merge={}".format(
        a.tag, names.describe(a.tag), len(groups), a.order, a.merge))
    print("photos   n     median km      mean     <25km    <200km"
          "     vs 1 photo, 95% CI")
    print("-" * 86)
    base = None
    errs = {}
    for N in sizes:
        anchors, nidx, nsim = [], [], []
        for members in groups:
            # members[0] is the anchor and always leads, so every arm scores
            # the *same* photograph against the same ground truth and the only
            # thing that changes is how many others voted with it.
            take = members[:N]
            ci, cs = merge_candidates(take, idx32, sim32, K_use, a.merge)
            anchors.append(members[0])
            nidx.append(ci)
            nsim.append(cs)
        anchors = np.array(anchors)
        ext = ExternalSet(V[anchors], lat[anchors], lon[anchors],
                          np.stack(nidx), np.stack(nsim).astype(np.float32),
                          bx, by)
        m = evaluate(model, ext, src, dev, None, beam_k=a.beam,
                     top_m=max(4, a.beam), greedy=(a.beam == 1),
                     score_steps=a.score_steps, street_gpu=tbl)
        e = m["err"]
        errs[N] = e
        hit = float((e < 25).mean())
        if base is None:
            base = hit
        # Paired over groups: every arm scored the same anchors, so resample
        # groups, not arms. Without this the curve is point estimates with no
        # error bars, and at a few hundred groups a 1 pp step is a handful of
        # images changing -- which is how the `near` control produced a
        # non-monotone 8.0 -> 7.5 -> 6.4 -> 9.5 that looked like a finding.
        ci = ""
        if N != sizes[0]:
            b = errs[sizes[0]]
            rng = np.random.default_rng(0)
            ix = rng.integers(0, len(e), (3000, len(e)))
            d = (e < 25).astype(np.float64) - (b < 25)
            bs = d[ix].mean(1) * 100
            lo, hi = np.percentile(bs, [2.5, 97.5])
            # The median moves far more than the hit rate here, so it needs an
            # interval too -- resampled on the same groups, as a difference of
            # medians rather than a median of differences.
            md = np.median(e[ix], 1) - np.median(b[ix], 1)
            mlo, mhi = np.percentile(md, [2.5, 97.5])
            ci = ("   {:+.1f} pp [{:+.2f}, {:+.2f}] {:<12} "
                  "median {:+.0f} km [{:+.0f}, {:+.0f}] {}".format(
                      100 * (hit - base), lo, hi,
                      "separated" if lo * hi > 0 else "inside noise",
                      float(np.median(e) - np.median(b)), mlo, mhi,
                      "separated" if mlo * mhi > 0 else "inside noise"))
        print("{:>4}  {:>5,} {:>10.1f} {:>10.1f} {:>9.1%} {:>9.1%}{}".format(
            N, len(e), float(np.median(e)), float(e.mean()), hit,
            float((e < 200).mean()), ci), flush=True)

    if a.export:
        np.savez(a.export, sizes=np.array(sizes),
                 **{"err{}".format(N): errs[N] for N in sizes})
        print("wrote {}  (per-group errors, so the intervals can be redone "
              "without re-running)".format(a.export))
    print("\nOnly the retrieval prior sees the extra photographs; the policy "
          "still\nreads one street vector. No weights changed.")


if __name__ == "__main__":
    main()
