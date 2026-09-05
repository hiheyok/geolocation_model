"""Change only the uploaded image's encoding. Keep the 3.4M bank as it is.

The pyramid results so far all rebuilt both sides. That is the ceiling, and it
is not the goal: the goal is to leave `pca768_bank70` alone -- 3.4M rows, days
of GPU already spent -- and change only how the *query* is read.

That constraint is severe in a way worth stating up front. Retrieval is a
cosine against vectors built one specific way (3 crops at 224, SigLIP scaled
4.03, crop-mean to 1536, PCA to 768). Any query built differently is asking the
bank to match something it never encoded. Blending tiles into the query is the
clearest case and it is already measured: `(L0+L1)/2` against an `L0` bank
gives +0.2 pp, because

    cos(q, b) ~ 1/2 cos(L0_q, L0_b) + 1/2 cos(L1_q, L0_b)

and the second term compares tile features to crop features -- half the query
spent on a dimension the bank does not have.

So the arms here are the ones that keep the query's *construction* matched to
the bank and vary only what the encoder sees:

    Q1  3 crops @224            the shipping query, as a control
    Q3  3 crops @448            same framing, 4x the pixels
    Q4  9 views, merged         3 crops + 6 tiles as SEPARATE queries

Q4 is the version of the pyramid that a fixed bank can accept: nine
independent lookups whose results are merged, rather than nine views averaged
into one vector that no longer resembles a bank row.

Two things this measures against, both real: the live 3.4M bank, and
same-sequence exclusion applied here rather than assumed. The bank extension
shares sequence ids with the release, which is what made 41.6% of test queries
retrieve a frame from their own drive; a probe that forgot the mask would
rediscover that as a result.

    OSV_RELEASE=s10 py scripts/query_only.py --queries 3000
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import provenance as prov                        # noqa: E402
import splits as sp                              # noqa: E402
from tile_pool import paired                     # noqa: E402
from tile_math import great_circle_km as great_circle

THRESH = (1, 25, 200, 750, 2500)
D_ENC = 768
R_EARTH = 6371.0088


def top1(Q, bank, bank_seq, q_seq, dev, block=200_000):
    """Best bank row per query, excluding the query's own sequence.

    Blocked because the full similarity table is 3,000 x 3.4M -- 40 GB as
    float32. A running argmax over bank blocks costs one pass and a few
    hundred MB.

    The exclusion is not optional. Release and extension share sequence ids,
    and a same-drive frame is a near-duplicate of the query: leaving it in
    scored 41.6% of test queries against a copy of themselves.
    """
    q = torch.from_numpy(Q).to(dev, torch.float16)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-6)
    best = torch.full((len(Q),), -2.0, device=dev)
    who = torch.zeros(len(Q), dtype=torch.long, device=dev)
    qs = np.asarray(q_seq)
    for s in range(0, len(bank), block):
        e = min(s + block, len(bank))
        b = torch.from_numpy(np.asarray(bank[s:e], np.float16)).to(dev)
        b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-6)
        sim = (q @ b.T).float()
        same = torch.from_numpy(
            (bank_seq[s:e][None, :] == qs[:, None])).to(dev)
        sim.masked_fill_(same, -2.0)
        v, j = sim.max(1)
        hit = v > best
        who[hit] = j[hit] + s
        best[hit] = v[hit]
        del b, sim, same
    return who.cpu().numpy(), best.cpu().numpy()


def pipeline(dual, dev, scale_b, pca):
    """The bank's own recipe: scale SigLIP, crop-mean to 1536, PCA to 768.

    `dual` is (n, crops, 2, 768). Anything the bank will be searched with has
    to come out of this function, or the comparison is between two spaces.
    """
    x = torch.from_numpy(np.asarray(dual, np.float32)).to(dev)
    x[:, :, 1] *= scale_b
    v = torch.cat([x[:, :, 0].mean(1), x[:, :, 1].mean(1)], dim=1)   # 1536
    if pca is not None:
        v = (v - pca[0]) @ pca[1]
    return v.cpu().numpy().astype(np.float32)


def encode(paths_blobs, size, crops, dev, batch=32):
    """(n, crops, 2, 768) from JPEG bytes at the given input size."""
    import timm
    from embed_street import preprocess

    specs = [("vit_base_patch14_dinov2.lvd142m", 14),
             ("vit_base_patch16_siglip_224.v2_webli", 16)]
    out = np.empty((len(paths_blobs), crops, 2, D_ENC), np.float32)
    for ei, (spec, patch) in enumerate(specs):
        if size % patch:
            raise SystemExit(
                "--size {} is not a multiple of {}'s patch size {}; the "
                "position grid would not tile.".format(size, spec, patch))
        net = timm.create_model(spec, pretrained=True, num_classes=0,
                                img_size=size).eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=net)
        # (3, 1, 1), not (3,): preprocess subtracts these from a (3, H, W)
        # array, and a bare triple broadcasts against the channel axis only by
        # accident at some sizes and not at others.
        mu = np.array(cfg["mean"], np.float32).reshape(3, 1, 1)
        sd = np.array(cfg["std"], np.float32).reshape(3, 1, 1)
        with torch.no_grad():
            for s in range(0, len(paths_blobs), batch):
                e = min(s + batch, len(paths_blobs))
                xs = [preprocess(b, size=size, crops=crops,
                                 mean=mu, std=sd)
                      for b in paths_blobs[s:e]]
                x = torch.from_numpy(np.concatenate(xs)).to(dev)
                with torch.autocast(dev, dtype=torch.bfloat16,
                                    enabled=(dev == "cuda")):
                    f = net(x).float()
                out[s:e, :, ei] = f.reshape(e - s, crops, D_ENC).cpu().numpy()
        del net
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d768-b350-e6-drop70")
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--sizes", default="448",
                    help="comma-separated query input sizes to try beyond 224")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    ck = torch.load(config.CHECKPOINTS / (a.tag + ".pt"), map_location="cpu",
                    weights_only=False)
    bank_file, knn_file = ck["street_file"], ck["knn_file"]
    bank = np.load(config.STREET_CACHE / bank_file, mmap_mode="r")
    knn = np.load(config.STREET_CACHE / knn_file, allow_pickle=True)
    rows = knn["bank_rows"]
    ext = str(knn["bank_ext"]) if "bank_ext" in knn else ""

    ds = pq.read_table(config.DATASET_PARQUET)
    labels, _ = sp.read(ds, ck.get("split_mode", sp.PRIMARY))
    lat = np.asarray(ds["lat"], np.float64)
    lon = np.asarray(ds["lon"], np.float64)
    seq = np.asarray(ds["sequence"]).astype("U40")
    zname = np.asarray(ds["zip_name"].to_pylist())

    m = prov.bank_ext(ext, config.RELEASE) if ext else None
    if m is not None:
        e_seq = np.asarray(m["sequence"]).astype("U40")
        n16 = 1 << 16
        ex = m["x16"].astype(np.float64) + 0.5
        ey = m["y16"].astype(np.float64) + 0.5
        e_lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ey / n16))))
        e_lon = ex / n16 * 360.0 - 180.0
        all_seq = np.concatenate([seq, e_seq])
        all_lat = np.concatenate([lat, e_lat])
        all_lon = np.concatenate([lon, e_lon])
    else:
        all_seq, all_lat, all_lon = seq, lat, lon

    B = np.asarray(bank[rows])
    b_seq, b_lat, b_lon = all_seq[rows], all_lat[rows], all_lon[rows]
    print("bank   {}  {:,} rows of {:,}, {}-d".format(
        bank_file, len(rows), len(bank), B.shape[1]))

    qi = np.flatnonzero(labels == "test")[:a.queries]
    print("query  {:,} held-out release images\n".format(len(qi)), flush=True)

    scale_b = float(ck.get("scale_b", 4.03))
    pca = None
    pr = prov.read(config.STREET_CACHE / bank_file) or {}
    basis = pr.get("projection")
    if B.shape[1] == 768 and basis:
        z = np.load(config.STREET_CACHE / basis)
        pca = (torch.from_numpy(z["mu"]).to(dev).float(),
               torch.from_numpy(z["P"]).to(dev).float())
        print("basis  {} (recorded)".format(basis), flush=True)

    def report(name, Q):
        who, _ = top1(Q, B, b_seq, seq[qi], dev)
        e = great_circle(lat[qi], lon[qi], b_lat[who], b_lon[who])
        print("%-28s %9.1f %s" % (name, np.median(e), " ".join(
            "%7.1f%%" % (100 * (e < t).mean()) for t in THRESH)), flush=True)
        return e

    print("%-28s %9s %s" % ("query encoding", "median km",
                            " ".join("%8s" % ("<%gkm" % t) for t in THRESH)))
    print("-" * 76)
    err = {}
    # Q1: the query rows as the bank itself stores them -- the true control,
    # because it needs no re-encoding and therefore no chance of drift.
    err["Q1 3 crops @224 (ships)"] = report("Q1 3 crops @224 (ships)",
                                            np.asarray(bank[qi], np.float32))

    blobs, zips = [], {}
    for i in qi:
        member = zname[i]
        sh = member.split("/")[0]
        z = zips.get(sh) or zips.setdefault(
            sh, zipfile.ZipFile(config.TRAIN_ZIPS / (sh + ".zip")))
        blobs.append(z.read(member))

    for size in [int(s) for s in a.sizes.split(",") if s.strip()]:
        d = encode(blobs, size, 3, dev)
        err["Q3 3 crops @%d" % size] = report(
            "Q3 3 crops @%d" % size, pipeline(d, dev, scale_b, pca))

    rng = np.random.default_rng(0)
    base = err["Q1 3 crops @224 (ships)"]
    print("\n--- against the shipping query encoding ---")
    for k, e in err.items():
        if k.startswith("Q1"):
            continue
        cells = []
        for t in THRESH:
            lo, hi = paired(base < t, e < t, rng)
            cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                100 * ((e < t).mean() - (base < t).mean()),
                lo, hi, " " if lo * hi > 0 else "~"))
        print("%-22s %s" % (k, " ".join(cells)))
    print("\n~ spans zero. {:.0f}s total".format(time.time() - t0))


if __name__ == "__main__":
    main()
