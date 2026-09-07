"""Web demo: upload a photograph, watch the agent navigate the map to it.

The interesting thing to show is not a pin on a map -- it is the descent. The
agent starts holding the whole world in one tile and picks one of 256 children
four times, so the answer comes with its own explanation: four map views, each
with the cell it chose. That is what this serves.

Run:
    python scripts/serve.py --tag s10_n400k_e2          # 3.7 GB bank, quick
    python scripts/serve.py --tag s10_n400k_bank25      # 10.6 GB bank, best

Everything is loaded once at startup: both encoders, the policy, and the
retrieval bank the checkpoint was trained against. A request then costs one
encoder pass, one matmul against the bank, and four map fetches.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import knnmeta
import shortlist as shortlist_mod
import provenance as prov
import splits as sp
import tile_math as tm
import tiles as T
from beam import source_for, search
from embed_street import preprocess
from evaluate import load_model, street_file_for
from tile_math import great_circle_km as great_circle_km

STATE = {}


def load_everything(tag, dev, bank_gpu, shortlist=0):
    """Model, encoders and bank. Done once; a request must not touch disk."""
    t0 = time.time()
    model, ck, d_street = load_model(tag, dev)
    sf = street_file_for(ck, d_street)
    mode = ck.get("split_mode", sp.PRIMARY)
    print("checkpoint {}  release {}  split {}  street {}".format(
        tag, ck.get("release", "s01"), mode, sf), flush=True)

    import pyarrow.parquet as pq
    ds = pq.read_table(config.DATASET_PARQUET)
    labels, _ = sp.read(ds, mode)
    tg = pq.read_table(config.TARGETS_PARQUET)
    per = tm.STEPS + 1
    tx = np.asarray(tg["tile_x"]).astype(np.int64).reshape(-1, per)[:, tm.STEPS]
    ty = np.asarray(tg["tile_y"]).astype(np.int64).reshape(-1, per)[:, tm.STEPS]
    lat = np.asarray(ds["lat"], dtype=np.float64)
    lon = np.asarray(ds["lon"], dtype=np.float64)

    # the bank is whatever this checkpoint's kNN cache was built from: the
    # split's train side, plus a bank extension if it had one
    if not ck.get("knn_file"):
        # A bare ck["knn_file"] made a non-retrieval checkpoint fail with an
        # incidental KeyError three frames deep. Say what is wrong instead.
        raise SystemExit(
            "this checkpoint records no knn_file, so it was trained without "
            "retrieval; serve.py needs a retrieval arm. Train with --retr or "
            "point --tag at one that has it.")
    # Every cache here is addressed by row order, so serving a checkpoint
    # from another release pairs each image with a different image's
    # embedding. The evaluator has always checked this; the server did not.
    from evaluate import check_split
    check_split(ck, ck.get("split_mode", sp.PRIMARY), "test")
    knn = np.load(config.STREET_CACHE / ck["knn_file"], allow_pickle=True)
    ext = str(knn["bank_ext"]) if "bank_ext" in knn else ""
    if "bank_rows" in knn:
        # provenance.bank_ext, not a bare np.load: check_split above
        # validates the checkpoint against the release, but the extension
        # metadata is a separately replaceable file, and a mismatched one
        # attaches another release's z16 addresses to valid bank rows --
        # plausible coordinates, wrong place, nothing to notice it.
        m = prov.bank_ext(ext, config.RELEASE) if ext else None
        # Validated rather than read: the server had no check on this file at
        # all, so a stale or replaced cache of the right name selected a
        # different bank while serving carried on normally, and a negative row
        # wrapped to the end instead of being refused. REVIEW4 #14. The bound
        # is release + extension, not the release alone -- bank rows run past
        # len(labels) whenever there is an extension, which is what the n_ext
        # count below is counting.
        rows = knnmeta.bank_rows(
            knn, len(labels) + (len(m["x16"]) if m is not None else 0),
            ck["knn_file"])
    else:
        from build_knn import bank_rows_for
        rows, m = bank_rows_for(ds, labels, mode, ext or None)
    if ext:
        tx = np.concatenate([tx, m["x16"].astype(np.int64)])
        ty = np.concatenate([ty, m["y16"].astype(np.int64)])
        lat = np.concatenate([lat, np.full(len(m["x16"]), np.nan)])
        lon = np.concatenate([lon, np.full(len(m["x16"]), np.nan)])
        n_ext = int((rows >= len(labels)).sum())
        print("bank ext   {} contributes {:,} of its {:,} rows"
              .format(ext, n_ext, len(m["x16"])), flush=True)

    emb = np.load(config.STREET_CACHE / sf, mmap_mode="r")
    gb = len(rows) * emb.shape[1] * 2 / 1e9
    print("bank       {:,} images, {:.2f} GB fp16 -> {}".format(
        len(rows), gb, "gpu" if bank_gpu else "host"), flush=True)
    dst = dev if bank_gpu else "cpu"
    B = torch.empty((len(rows), emb.shape[1]), dtype=torch.float16, device=dst)
    for lo in range(0, len(rows), 8192):
        hi = min(lo + 8192, len(rows))
        blk = torch.from_numpy(np.asarray(emb[rows[lo:hi]], dtype=np.float32))
        B[lo:hi] = (blk / blk.norm(dim=1, keepdim=True).clamp_min(1e-6)).half().to(dst)

    # A resident coarse index over the same rows. The exact bank stays where it
    # is and is read only for the rows this proposes, so what crosses PCIe per
    # query goes from the whole 5.22 GB to a gather of `probe` rows.
    STATE["shortlist"] = None
    if shortlist:
        STATE["shortlist"] = shortlist_mod.Shortlist(B, dev, dim=shortlist)
        print("shortlist  {}-d index, {:.2f} GB on {}  (probe {})".format(
            STATE["shortlist"].dim, STATE["shortlist"].gb, dev,
            STATE["shortlist"].probe), flush=True)

    import timm
    from timm.data import resolve_model_data_config
    encs = []
    for name in ("vit_base_patch14_dinov2.lvd142m",
                 "vit_base_patch16_siglip_224.v2_webli"):
        m_ = timm.create_model(name, pretrained=True, num_classes=0,
                               img_size=224).eval().to(dev)
        cfg = resolve_model_data_config(m_)
        encs.append((m_,
                     np.array(cfg["mean"], np.float32).reshape(3, 1, 1),
                     np.array(cfg["std"], np.float32).reshape(3, 1, 1)))
        print("encoder    {}".format(name), flush=True)

    STATE.update(model=model, ck=ck, dev=dev, bank=B, rows=rows,
                 x16=tx, y16=ty, lat=lat, lon=lon, encs=encs,
                 source=source_for(ck),
                 client=T.TileClient(config.TILE_SERVER),
                 street=torch.from_numpy(np.asarray(emb)) if not bank_gpu else None,
                 k=ck.get("retr_k", 16))
    # sf is the checkpoint's own street file, resolved above
    # Both of these were decided by substrings in the filename: the SigLIP
    # scale by `"bal" in sf or "pca768" in sf`, and the basis by a hardcoded
    # name whatever produced the bank. Serving is the one place a wrong answer
    # reaches a user rather than a report, so it prefers the record and says
    # which it used.
    _bank = config.STREET_CACHE / sf
    _enc = prov.encoder_of(_bank)
    STATE["enc_scale"] = [1.0, float(_enc["siglip_scale"])
                          if "siglip_scale" in _enc
                          else (4.03 if "bal" in sf or "pca768" in sf else 1.0)]
    STATE["pca"] = None
    if STATE["bank"].shape[1] == 768:
        basis = prov.projection_of(_bank)
        if not basis:
            basis = "pca768_bank55_pca.npz"
            print("warning: {} does not record its PCA basis, so {} is "
                  "assumed. Width was the only thing ever checked, and "
                  "several bases here share one.".format(sf, basis),
                  flush=True)
        print("basis      {}".format(basis), flush=True)
        z = np.load(config.STREET_CACHE / basis)
        STATE["pca"] = (torch.from_numpy(z["mu"]).to(STATE["dev"]),
                        torch.from_numpy(z["P"]).to(STATE["dev"]))
    print("query     {}-d, siglip x{:.2f}{}".format(
        STATE["bank"].shape[1], STATE["enc_scale"][1],
        ", PCA basis loaded" if STATE["pca"] is not None else ""), flush=True)
    print("ready in {:.1f}s".format(time.time() - t0), flush=True)


D_ENC = 768


@torch.no_grad()
def load_panel(dev):
    """Two banks over the same rows: crops only, and crops + tiles.

    The point of the panel is that these differ ONLY in the representation.
    Same images, same split, same width (1536), so a difference in what comes
    back is the representation and nothing else.

    Restricted to the rows `tile_cache.py` has actually tiled -- 120,000 of
    the release, of which the split's train side is the bank. The shipping
    bank is 3.4M rows and 2.8% of it has tiles, so this cannot be the agent's
    bank; it is a side-by-side lookup, and it is labelled as one.
    """
    import pyarrow.parquet as pq
    from osv_pyramid import complete_rows, load_tokens_blocked, level_vectors

    # Finished rows only. tile_cache grows in place, so a cache caught during
    # a pass is mostly zero fill, and a zero row L2-normalises to a
    # unit-length nothing that ranks like a real vector.
    sel, pos = complete_rows("tile6")
    ds = pq.read_table(config.DATASET_PARQUET)
    labels, _ = sp.read(ds, STATE["ck"].get("split_mode", sp.PRIMARY))
    keep = np.flatnonzero(labels[sel] == "train")

    X = load_tokens_blocked(sel, pos)
    lv = level_vectors(X, [0] * 3 + [1] * (X.shape[1] - 3), dev)
    crops = lv[0][keep]
    mix = (lv[0][keep] + lv[1][keep]) / 2.0
    mix /= np.linalg.norm(mix, axis=1, keepdims=True).clip(1e-6)

    rows = sel[keep]
    STATE["panel"] = {
        "crops": torch.from_numpy(crops.astype(np.float32)).to(dev),
        "both": torch.from_numpy(mix.astype(np.float32)).to(dev),
        "lat": np.asarray(ds["lat"], np.float64)[rows],
        "lon": np.asarray(ds["lon"], np.float64)[rows],
        "n": len(rows),
    }
    print("panel      {:,} tiled bank rows, crops and crops+tiles at 1536-d"
          .format(len(rows)), flush=True)


def panel_query(blob, dev):
    """The uploaded image as both representations, 1536-d each.

    Assembled exactly the way the bank rows were: per-token L2, then the mean
    within a level, then L2 per encoder. Query and bank have to be built the
    same way or the comparison is between two spaces rather than two
    representations.
    """
    from tile_cache import tile_uint8

    nrm = lambda v: v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    per = []
    # no_grad, not merely eval(): without it every request builds a graph it
    # never frees, and the first .numpy() on the result raises instead of
    # answering -- which is exactly how this failed the first time.
    with torch.no_grad():
        for (m_, mean, std) in STATE["encs"]:
            mu = torch.as_tensor(mean, device=dev).view(1, 3, 1, 1)
            sd = torch.as_tensor(std, device=dev).view(1, 3, 1, 1)
            x = torch.from_numpy(
                preprocess(blob, crops=3, mean=mean, std=std)).to(dev)
            t = torch.from_numpy(
                tile_uint8(blob, 3, 2).astype(np.float32) / 255.0)
            t = ((t.permute(0, 3, 1, 2).to(dev)) - mu) / sd
            with torch.autocast(dev, dtype=torch.bfloat16,
                                enabled=(dev == "cuda")):
                fc = m_(x).float()
                ft = m_(t).float()
            per.append((nrm(nrm(fc).mean(0, keepdim=True)),
                        nrm(nrm(ft).mean(0, keepdim=True))))

    l0 = nrm(torch.cat([c for c, _ in per], dim=1))
    l1 = nrm(torch.cat([t for _, t in per], dim=1))
    return l0, nrm((l0 + l1) / 2.0)


def embed(blob):
    """Upload bytes -> exactly the vector this checkpoint's bank is made of.

    This has to track the bank, not a fixed width. Three schemes are in use:

        4608  three crops concatenated, both encoders
        1536  the same, mean-pooled over crops (pool_street.py)
         768  that, through the saved PCA basis (project_street.py)

    and a `_bal` bank additionally scales SigLIP by 4.03 before joining, because
    the two encoders' raw norms are 83 and 21 and joining them unscaled weights
    the cosine 81/19 toward DINOv2. Getting any of this wrong produces a vector
    in the wrong space, and the failure looks like a bad photograph rather than
    a bug -- this server predated pooling and silently had no scaling at all.
    """
    dev = STATE["dev"]
    parts = []
    for (m_, mean, std), scale in zip(STATE["encs"], STATE["enc_scale"]):
        x = preprocess(blob, crops=3, mean=mean, std=std)
        x = torch.from_numpy(x).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            f = m_(x)
        parts.append(scale * f.float().reshape(1, -1))   # 3 crops concatenated
    q = torch.cat(parts, dim=1)                          # dinov2 | siglip
    want = STATE["bank"].shape[1]
    if want == q.shape[1]:
        return q
    half = q.shape[1] // 2
    nc = half // D_ENC
    pooled = torch.cat([q[:, :half].reshape(1, nc, D_ENC).mean(1),
                        q[:, half:].reshape(1, nc, D_ENC).mean(1)], dim=1)
    if want == pooled.shape[1]:
        return pooled
    P = STATE.get("pca")
    if P is None or want != P[1].shape[1]:
        raise RuntimeError("bank is {}-d; no projection to match".format(want))
    return (pooled - P[0]) @ P[1]


def _neighbours(q):
    """Top-k bank rows for one query vector, as `(scores, rows)`.

    With a shortlist this is one pass over a resident 128-d index plus an
    exact rescore of the candidates; without one it is the full scan, kept as
    the reference path and reachable with `--shortlist 0`.
    """
    dev, k = STATE["dev"], STATE["k"]
    qn = (q / q.norm(dim=1, keepdim=True).clamp_min(1e-6)).half()
    sl = STATE.get("shortlist")
    if sl is not None:
        s, rows = sl.topk(qn, k)
        return s.cpu(), rows.cpu()
    B = STATE["bank"]
    sims = torch.empty(len(STATE["rows"]), dtype=torch.float32)
    step = 200000
    for lo in range(0, B.shape[0], step):
        blk = B[lo:lo + step]
        blk = blk if blk.device.type == dev else blk.to(dev, non_blocking=True)
        sims[lo:lo + step] = (qn.to(dev) @ blk.T).float().flatten().cpu()
    # A request for more neighbours than the bank holds is a RuntimeError
    # from torch, not a useful answer.
    return sims.topk(min(k, sims.shape[-1]))


@torch.no_grad()
def locate(blobs, beam_k=4, score_steps=3):
    """One or more photographs of the same place.

    The benchmark scores 2.7 km on OSV-5M and 442 km on novel photographs, and
    the cause is bank coverage: OSV-5M's test images come from streets the bank
    covers densely. Density on the bank side costs hours of encoding. Density on
    the *query* side is free -- several photographs taken from one spot agree on
    the true location and disagree on their spurious matches.

    Nothing was retrained for this. The retrieval prior takes k neighbours as a
    similarity-weighted set over their z16 addresses and cannot tell which image
    produced them, so extra photographs simply contribute more candidates. The
    policy still reads a single street vector -- the first image.
    """
    if isinstance(blobs, (bytes, bytearray)):
        blobs = [blobs]
    dev, k = STATE["dev"], STATE["k"]
    B = STATE["bank"]
    qs = [embed(b) for b in blobs]
    q = qs[0]                                   # the primary drives the policy

    per = [_neighbours(qi) for qi in qs]

    # Each photograph gets its own share of the k slots, round-robin by rank,
    # rather than the k best similarities overall. Similarities are not
    # calibrated across photographs -- one view is simply more typical of the
    # bank than another -- so a global top-k lets the strongest photograph fill
    # every slot and the rest contribute nothing. Measured: four photographs
    # through a global top-k changed the answer in one case out of five.
    seen, picks = set(), []
    for rank in range(k):
        for si, ji in per:
            if len(picks) == k:
                break
            if rank >= len(ji):
                continue
            r = int(ji[rank])
            if r in seen:                      # two photographs, one bank image
                continue
            seen.add(r)
            picks.append((float(si[rank]), r))
        if len(picks) == k:
            break
    picks.sort(key=lambda t: -t[0])            # the prior still sees them ranked
    s = torch.tensor([p[0] for p in picks], dtype=torch.float32)
    j = torch.tensor([p[1] for p in picks], dtype=torch.long)
    rows = STATE["rows"][j.numpy()]

    nbrs = [torch.from_numpy(STATE["x16"][rows][None, :]).to(dev),
            torch.from_numpy(STATE["y16"][rows][None, :]).to(dev),
            s[None, :].to(dev)]
    if STATE["ck"].get("retr_mode") in ("pos", "dual"):
        nbrs.append(B[j].float()[None, :].to(dev))

    res = search(STATE["model"], q, STATE["source"], dev, beam_k,
                 max(4, beam_k), score_steps=score_steps, nbrs=nbrs)[0]
    best = res["best"]

    # rebuild the tile the agent was looking at before each choice, so the page
    # can show the descent rather than only its conclusion
    steps, z, x, y = [], 0, 0, 0
    for t, a in enumerate(best["path"]):
        steps.append({"step": t, "z": z, "x": x, "y": y, "action": int(a),
                      "row": int(a) // tm.G, "col": int(a) % tm.G,
                      "km": [2504, 156, 9.8, 0.611][t]})
        z, x, y = tm.descend(z, x, y, int(a), tm.G)
    steps.append({"step": tm.STEPS, "z": z, "x": x, "y": y, "action": None,
                  "row": None, "col": None, "km": 0.611})

    return {
        "lat": best["lat"], "lon": best["lon"],
        "radius_km": res["confidence_radius_km"],
        "path": [int(a) for a in best["path"]],
        "photos": len(blobs),
        "steps": steps,
        "candidates": [{"lat": c["lat"], "lon": c["lon"], "score": c["score"]}
                       for c in res["candidates"][:beam_k]],
        "neighbours": [
            {"sim": float(sv),
             "lat": None if np.isnan(STATE["lat"][r]) else float(STATE["lat"][r]),
             "lon": None if np.isnan(STATE["lon"][r]) else float(STATE["lon"][r])}
            for sv, r in zip(s.tolist(), rows)],
    }


def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    app = FastAPI(title="Geolocation agent")
    page = (Path(__file__).resolve().parent / "static" / "index.html")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return page.read_text(encoding="utf-8")

    @app.post("/locate")
    async def do_locate(request: Request):
        ct = request.headers.get("content-type", "")
        blobs = []
        if ct.startswith("multipart/form-data"):
            form = await request.form()
            for key in form:
                for v in form.getlist(key):
                    if hasattr(v, "read"):
                        blobs.append(await v.read())
        else:
            raw = await request.body()
            if raw:
                blobs = [raw]
        blobs = [b for b in blobs if b]
        if not blobs:
            return JSONResponse({"error": "no image"}, status_code=400)
        t0 = time.time()
        try:
            out = locate(blobs)
        except Exception as e:                     # a bad upload must not 500
            return JSONResponse({"error": str(e)}, status_code=400)
        out["secs"] = round(time.time() - t0, 2)
        return JSONResponse(out)

    @app.post("/compare")
    async def do_compare(request: Request):
        """Same image, same rows, two representations -- nothing else differs.

        Deliberately NOT the agent. The agent runs on the 3.4M shipping bank,
        of which 2.8% has tiles; this looks up a 96k tiled subset. So the two
        lists here are comparable to each other and not to /locate.
        """
        if "panel" not in STATE:
            return JSONResponse({"error": "started without --tiles-panel"},
                                status_code=503)
        ct = request.headers.get("content-type", "")
        if ct.startswith("multipart/form-data"):
            form = await request.form()
            blob = await next(iter(form.values())).read()
        else:
            blob = await request.body()
        t0 = time.time()
        P, dev = STATE["panel"], STATE["dev"]
        q_crops, q_both = panel_query(blob, dev)
        out = {}
        for name, q, B in (("crops", q_crops, P["crops"]),
                           ("crops_tiles", q_both, P["both"])):
            sims = (B @ q.reshape(-1, 1)).reshape(-1)
            v, j = torch.topk(sims, 5)
            j = j.cpu().numpy()
            out[name] = [{"lat": float(P["lat"][r]), "lon": float(P["lon"][r]),
                          "sim": float(x)} for r, x in zip(j, v.cpu().numpy())]
        # how far apart the two answers are, which is the thing to look at
        a, b = out["crops"][0], out["crops_tiles"][0]
        out["top1_gap_km"] = round(float(great_circle_km(
            a["lat"], a["lon"], b["lat"], b["lon"])), 1)
        out["bank"] = P["n"]
        out["secs"] = round(time.time() - t0, 3)
        return JSONResponse(out)

    @app.get("/map/{z}/{x}/{y}.png")
    def map_tile(z: int, x: int, y: int):
        # proxied so the page never needs to reach the tile server itself
        try:
            raw = STATE["client"].png(z, x, y)
        except Exception:
            return Response(status_code=404)
        return Response(raw, media_type="image/png")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s10_n400k_e2")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tiles-panel", action="store_true",
                    help="also serve /compare: the same image looked up under "
                         "crops and under crops+tiles, over the 96k bank rows "
                         "that have tiles cached. A side-by-side of the "
                         "representation, not a change to the agent.")
    ap.add_argument("--bank-gpu", action="store_true",
                    help="hold the exact bank in VRAM; only for smaller banks")
    ap.add_argument("--shortlist", type=int, default=shortlist_mod.DIM,
                    help="width of the resident coarse index; 0 searches the "
                         "bank exhaustively, which is the reference path. The "
                         "default fits the 3.4M bank in 0.87 GB of VRAM where "
                         "the exact one needs 5.22 GB, and returns the same "
                         "rows -- see src/shortlist.py for the measurements.")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    load_everything(a.tag, dev, a.bank_gpu, a.shortlist)
    if a.tiles_panel:
        load_panel(dev)
    import uvicorn
    print("\n  http://{}:{}\n".format(a.host, a.port), flush=True)
    uvicorn.run(build_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
