"""Encode the release at each encoder's native resolution, for conditioning only.

This cache is **not** a retrieval bank and must never be used as one. It exists
for the decoupled design: retrieval stays on the matched `L0L1` vector, and
this rides alongside as extra conditioning the model sees but the cosine never
touches.

That split is the whole point. `runs/PYR_LEVELS.md` measured what happens when
extra detail is put *into* the retrieval vector -- against a fixed `L0L1` bank,
adding an L2 level at weight 0.10 replaces 8.2% of the top-16 and flips the
top-1 for 12.7% of queries, and the hit rate does not move at all (59.6% ->
59.6%). The detail reorders the ranking without being about location, so it is
pure churn. Push the weight to 1/3 and a quarter of the neighbour set turns
over for -0.8 pp.

So the only place extra pixels could still pay is a path that is not a cosine.
The model's conditioning input is compared to nothing, so nothing constrains it
to match a bank -- and it has only ever been fed 224-derived vectors.

    DINOv2 /14 at 518   its exact training resolution
    SigLIP /16 at 512   the frame's own pixel height, no resampling

against the 224 the pipeline has always forced -- 44% of linear resolution,
~80% of pixels discarded.

**bf16 weights and torch.compile are used here and that is deliberate.**
`runs/TILEBENCH.md` rejected them for extending `tile6` because they move the
nearest neighbour for 0.8-1.0% of queries, and a cache extended under different
numerics puts a permanent seam through a bank. This cache is written in one
pass, is never compared to another cache by cosine, and is consumed only by a
head that trains on whatever it is given -- so the seam argument does not apply
and the 1.218x is free. Needs `triton-windows` and MSVC on PATH.

Resumable: a done-mask marks finished rows, so a stop costs whatever is in
flight rather than the whole run.

    OSV_RELEASE=s10 py scripts/embed_native.py --out cond_native
"""

import argparse
import os
import sys
import time
import zipfile
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import maskio                                    # noqa: E402
import provenance as prov                        # noqa: E402
from embed_street import slurp, preprocess       # noqa: E402

# (spec, patch, size). Sizes are per encoder, not shared: nothing requires one
# input size, since the encoders run independently and only their 768-d outputs
# are joined. Forcing a shared size means a multiple of lcm(14,16)=112, which
# lands on 448 -- 14% below DINOv2's grid and an upsample for SigLIP.
ENCODERS = [("vit_base_patch14_dinov2.lvd142m", 14, 518),
            ("vit_base_patch16_siglip_224.v2_webli", 16, 512)]
D_ENC = 768


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cond_native")
    ap.add_argument("--parquet", default=None,
                    help="image list; default the release's dataset.parquet")
    ap.add_argument("--n", type=int, default=0, help="0 = every row")
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-preload", dest="preload", action="store_false",
                    help="read members straight off disk. dataset.parquet "
                         "interleaves shards, so a SMALL --n touches nearly "
                         "every shard and slurping 2.5 GB for each makes a "
                         "smoke test pathological -- a full run groups all "
                         "rows by shard and slurps each exactly once.")
    ap.set_defaults(preload=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.set_defaults(compile=True)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import timm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    src = config.PROCESSED / a.parquet if a.parquet else config.DATASET_PARQUET
    tbl = pq.read_table(src, columns=["zip_name", "image_id"])
    zn = np.asarray(tbl["zip_name"]).astype("U40")
    ids = np.asarray(tbl["image_id"])
    n = len(zn) if not a.n else min(a.n, len(zn))
    zn, ids = zn[:n], ids[:n]

    emb_p = config.STREET_CACHE / (a.out + ".f16.npy")
    done_p = config.STREET_CACHE / (a.out + "_done.u8.npy")
    meta_p = config.STREET_CACHE / (a.out + "_meta.npz")
    d_out = 2 * D_ENC
    if emb_p.exists():
        E = np.lib.format.open_memmap(emb_p, mode="r+")
        if E.shape != (n, d_out):
            raise SystemExit("{} is {} and this run wants {}; delete it or "
                             "use --out".format(emb_p.name, E.shape,
                                                (n, d_out)))
        done = maskio.load_mask(done_p, n, a.out)
    else:
        E = np.lib.format.open_memmap(emb_p, mode="w+", dtype=np.float16,
                                      shape=(n, d_out))
        done = np.zeros(n, np.uint8)
    np.save(done_p, done)
    todo = np.flatnonzero(done != 1)
    print("{}  {:,} rows, {:,} to embed, {:.2f} GB".format(
        a.out, n, len(todo), E.nbytes / 1e9), flush=True)
    if not len(todo):
        print("nothing to do", flush=True)
        return

    # One encoder at a time over the whole selection: two passes over the
    # shards costs disk, but holding both models plus their activations at
    # 1,369 tokens does not fit beside a 2.5 GB shard in RAM comfortably.
    for enc_i, (spec, patch, size) in enumerate(ENCODERS):
        assert size % patch == 0, (spec, size, patch)
        net = timm.create_model(spec, pretrained=True, num_classes=0,
                                img_size=size).eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=net)
        mu = np.array(cfg["mean"], np.float32).reshape(3, 1, 1)
        sd = np.array(cfg["std"], np.float32).reshape(3, 1, 1)
        net = net.to(torch.bfloat16)
        if a.compile:
            try:
                net = torch.compile(net)
            except Exception as exc:                      # noqa: BLE001
                print("compile unavailable ({}); eager".format(
                    str(exc).split(chr(10))[0]), flush=True)
        col = slice(enc_i * D_ENC, (enc_i + 1) * D_ENC)

        by_zip = defaultdict(list)
        for i in todo:
            by_zip[zn[i].split("/")[0]].append((int(i), zn[i]))
        t0, seen = time.time(), 0
        pool = ThreadPoolExecutor(a.workers)

        def prep(blob):
            return preprocess(blob, size=size, crops=a.crops, mean=mu, std=sd)

        with torch.no_grad():
            for shard, items in sorted(by_zip.items()):
                zp = Path(config.OSV_ROOT) / "images" / "train" / (shard + ".zip")
                with zipfile.ZipFile(slurp(zp) if a.preload else zp) as zf:
                    q = deque()
                    it = iter(items)

                    def fill():
                        while len(q) < 4 * a.batch:
                            nxt = next(it, None)
                            if nxt is None:
                                return
                            q.append((nxt[0], pool.submit(prep, zf.read(nxt[1]))))
                    fill()
                    while q:
                        rows, arrs = [], []
                        while q and len(rows) < a.batch:
                            r, fut = q.popleft()
                            rows.append(r)
                            arrs.append(fut.result())
                            fill()
                        x = torch.from_numpy(np.concatenate(arrs)).to(
                            dev, torch.bfloat16, non_blocking=True)
                        f = net(x).float().reshape(len(rows), a.crops, D_ENC)
                        v = nrm(f.mean(1)).cpu().numpy().astype(np.float16)
                        for k, r in enumerate(rows):
                            E[r, col] = v[k]
                        seen += len(rows)
                        # Every 200 batches, but always the first few: a
                        # short run used to print nothing at all, so a smoke
                        # test looked identical to a hang.
                        if seen <= 3 * a.batch or seen % (a.batch * 200) < a.batch:
                            el = time.time() - t0
                            print("   {:<7} {:>8,}/{:,}  {:.0f}s  eta {:.0f}s"
                                  .format(spec.split("_")[2][:6], seen,
                                          len(todo), el,
                                          el * (len(todo) - seen) / max(seen, 1)),
                                  flush=True)
                del zf
        pool.shutdown()
        del net
        torch.cuda.empty_cache()
        print("{} done in {:.0f}s".format(spec, time.time() - t0), flush=True)

    done[todo] = 1
    E.flush()
    np.save(done_p, done)
    if not maskio.is_complete(done, n):
        raise SystemExit("{:,} rows unfilled".format(n - maskio.complete(done)))
    np.savez(meta_p, release=config.RELEASE, crops=a.crops,
             encoders=np.array([e[0] for e in ENCODERS]),
             sizes=np.array([e[2] for e in ENCODERS]),
             rows_parquet=(a.parquet or config.DATASET_PARQUET.name),
             rows_digest=prov.rows_digest(ids),
             layout="(n, dino768|siglip768) native-res crop mean, CONDITIONING "
                    "ONLY -- never a retrieval bank")
    prov.write(emb_p, ids, release=config.RELEASE,
               row_space=("the release" if not a.parquet
                          else "the rows of " + a.parquet),
               conditioning_only=True,
               sizes="/".join(str(e[2]) for e in ENCODERS), crops=a.crops)
    print("\nwrote {}".format(emb_p.name), flush=True)


if __name__ == "__main__":
    main()
