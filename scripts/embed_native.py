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
    SigLIP /16 at 512   `..._siglip_512.v2_webli`, trained at 512

against the 224 the pipeline has always forced -- 44% of linear resolution,
~80% of pixels discarded.

**Both encoders are checkpoints actually trained at these sizes.** The obvious
mistake here, and one I made first, is to keep the shipping
`vit_base_patch16_siglip_224.v2_webli` and pass `img_size=512`: timm accepts it
and interpolates the position embeddings, so it runs and produces plausible
vectors from a model that has never seen a 32x32 token grid. That is not native
resolution, it is a 224 model extrapolating, and calling it native would make
any null result unreadable -- "more pixels do not help" and "this checkpoint
cannot use them" look identical. timm ships a real 512 variant of the same
family and the same v2_webli weights, so there is no reason to interpolate.

**bf16 weights and torch.compile are used here and that is deliberate.**
`runs/TILEBENCH.md` rejected them for extending `tile6` because they move the
nearest neighbour for 0.8-1.0% of queries, and a cache extended under different
numerics puts a permanent seam through a bank. This cache is written in one
pass, is never compared to another cache by cosine, and is consumed only by a
head that trains on whatever it is given -- so the seam argument does not apply
and the 1.218x is free. Needs `triton-windows` and MSVC on PATH.

The feeding path is tuned but NOT yet verified under load. The first full pass
measured the card oscillating on a ~20 s cycle -- gradual drain to under 50%,
floor, gradual refill -- at 85% mean utilisation, while total CPU sat at 26% of
16 cores and *rose* during the dips. That shape is a producer/consumer
oscillation against the bounded queue with capacity to spare, not a stall. The
response is more decode threads and a reused pinned staging buffer in place of
a 154 MB per-batch concatenate on the thread that feeds the GPU. Aggregate
throughput was 31 img/s against a fed-card estimate of ~36, so the ceiling is
about 15%. None of it has been A/B'd -- it was written while the GPU was busy
with the very run that motivated it, and it should be measured before it is
believed.

Resumable per encoder and per shard. The done-mask carries one column per
encoder, because the two passes are sequential and a single column cannot say
that DINOv2 finished a row while SigLIP has not. It is flushed at every shard
boundary: written only at the end, a nine-hour pass is all-or-nothing, and a
death at hour eight loses all of it.

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
            ("vit_base_patch16_siglip_512.v2_webli", 16, 512)]
D_ENC = 768


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def settings(a, ids):
    """Everything that decides what a row of this cache contains."""
    return {"rows_parquet": (a.parquet or config.DATASET_PARQUET.name),
            "rows_digest": prov.rows_digest(ids),
            "crops": str(a.crops),
            "encoders": ",".join(e[0] for e in ENCODERS),
            "sizes": ",".join(str(e[2]) for e in ENCODERS)}


def check_resume(meta, want, stem):
    """Refuse to keep rows that were built to different terms.

    A resume retains every row the mask marks done and fills only the rest, so
    the retained rows have to mean what this run means. The shape cannot say:
    every encoder is mean-pooled to 768 regardless of crop count, so a 3-crop
    and a 5-crop cache are both (n, 1536), and any same-length image list gives
    the same n.

    Two silent corruptions that closes. Re-using `--out` with a different
    same-length `--parquet` keeps completed rows from the *old* image list and
    fills the rest from the new one -- one cache, two corpora, a valid binary
    mask. And re-using it with a different `--crops` returns "nothing to do"
    when the mask is full, so the requested representation is never built and
    the stale one is served under its name.

    Metadata is written before the first forward for exactly this reason: if it
    were only written on success, an interrupted run -- the one case resume
    exists for -- would have nothing to check against.
    """
    if meta is None:
        raise SystemExit(
            "{} exists but has no metadata, so what its finished rows contain "
            "cannot be established. It predates this check or its run was "
            "killed before writing one. Delete it and rebuild.".format(stem))
    for k, v in sorted(want.items()):
        had = str(meta[k]) if k in meta.files else None
        if had != v:
            raise SystemExit(
                "{} was built with {}={} and this run wants {}. Resuming would "
                "keep the finished rows on the old terms and fill the rest on "
                "the new ones, in one cache with a valid mask. Delete it or "
                "use --out.".format(stem, k, had, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cond_native")
    ap.add_argument("--parquet", default=None,
                    help="image list; default the release's dataset.parquet")
    ap.add_argument("--n", type=int, default=0, help="0 = every row")
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12,
                    help="decode threads. The first full pass measured the GPU "
                         "oscillating on a ~20 s cycle at 85%% mean while total "
                         "CPU sat at 26%% of 16 cores and ROSE during the dips "
                         "-- a producer/consumer oscillation against the "
                         "bounded queue, with capacity to spare.")
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
    # range(0, n, 0) raises, but range(0, n, -1) is empty: a negative batch
    # would embed nothing, fall through, and mark every row done. REVIEW6 #7.
    if a.batch < 1 or a.workers < 1 or a.crops < 1:
        raise SystemExit("--batch, --workers and --crops must be >= 1")

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
    want = settings(a, ids)
    if emb_p.exists():
        E = np.lib.format.open_memmap(emb_p, mode="r+")
        if E.shape != (n, d_out):
            raise SystemExit("{} is {} and this run wants {}; delete it or "
                             "use --out".format(emb_p.name, E.shape,
                                                (n, d_out)))
        check_resume(np.load(meta_p, allow_pickle=True) if meta_p.exists()
                     else None, want, a.out)
        done = maskio.load_mask(done_p, n, a.out)
        if done.ndim == 1:                      # pre-2026-09-05 single column
            done = np.repeat(done[:, None], len(ENCODERS), axis=1)
    else:
        E = np.lib.format.open_memmap(emb_p, mode="w+", dtype=np.float16,
                                      shape=(n, d_out))
        # One column per encoder. A row is finished only when both have
        # written it, and the two passes are sequential, so a single column
        # cannot express the state this run is actually in.
        done = np.zeros((n, len(ENCODERS)), np.uint8)
    np.save(done_p, done)
    todo = np.flatnonzero(done.min(axis=1) != 1)
    # Published before the first forward, not after the last: the metadata is
    # what a resume checks against, and a run that only writes it on success
    # leaves the interrupted case -- the only case resume is for -- unverifiable.
    #
    # But only when there is work to do. Writing complete=False unconditionally
    # meant that merely *checking* a finished cache downgraded truthful
    # metadata and then returned before ever restoring it, so a harmless
    # no-op left the artifact claiming to be unfinished. REVIEW6 #8.
    if len(todo):
        np.savez(meta_p, release=config.RELEASE, complete=False,
                 layout="(n, dino768|siglip768) native-res crop mean, "
                        "CONDITIONING ONLY -- never a retrieval bank", **want)
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
            eager = net
            try:
                net = torch.compile(net)
                # torch.compile defers everything to the first call, so
                # wrapping it never raises -- the Triton/inductor failure
                # arrives on the first forward, hours into a run, past this
                # handler. Trip it here on one tiny batch instead. REVIEW6 #9.
                with torch.no_grad():
                    net(torch.zeros(1, 3, size, size,
                                    device=dev, dtype=torch.bfloat16))
            except Exception as exc:                      # noqa: BLE001
                print("compile unavailable ({}); eager".format(
                    str(exc).strip().split(chr(10))[0]), flush=True)
                net = eager
        col = slice(enc_i * D_ENC, (enc_i + 1) * D_ENC)

        # Per encoder: only the rows THIS encoder still owes. Resuming after
        # DINOv2 finished must not re-run it.
        mine = np.flatnonzero(done[:, enc_i] != 1)
        if not len(mine):
            print("{} already complete".format(spec), flush=True)
            del net
            torch.cuda.empty_cache()
            continue
        by_zip = defaultdict(list)
        for i in mine:
            by_zip[zn[i].split("/")[0]].append((int(i), zn[i]))
        t0, seen = time.time(), 0
        host = None                       # reused pinned staging buffer
        pool = ThreadPoolExecutor(a.workers)

        def prep(blob):
            # float16 in the worker. preprocess returns float32, so a batch of
            # 16 images at 3 crops of 518 is a 154 MB host allocation and copy
            # on the main thread before every submission -- which is where the
            # GPU dips to 60%. Casting here halves it and moves the work off
            # the thread that feeds the card.
            return preprocess(blob, size=size, crops=a.crops,
                              mean=mu, std=sd).astype(np.float16)

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
                        # Into a preallocated pinned buffer, not a fresh
                        # concatenate. At 3 crops of 518 a batch of 16 is a
                        # 154 MB host allocation and copy on the very thread
                        # that feeds the card, and an unpinned source makes the
                        # transfer synchronous however non_blocking is set.
                        need = sum(len(v) for v in arrs)
                        if host is None or host.shape[0] < need:
                            host = torch.empty((need,) + arrs[0].shape[1:],
                                               dtype=torch.float16).pin_memory()
                        off = 0
                        for v in arrs:
                            host[off:off + len(v)] = torch.from_numpy(v)
                            off += len(v)
                        x = host[:need].to(dev, torch.bfloat16,
                                           non_blocking=True)
                        f = net(x).float().reshape(len(rows), a.crops, D_ENC)
                        v = nrm(f.mean(1)).cpu().numpy().astype(np.float16)
                        for k, r in enumerate(rows):
                            E[r, col] = v[k]
                            done[r, enc_i] = 1
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
                # At every shard boundary, not once at the end. Writing the
                # mask only on success made a 9 h pass all-or-nothing: a death
                # at hour eight lost everything, and the docstring's promise
                # that a stop costs "whatever is in flight" was simply false.
                E.flush()
                np.save(done_p, done)
        pool.shutdown()
        del net
        torch.cuda.empty_cache()
        print("{} done in {:.0f}s".format(spec, time.time() - t0), flush=True)

    E.flush()
    np.save(done_p, done)
    if not maskio.is_complete(done, n):
        raise SystemExit("{:,} rows unfilled".format(n - maskio.complete(done)))
    np.savez(meta_p, release=config.RELEASE, complete=True,
             layout="(n, dino768|siglip768) native-res crop mean, CONDITIONING "
                    "ONLY -- never a retrieval bank", **want)
    prov.write(emb_p, ids, release=config.RELEASE,
               row_space=("the release" if not a.parquet
                          else "the rows of " + a.parquet),
               conditioning_only=True,
               sizes="/".join(str(e[2]) for e in ENCODERS), crops=a.crops)
    print("\nwrote {}".format(emb_p.name), flush=True)


if __name__ == "__main__":
    main()
