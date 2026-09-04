"""Cache a 33-token scale pyramid over the high-resolution harvest.

`res_probe.py` already builds the geometry and answered the mean-pooling
question: three levels lose to two, and level-weighting beats token-weighting by
4 pp at three levels. But it reduces each level to a single 768-d vector before
storing anything, so the tokens are gone and the only question that can be asked
of its output is how to average them.

That is exactly the question attention is supposed to replace. Under a mean, a
3+6+24 union hands the deepest level **73% of the vector by token count** -- the
level that loses on its own dominates the result -- which is a property of the
pooling, not of the information. So "L2 hurts" does not license "L2 is useless",
and settling it needs the tokens kept.

    L0   3 crops   short side to 224, three 224 windows across the width
    L1   6 tiles   resize to 672x448, six disjoint 224 tiles      (2x linear)
    L2  24 tiles   resize to 1344x896, twenty-four 224 tiles      (4x linear)

Same encoder input at every level, different source scale -- that is what makes
it a pyramid rather than a resolution sweep. Raising the encoder input to 448 is
a separate axis, worth +2.08 pp at `<1 km` on its own, and is deliberately not
mixed in here.

**This only exists on high-resolution data.** A 682x512 OSV-5M frame bottoms out
at 3x2, so L2 is unreachable there; a 6 MP KartaView frame supports roughly
11x9. Which is also the hazard: 1.45% of OSV-5M frames carry their true GPS
burned in, harmless only because everything is resized to 224. Run
`scripts/screen_leak.py` over the harvest before believing any number this
produces.

Output is `<stem>.f16.npy` shaped (n, 33, 2, 768) -- token, encoder, dim -- plus
a metadata npz. Resumable through a done-mask, like `tile_cache.py`.
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
from res_probe import variants, MODELS

LEVELS = ["crop3_224", "tile6", "tile24"]      # 3 + 6 + 24 = 33 tokens
NTOK = {"crop3_224": 3, "tile6": 6, "tile24": 24}
D_ENC = 768


def level_index():
    """Which level each of the 33 token slots belongs to -- the fusion head
    needs it for a level embedding, and the probes need it for weighting."""
    return np.concatenate([np.full(NTOK[k], i, np.int64)
                           for i, k in enumerate(LEVELS)])


def _load_done(path, n):
    """Per-encoder completion, (n, 2). A 1-D file is a pre-2026-09-02 cache
    where a row could only be written after both passes, so both are done."""
    if not path.exists():
        return np.zeros((n, 2), np.uint8)
    d = np.load(path)
    if d.ndim == 1:
        d = np.repeat(d.reshape(-1, 1), 2, axis=1)
    # A mask of the wrong length, or holding anything but 0/1, is not a resume
    # point -- it is another run's file under this one's name, and trusting it
    # blesses rows that were never written.
    if d.shape != (n, 2):
        raise SystemExit(
            "resume mask is {} but this cache wants {}; it belongs to a "
            "different build. Delete it to start fresh.".format(d.shape, (n, 2)))
    if not np.isin(d, (0, 1)).all():
        raise SystemExit(
            "resume mask holds values outside {0, 1}; it is not a completion "
            "mask. Delete it to start fresh.")
    return d.astype(np.uint8)


def _mark(done, loaded, ei, path, mm=None):
    """Persist progress for one encoder, after its data is on disk.

    The ordering matters and used to be backwards: the mask was written
    durably through tmp+replace while the token memmap was never flushed, so a
    crash could leave rows marked complete whose tokens never reached disk --
    and an unwritten row is the zero fill, which reads as a valid embedding.

    The mask used to be written once, after *both* passes. Fourteen attempts
    were logged and one reached 14,000 images; every restart resumed from zero
    because nothing had been recorded. Saving per flush makes an interruption
    cost one batch instead of the whole run.
    """
    if not loaded:
        return
    if mm is not None:
        mm.flush()
    done[np.array(sorted(loaded), np.int64), ei] = 1
    loaded.clear()
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, done)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/data/kartaview_hr")
    ap.add_argument("--out", default="pyr33")
    ap.add_argument("--n", type=int, default=0, help="0 = every image held")
    ap.add_argument("--batch", type=int, default=24, help="views per forward")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import timm
    from PIL import Image
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    root = Path(a.data)
    recs = {}
    for line in (root / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        recs[r["id"]] = r                       # append-only; last wins
    ids = sorted(recs)
    rng = np.random.default_rng(a.seed)
    # A permutation prefix, so raising --n later extends the same cache rather
    # than invalidating it -- the trick tile_cache.py uses.
    order = np.asarray(ids)[rng.permutation(len(ids))]
    if a.n:
        order = order[:a.n]
    n = len(order)
    ntok = sum(NTOK[k] for k in LEVELS)
    print("{:,} images held, caching {:,}  ->  {} tokens x 2 encoders"
          .format(len(ids), n, ntok), flush=True)

    out = config.STREET_CACHE / (a.out + ".f16.npy")
    donep = config.STREET_CACHE / (a.out + "_done.u8.npy")
    config.STREET_CACHE.mkdir(parents=True, exist_ok=True)
    shape = (n, ntok, 2, D_ENC)
    if out.exists() and np.load(out, mmap_mode="r").shape == shape:
        X = np.load(out, mmap_mode="r+")
        done = _load_done(donep, n)
        print("resuming: {:,}/{:,} complete  (dinov2 {:,}, siglip {:,})".format(
            int(done.all(1).sum()), n,
            int(done[:, 0].sum()), int(done[:, 1].sum())))
    else:
        X = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                      shape=shape)
        done = np.zeros((n, 2), np.uint8)
    print("{}  {:.2f} GB".format(out.name, X.nbytes / 1e9), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # fixed encoder order, since the axis index is baked into the cache
    enc_order = ["dinov2", "siglip"]

    def load_img(k):
        """Decode AND cut the 33 views, both in the worker.

        Cutting them on the consumer side serialised the expensive half onto
        one thread: `variants` resizes a 6 MP image up to 1344x896 and cuts 33
        tiles, which is far more work than the JPEG decode. Measured mid-run,
        neither resource was saturated -- E: queue length 0.5, CPU 13% -- while
        the GPU alternated 100/0, which is the signature of a single-threaded
        stage in the middle of the pipeline.

        This also shrinks what the queue holds: 33 x 224 x 224 x 3 is 4.96 MB
        against roughly 18 MB for the decoded 6 MP image.
        """
        try:
            im = Image.open(root / "img" / ("%s.jpg" % order[k]))
            im.load()
            v = variants(im.convert("RGB"), LEVELS)
            return k, np.concatenate([v[name] for name in LEVELS])
        except Exception:
            return k, None

    # Do NOT return here. A crash between the final _mark and the metadata
    # write leaves a cache whose mask says complete and whose _meta.npz does
    # not exist, and every later run took this branch and exited -- so the
    # cache permanently reported "nothing to do" and never became usable.
    # Fall through instead: the encoder loops are already no-ops when done,
    # and the verification and metadata write below are cheap and idempotent.
    if done.all():
        print("all rows already cached; verifying and rewriting metadata")

    for ei, mname in enumerate(enc_order):
        # per encoder, so a run interrupted during siglip does not redo dinov2
        todo = np.flatnonzero(done[:, ei] == 0)
        if not len(todo):
            print("  {:<7} already complete".format(mname))
            continue
        loaded = set()
        # img_size must be given: DINOv2's timm default is 518, and every
        # token here is 224. mean/std come from the model's own config -- they
        # differ between these two encoders (ImageNet vs 0.5), and hardcoding
        # one silently degrades the other rather than failing.
        net = timm.create_model(MODELS[mname], pretrained=True, num_classes=0,
                                img_size=224).eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=net)
        mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
        print("  {:<7} input {} mean {} std {}".format(
            mname, cfg["input_size"], cfg["mean"], cfg["std"]), flush=True)
        t0, seen = time.time(), 0
        with ThreadPoolExecutor(a.workers) as pool, torch.no_grad():
            def decoded(keys, window):
                """Decode on the pool, at most `window` images in flight.

                Not ThreadPoolExecutor.map: that submits every task at once and
                buffers every result, so it would hold all 18,812 decoded 6 MP
                images -- roughly 340 GB -- and die on an 882 KiB allocation.
                tile_cache.py carries the same warning; this is the same bug.
                """
                q, it = deque(), iter(keys)
                def top_up():
                    while len(q) < window:
                        nxt = next(it, None)
                        if nxt is None:
                            return
                        q.append(pool.submit(load_img, nxt))
                top_up()
                while q:
                    yield q.popleft().result()
                    top_up()

            buf_v, buf_k = [], []

            def flush():
                if not buf_v:
                    return
                arr = np.concatenate(buf_v)             # (m*33, 224, 224, 3)
                t = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float()
                t = (t / 255.0 - mean) / std
                z = []
                for s in range(0, len(t), a.batch):
                    z.append(net(t[s:s + a.batch]).float().cpu().numpy())
                z = np.concatenate(z).reshape(len(buf_k), ntok, D_ENC)
                for j, k in enumerate(buf_k):
                    X[k, :, ei, :] = z[j].astype(np.float16)
                buf_v.clear()
                buf_k.clear()

            for k, views in decoded(todo.tolist(), 2 * a.workers):
                if views is None:
                    continue
                buf_v.append(views)
                buf_k.append(k)
                loaded.add(int(k))
                seen += 1
                if len(buf_k) >= 4:
                    flush()
                if seen % 500 == 0:
                    el = time.time() - t0
                    print("  {:<7} {:>7,}/{:,}  {:.1f} img/s  eta {:.0f} min"
                          .format(mname, seen, len(todo), seen / el,
                                  (len(todo) - seen) / max(seen / el, 1e-9) / 60),
                          flush=True)
            flush()
            _mark(done, loaded, ei, donep, X)
        del net
        torch.cuda.empty_cache()
        print("  {} done in {:.0f} min".format(mname, (time.time() - t0) / 60),
              flush=True)

    # Mark only the images that actually decoded. Marking all of `todo` would
    # stamp a failed decode as cached and leave 33 all-zero tokens behind --
    # the same silent-null failure the GeoMem table had, and it would look like
    # an honest result rather than an error.
    ok = np.flatnonzero(done.all(1))
    # Validate BOTH encoder slices. Checking only slice 0 would pass an image
    # that decoded for dinov2 and failed for siglip, leaving half its tokens
    # zero behind a "done" flag -- the silent-null failure this check exists
    # to prevent.
    # Chunked, and the encoder axis sliced BEFORE the cast. `X[ok][:, :, ei]`
    # is fancy indexing: it materialises both encoders for every complete row
    # as float32 first -- (n, 33, 2, 768), over 10 GB at full scale -- and
    # only then throws half away. Same shape of bug as the embed_street scan.
    for ei, mname in enumerate(enc_order):
        z = 0
        for s in range(0, len(ok), 20000):
            blk = np.asarray(X[ok[s:s + 20000]][:, :, ei, :], np.float32)
            z += int((np.abs(blk).sum(-1) == 0).sum())
        if z:
            raise SystemExit("{:,} all-zero {} tokens among images marked "
                             "complete".format(z, mname))
    if len(ok) < n:
        print("{:,} of {:,} images are not complete for both encoders"
              .format(n - len(ok), n))
    lat = np.array([recs[i]["lat"] for i in order], np.float64)
    lon = np.array([recs[i]["lon"] for i in order], np.float64)
    seq = np.array([str(recs[i].get("sequence_id", i)) for i in order])
    np.savez(config.STREET_CACHE / (a.out + "_meta.npz"),
             image_id=order, lat=lat, lon=lon, sequence=seq,
             levels=np.array(LEVELS), level_of=level_index(),
             encoders=np.array(enc_order), layout="(n, 33, 2, 768) "
             "token-major: 3 crops, 6 tiles, 24 tiles; encoder 0 dinov2")
    print("\nwrote {} and {}_meta.npz  ({:,} images)".format(
        out.name, a.out, int(done.sum())))
    print("levels per token:", np.bincount(level_index()))

    short = n - int(done.all(1).sum())
    if short:
        # The stage runner keys its markers on the exit code, so exiting 0
        # with rows outstanding marked a partial cache finished and it was
        # never retried. Unwritten rows are zero embeddings that read as valid.
        sys.exit("\nINCOMPLETE: {:,} of {:,} rows are missing an "
                 "encoder pass. Re-run to resume.".format(short, n))


if __name__ == "__main__":
    main()
