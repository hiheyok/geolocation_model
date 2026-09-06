"""One pass over the shard zip -> frozen DINOv2 embeddings.

Nothing is extracted to disk, and after this pass training never opens an image
again -- that is what keeps the training loop offline and reproducible.

Two things make it fast on this machine:

  * The shard lives on a spinning disk.  50k random reads into a 2.5 GB archive
    is seek-bound at ~6 MB/s with the disk pegged at 99%.  One sequential read of
    the whole file into RAM turns every later read into a memory hit, so
    --preload is the default; pass --no-preload if RAM is tight.
  * Decode, not the GPU, is then the bottleneck, so it runs on a thread pool
    (PIL releases the GIL) using JPEG draft mode -- libjpeg DCT downscaling
    during decode -- which is ~2.3x faster than decoding full size then resizing.

Row order matches dataset.parquet exactly: the embedding for image i is
embeddings[i].
"""

import argparse
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import provenance as prov
import safeio
import shards

EMB = config.STREET_CACHE / "embeddings.f16.npy"
EMB_IDS = config.STREET_CACHE / "image_ids.i64.npy"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(blob, size=224, crops=1, mean=None, std=None):
    """JPEG bytes -> (crops, 3, size, size) normalised float32.  224 = 16*14.

    OSV-5M frames are 910x512.  A single centre crop keeps only 224/398 = 56%
    of the horizontal field of view, discarding the periphery -- signage,
    vegetation, road furniture -- which is exactly what fine-grained geolocation
    runs on.  crops=3 tiles the full width (left, centre, right) instead.
    """
    img = Image.open(io.BytesIO(blob))
    img.draft("RGB", (size, size))          # decode at a reduced DCT scale
    img = img.convert("RGB")
    w, h = img.size
    # Scale so the short side is exactly `size`. OSV-5M frames are 910x512 so
    # this only ever shrinks them, but the demo server feeds this arbitrary
    # uploads and an image smaller than 224 has to be grown or the crop below
    # would run off the edge.
    sc = size / min(w, h)
    if sc != 1.0:
        img = img.resize((max(size, round(w * sc)), max(size, round(h * sc))),
                         Image.BILINEAR)
    w, h = img.size
    t = (h - size) // 2
    if crops == 1:
        lefts = [(w - size) // 2]
    else:
        lefts = [round(i * (w - size) / (crops - 1)) for i in range(crops)]
    out = np.empty((len(lefts), 3, size, size), dtype=np.float32)
    for i, l in enumerate(lefts):
        c = img.crop((l, t, l + size, t + size))
        x = np.array(c, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32) / 255.0
        out[i] = (x - (MEAN if mean is None else mean)) / (STD if std is None else std)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=224,
                    help="crop size and model input size. 224 is what every "
                         "cached artefact uses; raise it only to test an "
                         "encoder at its native resolution.")
    ap.add_argument("--crops", type=int, default=1,
                    help="horizontal crops per image; embeddings are concatenated")
    ap.add_argument("--patch-grid", type=int, default=0,
                    help="keep a PxP pooled grid of DINOv2 patch tokens per crop "
                         "instead of the CLS vector; 0 = CLS as before")
    ap.add_argument("--out", default=None, help="override output filename stem")
    ap.add_argument("--parquet", default=None,
                    help="image list to embed instead of the release's "
                         "dataset.parquet; needs image_id and zip_name. Used "
                         "for bank-only shards, which never enter the release.")
    ap.add_argument("--no-preload", dest="preload", action="store_false",
                    help="stream from disk instead of loading the zip into RAM")
    ap.set_defaults(preload=True)
    a = ap.parse_args()

    # The default output is the release's canonical embedding cache, which is
    # what every arm on record was trained against. A run that embeds a
    # different image list, a different encoder or a different geometry is not
    # that cache, and without --out it overwrote it in place -- leaving a file
    # of the right name, the right length and the wrong contents.
    custom = [f for f, v in (("--parquet", a.parquet),
                             ("--crops > 1", a.crops > 1),
                             ("--patch-grid", a.patch_grid),
                             ("--size != 224", a.size != 224),
                             ("--model", a.model != ap.get_default("model")))
              if v]
    if not a.out and custom:
        raise SystemExit(
            "this run differs from the canonical cache ({}) but has no --out, "
            "so it would overwrite {} -- the file every checkpoint on record "
            "was trained against. Name it: --out <stem>."
            .format(", ".join(custom), EMB.name))

    ds = pq.read_table(config.PROCESSED / a.parquet if a.parquet
                       else config.DATASET_PARQUET)
    image_ids = np.asarray(ds["image_id"])
    names = ds["zip_name"].to_pylist()
    n = len(names)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device     {} ({})".format(
        dev, torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"), flush=True)

    import timm
    # DINOv2 checkpoints default to 518px; 224 = 16*14 keeps it cheap and timm
    # interpolates the position embeddings for us.
    # 224 is the shipping size and the tile/crop geometry everywhere else, but
    # it is not every encoder's native one -- timm reports 518 for DINOv2 ViT-B,
    # 256 for DINOv3 ViT-B and 224 for SigLIP. Running off-native interpolates
    # the position embeddings and changes the patch grid (DINOv3 at 224 is 14x14
    # rather than its native 16x16), which is a handicap of unknown size. Make
    # it a flag so a comparison between encoders can be checked rather than
    # assumed.
    model = timm.create_model(a.model, pretrained=True, num_classes=0,
                              img_size=a.size).eval().to(dev)
    dim = model.num_features
    # Normalisation must follow the model, not a hardcoded constant: DINOv2
    # wants ImageNet statistics and SigLIP wants 0.5/0.5, and feeding one the
    # other's silently degrades it -- which would make an encoder comparison
    # measure preprocessing rather than encoders.
    from timm.data import resolve_model_data_config
    cfg = resolve_model_data_config(model)
    nmean = np.array(cfg["mean"], dtype=np.float32).reshape(3, 1, 1)
    nstd = np.array(cfg["std"], dtype=np.float32).reshape(3, 1, 1)
    print("model      {}  dim {}  input {}".format(a.model, dim, a.size),
          flush=True)
    print("normalise  mean {}  std {}".format(cfg["mean"], cfg["std"]), flush=True)

    out_path = (config.STREET_CACHE / (a.out + ".f16.npy")) if a.out else EMB
    # Patch mode keeps a spatial set instead of one vector per image: the CLS
    # token is a learned summary, and the open question is whether the detail it
    # drops is what the fine zoom steps need.
    n_tok = a.crops * a.patch_grid ** 2 if a.patch_grid else 0
    shape = (n, n_tok, dim) if a.patch_grid else (n, dim * a.crops)
    # Written to a temporary name and published only once every row has been
    # proved non-zero. The sidecar used to be written here, before the first
    # image was embedded, so an interrupted run left a full-shaped array whose
    # tail was the zero fill AND an authoritative-looking `basis="built"`
    # record beside it. The all-row scan added for item 45 cannot catch that,
    # because the process that was killed never reaches it -- and a zero row is
    # a legal-looking embedding, so nothing downstream notices either.
    # keeps the .npy extension open_memmap expects, and `*.partial.npy` is
    # visibly not a cache anyone should reach for
    tmp_path = out_path.with_name(out_path.name[:-len(".npy")] + ".partial.npy")
    emb = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16,
                                    shape=shape)
    if a.patch_grid:
        print("patch grid  {}x{} per crop -> {} tokens x {} dims  ({:.1f} GB)"
              .format(a.patch_grid, a.patch_grid, n_tok, dim,
                      n * n_tok * dim * 2 / 1e9), flush=True)
    if a.crops > 1:
        print("crops      {} horizontal -> embedding dim {}"
              .format(a.crops, dim * a.crops), flush=True)

    # Rows are grouped by their zip, because the join in build_dataset reorders
    # them and only one 2.5 GB shard is held in RAM at a time.  Writes are
    # scattered back to the row each image actually occupies.
    by_shard = {}
    for i, name in enumerate(names):
        by_shard.setdefault(name.split("/")[0], []).append(i)
    print("shards     {}  ({:,} images)  batch {}  workers {}\n"
          .format(", ".join(sorted(by_shard)), n, a.batch, a.workers), flush=True)

    t0 = time.time()
    seen = 0
    for sh in sorted(by_shard):
        rows = np.array(by_shard[sh], dtype=np.int64)
        print("shard {}  {:,} images".format(sh, len(rows)), flush=True)
        with (shards.archive(sh, preload=a.preload) as z,
              ThreadPoolExecutor(max_workers=a.workers) as pool,
              torch.inference_mode()):
            for lo in range(0, len(rows), a.batch):
                sel = rows[lo:lo + a.batch]
                blobs = []
                for r in sel:
                    with z.open(names[r]) as f:
                        blobs.append(f.read())
                arrs = list(pool.map(
                    lambda b: preprocess(b, size=a.size, crops=a.crops, mean=nmean, std=nstd),
                    blobs))
                x = torch.from_numpy(np.concatenate(arrs)).to(dev, non_blocking=True)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    if a.patch_grid:
                        ft = model.forward_features(x)
                        ft = ft[:, getattr(model, "num_prefix_tokens", 1):]
                        gg = int(round(ft.shape[1] ** 0.5))
                        ft = ft.transpose(1, 2).reshape(-1, dim, gg, gg)
                        ft = torch.nn.functional.adaptive_avg_pool2d(ft, a.patch_grid)
                        f = ft.flatten(2).transpose(1, 2)
                    else:
                        f = model(x)
                f = (f.float().reshape(len(sel), n_tok, dim) if a.patch_grid
                     else f.float().reshape(len(sel), dim * a.crops))
                emb[sel] = f.cpu().numpy().astype(np.float16)
                seen += len(sel)
                if (lo // a.batch) % 40 == 0 and lo:
                    el = time.time() - t0
                    rate = seen / max(el, 1e-6)
                    print("  {:>7,}/{:,}  {:5.0f} img/s  eta {:4.1f} min"
                          .format(seen, n, rate, (n - seen) / max(rate, 1e-6) / 60),
                          flush=True)
    emb.flush()
    el = time.time() - t0
    print("\nembedded {:,} in {:.1f} min ({:.0f} img/s)".format(n, el / 60, n / el))
    print("wrote {}  {:.0f} MB".format(out_path.name, emb.nbytes / 1e6))
    # The first 512 rows only prove the run started. An interruption
    # anywhere later leaves a full-shaped memmap whose tail is the zero fill,
    # which is exactly what this check exists to catch, so it has to look at
    # every row. A norm scan over the whole file is seconds.
    dead = 0
    for lo in range(0, len(emb), 200_000):
        blk = np.asarray(emb[lo:lo + 200_000], dtype=np.float32)
        dead += int((np.linalg.norm(blk, axis=1) == 0).sum())
    if dead:
        raise SystemExit(
            "{:,} of {:,} embedding rows are all zero, so this file is "
            "incomplete -- most likely an interrupted run. Re-run; a "
            "zero row is a legal-looking embedding and nothing downstream "
            "would notice.".format(dead, len(emb)))
    sample = np.asarray(emb[:512], dtype=np.float32)
    flat = sample.reshape(len(sample), -1)
    print("sanity: mean L2 {:.3f}   zero rows in first 512: {}".format(
        float(np.linalg.norm(flat, axis=1).mean()),
        int((np.abs(flat).sum(1) == 0).sum())))

    # Publish only now: the data is complete, so the name and the sidecar can
    # start meaning something. Windows will not rename a mapped file, so the
    # memmap is dropped first.
    del emb, sample, flat
    safeio.replace_from(tmp_path, out_path)
    # a bank extension is not the release: do not overwrite its id map
    if not a.parquet:
        np.save(EMB_IDS, image_ids)
    # Record the rows this file describes, beside the file. `image_ids.i64.npy`
    # has existed for a while and nothing ever read it -- and it is a single
    # global path, so it says nothing about which of the many caches in this
    # directory it belongs to.
    prov.write(out_path, image_ids, release=config.RELEASE,
               row_space=("the release" if not a.parquet
                          else "the rows of " + Path(a.parquet).name),
               model=a.model, crops=a.crops, size=a.size)


if __name__ == "__main__":
    main()
