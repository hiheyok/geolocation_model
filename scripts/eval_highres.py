"""Run the shipping model on high-resolution KartaView images it never saw.

Everything measured so far is OSV-5M evaluating on OSV-5M. That answers "does
this work on held-out images of the same kind", not "does this work". KartaView
is one of the two sources OSV-5M was built from, so it is the same *kind* of
imagery from a different pipeline -- close enough that a large gap would be
informative rather than trivially explained by domain.

**The images are put through the shipping pipeline exactly, not their native
resolution.** Short side to 224, three full-height crops across the width,
DINOv2 and SigLIP, joined at equal norm (x4.03), mean-pooled to 1536, then the
saved PCA basis to 768. Using the *saved* basis matters: fitting a new one would
put the queries in a different space from the bank and the failure would look
like a domain gap.

Retrieval is against the same 2.65M bank. These images are not in it, so no
same-sequence masking is needed -- and unlike the OSV-5M test split there is no
possibility of a near-duplicate from the same drive sitting in the bank, which
if anything makes this harder than the benchmark.

The model is an agent, not a classifier, so this fetches map tiles and runs the
real beam rollout. `evaluate()` wants a dataset; the contract it actually uses is
small -- street, rows, knn_*, all_x16/y16, lat, lon, action -- so this supplies a
shim rather than pretending the harvest is a release.
"""

import argparse
import json
import io
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    # Declared, not silent. config refuses to guess a release precisely
    # because guessing produced five wrong-release runs; a tool that picks one
    # for you should at least say so in the output it is about to print.
    os.environ["OSV_RELEASE"] = "s10"
    print("OSV_RELEASE not set; using s10 (this tool's bank and harvest are "
          "s10-only)", flush=True)

import config
import knnmeta
import provenance as prov
import safeio
import shards
import names
import tile_math as tm
from dataset import street_table
from evaluate import evaluate, load_model
from beam import source_for

D_ENC = 768
DINO = "vit_base_patch14_dinov2.lvd142m"
SIGLIP = "vit_base_patch16_siglip_224.v2_webli"


class ExternalSet:
    """The slice of GeoStepDataset that `evaluate` actually touches."""

    def __init__(self, street, lat, lon, knn_idx, knn_sim, all_x16, all_y16):
        self.street = street
        self.rows = np.arange(len(street))
        self.lat, self.lon = lat, lon
        self.knn_idx, self.knn_sim = knn_idx, knn_sim
        self.knn_k = knn_idx.shape[1]
        self.all_x16, self.all_y16 = all_x16, all_y16
        # per-step target actions, read off the true z16 address exactly as
        # tile_math.target_actions does for the benchmark
        x16, y16 = tile_for_vec(lat, lon, 4 * tm.STEPS)
        act = []
        for t in range(tm.STEPS):
            shift = tm.G ** (tm.STEPS - 1 - t)
            act.append(((y16 // shift) % tm.G) * tm.G + ((x16 // shift) % tm.G))
        self.action = np.stack(act, 1)

    def __len__(self):
        return len(self.street)


def tile_for_vec(lat, lon, z):
    lat = np.clip(lat, -tm.MAX_LAT, tm.MAX_LAT)
    x = (lon + 180.0) / 360.0
    sn = np.sin(np.radians(lat))
    y = 0.5 - np.log((1.0 + sn) / (1.0 - sn)) / (4.0 * np.pi)
    n = 1 << z
    return (np.clip((x * n).astype(np.int64), 0, n - 1),
            np.clip((y * n).astype(np.int64), 0, n - 1))


def embed(paths, dev, batch=16, match_bank=False):
    """The shipping street pipeline: 3 crops at 224 through both encoders.

    match_bank reproduces what embed_street.py does when it builds the
    bank: a JPEG draft decode at a reduced DCT scale, and a bf16 autocast
    forward. Without it the queries reach the same similarity space
    through a different pipeline than the rows they are matched against.
    Measured over 300 images, the two embeddings differ by cosine 0.993
    on average and 0.926 at worst -- small next to a top-1 bank
    similarity of 0.71, but systematic, and only a paired run says
    whether it moves the metric.
    """
    import timm
    from PIL import Image
    out = {}
    for mname, spec in (("dinov2", DINO), ("siglip", SIGLIP)):
        net = timm.create_model(spec, pretrained=True, num_classes=0,
                                img_size=224).eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=net)
        mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
        Z, t0 = [], time.time()
        with torch.no_grad():
            for s in range(0, len(paths), batch):
                crops = []
                for p in paths[s:s + batch]:
                    im = Image.open(p)
                    if match_bank:
                        im.draft("RGB", (224, 224))
                    im = im.convert("RGB")
                    w, h = im.size
                    sc = 224 / min(w, h)
                    im = im.resize((max(224, round(w * sc)),
                                    max(224, round(h * sc))), Image.BILINEAR)
                    w, h = im.size
                    top = (h - 224) // 2
                    lefts = [round(i * (w - 224) / 2) for i in range(3)]
                    crops += [np.asarray(im.crop((l, top, l + 224, top + 224)),
                                         np.uint8) for l in lefts]
                t = torch.from_numpy(np.stack(crops)).to(dev)
                t = t.permute(0, 3, 1, 2).float() / 255.0
                with torch.autocast(dev, dtype=torch.bfloat16,
                                    enabled=(match_bank and dev == "cuda")):
                    z = net((t - mean) / std)
                Z.append(z.float().cpu().numpy())
        Z = np.concatenate(Z).reshape(len(paths), 3, D_ENC)
        out[mname] = Z
        print("  {:<7} {:,} images in {:.0f}s".format(mname, len(paths),
                                                      time.time() - t0),
              flush=True)
        del net
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="d768-b265-e6")
    ap.add_argument("--bank", default=None,
                    help="defaults to the checkpoint's own street file")
    ap.add_argument("--export", default="",
                    help="npz of per-image errors and ids, so two arms on the "
                         "same sample can be compared with a paired interval "
                         "instead of by eyeballing two point estimates")
    ap.add_argument("--data", default="E:/data/kartaview_hr")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--score-steps", type=int, default=3)
    ap.add_argument("--scale-b", type=float, default=4.03)
    ap.add_argument("--basis", default="",
                    help="PCA basis; empty means read it from the bank's "
                         "provenance sidecar, which is where it belongs. "
                         "The old default was pca768_bank55_pca.npz whatever "
                         "bank was loaded, checked on output width alone.")
    ap.add_argument("--basis-legacy-default", dest="basis_fallback",
                    default="pca768_bank55_pca.npz",
                    help="PCA basis, used only when the bank is narrower than "
                         "the pooled vector")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohort", default="",
                    help="frozen list of image ids to score. Defaults to "
                         "<data>/cohort_seed<seed>_n<n>.json, written on the "
                         "first run and replayed after. Hash-ranking alone is "
                         "not append-stable: taking the n smallest hashes "
                         "means a newly harvested id that hashes low joins and "
                         "evicts someone, so two runs at the same --n and "
                         "--seed can score different benchmarks.")
    ap.add_argument("--match-bank", action="store_true",
                    help="embed queries the way embed_street builds the "
                         "bank: JPEG draft decode and bf16 autocast")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = {}
    for line in (Path(a.data) / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        recs[r["id"]] = r
    # Drop what the leak screen found. An image with its own coordinates
    # burned in is not a geolocation test, it is an OCR test, and until now
    # the blocklist was written and never read (item 14).
    _block, _screened = prov.leak_blocklist(a.data)
    if _block:
        _before = len(recs)
        recs = {i: r for i, r in recs.items() if str(i) not in _block}
        print("leak screen dropped {:,} of {:,} images with burned-in "
              "coordinates ({:,} screened)".format(_before - len(recs),
                                                   _before, _screened),
              flush=True)
    elif _screened:
        print("leak screen: {:,} images screened, none blocked"
              .format(_screened), flush=True)
    else:
        print("leak screen: no blocklist at {} -- nothing has been screened, "
              "which is not the same as nothing leaking "
              "(scripts/screen_leak.py --write-blocklist)".format(a.data),
              flush=True)
    ids = sorted(recs)
    # Pick by hashing the id, not by permuting the manifest: `permutation(
    # len(ids))[:n]` silently selected a different thousand images when the
    # harvest grew 18,812 -> 47,646, so two runs both reporting "n=1000,
    # seed 0" scored different sets and were compared as one benchmark.
    #
    # Hashing fixes the ordering but NOT the membership, and the comment here
    # used to claim otherwise. Taking the n smallest hashes means membership
    # depends on the global nth value: every newly harvested id that hashes
    # below it joins and evicts someone. At the sizes above with n=1000 that
    # is an expected 39.5% overlap, not stability. A hash *threshold* would be
    # append-stable but gives a variable count; you cannot have both from a
    # rule alone.
    #
    # So the cohort is frozen as an artifact instead. The first run at a given
    # (seed, n) writes the ids it chose; every later run replays that file, and
    # the digest is printed and exported so two numbers can be shown to be the
    # same benchmark rather than assumed to be.
    cohort_p = Path(a.cohort) if a.cohort else \
        Path(a.data) / "cohort_seed{}_n{}.json".format(a.seed, a.n)
    if cohort_p.exists():
        pick = [str(i) for i in json.loads(cohort_p.read_text(encoding="utf-8"))]
        missing = [i for i in pick if i not in recs]
        if missing:
            raise SystemExit(
                "{} names {:,} images the manifest no longer holds (e.g. {}). "
                "The frozen cohort cannot be scored; re-harvest them or start "
                "a new cohort under a different --seed/--n."
                .format(cohort_p.name, len(missing), missing[0]))
        how = "replayed from " + cohort_p.name
    else:
        key = np.array([zlib.crc32((str(a.seed) + i).encode()) for i in ids])
        pick = [ids[i] for i in np.argsort(key)[:a.n]]
        safeio.write_text(cohort_p, json.dumps(pick))
        how = "selected by hash and frozen into " + cohort_p.name
    cohort_digest = prov.rows_digest(np.asarray(pick))
    print("cohort     {:,} ids, digest {}  ({}; manifest holds {:,})"
          .format(len(pick), cohort_digest, how, len(ids)), flush=True)
    # The manifest's own `file`, not a reconstructed "<id>.jpg": once
    # pack_kartaview has run, the image is a member of a pack and the loose
    # path does not exist. `embed` opens whatever it is handed, and PIL takes
    # a file object, so the bytes go straight in.
    with shards.Reader(Path(a.data) / "img") as _rd:
        paths = [io.BytesIO(_rd.read(recs[i].get("file") or (i + ".jpg")))
                 for i in pick]
    lat = np.array([recs[i]["lat"] for i in pick], np.float64)
    lon = np.array([recs[i]["lon"] for i in pick], np.float64)
    print("{:,} high-resolution images, evaluated as {} on the {} pipeline\n"
          .format(len(pick), a.tag, "224px 3-crop"), flush=True)

    model, ck, _ = load_model(a.tag, dev)
    # The bank must be the one this checkpoint was TRAINED with. Hardcoding it
    # silently evaluated d768-b350-e6 against the 2.75M bank it had never seen
    # and reported the number as that arm's external score.
    bank_file = a.bank or ck.get("street_file")
    # The record first, the filename only as a fallback. Scanning the name for
    # "70"/"55"/"40" picked another extension's metadata the moment a bank was
    # renamed, and every dimensional check still passed.
    meta_stem, how = prov.ext_for_bank(config.STREET_CACHE / bank_file,
                                       bank_file)
    print("bank {}   addresses {}  ({})".format(bank_file, meta_stem, how),
          flush=True)

    E = embed(paths, dev, match_bank=a.match_bank)
    # equal-norm join, mean-pool over crops, then the SAVED PCA basis -- but
    # only for a projected bank. A 1536-d arm (pool_bal_*) stores the pooled
    # vector itself, so projecting it would compare a 768-d query against a
    # 1536-d bank; the width is read off the bank rather than assumed.
    tok = np.concatenate([E["dinov2"], a.scale_b * E["siglip"]], axis=2)
    pooled = tok.mean(1).astype(np.float32)          # (n, 1536)
    bank_dim = np.load(config.STREET_CACHE / bank_file, mmap_mode="r").shape[1]
    if bank_dim == pooled.shape[1]:
        q = pooled.astype(np.float16)
        print("")
        print("query vectors {}  pooled, no projection (bank is {}-d)"
              .format(q.shape, bank_dim), flush=True)
    else:
        basis = a.basis or prov.projection_of(
            config.STREET_CACHE / bank_file)
        if not basis:
            basis = a.basis_fallback
            print("warning: {} does not record the PCA basis it was "
                  "projected with, so {} is assumed. Output width was the "
                  "only thing ever checked, and several bases here share a "
                  "width. Re-run scripts/project_street.py to record it."
                  .format(bank_file, basis), flush=True)
        print("basis      {}".format(basis), flush=True)
        z = np.load(config.STREET_CACHE / basis)
        q = ((pooled - z["mu"]) @ z["P"]).astype(np.float16)
        if q.shape[1] != bank_dim:
            sys.exit("projected to {}-d but the bank is {}-d"
                     .format(q.shape[1], bank_dim))

    bank = np.load(config.STREET_CACHE / bank_file, mmap_mode="r")
    # Search the rows the checkpoint's k-NN was actually built over, and take
    # the neighbour count it trained with. Searching the whole embedding file
    # added 99,820 rows the training bank excluded, and a hardcoded K=32 fed
    # the prior twice the neighbours it saw in training -- which is a direct
    # confound for anything studying how much the policy leans on retrieval.
    keep = None
    kf = ck.get("knn_file")
    if a.bank and a.bank != ck.get("street_file"):
        # --bank swaps the corpus out from under the checkpoint, which is the
        # whole point of the 2x2. But the checkpoint's bank_rows index its OWN
        # bank file, and applying them to a different one selects a different
        # set of images -- in range, so silently. Ask for the rows built over
        # the bank actually in use, via the same constructor build_knn writes
        # with, and refuse rather than guess if there is no such cache.
        kf = config.knn_name(bank_file, ck.get("split_mode", "sequence"),
                             ext=meta_stem)
        if not (config.STREET_CACHE / kf).exists():
            sys.exit("--bank {} needs the kNN cache built over it ({}) to know "
                     "which rows that bank contains; the checkpoint's own "
                     "{} describes a different file."
                     .format(bank_file, kf, ck.get("knn_file")))
        print("bank override: rows from {} (not the checkpoint's {})"
              .format(kf, ck.get("knn_file")), flush=True)
    if kf and (config.STREET_CACHE / kf).exists():
        meta = np.load(config.STREET_CACHE / kf, allow_pickle=True)
        # Validated, not read. A raw fancy-index assignment into the mask
        # accepts a negative row and wraps it to the end of the bank, marking
        # images nobody selected -- silently, at the right shape. REVIEW4 #14.
        keep = knnmeta.check(meta, kf, n_bank=bank.shape[0],
                             street_path=config.STREET_CACHE / bank_file)
    if keep is None:
        keep = np.arange(bank.shape[0], dtype=np.int64)
        print("bank rows  {:,} (checkpoint records none)".format(len(keep)))
    else:
        print("bank rows  {:,} of {:,} in the file, from {}"
              .format(len(keep), bank.shape[0], kf))
    keep_set = np.zeros(bank.shape[0], bool)
    keep_set[keep] = True

    qn = torch.from_numpy(q.astype(np.float32)).to(dev)
    qn = torch.nn.functional.normalize(qn, dim=1)
    K = int(ck.get("retr_k", 16)) if ck.get("retr") else 16
    print("neighbours {} (the checkpoint's retr_k)".format(K))
    best_v = torch.full((len(q), K), -2.0, device=dev)
    best_i = torch.zeros((len(q), K), dtype=torch.long, device=dev)
    t0 = time.time()
    for s in range(0, bank.shape[0], 200000):
        m = keep_set[s:s + 200000]
        if not m.any():
            continue
        gidx = torch.from_numpy(np.flatnonzero(m).astype(np.int64) + s).to(dev)
        B = torch.from_numpy(
            np.asarray(bank[s:s + 200000][m], np.float32)).to(dev)
        B = torch.nn.functional.normalize(B, dim=1)
        sim = qn @ B.T
        v, i = torch.topk(sim, min(K, sim.shape[1]), dim=1)
        cat_v = torch.cat([best_v, v], 1)
        cat_i = torch.cat([best_i, gidx[i]], 1)   # local -> global row id
        best_v, sel = torch.topk(cat_v, K, dim=1)
        best_i = torch.gather(cat_i, 1, sel)
        del B, sim
    # len(keep), not bank.shape[0]: the search is masked to the checkpoint's own
    # bank, and reporting the file's row count here restates a number instead of
    # deriving it -- which is the bug this whole restriction exists to undo.
    print("kNN over {:,} bank rows in {:.0f}s   top-1 sim {:.4f}".format(
        len(keep), time.time() - t0, float(best_v[:, 0].mean())), flush=True)

    # z16 address of every bank row: release rows from dataset.parquet, then
    # each extension's saved addresses, in the order stack_bank laid them down
    import pyarrow.parquet as pq
    ds = pq.read_table(config.DATASET_PARQUET, columns=["lat", "lon"])
    bx, by = tile_for_vec(np.asarray(ds["lat"], np.float64),
                          np.asarray(ds["lon"], np.float64), 4 * tm.STEPS)
    m = prov.bank_ext(meta_stem, config.RELEASE)
    bx = np.concatenate([bx, m["x16"].astype(bx.dtype)])
    by = np.concatenate([by, m["y16"].astype(by.dtype)])
    assert len(bx) == bank.shape[0], (meta_stem, len(bx), bank.shape[0])

    ext = ExternalSet(q, lat, lon, best_i.cpu().numpy(),
                      best_v.cpu().numpy().astype(np.float32), bx, by)

    tbl = None
    if ck.get("retr_mode") in ("pos", "dual"):
        tbl = street_table(config.STREET_CACHE / bank_file, dev)
    src = source_for(ck)
    m = evaluate(model, ext, src, dev, None, beam_k=a.beam,
                 top_m=max(4, a.beam), greedy=(a.beam == 1),
                 score_steps=a.score_steps, street_gpu=tbl)

    e = m["err"]
    # names.describe() reports the bank the arm TRAINED on. Under --bank
    # that is not the bank just searched, and a line reading "2.65M bank"
    # over a 3.40M result is the confusion the 2x2 exists to resolve.
    swapped = ("   [evaluated against {}]".format(bank_file)
               if a.bank and a.bank != ck.get("street_file") else "")
    print("\n{}  ({}){}".format(a.tag, names.describe(a.tag), swapped))
    print("{:,} high-res images   median {:.1f} km   mean {:.1f}   <25km {:.1%}"
          .format(len(e), float(np.median(e)), float(e.mean()),
                  float((e < 25).mean())))
    for t in (1, 25, 200, 750, 2500):
        print("   <{:>5} km  {:5.1%}".format(t, float((e < t).mean())))
    print("\nper-step tile accuracy " + "  ".join(
        "s{} {:.1%}".format(i, v) for i, v in enumerate(m["step_acc"])))
    if a.export:
        np.savez(a.export, err=e,
                 image_id=np.array(pick),
                 # so a later paired comparison can prove the two runs scored
                 # the same benchmark rather than assume it from --n/--seed
                 cohort=cohort_digest,
                 tag=a.tag, bank=bank_file)
        print("wrote " + a.export)
    print("")
    print("Compare against this checkpoint's own row in the matching "
          "BOOTSTRAP_*.md. Do not hardcode it here: that printed one "
          "arm's OSV-5M number under another arm's name.")
    # This used to end "a gap here is domain shift, not overfitting". Half of
    # that reading is now known to be wrong: on 2026-09-04 the OSV-5M side was
    # found to serve same-sequence frames as neighbours (41.6% of test queries
    # had a same-drive top-1, at a median 0.31 km), so a large part of the gap
    # is the OSV-5M number being inflated rather than this one being depressed.
    # The clause that IS still true is the one about these images.
    print("These images are not in the bank and share no sequence with "
          "anything in it,")
    print("so this number carries no same-sequence leakage. Until the kNN "
          "cache is rebuilt")
    print("with the fix from review item 61, the OSV-5M side does, and the "
          "gap between")
    print("them is not purely domain shift.")


if __name__ == "__main__":
    main()
