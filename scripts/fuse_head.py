"""A learned fusion head over the 9-region, 2-encoder token set.

Everything about this design was fixed by measurement rather than taste:

* **9 regions, because the training data cannot support more.** OSV-5M frames
  are 682x512, so at 224px tiles they bottom out at exactly 3x2 -- a 6x4 grid
  there means upsampling 1344x896 out of 682x512, which is inventing pixels.
  The 3-level pyramid did test worse, but note that was measured under *mean
  pooling*, where a weak token necessarily dilutes the average. Attention can
  ignore a token; a mean cannot. So that result does not license 9 here, and
  33 is worth testing on high-resolution source once there is enough of it to
  train on.
* **18 tokens, not 9.** One token per (region, encoder). With the encoders
  concatenated into a single 1536-d token they are channels: a projection can
  mix them but attention cannot *select* between them, and cross-attention
  between encoders is not expressible at all.
* **A level embedding, because level weighting is worth 4 pp.** Weighting
  tokens equally against weighting levels equally swung the 3-level pyramid by
  4 pp. A hand-picked pooling constant doing that much work is the actual
  argument for learning it -- and for learning it *per image*, since a frame
  with a legible storefront wants its tiles and a featureless highway wants its
  scene.
* **Output at 768-d**, the same width as the mean-pooled baseline, so the
  comparison is equal-bytes. The head has to win on what it packs into the
  vector, not on carrying a bigger one.

The bar is set by measurement too. Mean-pooling this same token set converts
+1.50 pp [+0.3, +2.8] of the +5.00 pp [+4.23, +5.80] of oracle headroom the
complementarity analysis found. So the head is not competing against crops --
it is competing against `L0+L1` level-weighted, with about 3.5 pp left on the
table.

The head is a **residual on the mean-pooled baseline** with its last layer
zero-initialised, so it emits exactly that baseline at init and can only depart
if it pays. Starting from random makes it rediscover mean pooling before it can
beat it, which confounds "the head is bad" with "the head is untrained" -- the
first run of this did exactly that and read -23 pp. Only the last layer is
zeroed: zeroing a gate that multiplies a zero-init tensor deadlocks, each being
the other's gradient.

Training is geographic contrastive: an anchor's positive is another image within
`--pos-km` from a *different sequence*, negatives are the rest of the batch.
Same-sequence positives would be near-duplicate frames of one drive and the head
would learn to match those instead of learning geography -- the same trap
`build_knn` guards against.
"""

import argparse
import zlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config
import safeio
import splits as sp
from tile_pool import l2, paired
from tile_match import dense_sim, topk_stats

D_ENC = 768
THRESH = (1, 25, 200, 750, 2500)


class FuseHead(nn.Module):
    """Cross-attention between encoder groups, then self-attention, then pool."""

    def __init__(self, d=256, out=768, heads=4, n_reg=9, drop=0.1,
                 level_of=None):
        super().__init__()
        # Which pyramid level each token belongs to. Defaults to the
        # two-level OSV-5M layout (3 crops then 6 tiles) so existing runs
        # are unchanged; a high-resolution pyramid passes 3+6+24.
        if level_of is None:
            level_of = [0] * 3 + [1] * (n_reg - 3)
        lv = torch.as_tensor(level_of, dtype=torch.long)
        assert len(lv) == n_reg, "level_of must cover every token"
        self.register_buffer("level_of", lv, persistent=True)
        self.n_lvl = int(lv.max()) + 1
        # separate input projections: the two encoders occupy different native
        # spaces and nothing makes them comparable a priori
        self.proj = nn.ModuleList([nn.Linear(D_ENC, d) for _ in range(2)])
        self.reg = nn.Embedding(n_reg, d)       # which region
        self.lvl = nn.Embedding(self.n_lvl, d)  # which scale -- worth 4 pp
        self.enc = nn.Embedding(2, d)           # which encoder
        self.n_cross = nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=drop,
                                           batch_first=True)
        self.n_self = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=drop,
                                               batch_first=True)
        self.n_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(4 * d, d))
        self.q = nn.Parameter(torch.randn(4, d) * d ** -0.5)
        # a fixed, non-learned map from the 1536-d level-weighted mean pool to
        # the output width, so "start at the baseline" is exact rather than
        # something the head has to fit
        g = torch.Generator().manual_seed(0)
        w = torch.randn(2 * D_ENC, out, generator=g) / (2 * D_ENC) ** 0.5
        self.register_buffer("base_w", w, persistent=True)
        self.pool = nn.MultiheadAttention(d, heads, dropout=drop,
                                          batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, out))
        # The head is a RESIDUAL on the mean-pooled baseline, and this last
        # layer is zero-initialised, so at init it emits exactly that baseline
        # and can only depart if it pays. Starting from random would make it
        # rediscover mean pooling before it could beat it, which is a waste of
        # the first several epochs and confounds "the head is bad" with "the
        # head is untrained". Only the LAST layer is zeroed: zeroing a gate that
        # multiplies a zero-init tensor deadlocks, since each is the other's
        # gradient.
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def baseline(self, x):
        """Level-weighted mean pool: the arm the head must beat.

        The mean of the per-level means, not the mean over tokens. Those differ
        badly once a level is large: a 3+6+24 pyramid has 73% of its tokens at
        the deepest level, so token-weighting hands that level the vector, and
        it measured 4 pp worse at three levels.
        """
        per = [torch.nn.functional.normalize(
            x[:, self.level_of == l].mean(1), dim=-1)
            for l in range(self.n_lvl)]
        return torch.stack(per).mean(0).flatten(1)      # (B, 2*768)

    def forward(self, x):
        """x: (B, n_reg, 2, 768) -> (B, out), L2-normalised."""
        B, R, E, _ = x.shape
        reg = torch.arange(R, device=x.device)
        lvl = self.level_of                     # set from the cache
        h = torch.stack([self.proj[e](x[:, :, e]) for e in range(E)], dim=2)
        h = h + (self.reg(reg) + self.lvl(lvl)).unsqueeze(0).unsqueeze(2)
        h = h + self.enc(torch.arange(E, device=x.device)).view(1, 1, E, -1)
        a, b = h[:, :, 0], h[:, :, 1]           # dino group, siglip group
        # bidirectional cross-attention: each group reads the other
        an = self.n_cross(a)
        bn = self.n_cross(b)
        a = a + self.cross(an, bn, bn, need_weights=False)[0]
        b = b + self.cross(bn, an, an, need_weights=False)[0]
        t = torch.cat([a, b], dim=1)            # (B, 2R, d)
        tn = self.n_self(t)
        t = t + self.self_attn(tn, tn, tn, need_weights=False)[0]
        t = t + self.ffn(self.n_ffn(t))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        p = self.pool(q, t, t, need_weights=False)[0].reshape(B, -1)
        return torch.nn.functional.normalize(
            self.base(x) + self.out(p), dim=-1)

    def base(self, x):
        """The baseline projected to the output width, frozen in place."""
        b = self.baseline(x)
        return b @ self.base_w


def load_tokens(sel):
    """(n, 9, 2, 768) float16: 3 crops then 6 tiles, split by encoder."""
    C = np.asarray(np.load(config.STREET_CACHE / "dual_c3.f16.npy",
                           mmap_mode="r")[sel], np.float32)
    h = C.shape[1] // 2
    nc = h // D_ENC
    C = np.stack([C[:, :h].reshape(-1, nc, D_ENC),
                  C[:, h:].reshape(-1, nc, D_ENC)], axis=2)
    T = np.asarray(np.load(config.STREET_CACHE / "tile6.f16.npy",
                           mmap_mode="r"), np.float32)
    T = np.stack([T[:, :, :D_ENC], T[:, :, D_ENC:]], axis=2)
    X = np.concatenate([C, T], axis=1)
    X /= np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-6)   # per-token L2
    return X.astype(np.float16)


def unit3(lat, lon):
    p = np.pi / 180
    return np.stack([np.cos(lat * p) * np.cos(lon * p),
                     np.cos(lat * p) * np.sin(lon * p), np.sin(lat * p)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--pos-km", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--hard", action="store_true",
                    help="draw each batch from ONE geographic bucket, so the "
                         "in-batch negatives are a few hundred km away rather "
                         "than global. Without it the task is 'same continent?' "
                         "-- the first run learned exactly that, gaining "
                         "+3.97 pp at 2500 km while losing 4.10 pp at 25 km.")
    ap.add_argument("--bucket-z", type=int, default=6,
                    help="zoom of the bucket grid; z6 cells are ~626 km")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pyr-stem", default="pyr33",
                    help="which pyramid cache --tokens pyr33 reads; the "
                         "18,812-image pyr33 run was confounded by data "
                         "volume, so the re-run needs its own cache rather "
                         "than overwriting the one it is compared against")
    ap.add_argument("--tokens", default="osv", choices=("osv", "pyr33"),
                    help="osv: 3 crops + 6 tiles over OSV-5M, two levels. "
                         "pyr33: 3 + 6 + 24 over the high-resolution harvest, "
                         "three levels -- the arm mean pooling cannot judge")
    ap.add_argument("--export", default="",
                    help="stem to write the fused vectors and the trained head "
                         "to, so the agent can be trained on them")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree

    torch.manual_seed(a.seed)
    t0 = time.time()
    if a.tokens == "pyr33":
        m = np.load(config.STREET_CACHE / (a.pyr_stem + "_meta.npz"),
                    allow_pickle=True)
        lat, lon = m["lat"], m["lon"]
        seq = m["sequence"].astype("U40")
        level_of = m["level_of"].tolist()
        # The harvest has no split of its own. Split whole *sequences*, as the
        # main benchmark does: frames from one drive are near-duplicates, so a
        # row-wise split would put a near-copy of every test image in train and
        # the numbers would be meaningless.
        # crc32, not hash(): Python salts string hashing per process, so
        # hash() would silently reshuffle the split on every run and no two
        # arms would be comparable.
        h = np.array([zlib.crc32(x.encode()) % 10
                      for x in seq.tolist()])
        tr = np.flatnonzero(h < 8)
        te = np.flatnonzero(h >= 8)
        X = np.asarray(np.load(config.STREET_CACHE
                               / (a.pyr_stem + ".f16.npy"),
                               mmap_mode="r"), np.float32)
        X /= np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-6)
        X = X.astype(np.float16)
        print("{:,} pyramid rows: {:,} train, {:,} test   levels {}"
              .format(len(lat), len(tr), len(te), np.bincount(level_of)),
              flush=True)
    else:
        level_of = None
        sel = np.load(config.STREET_CACHE / "tile6_rows.i64.npy")
        ds = pq.read_table(config.DATASET_PARQUET)
        lat = np.asarray(ds["lat"], np.float64)[sel]
        lon = np.asarray(ds["lon"], np.float64)[sel]
        seq = np.asarray(ds["sequence"]).astype("U40")[sel]
        spl = sp.read(ds, "sequence")[0][sel]
        tr = np.flatnonzero(spl == "train")
        te = np.flatnonzero(spl == "test")
        print("{:,} cached rows: {:,} train, {:,} test".format(
            len(sel), len(tr), len(te)), flush=True)
        X = load_tokens(sel)
    print("tokens {}  {:.2f} GB  in {:.0f}s".format(X.shape, X.nbytes / 1e9,
                                                    time.time() - t0),
          flush=True)

    # positives: a different image within pos-km, from a different sequence
    R = 6371.0088
    tree = cKDTree(unit3(lat[tr], lon[tr]))
    rad = 2 * np.sin(a.pos_km / (2 * R))
    pairs = []
    for i, js in enumerate(tree.query_ball_point(unit3(lat[tr], lon[tr]), rad)):
        for j in js:
            if j != i and seq[tr[j]] != seq[tr[i]]:
                pairs.append((tr[i], tr[j]))
    pairs = np.array(pairs, np.int64)
    print("{:,} anchor-positive pairs within {:.0f} km, different sequence "
          "({:.1f}% of train images have one)".format(
              len(pairs), a.pos_km,
              100 * len(np.unique(pairs[:, 0])) / len(tr)), flush=True)
    if len(pairs) < 1000:
        sys.exit("too few positives; raise --pos-km")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FuseHead(d=a.d, n_reg=X.shape[1],
                     level_of=level_of).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * max(1, len(pairs) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps,
                                                pct_start=0.1)
    print("head {:,} params   {:,} steps".format(n_par, steps), flush=True)

    # The token table is read-only and small enough to live on the card at
    # this size, which removes the host gather and the PCIe copy entirely --
    # indexing then happens in VRAM. Falls back to a pinned host gather when it
    # does not fit, and the two lookups per step are combined into one so the
    # expensive fancy-index runs once rather than twice.
    Xg = None
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        need = X.nbytes + (1 << 30)          # table plus a gigabyte of headroom
        if need < free:
            Xg = torch.from_numpy(X).to(dev)
            print("tokens on GPU  {:.2f} GB resident, {:.2f} GB free"
                  .format(X.nbytes / 1e9, free / 1e9), flush=True)
        else:
            print("tokens stay on host: {:.2f} GB needed, {:.2f} GB free"
                  .format(need / 1e9, free / 1e9), flush=True)

    def take_pair(idx):
        """Both halves of the batch in one gather -> (2B, ntok, 2, D)."""
        flat = np.concatenate([idx[:, 0], idx[:, 1]])
        if Xg is not None:
            t = Xg[torch.from_numpy(flat).to(dev)].float()
        else:
            t = torch.from_numpy(X[flat]).pin_memory().to(
                dev, non_blocking=True).float()
        return t[:len(idx)], t[len(idx):]

    def take(rows):
        return torch.from_numpy(X[rows]).to(dev).float()

    rng = np.random.default_rng(a.seed)

    # Hard negatives by construction: bucket the anchors geographically and draw
    # each batch from one bucket. In-batch negatives are then a few hundred km
    # away instead of on another continent, which is the difference between
    # learning "same region?" and learning "same continent?".
    if a.hard:
        n = 1 << a.bucket_z
        px = ((lon[pairs[:, 0]] + 180.0) / 360.0 * n).astype(np.int64)
        s_ = np.sin(np.radians(np.clip(lat[pairs[:, 0]], -85.05, 85.05)))
        py = ((0.5 - np.log((1 + s_) / (1 - s_)) / (4 * np.pi)) * n).astype(np.int64)
        key = py * n + px
        buckets = [np.flatnonzero(key == k) for k in np.unique(key)]
        buckets = [b for b in buckets if len(b) >= a.batch]
        print("hard negatives: {:,} buckets of >= {} pairs at z{} "
              "(~{:.0f} km cells), covering {:,} of {:,} pairs".format(
                  len(buckets), a.batch, a.bucket_z, 40075.0 / n,
                  sum(len(b) for b in buckets), len(pairs)), flush=True)
        if not buckets:
            sys.exit("no bucket holds a full batch; lower --batch or --bucket-z")

    def batches():
        if not a.hard:
            order = rng.permutation(len(pairs))
            for s in range(0, len(order) - a.batch + 1, a.batch):
                yield pairs[order[s:s + a.batch]]
            return
        total = sum(len(b) // a.batch for b in buckets)
        w = np.array([len(b) for b in buckets], np.float64)
        w /= w.sum()
        for _ in range(total):
            b = buckets[rng.choice(len(buckets), p=w)]
            yield pairs[rng.choice(b, a.batch, replace=False)]

    step = 0
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for idx in batches():
            ta, tp = take_pair(idx)
            za, zp = model(ta), model(tp)
            logits = za @ zp.T / a.tau
            tgt = torch.arange(len(idx), device=dev)
            loss = 0.5 * (nn.functional.cross_entropy(logits, tgt) +
                          nn.functional.cross_entropy(logits.T, tgt))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
            step += 1
        print("  epoch {:>2}  loss {:.4f}  {:.0f}s".format(ep, tot / max(nb, 1),
                                                           time.time() - t0),
              flush=True)

    # ---- evaluation: test queries against a train bank, same-sequence masked
    model.eval()
    Z = np.zeros((len(X), 768), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 1024):
            Z[s:s + 1024] = model(take(np.arange(s, min(s + 1024, len(X))))
                                  ).cpu().numpy()

    Xf = X.astype(np.float32)
    base_l0 = Xf[:, :3].reshape(len(X), -1, D_ENC).mean(1)
    base_l0 = np.concatenate([l2(Xf[:, :3, 0].mean(1)),
                              l2(Xf[:, :3, 1].mean(1))], 1)
    # The level-weighted mean pool, over however many levels the cache has.
    # This MUST match FuseHead.baseline: the head is a residual on it, so if the
    # printed baseline is computed differently the head is being scored against
    # an arm it never started from. Hardcoding two levels here silently did
    # exactly that on a three-level pyramid -- "3 crops vs all 30 tiles".
    lo = np.asarray(model.level_of.cpu())
    lvl = lambda i: np.stack([l2(Xf[:, lo == l, i].mean(1))
                              for l in range(model.n_lvl)]).mean(0)
    base_un = np.concatenate([l2(lvl(0)), l2(lvl(1))], 1)
    LBL = "L0+L1 level (mean)" if model.n_lvl == 2 else           "+".join("L%d" % l for l in range(model.n_lvl)) + " level (mean)"

    nq = min(a.queries, len(te))
    qi, bi = te[:nq], tr
    same = seq[qi][:, None] == seq[bi][None, :]
    print("\n{:,} test queries  {:,} train bank  {:,} same-sequence masked"
          .format(nq, len(bi), int(same.sum())), flush=True)

    err = {}
    print("\n%-26s %10s %s" % ("arm", "median km",
                               " ".join("%9s" % ("<%gkm" % t) for t in THRESH)))
    print("-" * 88)
    # The head and the mean pool win at opposite ends -- the head is better
    # coarse, the mean better fine -- so score the union too. Same reasoning
    # that made crops+tiles work: complementary errors beat either alone.
    combo = np.concatenate([l2(base_un), l2(Z)], axis=1)

    def pca_to(X, d, rng, fit=40000):
        F = X[rng.choice(len(X), min(fit, len(X)), replace=False)]
        mu = F.mean(0, keepdims=True)
        Fc = F - mu
        Om = rng.standard_normal((Fc.shape[1], d + 64)).astype(np.float32)
        Y = Fc @ Om
        for _ in range(2):
            Y = Fc @ (Fc.T @ Y)
        Q, _ = np.linalg.qr(Y)
        _, _, Vt = np.linalg.svd(Q.T @ Fc, full_matrices=False)
        return (X - mu) @ Vt[:d].T

    combo_eq = pca_to(combo, base_un.shape[1], np.random.default_rng(0))

    if a.export:
        # Everything above is a retrieval probe: 3,000 queries against a 96k
        # bank, never the agent's own metric.  Pooling looked free at this level
        # too and still needed a training run to confirm it, so the fused vector
        # has to reach train.py before it is believed.  `sel` is already parquet
        # row indices, so a street file can be assembled from these two files
        # without re-running the head.
        stem = config.STREET_CACHE / a.export
        np.save(str(stem) + ".f16.npy", combo_eq.astype(np.float16))
        np.save(str(stem) + "_rows.i64.npy",
                sel if a.tokens == "osv" else np.arange(len(X)))
        safeio.save_torch({"state": model.state_dict(), "d": a.d, "tau": a.tau,
                    "pos_km": a.pos_km, "seed": a.seed,
                    "n_rows": int(len(X))}, str(stem) + "_head.pt")
        print("")
        print("exported {}.f16.npy  {} x {}   (+ _rows.i64.npy, _head.pt)"
              .format(a.export, len(X), combo_eq.shape[1]),
              flush=True)
        print("NOTE: PCA output is not L2-normalised, like the pooled caches.")
    for name, V in (("L0 crops (mean)", base_l0),
                    (LBL, base_un),
                    ("fusion head (learned)", Z),
                    ("mean + head (concat)", combo),
                    ("mean + head, PCA to 1536", combo_eq)):
        e, _ = topk_stats(dense_sim(V[qi], V[bi], dev), lat, lon, qi, bi, same)
        err[name] = e
        print("%-26s %5d-d %10.1f %s" % (name, V.shape[1], np.median(e), " ".join(
            "%8.1f%%" % (100 * (e < t).mean()) for t in THRESH)), flush=True)

    rng2 = np.random.default_rng(0)
    for arm in ("fusion head (learned)", "mean + head (concat)",
                "mean + head, PCA to 1536"):
        if arm not in err:
            continue
        for base in ("L0 crops (mean)", LBL):
            cells = []
            for t in THRESH:
                lo, hi = paired(err[base] < t, err[arm] < t, rng2)
                cells.append("%+5.2f[%+.1f,%+.1f]%s" % (
                    100 * ((err[arm] < t).mean() - (err[base] < t).mean()),
                    lo, hi, " " if lo * hi > 0 else "~"))
            print("\n{} against {}".format(arm, base))
            print("  " + " ".join(cells))


if __name__ == "__main__":
    main()
