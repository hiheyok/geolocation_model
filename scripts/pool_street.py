"""Mean-pool the crop tokens of a dual cache: 4608-d -> 1536-d.

The dual cache stores three crops through two encoders as one flat vector laid
out `[DINOv2 c0 c1 c2 | SigLIP c0 c1 c2]`.  Read as tokens that is three
1536-d `[DINOv2 | SigLIP]` vectors, one per crop, glued end to end.  Averaging
the three instead of concatenating them measured at **+0.20 pp [-0.30, +0.70]**
on any-of-32 `<25 km` against the equal-norm 4608-d vector -- indistinguishable,
at a third of the bytes.

Why that is worth a run rather than a footnote:

  * **The bank is the memory-bound part of the system.**  The neighbour table is
    11.52 GB for 1.25M images at 4608-d; at 1536-d the same RAM holds 3.75M.
    Corpus size is the axis that has moved this metric most (+20.4 pp for
    400k -> 1.15M), so a third of the bytes is three times the corpus.
  * **It shrinks the model too.**  `StreetProj` and the retrieval keys are both
    shaped by the 4608 input and are together 4.72M of the 9.22M trainable
    parameters.  Pooled they are roughly 1.6M, taking the whole model to ~6M.

What this script does NOT settle is whether the retrieval-level result survives
into the agent's own metric, which is why it exists: produce the cache, rebuild
the kNN, train one arm, compare paired.  Nothing here is reversible-by-accident
-- the pooled cache is written under its own name and the 4608-d files are left
alone.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config

D_ENC = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dual_bal.f16.npy")
    ap.add_argument("--out", default=None,
                    help="default: src with 'dual' replaced by 'pool'")
    ap.add_argument("--block", type=int, default=100000,
                    help="rows per pass; 100k x 4608 fp16 is ~0.9 GB")
    a = ap.parse_args()

    src = config.STREET_CACHE / a.src
    out = config.STREET_CACHE / (a.out or a.src.replace("dual", "pool", 1))
    if out == src:
        sys.exit("refusing to write over the source")

    X = np.load(src, mmap_mode="r")
    n, d = X.shape
    if d % (2 * D_ENC):
        sys.exit("{} is {}-d, not a whole number of 768-d blocks per encoder"
                 .format(a.src, d))
    crops = d // (2 * D_ENC)
    half = d // 2
    print("{}  {:,} x {} = {} crops x [dino {} | siglip {}]".format(
        a.src, n, d, crops, D_ENC, D_ENC), flush=True)
    print("{}  {:,} x {}   {:.2f} -> {:.2f} GB".format(
        out.name, n, 2 * D_ENC, X.nbytes / 1e9,
        n * 2 * D_ENC * 2 / 1e9), flush=True)

    Y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(n, 2 * D_ENC))
    t0 = time.time()
    for s in range(0, n, a.block):
        e = min(s + a.block, n)
        B = np.asarray(X[s:e], dtype=np.float32)
        # (rows, crops, 1536): the two encoder halves are each crop-major, so
        # they are un-interleaved separately and re-joined per crop.
        tok = np.concatenate([B[:, :half].reshape(-1, crops, D_ENC),
                              B[:, half:].reshape(-1, crops, D_ENC)], axis=2)
        Y[s:e] = tok.mean(1).astype(np.float16)
        el = time.time() - t0
        print("  {:>9,}/{:,}  {:.0f}s  eta {:.0f}s".format(
            e, n, el, el * (n - e) / max(e, 1)), flush=True)
    Y.flush()

    # Cheap correctness check rather than trust: the pooled vector must equal
    # the mean of the three crop tokens for a handful of rows, recomputed.
    rng = np.random.default_rng(0)
    for i in rng.choice(n, 5, replace=False):
        v = np.asarray(X[i], dtype=np.float32)
        want = np.concatenate([v[:half].reshape(crops, D_ENC),
                               v[half:].reshape(crops, D_ENC)], axis=1).mean(0)
        got = np.asarray(Y[i], dtype=np.float32)
        assert np.allclose(want, got, atol=2e-2), "row {} disagrees".format(i)
    print("\nwrote {} in {:.0f}s; 5 rows verified against a recompute".format(
        out.name, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
