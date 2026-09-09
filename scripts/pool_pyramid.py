"""Write the pyramid's 1536-d street vector: crops alone, or crops plus tiles.

`runs/RESMATCH.md` decided the rebuild is tiles at 224, on a retrieval probe.
The agent consumes neighbours through a *trained* retrieval prior, so only a
retrain converts that into an agent number -- and this project has twice had
inference-level reasoning predict the wrong sign. This script exists to make
that retrain a clean paired experiment: **one code path, one flag between the
arms.**

    py scripts/pool_pyramid.py --out pyr_l0.f16.npy                  # crops
    py scripts/pool_pyramid.py --tiles tile6 --out pyr_mix.f16.npy   # crops+tiles

The arithmetic is exactly what the probes measured, which is *not* what
`pool_street.py` does. `pool_street` takes a plain mean of the crop tokens; the
probes normalise per token, per encoder half, and again after the join:

    C   = l2(crop tokens)                      (n, crops, 2, 768)
    L0  = l2([ l2(mean_c C_dino) | l2(mean_c C_siglip) ])
    L1  = l2([ l2(mean_t T_dino) | l2(mean_t T_siglip) ])
    out = L0                     or     l2((1-w)*L0 + w*L1)

Those two happen to score the same (`pool_bal` 226.0 km / 27.8% against `L0`
223.0 km / 27.7%, identical any-of-32), which is why `L0` can stand in as the
untiled arm -- but "scores the same" is not "is the same", and a paired
experiment wants the untiled arm produced by this code rather than by the other
one. Otherwise the contrast carries a normalisation difference as well as the
tiles.

**Per-token normalisation subsumes `--scale-b`.** `nrm` divides each 768-d
token by its own norm, so any positive per-encoder rescaling cancels: measured,
`L0(dual_c3)` and `L0(dual_bal)` agree at **cosine 0.99999976** over 200 rows
(max abs difference 9.3e-05, which is fp16 storage noise) while their raw
SigLIP norms are 20.53 and 82.73. So it does not matter whether this is fed the
balanced or unbalanced dual cache -- including on the extension side, where
both `bank_ext_dual` and `bank_ext_bal` exist and picking wrong would otherwise
be a silent seam between the two halves of a bank.

`w` defaults to 0.5, the equal blend every pyramid number on record was
measured at. It is deliberately *not* the 0.04 of `runs/PYR_BLEND.md`: that
weight blends a learned fusion head against a level mean, which is a different
pair of things.

**A partly-tiled output is refused, not filled.** A bank whose rows are a mix
of `l2((L0+L1)/2)` and `L0` is not a smaller tiled bank, it is a broken one --
one cosine ranks both, so an untiled row has the query's `L1` half scored
against its `L0`, cross-space and systematically lower, and it is silently
demoted. Nothing downstream can see that: every row is a unit vector of the
right width. So every requested row must be present *and* marked done in the
tile cache, or this exits.
"""

import argparse
import sys
import time
from pathlib import Path

# BEFORE torch and before any large mmap, deliberately. `pq.read_table`
# imports `pyarrow.dataset` lazily on first use, and that DLL load takes an
# access violation once CUDA has been initialised OR a multi-gigabyte mmap is
# open -- either alone is enough, measured. The tiles arm hit both, so it died
# at 0xC0000005 with an empty log and three identical retries. Importing here
# costs nothing and the crash does not reproduce.
import pyarrow.dataset          # noqa: F401  (imported for its side effect)
import pyarrow.parquet as pq
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config                                        # noqa: E402
import embed_street                                  # noqa: E402
import maskio                                        # noqa: E402
import provenance as prov                            # noqa: E402

D_ENC = 768


AXES = ("depth", "row", "col")

# Small on purpose -- see `rope_angles`. The coordinates here span 0 to 2,
# so a language model's geometric spread would park most of the vector at
# under a degree per unit and carry no position in it.
ROPE_SPREAD = 10.0

# The frame the corpus was embedded from, and the window size slid across
# it. Every OSV-5M image is 910x512; `--frame` exists because the crop
# positions are a function of it and a wrong one is invisible downstream.
FRAME = (910, 512)
CROP_SIZE = 224


def tile_grid(stem):
    """(cols, rows) of the tile cache, or None if it does not record one.

    3x2 and 2x3 both hold six tiles, so the grid cannot be inferred from the
    tile count -- `tile_cache.check_resume` already refuses to resume across
    that boundary for the same reason. A rotation that guesses wrong assigns
    every tile the wrong row and column and still writes a bank of unit
    vectors, so this reads the recorded value rather than deriving one.
    """
    meta_p = config.STREET_CACHE / (stem + "_meta.npz")
    if not meta_p.exists():
        return None
    meta = np.load(meta_p, allow_pickle=True)
    if "grid" not in meta.files:
        return None
    gc, gr = (int(v) for v in meta["grid"])
    return gc, gr


def crop_columns(crops, gc, frame, size=CROP_SIZE):
    """Where the crops actually sit, measured in tile columns.

    **Not 0, 1, 2.** `embed_street.preprocess` scales the short side to `size`
    and slides a size-by-size window across the width, so on the 910x512
    OSV-5M frame the crops are 224 pixels of 398 -- 56% of the width -- placed
    at lefts 0, 87, 174 and overlapping by 61%. Their centres land at

        0.344, 1.000, 1.656   tile columns

    and the outer two are 1.31 columns apart, not 2.

    The difference is not cosmetic. Putting the outer crops at 0 and 2 claims
    they are as far apart as the outer tile columns, so a `col` rotation would
    separate two views that mostly show the same pixels -- and it would align
    each crop with a tile centre it does not sit on, which is precisely the
    cross-level matching the rotation exists to get right.

    A crop is an extent, not a point, and giving it one coordinate is an
    approximation either way. The centre is the honest choice; inventing a
    correspondence with the columns is not.
    """
    W, H = frame
    w = max(size, round(W * size / min(W, H)))
    return [gc * (l + size / 2.0) / w - 0.5
            for l in embed_street.crop_lefts(w, size, crops)]


def view_positions(crops, grid, cols=None):
    """(depth, row, col) for every view, crops first then tiles.

    The two levels share one coordinate frame, which is the only reason a
    depth rotation means anything. `cols` is where the crops sit along that
    frame, from `crop_columns`; the tiles are a uniform partition, so tile `t`
    is simply at `(t // gc, t % gc)`.

    Crops are full height -- `preprocess` takes a size-by-size window from an
    image whose short side is already `size` -- so a crop spans every tile row
    and takes the centre one, `(gr - 1) / 2`. Fractional rows are deliberate:
    rounding a crop onto one row would claim it is nearer that row than the
    other, which is a claim about the image nobody has evidence for.
    """
    gc, gr = grid if grid else (crops, 1)
    if cols is None:
        cols = list(range(crops))
    p0 = [(0.0, (gr - 1) / 2.0, float(c)) for c in cols]
    p1 = [(1.0, float(t // gc), float(t % gc)) for t in range(gc * gr)]
    return p0, p1


def degenerate(pos, axes):
    """Which of `axes` every view shares a coordinate on.

    A rotation is only information when two views DIFFER on an active axis.
    Views at the same coordinate turn through the same angle, so a flat axis
    does not confuse them -- it just contributes nothing while still taking
    its round-robin share of the dimension pairs.

    This replaced two hand-written special cases ("depth or row without
    --tiles" and "a square frame collapses the crops"). The second was also
    wrong in its stated reason: it claimed rotating identical crops would
    separate them, when identical coordinates produce identical rotations. One
    derived rule is both correct and shorter than the list of cases somebody
    thought of.
    """
    return [ax for ax in axes
            if len({p[AXES.index(ax)] for p in pos}) == 1]


def rope_angles(pos, n_pairs, axes, rope_max, spread):
    """(views, n_pairs) -- how far each dimension pair turns for each view.

    Pairs are dealt to the axes **round-robin**, not in contiguous sections,
    which is where this departs from a language model's mRoPE. Contiguous
    sections hand one axis the fast frequencies and another the slow ones; the
    axes here are 2, 2 and 3 positions long, so whichever axis drew the slow
    end would barely rotate at all and would silently contribute nothing.

    The angle is parameterised by the rotation itself rather than by a base
    period, because the usual base of 10,000 is built for sequences of
    thousands. Over coordinates 0 to 2 it turns every pair through a fraction
    of a degree, and the whole transform collapses to the identity that this
    is supposed to be an alternative to: at base 10,000 the median pair turns
    1.16 degrees over the whole range and 48% of pairs turn less than one.
    `rope_max` is the angle the fastest pair turns through per unit
    coordinate; `spread` is the ratio between the fastest pair and the
    slowest.

    **Spread has to be small here, and that is not the language-model
    setting.** A geometric spread leaves most dimensions slow, which is the
    right trade over thousands of positions and the wrong one over three: at
    spread 1000, 41% of the pairs turn under 1 degree per unit and contribute
    no positional information at any coordinate this pyramid has. At spread 10
    the median pair turns 18 degrees and the slowest still turns 5.7, so the
    whole vector carries position. The default is 10 for that reason and the
    knob is worth sweeping before `rope_max` is.

    `rope_max = 0` gives exactly zero angles, hence cos 1 and sin 0, hence the
    unrotated pooling bit for bit -- the identity check that makes an arm
    ladder interpretable.
    """
    ang = np.zeros((len(pos), n_pairs), np.float64)
    for i, ax in enumerate(axes):
        own = np.arange(i, n_pairs, len(axes))
        m = len(own)
        k = np.arange(m, dtype=np.float64)
        theta = rope_max * float(spread) ** (-k / max(m - 1, 1))
        coord = np.array([p[AXES.index(ax)] for p in pos], np.float64)
        ang[:, own] = coord[:, None] * theta[None, :]
    return ang


def apply_rope(V, ang):
    """Rotate (b, views, 2, d) pairwise by (views, d/2) angles.

    Consecutive dimensions are paired -- (0,1), (2,3) ... -- and each pair is
    turned in its own plane. The transform is orthogonal, so it preserves
    every norm and therefore commutes with `nrm`; it changes only the angles
    *between* views, which is the entire point.
    """
    b, v, e, d = V.shape
    P = V.reshape(b, v, e, d // 2, 2)
    c = torch.cos(ang).to(V.dtype).view(1, v, 1, d // 2)
    s = torch.sin(ang).to(V.dtype).view(1, v, 1, d // 2)
    x, y = P[..., 0], P[..., 1]
    return torch.stack([x * c - y * s, x * s + y * c],
                       dim=-1).reshape(b, v, e, d)


def nrm(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def levels(C, T=None, a0=None, a1=None):
    """(b, crops, 2, 768) and (b, tiles, 2, 768) -> L0, L1 as the probes pool.

    Both inputs are token-normalised first, which is what makes the DINOv2 and
    SigLIP halves contribute equally without a `--scale-b` constant: their raw
    norms are 82.96 and 20.58, so an unnormalised join gives DINOv2 81% of the
    cosine.

    `a0` and `a1` are the per-view rotation angles. Without them the mean over
    views is permutation-invariant: a feature in the left crop and the same
    feature in the right crop pool to the same vector, and the retrieval
    cosine cannot tell them apart. Rotating each view by its own position
    first makes the pooled inner product

        <sum_k R(p_k) x_k, sum_l R(p_l) y_l> = sum_kl <x_k, R(p_l - p_k) y_l>

    so a match at the same position keeps full credit and a match across
    positions is attenuated by how far apart they are. Nothing is learned; the
    rotation is fixed by the geometry.
    """
    if a0 is not None:
        C = apply_rope(C, a0)
    if a1 is not None and T is not None:
        T = apply_rope(T, a1)
    C = nrm(C)
    L0 = nrm(torch.cat([nrm(C[:, :, 0].mean(1)), nrm(C[:, :, 1].mean(1))], 1))
    if T is None:
        return L0, None
    T = nrm(T)
    L1 = nrm(torch.cat([nrm(T[:, :, 0].mean(1)), nrm(T[:, :, 1].mean(1))], 1))
    return L0, L1


def check_tile_identity(stem, rows):
    """Are this cache's rows still the photographs it was embedded from?

    `rows` holds *positions* in an image list, and positions are not identity:
    rebuild that list at the same length and every tile embedding stays put
    under a row that now names a different photograph (REVIEW4 #19). This is
    the consumption half of the check `tile_cache` writes -- a recorded digest
    nobody reads is not a guard.

    It matters most precisely here. This function pairs each row's crops with
    the tile embeddings at the matching cache position, so a stale cache
    produces one image's crops beside another image's tiles: right shape, unit
    norm, complete mask, wrong photograph.

    Missing metadata is reported and allowed -- `tile6` predates both fields
    and holds 500,000 finished rows -- but never treated as agreement.
    """
    import provenance as prov

    meta_p = config.STREET_CACHE / (stem + "_meta.npz")
    if not meta_p.exists():
        return
    meta = np.load(meta_p, allow_pickle=True)
    if "rows_digest" not in meta.files or "rows_parquet" not in meta.files:
        print("warning: {} records no row identity, so its {:,} rows cannot be "
              "shown to still name the images they were embedded from"
              .format(stem, len(rows)), flush=True)
        return
    src = str(meta["rows_parquet"])
    pq_path = (config.DATASET_PARQUET if src == config.DATASET_PARQUET.name
               else config.PROCESSED / src)
    ids = np.asarray(pq.read_table(pq_path, columns=["image_id"])["image_id"])
    got = prov.rows_digest(ids[rows])
    if got != str(meta["rows_digest"]):
        sys.exit(
            "{} was built over {} whose rows now digest {} against the "
            "recorded {}. The row numbers still line up, so every shape and "
            "every mask agrees -- the images behind them changed, and pooling "
            "would pair each image's crops with another image's tiles."
            .format(stem, src, got, str(meta["rows_digest"])))


def tile_positions(stem, want):
    """Position of each wanted row in the tile cache, or exit.

    The cache stores a *selection*: `rows` holds the release rows it covers, in
    order, and `done` marks which of those were actually written. A row that is
    absent, or present but unwritten, is a zero fill -- and a zero row
    L2-normalises to a unit-length nothing that ranks like a real vector. This
    is the check whose absence would have built a bank 76% zeros.
    """
    rows_p = config.STREET_CACHE / (stem + "_rows.i64.npy")
    done_p = config.STREET_CACHE / (stem + "_done.u8.npy")
    if not rows_p.exists():
        sys.exit("{} has no _rows.i64.npy; cannot tell which rows it covers"
                 .format(stem))
    rows = np.load(rows_p)
    done = maskio.load_mask(done_p, len(rows), stem)
    if done.ndim > 1:
        done = done.min(axis=1)
    check_tile_identity(stem, rows)
    have = rows[done == 1]
    # Sized by both, not just `want`: the cache may legitimately cover rows the
    # caller did not ask for (a 750k cache queried for 400k of it), and sizing
    # this by the request alone makes writing the map itself go out of bounds.
    pos = np.full(int(max(rows.max(initial=-1), want.max(initial=-1))) + 1,
                  -1, np.int64)
    pos[rows] = np.arange(len(rows))
    missing = np.setdiff1d(want, have, assume_unique=False)
    if len(missing):
        sys.exit(
            "{} covers {:,} finished rows but {:,} of the {:,} requested are "
            "missing or unwritten (first: {}). A partly-tiled cache cannot be "
            "pooled: the untiled rows would be L0 while the rest are the "
            "blend, one cosine would rank both, and the untiled ones would be "
            "silently demoted. Finish the pass or narrow the selection."
            .format(stem, len(have), len(missing), len(want), missing[:5]))
    return pos[want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dual_c3.f16.npy",
                    help="dual crop cache, (n, crops*2*768)")
    ap.add_argument("--tiles", default=None,
                    help="tile cache stem, e.g. tile6; omitted = crops only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=0.5,
                    help="weight on the tile level in l2((1-w)*L0 + w*L1)")
    ap.add_argument("--block", type=int, default=20000)
    ap.add_argument("--rope-axes", default="",
                    help="comma list of depth,row,col to rotate views on; "
                         "empty (the default) pools exactly as before")
    ap.add_argument("--rope-max", type=float, default=1.0,
                    help="radians the fastest dimension pair turns per unit "
                         "coordinate; 0 reproduces the unrotated pooling")
    ap.add_argument("--frame", type=int, nargs=2, default=list(FRAME),
                    metavar=("W", "H"),
                    help="the frame the crop cache was embedded from; sets "
                         "where the crops sit relative to the tiles")
    ap.add_argument("--rope-spread", type=float, default=ROPE_SPREAD,
                    help="ratio between the fastest and slowest pair; small, "
                         "because the coordinates here span 0 to 2")
    a = ap.parse_args()

    if not 0.0 <= a.w <= 1.0:
        sys.exit("--w is a blend weight in [0, 1], got {}".format(a.w))
    axes = [s.strip() for s in a.rope_axes.split(",") if s.strip()]
    bad = [x for x in axes if x not in AXES]
    if bad:
        sys.exit("--rope-axes takes {}, got {}".format(
            "/".join(AXES), ", ".join(bad)))
    if len(set(axes)) != len(axes):
        sys.exit("--rope-axes repeats an axis: {}".format(a.rope_axes))
    if a.rope_max < 0 or a.rope_spread <= 0:
        sys.exit("--rope-max must be >= 0 and --rope-spread > 0")
    src = config.STREET_CACHE / a.src
    out = config.STREET_CACHE / a.out
    if out == src:
        sys.exit("refusing to write over the source")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(src, mmap_mode="r")
    n, d = X.shape
    if d % (2 * D_ENC):
        sys.exit("{} is {}-d, not a whole number of 768-d blocks per encoder"
                 .format(a.src, d))
    crops, half = d // (2 * D_ENC), d // 2

    T_all, tpos = None, None
    if a.tiles:
        tpos = tile_positions(a.tiles, np.arange(n))
        T_all = np.load(config.STREET_CACHE / (a.tiles + ".f16.npy"),
                        mmap_mode="r")
        if T_all.shape[2] != 2 * D_ENC:
            sys.exit("{} is {}-d per tile, expected {}".format(
                a.tiles, T_all.shape[2], 2 * D_ENC))

    grid = tile_grid(a.tiles) if a.tiles else None
    a0 = a1 = None
    cols = None
    # Every geometry question below is a question the ROTATION asks. Without
    # `--rope-axes` this script pools exactly as it always did, and gating on
    # anything else here would refuse inputs it used to accept -- a 3-crop
    # cache against a 2x3 grid pooled fine before this flag existed and has to
    # keep doing so.
    if axes:
        if a.tiles and grid is None:
            sys.exit("{} records no grid, and 3x2 and 2x3 hold the same six "
                     "tiles, so row and column cannot be recovered. Rebuild "
                     "the cache or pool without --rope-axes.".format(a.tiles))
        gc = grid[0] if grid else crops
        cols = crop_columns(crops, gc, tuple(a.frame))
        p0, p1 = view_positions(crops, grid, cols)
        flat = degenerate(p0 + (p1 if a.tiles else []), axes)
        # Views that share a coordinate get the SAME rotation, so a flat axis
        # does not confuse them -- it simply carries no information, and takes
        # its round-robin share of the dimension pairs with it. That is a
        # warning. It is only fatal when EVERY active axis is flat: then every
        # view turns through the same angle, the pooled vector is a global
        # rotation of the unrotated one, every cosine downstream is invariant
        # to it, and the arm is the baseline under another name.
        #
        # `--rope-max 0` is that on purpose -- it is the identity arm the
        # ladder is anchored on -- so it is exempt.
        if flat and a.rope_max > 0:
            if len(flat) == len(axes):
                sys.exit(
                    "every view has the same {} over these {} crops and {} "
                    "tiles, so the rotation turns them all through the same "
                    "angle: the output would be a global rotation of the "
                    "unrotated pooling, identical under every cosine "
                    "downstream, written under a name claiming to be an arm. "
                    "Add an axis the views differ on, or pass --rope-max 0 if "
                    "the identity is what you wanted."
                    .format(" and ".join(flat), crops,
                            gc * grid[1] if grid else 0))
            print("warning: every view has the same {}, so that axis carries "
                  "nothing and still takes {:.0f}% of the dimension pairs; "
                  "the remaining {} are working at a coarser spread than "
                  "--rope-spread {:g} asks for"
                  .format(" and ".join(flat), 100.0 * len(flat) / len(axes),
                          " and ".join(x for x in axes if x not in flat),
                          a.rope_spread), flush=True)
        ang = [rope_angles(p, D_ENC // 2, axes, a.rope_max, a.rope_spread)
               for p in (p0, p1)]
        a0, a1 = (torch.from_numpy(x).to(dev) for x in ang)

    print("{}  {:,} x {} = {} crops x [dino {} | siglip {}]".format(
        a.src, n, d, crops, D_ENC, D_ENC), flush=True)
    if axes:
        print("mRoPE on {}  max {:g} rad/unit  spread {:g}  grid {}".format(
            ",".join(axes), a.rope_max, a.rope_spread,
            "{}x{}".format(*grid) if grid else "none"), flush=True)
        # Printed because a wrong `--frame` is invisible everywhere else: the
        # output is the right shape, unit length, and silently misplaced.
        print("  crops at columns {} (frame {}x{}){}".format(
            ", ".join("{:.3f}".format(c) for c in cols),
            a.frame[0], a.frame[1],
            ", tiles at 0..{}".format(grid[0] - 1) if a.tiles else
            ", no tiles"), flush=True)
    print("{}  {:,} x {}   {}".format(
        a.out, n, 2 * D_ENC,
        "crops only" if not a.tiles else
        "l2({:.2f}*L0 + {:.2f}*L1) over {} tiles from {}".format(
            1 - a.w, a.w, T_all.shape[1], a.tiles)), flush=True)

    Y = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                  shape=(n, 2 * D_ENC))
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, a.block):
            e = min(s + a.block, n)
            B = torch.from_numpy(np.asarray(X[s:e], np.float32)).to(dev)
            C = torch.stack([B[:, :half].reshape(-1, crops, D_ENC),
                             B[:, half:].reshape(-1, crops, D_ENC)], dim=2)
            T = None
            if a.tiles:
                Tb = torch.from_numpy(
                    np.asarray(T_all[tpos[s:e]], np.float32)).to(dev)
                T = torch.stack([Tb[:, :, :D_ENC], Tb[:, :, D_ENC:]], dim=2)
            L0, L1 = levels(C, T, a0, a1)
            V = L0 if L1 is None else nrm((1 - a.w) * L0 + a.w * L1)
            Y[s:e] = V.cpu().numpy().astype(np.float16)
            el = time.time() - t0
            print("  {:>9,}/{:,}  {:.0f}s  eta {:.0f}s".format(
                e, n, el, el * (n - e) / max(e, 1)), flush=True)
    Y.flush()

    # Cheap recompute rather than trust, on rows the loop did not special-case.
    rng = np.random.default_rng(0)
    for i in rng.choice(n, 5, replace=False):
        v = torch.from_numpy(np.asarray(X[i:i + 1], np.float32)).to(dev)
        C = torch.stack([v[:, :half].reshape(-1, crops, D_ENC),
                         v[:, half:].reshape(-1, crops, D_ENC)], dim=2)
        T = None
        if a.tiles:
            Tb = torch.from_numpy(
                np.asarray(T_all[tpos[i:i + 1]], np.float32)).to(dev)
            T = torch.stack([Tb[:, :, :D_ENC], Tb[:, :, D_ENC:]], dim=2)
        L0, L1 = levels(C, T, a0, a1)
        want = (L0 if L1 is None else nrm((1 - a.w) * L0 + a.w * L1))[0]
        got = torch.from_numpy(np.asarray(Y[i], np.float32)).to(dev)
        assert float((want - got).abs().max()) < 2e-3, "row {} disagrees".format(i)
    # Every unit vector is a plausible one, so a row-count check is all the
    # protection there is against pooling a cache that describes other images.
    # The rotation is part of what the vectors mean, so it belongs in the
    # sidecar. Two banks pooled at different angles are as incomparable as
    # two pooled from different caches, and neither is visible in a shape.
    prov.carry(src, out, n, pooled_from=a.src, crops=crops,
               tiles=a.tiles or "none", w=a.w if a.tiles else 0.0,
               rope_axes=",".join(axes) or "none",
               rope_max=a.rope_max if axes else 0.0,
               rope_spread=a.rope_spread if axes else 0.0,
               rope_grid="{}x{}".format(*grid) if (axes and grid) else "none",
               rope_frame="{}x{}".format(*a.frame) if axes else "none")
    print("\nwrote {} in {:.0f}s; 5 rows verified against a recompute".format(
        out.name, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
