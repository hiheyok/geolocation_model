"""Rows in, tensors out.  Reads only memmaps and parquet -- never the network.

Indexed by *image*, not by row: one item is all STEPS+1 rows for one image, so
every batch contains every step type and per-step accuracy can be logged each
iteration.  The teacher-forced prefix is a pure function of (lat, lon), so the
rows are independent and no rollout is needed.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import knnmeta
import provenance as prov
import safeio
import splits as sp
import tile_math as tm



def street_table(path, dev, budget_gb=1.5, ram_gb=8.0):
    """The neighbour lookup table, on the card if it fits and in host RAM if not.

    retr_mode pos/dual is keyed on neighbour embeddings, so this is gathered
    from every iteration.  At 50k images it is 0.46 GB of otherwise idle VRAM
    and belongs there; at 500k it is 4.6 GB and would push the process into
    Windows' shared-memory spill, where every access is a PCIe round trip and
    nothing reports an error.  The gather is ~9 MB per batch either way.
    """
    a = np.load(path, mmap_mode="r")
    gb = a.nbytes / 1e9
    if dev == "cuda" and gb <= budget_gb:
        print("nbr table  {:.2f} GB on cuda".format(gb), flush=True)
        return torch.from_numpy(np.asarray(a)).to(dev)
    # Second tier: 2 MB pages. Strictly better than the private copy below --
    # locked rather than pageable, so it cannot be evicted, and 5,127 page-table
    # entries for a 10.75 GB table instead of 2.6 million. Opportunistic: large
    # pages need physically contiguous memory and Windows never compacts, so
    # measured on this machine 10.75 GB succeeds right after a reboot and 0.5 GB
    # is the ceiling after days of uptime. Returns None when it cannot, and the
    # tiers below are unchanged.
    if os.environ.get("NO_LARGE_PAGES") != "1":
        try:
            import largepages
            buf = largepages.empty(a.shape, a.dtype)
        except Exception:
            buf = None
        if buf is not None:
            for lo in range(0, a.shape[0], 200000):     # chunked, no big temp
                hi = min(lo + 200000, a.shape[0])
                buf[lo:hi] = a[lo:hi]
            print("nbr table  {:.2f} GB on large pages, locked in RAM"
                  .format(gb), flush=True)
            return torch.from_numpy(buf)

    # Third tier, for a bank extension. Whether it fits is measured, not
    # assumed: a fixed threshold is either too timid on an idle machine or an
    # OOM on a busy one, and this decision is made while dataloader workers are
    # about to be spawned.
    #
    # The margin is deliberately wide because the tier below is cheap. Measured
    # on the 11.52 GB table against the same run held in RAM: 805.1s vs 737.8s
    # on the first epoch and 747.9 vs 720.3 warm, about 4%. A gather touches
    # 64x16 rows, so the OS pages in exactly those and the file stays shared
    # between the workers that also read it. Trading a 4% saving for a chance of
    # losing a half-hour arm is not a trade worth making.
    try:
        import psutil
        free = psutil.virtual_memory().available / 1e9
    except Exception:
        free = 0.0
    budget = max(ram_gb, 0.4 * free)
    if gb <= budget:
        print("nbr table  {:.2f} GB in host RAM  ({:.1f} GB free, too large "
              "for the card)".format(gb, free), flush=True)
        return torch.from_numpy(np.asarray(a))
    print("nbr table  {:.2f} GB as a memmap  ({:.1f} GB free; holding it would "
          "cost more than the ~4% paging does)".format(gb, free), flush=True)
    return torch.from_numpy(a)


def gather_nbr(table, rows, dev):
    """rows may live on either device; the table decides where the gather runs."""
    idx = rows if rows.device == table.device else rows.to(table.device)
    return table[idx].to(dev, non_blocking=True).float()


def _check_retrieval_prefix(path, street, retr_file, d_retr, n_probe=64,
                            want_digest=None):
    """The first `d_retr` columns must BE the retrieval cache, byte for byte.

    A name comparison cannot see the failure that matters here: a retrieval
    cache rebuilt under the same filename leaves the k-NN addressing one
    embedding space while the model's retrieval block is in another. Every
    shape still agrees and every neighbour index is in range.

    So the bytes are checked on a sample of rows rather than the string. It
    costs 64 reads and it is the only thing that actually establishes that this
    file's retrieval block and the bank's are the same vectors.
    """
    import config

    if not retr_file:
        print("warning: {} carries a conditioning block but does not record "
              "which retrieval cache it was joined to, so the retrieval half "
              "cannot be verified against the bank".format(Path(path).name),
              flush=True)
        return
    rp = config.STREET_CACHE / retr_file
    if not rp.exists():
        raise SystemExit(
            "{} declares its retrieval block came from {}, which is not on "
            "disk. Without it the block cannot be checked against the bank "
            "the k-NN was built over.".format(Path(path).name, retr_file))
    R = np.load(rp, mmap_mode="r")
    if R.shape != (street.shape[0], d_retr):
        raise SystemExit(
            "{} is {} but {} declares a {}-d retrieval block over {:,} rows"
            .format(retr_file, R.shape, Path(path).name, d_retr,
                    street.shape[0]))
    # The whole file, by content. The sampled comparison below still runs --
    # it is the only thing that checks the *join* itself -- but it cannot be
    # the authority: its seed is fixed, so the rows it skips are skipped
    # forever and a localized rebuild passes every run rather than eventually
    # being caught. REVIEW6 #2.
    if want_digest:
        got = safeio.content_digest(rp)
        if got != want_digest:
            raise SystemExit(
                "{} was joined over {} whose contents digest {}; that file now "
                "digests {}. It was rebuilt under the same name, so this "
                "file's retrieval block and the bank the k-NN addresses are "
                "different embedding spaces. Rebuild the joined cache."
                .format(Path(path).name, retr_file, want_digest, got))
    else:
        print("warning: {} records no content digest for its retrieval block, "
              "so only a {}-row sample can be checked".format(
                  Path(path).name, n_probe), flush=True)
    rng = np.random.default_rng(0)
    rows = np.unique(rng.integers(0, street.shape[0], min(n_probe,
                                                          street.shape[0])))
    a = np.asarray(street[rows, :d_retr], np.float32)
    b = np.asarray(R[rows], np.float32)
    if not np.array_equal(a, b):
        bad = int((a != b).any(1).sum())
        raise SystemExit(
            "{}'s retrieval block does not match {} on {} of {} probed rows. "
            "The k-NN addresses that file's embedding space; this one is a "
            "different space under the same name, so every neighbour would be "
            "the right row of the wrong geometry. Rebuild the joined cache."
            .format(Path(path).name, retr_file, bad, len(rows)))


def _check_street_rows(path, name, all_ids, n_street, digest):
    """Classify what row space a street cache claims, then check it.

    The rule used to be `if len(street) == len(dataset.parquet)`, and every
    other length was accepted unchecked. That is not a narrow gap. This
    repository holds standalone 750,000-row `bank_ext*_*.f16.npy` files which
    contain **no release rows at all** and are *longer* than the 500,000-row
    release, so a length heuristic cannot even tell them apart: passing one as
    `--street-file` made the first 500,000 extension rows serve as the release
    images. Shapes are valid, indices are in range, and every photograph is
    paired with another photograph's embedding.

    So classify from the recorded row space, which is the only thing that
    knows, and check all three cases:

    * exact release length -- must be this release's ids, in order;
    * a stack -- must be the release ids followed by each recorded
      extension's ids, in the recorded order;
    * anything else -- refused, because a dataset indexed by release row
      cannot address it at all.

    A missing sidecar stays a warning for the release-length case (every
    artifact predates the convention) but is a refusal for a longer file:
    there, the sidecar is not corroborating the layout, it is the only thing
    that *determines* it.
    """
    import config

    if n_street == len(all_ids):
        prov.check(path, all_ids, name, digest=digest)
        return
    if not prov.stacked_on_release(path):
        rec = prov.read(path)
        raise SystemExit(
            "{} holds {:,} rows but the release has {:,}, and its provenance "
            "does not say it is the release followed by bank extensions ({}). "
            "A cache that is not release-addressed cannot be a --street-file: "
            "row i here is not image i, so every image would be scored "
            "against another image's embedding with nothing out of range to "
            "notice. Stack it with scripts/stack_bank.py, or run "
            "scripts/backfill_prov.py if the file is right and only the "
            "sidecar is missing."
            .format(name, n_street, len(all_ids),
                    "no sidecar" if rec is None
                    else "row space: " + str(rec.get("row_space"))))
    parts, stems = [all_ids], prov.exts_of(path)
    for stem in stems:
        parts.append(np.asarray(prov.bank_ext(stem, config.RELEASE)["image_id"]))
    want = sum(len(p) for p in parts)
    if want != n_street:
        raise SystemExit(
            "{} holds {:,} rows but the release ({:,}) plus its recorded "
            "extensions {} come to {:,}. The stack and its metadata are from "
            "different builds; rebuild it with scripts/stack_bank.py."
            .format(name, n_street, len(all_ids), stems, want))
    prov.check_stack(path, parts, name)


def _check_fetched(done_p, tok_row, zs, n_neg, split):
    """An unfetched tile is a zero row, and a zero row is a legal histogram.

    fetch_tiles marks each row it completes in done.u8.npy and leaves the rest
    of the memmap at its zero fill.  Nothing here read that mask, so a cache
    interrupted part way through trained happily against all-zero map tokens
    for every tile it never got -- which no loss can show you, because a
    uniform-zero histogram is a perfectly well-formed input.  It just makes a
    quietly worse model, which is this project's most expensive failure shape.

    Only the rows this split can actually reach are checked.  With negatives
    enabled the reachable set widens to every z4 and z8 tile, because a sibling
    negative may descend into any of them.
    """
    if not done_p.exists():
        return                      # caches written before the mask existed
    done = np.load(done_p)
    if done.ndim != 1 or len(done) != len(zs):
        raise SystemExit(
            "map cache is inconsistent: {} index rows but a {}-row done mask "
            "({}). Rebuild the cache; a stale mask cannot be interpreted."
            .format(len(zs), getattr(done, "shape", "?"), done_p))
    # The producer writes 0 or 1 and validates it; this consumer defined
    # "fetched" as `!= 0`, so any other value -- a 2 from a half-migrated
    # writer, a 255 from a torn or misinterpreted file -- certified an
    # all-zero token row as complete. Beam inference already required == 1,
    # so the same cache could train on a blank tile and fetch it live at
    # evaluation, with the two paths disagreeing and neither complaining.
    bad = np.unique(done[(done != 0) & (done != 1)])
    if len(bad):
        raise SystemExit(
            "map cache completion mask holds {} that is neither 0 nor 1 ({}). "
            "A non-binary mask cannot be read as 'fetched': rebuild it with "
            "fetch_tiles, which writes only 0 or 1."
            .format("values " + ", ".join(str(int(v)) for v in bad[:5]),
                    done_p))

    used = np.unique(tok_row)
    if n_neg > 0:
        # sibling negatives descend into arbitrary z4/z8 tiles
        used = np.union1d(used, np.flatnonzero(zs <= 8))
    miss = used[done[used] != 1]
    if len(miss):
        raise SystemExit(
            "map cache is incomplete: {:,} of the {:,} tiles the {!r} split "
            "reads were never fetched ({}). Those rows are all-zero token "
            "histograms, which look valid and are not. Re-run fetch_tiles; it "
            "is resumable and will fetch exactly these."
            .format(len(miss), len(used), split, done_p))


def _check_neighbours(idx, z, what, n_addr, rel_seq, ext_seq):
    """In range is not in the bank, and not the query is not a different drive.

    The range check that was here established only that every neighbour id
    resolves to *some* row of the address tables. Three things it cannot see,
    all of which leave every index non-negative, in range, and ordinary
    (REVIEW6 #4):

    **Not in the bank.** `bank_rows` records which rows were searched. A cache
    can name a clean train bank there and still carry val or test rows in
    `idx`; the ids are perfectly valid rows of the release. Held-out images
    would then be retrieved into training and evaluation as neighbours.

    **The query itself.** A query that retrieves its own row scores a cosine of
    1.0 against a photograph whose z16 address is the answer. That is not a
    subtle leak; it is the label.

    **Its own sequence.** OSV-5M captures run consecutively along a road, so a
    same-sequence neighbour is a near-duplicate frame metres away. `build_knn`
    masks those to -2.0 before the top-k, over release and extension together,
    for every split mode. Nothing downstream ever confirmed it happened -- and
    when this exclusion was found to be missing from the *bank*, it was worth
    18 points of the headline. A silent regression in the builder would look
    exactly like a good result.

    Checked in full rather than sampled: the whole comparison is one gather and
    one equality over the 16M ids a 500k x 32 cache holds, which is under two
    seconds including the sequence encode.
    """
    rows = knnmeta.bank_rows(z, n_addr, what)
    if rows is None:
        print("warning: {} records no bank_rows, so its neighbours cannot be "
              "checked for membership in the bank it searched".format(what),
              flush=True)
    else:
        member = np.zeros(n_addr, bool)
        member[rows] = True
        off = ~member[idx]
        if off.any():
            bad = np.unique(idx[off])
            raise SystemExit(
                "{}: {:,} of {:,} cached neighbours are not in the {:,} rows "
                "the cache says it searched (e.g. {}). Every one is a valid, "
                "in-range row, so nothing else would have noticed -- but rows "
                "outside the bank are the split's held-out side."
                .format(what, int(off.sum()), idx.size, len(rows),
                        ", ".join(str(int(b)) for b in bad[:5])))

    q = np.arange(idx.shape[0], dtype=np.int64)
    if (idx == q[:, None]).any():
        n = int((idx == q[:, None]).any(1).sum())
        raise SystemExit(
            "{}: {:,} queries retrieve their own row. A self match is a cosine "
            "of 1.0 against the answer's own z16 address.".format(what, n))

    seq = rel_seq if ext_seq is None else np.concatenate([rel_seq, ext_seq])
    if len(seq) != n_addr:
        raise SystemExit(
            "{}: {:,} sequence labels for {:,} addressable rows; the release "
            "and the bank extension disagree about how many images there are."
            .format(what, len(seq), n_addr))
    _, sid = np.unique(seq, return_inverse=True)
    same = sid[idx] == sid[q][:, None]
    if same.any():
        raise SystemExit(
            "{}: {:,} of {:,} cached neighbours share their query's sequence. "
            "OSV-5M captures run consecutively along a road, so those are "
            "near-duplicate frames metres from the query -- build_knn masks "
            "them to -2.0 before the top-k, and this cache has them anyway."
            .format(what, int(same.sum()), idx.size))


class GeoStepDataset(Dataset):
    def __init__(self, split="train", g=tm.G, steps=tm.STEPS, cache=None,
                 street_file=None, n_neg=0, neg_seed=0,
                 neg_random=True, split_mode=sp.PRIMARY, knn_file=None,
                 knn_k=0):
        street_file = street_file or config.STREET_DEFAULT
        self.g, self.steps = g, steps
        self.n_neg, self.neg_seed, self.neg_random = n_neg, neg_seed, neg_random
        # keeps train and val negative streams apart even at equal seeds
        self._split_entropy = int.from_bytes(
            hashlib.sha256(str(split).encode()).digest()[:4], "big")
        self.n_actions = g * g
        cache = Path(cache) if cache else config.MAP_CACHE

        ds = pq.read_table(config.DATASET_PARQUET)
        all_ids = np.asarray(ds["image_id"])
        splits, self.split_hash = sp.read(ds, split_mode)
        self.split_mode = split_mode
        keep = np.flatnonzero(splits == split) if split != "all" else np.arange(len(all_ids))

        self.rows = keep.astype(np.int64)
        self.image_id = all_ids[keep]
        self.lat = np.asarray(ds["lat"], dtype=np.float64)[keep]
        self.lon = np.asarray(ds["lon"], dtype=np.float64)[keep]
        self.country = np.asarray(ds["country"].to_pylist(), dtype=object)[keep]

        # street embeddings: row order matches dataset.parquet -- asserted,
        # until now, by nothing at all. The digest is computed once here and
        # reused for every artifact checked below; it costs 2 ms at 500k rows.
        self.rows_digest = prov.rows_digest(all_ids)
        self._street_path = config.STREET_CACHE / street_file
        self._tokens_path, index_p, done_p = config.map_files(cache)
        self.street = np.load(self._street_path, mmap_mode="r")
        # A conditioned cache is [retrieval | conditioning] in one tensor, and
        # its sidecar says where the split is. Retrieval width is what the
        # k-NN, the retrieval keys and the bank all live in; the conditioning
        # block is seen only by the model. Absent the field the whole file is
        # retrieval, which is every cache written before this existed.
        _rec = prov.read(self._street_path) or {}
        self.retr_file = _rec.get("retrieval_file")
        self.dim_street = int(_rec.get("retrieval_dim") or self.street.shape[1])
        self.dim_cond = self.street.shape[1] - self.dim_street
        if self.dim_cond < 0:
            raise SystemExit(
                "{} records a {}-d retrieval block but is only {}-d"
                .format(street_file, self.dim_street, self.street.shape[1]))
        if self.dim_cond:
            _check_retrieval_prefix(self._street_path, self.street,
                                    self.retr_file, self.dim_street,
                                    want_digest=_rec.get("retrieval_digest"))
        _check_street_rows(self._street_path, street_file, all_ids,
                           len(self.street), self.rows_digest)

        # map token cache + (z,x,y) -> row
        self.tokens = np.load(self._tokens_path, mmap_mode="r")
        idx = pq.read_table(index_p)
        # one key per cached tile; tile_math.tile_key owns the bit layout
        key = tm.tile_key(np.asarray(idx["z"]).astype(np.int64),
                          np.asarray(idx["x"]).astype(np.int64),
                          np.asarray(idx["y"]).astype(np.int64))
        self.lut = dict(zip(key.tolist(),
                            np.asarray(idx["row"]).astype(np.int64).tolist()))
        lut = self.lut

        # targets, reshaped to [n_images_total, steps+1]
        tg = pq.read_table(config.TARGETS_PARQUET)
        per = steps + 1
        # targets.parquet carries image_id and nothing ever read it: the file
        # was reshaped to (n, steps+1) and paired with dataset.parquet purely
        # by position. Rebuilding one without the other -- which is two
        # commands, not one -- silently gives every image another image's
        # zoom path, and every metric stays in range. No sidecar needed here;
        # the evidence was already in the file.
        tg_ids = np.asarray(tg["image_id"]).reshape(-1, per)
        if len(tg_ids) != len(all_ids):
            raise SystemExit(
                "targets.parquet holds {:,} images but dataset.parquet holds "
                "{:,}; they are from different builds. Re-run "
                "scripts/build_dataset.py, which writes both."
                .format(len(tg_ids), len(all_ids)))
        if not (tg_ids == all_ids[:, None]).all():
            bad = int((tg_ids != all_ids[:, None]).any(1).sum())
            raise SystemExit(
                "targets.parquet and dataset.parquet disagree on which image "
                "is in {:,} of {:,} rows. Every step target would belong to a "
                "different image than the embedding it is paired with. Re-run "
                "scripts/build_dataset.py.".format(bad, len(all_ids)))
        tz = np.asarray(tg["tile_z"]).astype(np.int64).reshape(-1, per)
        tx = np.asarray(tg["tile_x"]).astype(np.int64).reshape(-1, per)
        ty = np.asarray(tg["tile_y"]).astype(np.int64).reshape(-1, per)
        act = np.asarray(tg["target_action"]).astype(np.int64).reshape(-1, per)
        u = np.asarray(tg["u"], dtype=np.float32).reshape(-1, per)
        v = np.asarray(tg["v"], dtype=np.float32).reshape(-1, per)
        x0 = np.asarray(tg["x0"], dtype=np.float32).reshape(-1, per)
        y0 = np.asarray(tg["y0"], dtype=np.float32).reshape(-1, per)

        tk = tm.tile_key(tz, tx, ty)
        self.tok_row = np.array(
            [[lut[int(k)] for k in row] for row in tk[keep]], dtype=np.int64)
        self.action = act[keep][:, :steps]
        self.tile = np.stack([tz[keep], tx[keep], ty[keep]], axis=-1)

        _check_fetched(done_p, self.tok_row,
                       np.asarray(idx["z"]).astype(np.int64),
                       n_neg, split)

        # Where inside each step's tile the true point falls, in units of grid
        # cells.  Hard CE cannot tell a neighbouring cell from the wrong
        # continent; soft labels need the sub-cell offset to weight by distance,
        # and targets.parquet only stores u,v for the final click.
        lat = np.clip(self.lat, -tm.MAX_LAT, tm.MAX_LAT)
        sin = np.sin(np.radians(lat))
        px = (self.lon + 180.0) / 360.0
        py = 0.5 - np.log((1.0 + sin) / (1.0 - sin)) / (4.0 * np.pi)
        zs = tz[keep][:, :steps]
        n = (1 << zs).astype(np.float64)
        self.cell_xy = np.stack([
            (px[:, None] * n - tx[keep][:, :steps]) * g,
            (py[:, None] * n - ty[keep][:, :steps]) * g], axis=-1).astype(np.float32)
        # z16 address of every image in the parquet, not just this split's --
        # a neighbour is a train image and the query may be val
        self.all_x16 = tx[:, steps].copy()
        self.all_y16 = ty[:, steps].copy()

        self.knn_k = knn_k
        if knn_k:
            z = np.load(config.STREET_CACHE / knn_file)
            ext = str(z["bank_ext"]) if "bank_ext" in z else ""
            ext_seq = None
            if ext:
                # neighbours may live past the release: extend the address
                # tables so nbr index n+i resolves to the extension row i
                m = prov.bank_ext(ext, config.RELEASE)
                self.all_x16 = np.concatenate(
                    [self.all_x16, m["x16"].astype(self.all_x16.dtype)])
                self.all_y16 = np.concatenate(
                    [self.all_y16, m["y16"].astype(self.all_y16.dtype)])
                ext_seq = np.asarray(m["sequence"]).astype("U40")
            # build_knn records five things about how the cache was made and
            # only one of them was ever checked.  Each of the others is a way
            # for the wrong neighbours to arrive silently, and none of them
            # would raise on its own: the arrays are the right dtype and a
            # plausible shape whatever they were built from.
            if str(z["split_mode"]) != split_mode:
                raise SystemExit(
                    "knn cache was built on split {!r} but the dataset is {!r}; "
                    "the bank must be that split's train side".format(
                        str(z["split_mode"]), split_mode))
            if "split_hash" in z and str(z["split_hash"]) != self.split_hash:
                # Every kNN cache on disk was built before 2026-09-03, when
                # split_hash still hashed only each label's first character.
                # check_split learned to recognise that digest so the 122
                # checkpoints keep loading; this sibling check did not, and it
                # refused the entire shipping retrieval path. Recognise it the
                # same way, and say exactly what the weak digest does not
                # prove -- it cannot tell a train/test swap from the real
                # assignment, so it pins the mode and the val positions only.
                if sp.hash_matches(split_mode, splits,
                                   z["split_hash"]) == "legacy":
                    print("note: {} carries the pre-2026-09-03 split digest, "
                          "which hashed only each label's first character and "
                          "so cannot distinguish train from test. The bank's "
                          "train side matches as far as that digest can tell."
                          .format(knn_file), flush=True)
                else:
                    raise SystemExit(
                        "knn cache was built against split hash {} but the "
                        "data on disk hashes to {}; dataset.parquet changed "
                        "since the bank was built, so its train side is no "
                        "longer that train side. Rebuild it."
                        .format(str(z["split_hash"]), self.split_hash))
            # For a conditioned cache the k-NN was built on the retrieval
            # *prefix*, not on this file -- that is the whole point of the
            # decoupling -- so the comparison targets what the sidecar declares.
            want_sf = self.retr_file or street_file
            if "street_file" in z and str(z["street_file"]) != want_sf:
                raise SystemExit(
                    "knn cache was built over {!r} but this dataset retrieves "
                    "in {!r}. Neighbours found in one embedding space do not "
                    "transfer to another.".format(str(z["street_file"]),
                                                  want_sf))
            # The name is not the space. That file can be rebuilt under its own
            # name -- a repooled bank, a retuned blend weight, a re-run of the
            # script -- and the cache's idx and sim would still describe the
            # old vectors while every check above passes. For a conditioned
            # run the joined-prefix check catches it on the retrieval block;
            # an unconditioned one has no prefix to check, so compare against
            # the bytes build_knn recorded (REVIEW6 #3).
            knnmeta.check_bytes(z, config.STREET_CACHE / want_sf, knn_file)
            # One query row per *release* image.  Not per row of the street
            # file: a bank file has its extension rows appended after the
            # release's, so that length is larger and comparing against it
            # rejects every legitimate cache.
            n_q, n_rel = z["idx"].shape[0], len(splits)
            if n_q != n_rel:
                raise SystemExit(
                    "knn cache holds {:,} query rows but the release has {:,} "
                    "images. Both are indexed by row order in dataset.parquet, "
                    "so every image would be given another image's neighbours."
                    .format(n_q, n_rel))
            have = z["idx"].shape[1]
            if have < knn_k:
                raise SystemExit(
                    "knn cache has {} neighbours per query, {} were asked for. "
                    "Slicing would silently train on fewer neighbours than the "
                    "run records.".format(have, knn_k))
            idx_all = z["idx"][:, :knn_k]
            # A negative index reads from the end of the address tables and
            # returns a neighbour that is not the one recorded -- silently,
            # because the result is a perfectly ordinary row.
            lo, hi = int(idx_all.min()), int(idx_all.max())
            if lo < 0 or hi >= len(self.all_x16):
                raise SystemExit(
                    "knn cache holds neighbour indices in [{}, {}] but the "
                    "address tables have {:,} rows; a negative index would "
                    "read from the end and a large one is out of range."
                    .format(lo, hi, len(self.all_x16)))
            _check_neighbours(idx_all, z, knn_file, len(self.all_x16),
                              np.asarray(ds["sequence"].to_pylist(),
                                         dtype=object).astype("U40"),
                              ext_seq)
            sim_all = z["sim"][:, :knn_k].astype(np.float32)
            if sim_all.shape != idx_all.shape:
                raise SystemExit(
                    "{}: idx is {} but sim is {}. They are read with the same "
                    "row index at item time, so the shorter one decides which "
                    "rows raise and which silently pair a neighbour with "
                    "another query's similarity."
                    .format(knn_file, idx_all.shape, sim_all.shape))
            if not np.isfinite(sim_all).all():
                raise SystemExit(
                    "{}: {:,} of the cached similarities are not finite. They "
                    "become retrieval weights through a softmax, where a NaN "
                    "silently takes the whole distribution with it."
                    .format(knn_file, int((~np.isfinite(sim_all)).sum())))
            self.knn_idx = idx_all
            self.knn_sim = sim_all

        self.uv = np.stack([u[keep][:, steps], v[keep][:, steps]], 1)
        self.x0 = x0[keep]
        self.y0 = y0[keep]
        self.step_ids = np.arange(per, dtype=np.int64)

    def __getstate__(self):
        """np.memmap pickles by value, so the default would send the whole
        street and token caches to every worker -- 8.5 GB at 500k images, which
        a Windows pipe refuses.  Send the paths; the worker reopens them and the
        OS shares the pages."""
        st = self.__dict__.copy()
        st["street"] = None
        st["tokens"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.street = np.load(self._street_path, mmap_mode="r")
        self.tokens = np.load(self._tokens_path, mmap_mode="r")

    def neg_rng(self, i):
        """The stream image `i`'s off-path tiles are drawn from.

        A pure function of (split, seed, index) -- deliberately not of the
        epoch or of the worker. This runs inside spawned, persistent loader
        workers, so anything epoch-varying would have to cross a process
        boundary that is set up once per run, and anything worker-local makes
        the result depend on `--workers`. The cost is that an image sees the
        same off-path tiles every epoch; the benefit is that a recorded seed
        reproduces the run, which it did not before: the train set took
        `neg_random=True`, so `default_rng(None)` seeded from OS entropy for
        every item, independent of both `np.random.seed` and
        `torch.manual_seed`.

        The split is mixed in so train and val do not draw the same stream at
        equal seeds.
        """
        if self.neg_random:
            return np.random.default_rng(None)
        return np.random.default_rng(
            np.random.SeedSequence([self.neg_seed, i, self._split_entropy]))

    def sample_negatives(self, i, n_neg, rng):
        """Off-path views: sibling tiles that do NOT contain the true point.

        Every teacher-forced row is on-path by construction, so the sink class
        has no positives to learn from without these.  Siblings are the right
        negative because they are exactly what a beam expands into: the other
        children of the previous step's tile.

        Restricted to steps 1 and 2, whose inputs are z4 and z8 tiles -- both
        cached exhaustively, so a negative costs no fetch.  Step 3 reads z12,
        where only true-path tiles exist on disk.
        """
        rows, x0s, y0s, sts = [], [], [], []
        for _ in range(n_neg):
            t = int(rng.integers(1, 3))
            pz, px, py = self.tile[i, t - 1]
            true_a = int(self.action[i, t - 1])
            a = int(rng.integers(0, self.n_actions - 1))
            a = a + 1 if a >= true_a else a          # any sibling but the right one
            z, x, y = tm.descend(int(pz), int(px), int(py), a, self.g)
            rows.append(self.lut[tm.tile_key(z, x, y)])
            nx, ny = tm.norm_corner(z, x, y)
            x0s.append(nx); y0s.append(ny); sts.append(t)
        return (np.array(rows, dtype=np.int64), np.array(x0s, dtype=np.float32),
                np.array(y0s, dtype=np.float32), np.array(sts, dtype=np.int64))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        out = {
            "street": torch.from_numpy(
                np.asarray(self.street[self.rows[i]], dtype=np.float32)),
            "tokens": torch.from_numpy(
                np.asarray(self.tokens[self.tok_row[i]], dtype=np.float32)),
            "x0": torch.from_numpy(self.x0[i]),
            "y0": torch.from_numpy(self.y0[i]),
            "step": torch.from_numpy(self.step_ids),
            "action": torch.from_numpy(self.action[i]),
            "cell_xy": torch.from_numpy(self.cell_xy[i]),
            "uv": torch.from_numpy(self.uv[i].astype(np.float32)),
            "index": i,
        }
        if self.knn_k:
            nb = self.knn_idx[self.rows[i]]
            out["nbr_x"] = torch.from_numpy(self.all_x16[nb].astype(np.int64))
            out["nbr_y"] = torch.from_numpy(self.all_y16[nb].astype(np.int64))
            out["nbr_sim"] = torch.from_numpy(self.knn_sim[self.rows[i]])
            # row ids rather than embeddings: the caller gathers from a
            # GPU-resident copy of the street cache, which costs 0.46 GB of
            # otherwise idle VRAM and keeps the batch small
            out["nbr_row"] = torch.from_numpy(nb.astype(np.int64))
            out["self_row"] = int(self.rows[i])
        if self.n_neg:
            rng = self.neg_rng(i)
            r, nx, ny, st = self.sample_negatives(i, self.n_neg, rng)
            out["neg_tokens"] = torch.from_numpy(
                np.asarray(self.tokens[r], dtype=np.float32))
            out["neg_x0"] = torch.from_numpy(nx)
            out["neg_y0"] = torch.from_numpy(ny)
            out["neg_step"] = torch.from_numpy(st)
        return out
