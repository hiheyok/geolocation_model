"""Rows in, tensors out.  Reads only memmaps and parquet -- never the network.

Indexed by *image*, not by row: one item is all STEPS+1 rows for one image, so
every batch contains every step type and per-step accuracy can be logged each
iteration.  The teacher-forced prefix is a pure function of (lat, lon), so the
rows are independent and no rollout is needed.
"""

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import splits as sp
import tile_math as tm



def street_table(path, dev, budget_gb=1.5):
    """The neighbour lookup table, on the card if it fits and in host RAM if not.

    retr_mode pos/dual is keyed on neighbour embeddings, so this is gathered
    from every iteration.  At 50k images it is 0.46 GB of otherwise idle VRAM
    and belongs there; at 500k it is 4.6 GB and would push the process into
    Windows' shared-memory spill, where every access is a PCIe round trip and
    nothing reports an error.  The gather is ~9 MB per batch either way.
    """
    a = np.load(path, mmap_mode="r")
    gb = a.nbytes / 1e9
    t = torch.from_numpy(np.asarray(a))          # fp16; cast at the gather
    on_gpu = dev == "cuda" and gb <= budget_gb
    print("nbr table  {:.2f} GB on {}{}".format(
        gb, "cuda" if on_gpu else "cpu",
        "" if on_gpu else "  (too large for the card; gathered host-side)"),
        flush=True)
    return t.to(dev) if on_gpu else t


def gather_nbr(table, rows, dev):
    """rows may live on either device; the table decides where the gather runs."""
    idx = rows if rows.device == table.device else rows.to(table.device)
    return table[idx].to(dev, non_blocking=True).float()


class GeoStepDataset(Dataset):
    def __init__(self, split="train", g=tm.G, steps=tm.STEPS, cache=None,
                 street_file="embeddings.f16.npy", n_neg=0, neg_seed=0,
                 neg_random=True, split_mode=sp.PRIMARY, knn_file=None,
                 knn_k=0):
        self.g, self.steps = g, steps
        self.n_neg, self.neg_seed, self.neg_random = n_neg, neg_seed, neg_random
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

        # street embeddings: row order matches dataset.parquet
        self._street_path = config.STREET_CACHE / street_file
        self._tokens_path = cache / "tokens.f16.npy"
        self.street = np.load(self._street_path, mmap_mode="r")
        self.dim_street = self.street.shape[1]

        # map token cache + (z,x,y) -> row
        self.tokens = np.load(self._tokens_path, mmap_mode="r")
        idx = pq.read_table(cache / "index.parquet")
        key = (np.asarray(idx["z"]).astype(np.int64) << 58
               | np.asarray(idx["x"]).astype(np.int64) << 29
               | np.asarray(idx["y"]).astype(np.int64))
        self.lut = dict(zip(key.tolist(),
                            np.asarray(idx["row"]).astype(np.int64).tolist()))
        lut = self.lut

        # targets, reshaped to [n_images_total, steps+1]
        tg = pq.read_table(config.TARGETS_PARQUET)
        per = steps + 1
        tz = np.asarray(tg["tile_z"]).astype(np.int64).reshape(-1, per)
        tx = np.asarray(tg["tile_x"]).astype(np.int64).reshape(-1, per)
        ty = np.asarray(tg["tile_y"]).astype(np.int64).reshape(-1, per)
        act = np.asarray(tg["target_action"]).astype(np.int64).reshape(-1, per)
        u = np.asarray(tg["u"], dtype=np.float32).reshape(-1, per)
        v = np.asarray(tg["v"], dtype=np.float32).reshape(-1, per)
        x0 = np.asarray(tg["x0"], dtype=np.float32).reshape(-1, per)
        y0 = np.asarray(tg["y0"], dtype=np.float32).reshape(-1, per)

        tk = (tz << 58) | (tx << 29) | ty
        self.tok_row = np.array(
            [[lut[int(k)] for k in row] for row in tk[keep]], dtype=np.int64)
        self.action = act[keep][:, :steps]
        self.tile = np.stack([tz[keep], tx[keep], ty[keep]], axis=-1)

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
            if str(z["split_mode"]) != split_mode:
                raise SystemExit(
                    "knn cache was built on split {!r} but the dataset is {!r}; "
                    "the bank must be that split's train side".format(
                        str(z["split_mode"]), split_mode))
            self.knn_idx = z["idx"][:, :knn_k]
            self.knn_sim = z["sim"][:, :knn_k].astype(np.float32)

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
            rows.append(self.lut[(z << 58) | (x << 29) | y])
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
            rng = np.random.default_rng(None if self.neg_random
                                        else (self.neg_seed * 1000003 + i))
            r, nx, ny, st = self.sample_negatives(i, self.n_neg, rng)
            out["neg_tokens"] = torch.from_numpy(
                np.asarray(self.tokens[r], dtype=np.float32))
            out["neg_x0"] = torch.from_numpy(nx)
            out["neg_y0"] = torch.from_numpy(ny)
            out["neg_step"] = torch.from_numpy(st)
        return out
