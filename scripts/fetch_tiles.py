"""Fetch every mask the training rows need and cache it as a token grid.

Writes a single [N, grid*grid, 12] float16 memmap plus an index mapping
(z, x, y) -> row.  Resumable: rerunning only fetches rows not yet marked done,
so topping up later (exhaustive z8 for beam search, or another shard) is
incremental rather than a restart.
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import tile_math as tm
import tiles as T

TOKENS, INDEX, DONE = config.map_files()


def needed_keys(grid, exhaustive_z8):
    tg = pq.read_table(config.TARGETS_PARQUET)
    z = np.asarray(tg["tile_z"]).astype(np.int64)
    x = np.asarray(tg["tile_x"]).astype(np.int64)
    y = np.asarray(tg["tile_y"]).astype(np.int64)
    keys = set(zip(z.tolist(), x.tolist(), y.tolist()))

    b = tm.bits(grid)
    keys.add((0, 0, 0))
    for cx in range(1 << b):                      # all of z4: only 256 tiles
        for cy in range(1 << b):
            keys.add((b, cx, cy))
    if exhaustive_z8:
        n = 1 << (2 * b)
        for cx in range(n):
            for cy in range(n):
                keys.add((2 * b, cx, cy))
    return sorted(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=tm.G)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--exhaustive-z8", action="store_true",
                    help="also fetch all 65,536 z8 tiles (only helps beam latency)")
    ap.add_argument("--seed-from", default=None,
                    help="another release's map cache to copy already-fetched "
                         "tiles from, matched by (z,x,y). A token grid is a "
                         "pure function of its address and so is valid across "
                         "releases; only the row numbering differs.")
    a = ap.parse_args()

    keys = needed_keys(a.grid, a.exhaustive_z8)
    n, d = len(keys), a.grid * a.grid
    print("tiles needed   {:,}".format(n))
    print("cache size     {:.0f} MB fp16".format(n * d * T.N_CLASSES * 2 / 1e6))

    idx = pa.table({
        "z": pa.array([k[0] for k in keys], pa.int8()),
        "x": pa.array([k[1] for k in keys], pa.int32()),
        "y": pa.array([k[2] for k in keys], pa.int32()),
        "row": pa.array(list(range(n)), pa.int32()),
    })

    reuse = TOKENS.exists() and DONE.exists() and INDEX.exists()
    if reuse:
        old = pq.read_table(INDEX)
        reuse = (old.num_rows == n
                 and np.array_equal(np.asarray(old["x"]), np.asarray(idx["x"]))
                 and np.array_equal(np.asarray(old["y"]), np.asarray(idx["y"]))
                 and np.array_equal(np.asarray(old["z"]), np.asarray(idx["z"])))
        if not reuse:
            print("index changed -- starting a fresh cache")

    tok = np.lib.format.open_memmap(
        TOKENS, mode="r+" if reuse else "w+",
        dtype=np.float16, shape=(n, d, T.N_CLASSES))
    done = (np.load(DONE) if reuse
            else np.zeros(n, dtype=np.uint8))
    pq.write_table(idx, INDEX)

    if a.seed_from:
        # Growing a release adds sparse z12/z16 rows and renumbers everything,
        # which would otherwise refetch the 65,792 exhaustive tiles the previous
        # release already holds.  Addresses are stable, rows are not.
        src = Path(a.seed_from)
        stok_p, sidx_p, sdone_p = config.map_files(src)
        sidx = pq.read_table(sidx_p)
        sdone = np.load(sdone_p)
        stok = np.load(stok_p, mmap_mode="r")
        skey = {}
        for z, x, y, r in zip(np.asarray(sidx["z"]), np.asarray(sidx["x"]),
                              np.asarray(sidx["y"]), np.asarray(sidx["row"])):
            skey[(int(z), int(x), int(y))] = int(r)
        moved = 0
        for row in np.flatnonzero(done == 0):
            sr = skey.get(keys[row])
            if sr is not None and sdone[sr]:
                tok[row] = stok[sr]
                done[row] = 1
                moved += 1
        tok.flush()
        np.save(DONE, done)
        print("seeded         {:,} tiles from {}".format(moved, src))

    todo = np.flatnonzero(done == 0).tolist()
    print("already cached {:,}".format(n - len(todo)))
    print("to fetch       {:,}\n".format(len(todo)))
    if not todo:
        print("nothing to do")
        return

    client = T.TileClient(config.TILE_SERVER)
    lock = threading.Lock()
    state = {"ok": 0, "err": 0, "t0": time.time()}
    errors = []

    def work(row):
        z, x, y = keys[row]
        try:
            m = client.mask(z, x, y)
            tok[row] = T.to_tokens(m, a.grid).astype(np.float16)
            with lock:
                done[row] = 1
                state["ok"] += 1
        except Exception as e:
            with lock:
                state["err"] += 1
                if len(errors) < 10:
                    errors.append("z{}/{}/{}: {}".format(z, x, y, e))
        with lock:
            n_done = state["ok"] + state["err"]
            if n_done % 2000 == 0:
                el = time.time() - state["t0"]
                rate = n_done / max(el, 1e-6)
                eta = (len(todo) - n_done) / max(rate, 1e-6)
                print("  {:>7,}/{:,}  {:6.0f} tiles/s  eta {:5.1f} min  err {}"
                      .format(n_done, len(todo), rate, eta / 60, state["err"]))

    try:
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            list(ex.map(work, todo))
    finally:
        tok.flush()
        np.save(DONE, done)

    el = time.time() - state["t0"]
    print("\nfetched {:,} in {:.1f} min ({:.0f} tiles/s), {} errors"
          .format(state["ok"], el / 60, state["ok"] / max(el, 1e-6), state["err"]))
    for e in errors:
        print("  " + e)
    print("cached total {:,}/{:,}".format(int(done.sum()), n))


if __name__ == "__main__":
    main()
