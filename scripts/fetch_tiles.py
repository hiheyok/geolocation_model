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
import maskio
import tile_math as tm
import tiles as T

TOKENS, INDEX, DONE = config.map_files()


def _load_mask(path, n):
    """A completion mask is only a resume point if it describes this cache.

    Loaded blind, a short mask makes the run unrecoverable without deleting it
    by hand, and values above one corrupt the completed count that decides
    whether the build is finished.
    """
    # One implementation, in src/maskio.py. Four producers grew four copies of
    # this check and three of them were wrong at some point; the copies were
    # the reason nobody noticed when one drifted.
    return maskio.load_mask(path, n, "tile cache")


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
    ap.add_argument("--sub", type=int, default=1,
                    help="split each patch into sub x sub cells before "
                         "histogramming, so within-patch layout survives. "
                         "sub=1 is the original 12-d token")
    ap.add_argument("--out-cache", default=None,
                    help="write to this directory instead of the release's "
                         "own map cache; use it for a sub>1 cache so the "
                         "existing one stays valid for existing arms")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--exhaustive-z8", action="store_true",
                    help="also fetch all 65,536 z8 tiles (only helps beam latency)")
    ap.add_argument("--seed-from", default=None,
                    help="another release's map cache to copy already-fetched "
                         "tiles from, matched by (z,x,y). A token grid is a "
                         "pure function of its address and so is valid across "
                         "releases; only the row numbering differs.")
    a = ap.parse_args()

    # Resolve the cache here, not at import: --out-cache writes a parallel
    # cache so a sub>1 build cannot overwrite the one every existing arm was
    # trained against.
    global TOKENS, INDEX, DONE
    if a.out_cache:
        Path(a.out_cache).mkdir(parents=True, exist_ok=True)
        TOKENS, INDEX, DONE = config.map_files(a.out_cache)

    # OSV_RELEASE decides which targets.parquet defines "needed", and it
    # defaults to s01. Building an s10 cache under the default silently fetches
    # the 50k release's tiles, reports "0 errors", and fails much later inside
    # GeoStepDataset with a KeyError on a tile that was never asked for. Print
    # it where it cannot be missed.
    print("release       {}  ({})".format(config.RELEASE,
                                          config.TARGETS_PARQUET))
    keys = needed_keys(a.grid, a.exhaustive_z8)
    n, d = len(keys), a.grid * a.grid
    width = T.N_CLASSES * a.sub * a.sub
    print("tiles needed   {:,}".format(n))
    print("token width    {} ({} classes x {}x{} cells)".format(
        width, T.N_CLASSES, a.sub, a.sub))
    print("cache          {}".format(TOKENS))
    print("cache size     {:,.0f} MB fp16".format(n * d * width * 2 / 1e6))

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
        if reuse:
            # The address index says nothing about token width. Reusing a
            # 12-d cache for a --sub 2 build would keep the old memmap and an
            # all-complete done mask, then report "nothing to do".
            try:
                shp = np.load(TOKENS, mmap_mode="r").shape
                w, cells = shp[-1], shp[1]
            except Exception:
                w, cells = None, None
            # Cell count as well as width: the memmap header would catch a
            # mismatch on open, but only by raising somewhere less obvious
            # than here.
            if cells not in (None, d):
                reuse = False
                print("cache holds {} cells per tile, this build wants {} "
                      "-- starting fresh".format(cells, d))
            if w != width:
                reuse = False
                print("cache is {}-d, this build is {}-d -- starting fresh"
                      .format(w, width))
        if not reuse:
            print("index changed -- starting a fresh cache")

    if not reuse and DONE.exists():
        # Drop the old mask BEFORE the memmap below truncates the tokens.
        # Otherwise a crash in between leaves a zeroed token file beside a
        # mask that still says every row is complete, and the next run
        # blesses rows that hold nothing.
        DONE.unlink()

    tok = np.lib.format.open_memmap(
        TOKENS, mode="r+" if reuse else "w+",
        dtype=np.float16, shape=(n, d, width))
    done = (_load_mask(DONE, n) if reuse
            else np.zeros(n, dtype=np.uint8))
    pq.write_table(idx, INDEX)

    if a.seed_from:
        # Growing a release adds sparse z12/z16 rows and renumbers everything,
        # which would otherwise refetch the 65,792 exhaustive tiles the previous
        # release already holds.  Addresses are stable, rows are not.
        src = Path(a.seed_from)
        stok_p, sidx_p, sdone_p = config.map_files(src)
        sidx = pq.read_table(sidx_p)
        stok = np.load(stok_p, mmap_mode="r")
        if stok.shape[-1] != width:
            sys.exit("--seed-from cache is {}-d, this build is {}-d; a token "
                     "grid is only reusable at the same --sub"
                     .format(stok.shape[-1], width))
        # The destination mask goes through `_load_mask`; the SOURCE used to go
        # through nothing but that width check. Everything it is trusted about
        # is load-bearing, and each failure is silent rather than loud:
        #
        #   * `if sdone[sr]` accepts any nonzero value, so a 2 or a 255 blesses
        #     a zero-filled source row -- and blesses it as a 1 here, which
        #     launders it permanently into a cache that then looks clean;
        #   * a negative `row` indexes from the END in numpy, so it passes any
        #     `max()` bound check and copies some other tile's perfectly valid
        #     tensor;
        #   * a row past the token array is the only one that would have
        #     raised, and only sometimes.
        srow = np.asarray(sidx["row"]).astype(np.int64)
        sdone = maskio.load_mask(sdone_p, len(srow), "--seed-from")
        if len(srow) and (srow.min() < 0 or srow.max() >= len(stok)):
            sys.exit("--seed-from index addresses rows {}..{} but its token "
                     "array holds {:,}; a negative row wraps from the end in "
                     "numpy and would copy another tile's tensor."
                     .format(int(srow.min()), int(srow.max()), len(stok)))
        if len(np.unique(srow)) != len(srow):
            sys.exit("--seed-from index maps several addresses to one token "
                     "row; the cache it describes cannot be read back "
                     "unambiguously.")
        skey = {}
        for z, x, y, r in zip(np.asarray(sidx["z"]), np.asarray(sidx["x"]),
                              np.asarray(sidx["y"]), srow):
            skey[(int(z), int(x), int(y))] = int(r)
        if len(skey) != len(srow):
            sys.exit("--seed-from index holds duplicate tile addresses; which "
                     "row an address means is then decided by iteration order.")
        moved = 0
        for row in np.flatnonzero(done != 1):
            sr = skey.get(keys[row])
            if sr is not None and sdone[sr] == 1:
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
            tok[row] = T.to_tokens(m, a.grid, a.sub).astype(np.float16)
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

    # Exiting 0 here is what let a partial cache look finished: the stage runner
    # keys its markers on the exit code, so a run that lost tiles to a flaky
    # server wrote a success marker and was never retried.  The unfetched rows
    # stay at the memmap's zero fill, which is a legal token histogram, so
    # nothing downstream complained either.  Re-running is safe and resumable.
    short = n - int(done.sum())
    if short:
        print()
        sys.exit("INCOMPLETE: {:,} of {:,} tiles were not fetched. The cache "
                 "is not usable as it stands -- unfetched rows are all-zero "
                 "token histograms that read as valid. Re-run to resume."
                 .format(short, n))


if __name__ == "__main__":
    main()
