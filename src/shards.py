"""The one way this project reads images out of the release zips.

`embed_street.slurp` already existed, was exported, and was adopted by four
scripts. `resmatch` still opens `zipfile.ZipFile(path)` on the raw file and
reads members by name -- 43,000 seeks on a 5900 RPM drive at 178 reads/s,
6.6 MB/s, against 300 MB/s sequential. The GPU waits through all of it.

That is the failure mode this module exists to close, and it is not a missing
optimisation. **An optional helper gets left behind.** The same evening
produced three instances of the identical shape: `fuse_flat` shared while the
scoring composition was written out three times; `knnmeta.check` shared while
each caller passed its own arguments and activated none of the checks;
`embed_street.preprocess` imported by `query_only`, which then wrote its own
loop around it and dropped the prefetch pool. In every case the leaf was
shared and the loop was copied.

So this module owns **opening the archive**, not a step inside it, and
`tests/test_one_reader.py` fails when a new file calls `ZipFile` itself. The
allowlist there can only shrink.

Concurrency is deliberately not the headline. `resmatch` has a thread pool and
still stalls, because the bottleneck is the access pattern rather than the
worker count: on this hardware sequential reading is worth ~46x and queue
depth is worth ~2x.

**The hardware this is tuned for**, measured 2026-09-05:

    E:  ST32000542AS   5900 RPM HDD   the release zips
    C:  MSI M450       SSD            the embedding caches

Random member reads run at 178/s (5.6 ms each). A cold sequential pass over a
2.5 GB shard runs at 100-300 MB/s, so 8-25 s. Slurping therefore pays above
roughly 1,500-4,500 members from one shard, and loses badly below it -- the
server answering a single query must not read 2.5 GB. `MIN_SLURP` sits inside
that range rather than at either end, and the choice is made here so that no
caller has to make it.
"""

import io
import sys
import time
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

# Above this many members from one shard, read the whole shard sequentially.
# Derived above from measured seek and streaming rates; it is a property of
# this machine's disks, not a universal constant.
MIN_SLURP = 2048


def slurp(path, chunk=32 << 20, every=4, quiet=False):
    """Read a file sequentially into RAM, reporting throughput as it goes.

    Moved here from `scripts/embed_street.py`: a helper that four scripts
    import and a fifth forgot belongs in the library, not in one of its
    consumers.
    """
    path = Path(path)
    size = path.stat().st_size
    if not quiet:
        print("preload    {:.2f} GB sequentially into RAM".format(size / 1e9),
              flush=True)
    buf = bytearray(size)
    view = memoryview(buf)
    off, t0 = 0, time.time()
    with open(path, "rb", buffering=0) as fh:
        while off < size:
            got = fh.readinto(view[off:off + chunk])
            if not got:
                break
            off += got
            if not quiet and (off // chunk) % every == 0:
                el = time.time() - t0
                print("           {:.2f}/{:.2f} GB   {:6.1f} MB/s".format(
                    off / 1e9, size / 1e9, off / max(el, 1e-9) / 1e6),
                    flush=True)
    el = time.time() - t0
    if not quiet:
        print("           done in {:.1f}s ({:.0f} MB/s)".format(
            el, size / max(el, 1e-9) / 1e6), flush=True)
    return io.BytesIO(buf)


@contextmanager
def archive(shard, root=None, preload=True, quiet=False):
    """One shard's archive, opened over a sequential read of the whole file.

    `blobs` is for callers that want the bytes and nothing else. These four
    want the open `ZipFile` -- they each wrap it in a different tuned pipeline
    (a decode thread pool, a bounded sliding window, a pinned staging buffer),
    and those pipelines are genuinely different and should stay where they
    are. What was duplicated is not the loop but the POLICY in front of it:

        zipfile.ZipFile(slurp(zp) if preload else zp)

    written out four times, over a zip path derived by hand three times. That
    is the piece worth having once, because it is the piece that is wrong when
    it is wrong: a shard opened without the preload is 12,000 scattered reads
    into a 2.5 GB archive on a 5900 RPM disk.
    """
    zp = (Path(root) if root else config.TRAIN_ZIPS) / (shard + ".zip")
    if not zp.exists():
        raise SystemExit(
            "shard {} is not on disk at {}. Reading members by name from a "
            "missing archive is the one case where a partial result would "
            "look complete.".format(shard, zp))
    with zipfile.ZipFile(slurp(zp, quiet=quiet) if preload else zp) as zf:
        yield zf


def shard_of(member):
    """`"07/1234.jpg"` -> `"07"`. The shard is the archive it lives in."""
    return str(member).split("/")[0]


def _archive_order(zf, idxs, members):
    """`idxs` sorted by position in the archive, so an in-RAM read walks
    forward. Free once the file is resident, and it is what the caller's own
    order would otherwise scramble."""
    pos = {n: k for k, n in enumerate(zf.namelist())}
    return sorted(idxs, key=lambda i: pos.get(members[i], 1 << 30))


def _read(zf, member, shard):
    """One member, or an error that names it.

    A bare `zf.read` raises `KeyError: "There is no item named ..."` from three
    frames down, which does not say which shard was open or that everything
    else read fine.
    """
    try:
        return zf.read(member)
    except KeyError:
        raise SystemExit(
            "{} is not in shard {}. The archive opened and its other members "
            "read, so nothing else would have reported this."
            .format(member, shard)) from None


def blobs(members, root=None, min_slurp=MIN_SLURP, quiet=False):
    """JPEG bytes for `members`, in the caller's order.

    `members` are `"<shard>/<image_id>.jpg"` paths. They are grouped by shard
    and each shard is opened once; a shard contributing at least `min_slurp`
    members is read sequentially into RAM first, which is the entire point of
    this function.

    Returned in the caller's order, not archive order, so a caller can index
    the result against its own arrays without a second mapping -- getting that
    wrong silently pairs each image with another image's label, which is this
    project's most expensive failure shape.
    """
    members = [str(m) for m in members]
    want = defaultdict(list)
    for i, m in enumerate(members):
        want[shard_of(m)].append(i)

    out = [None] * len(members)
    for shard, idxs in sorted(want.items()):
        big = len(idxs) >= min_slurp
        with archive(shard, root=root, preload=big, quiet=quiet) as zf:
            if big:
                idxs = _archive_order(zf, idxs, members)
            for i in idxs:
                out[i] = _read(zf, members[i], shard)
    missing = [members[i] for i, b in enumerate(out) if b is None]
    if missing:
        raise SystemExit("{:,} members were never read, e.g. {}".format(
            len(missing), ", ".join(missing[:3])))
    return out


def iter_blobs(members, batch=256, root=None, min_slurp=MIN_SLURP, quiet=False):
    """`blobs` in batches, for callers that cannot hold every image at once.

    Yields `(indices, blobs)`, where `indices` are positions in `members` and
    pair positionally with `blobs`, so the caller indexes its own arrays with
    them. Batches do not span shards and are not in caller order.

    **One shard, one open** -- which the previous version claimed and did not
    do. It sorted the members shard-major, then sliced that global list into
    fixed `batch` windows and called `blobs` on each, so a shard contributing
    more than `batch` members was reopened once per window: 196 opens of the
    same 2.5 GB archive for a 50,000-member shard at the default batch.

    Worse, `blobs` decides the sequential read from the member count it is
    handed, and each window handed it `batch`. With any batch below
    `min_slurp` -- 2048, and the default here is 256 -- the preload could
    never fire at all, so every `iter_blobs` caller was doing precisely the
    scattered per-member reads this module exists to stop.

    So the grouping happens first and the batching happens inside it: the
    slurp decision sees the shard's whole contribution, and the archive stays
    open across the batches yielded from it.
    """
    members = [str(m) for m in members]
    want = defaultdict(list)
    for i, m in enumerate(members):
        want[shard_of(m)].append(i)

    for shard, idxs in sorted(want.items()):
        big = len(idxs) >= min_slurp
        with archive(shard, root=root, preload=big, quiet=quiet) as zf:
            if big:
                idxs = _archive_order(zf, idxs, members)
            for s in range(0, len(idxs), batch):
                sel = idxs[s:s + batch]
                yield sel, [_read(zf, members[i], shard) for i in sel]


