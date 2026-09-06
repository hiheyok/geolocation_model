"""Pack KartaView's loose JPEGs into ordered archives.

47,646 images sit as individual files in one directory on `E:` -- a 5900 RPM
HDD. Reading them costs a seek each, measured at 178 reads/s and 6.6 MB/s
against 300 MB/s sequential, so a single pass over the cohort is ~4.5 minutes
of the GPU waiting. Listing the directory alone exceeded a two-minute timeout.

Packed, the same read is a sequential scan through `shards.blobs`, which is
already how the OSV-5M release is read. The member path becomes
`"<pack>/<id>.jpg"`, identical in form to `"<shard>/<id>.jpg"`, so no reader
gains a second code path -- the layout change is a data change.

**Grouping by id is not arbitrary.** Adjacent KartaView ids are the same
drive: 64.0% of consecutive ids are within 1 km of each other and 51.7% within
100 m, median 67 m. A pack of 1,000 consecutive ids is therefore a set of whole
drives, so a sequential read over a pack is a sequential read over a region,
and the pack boundary is also the natural unit for grouped resampling.

Safety, in order:

* every blob is read back out of the pack and compared byte for byte before
  the manifest is rewritten;
* the manifest is rewritten atomically, with the old one kept as
  `manifest.jsonl.loose`;
* **originals are kept** unless `--delete-originals` is passed, and that flag
  refuses to act unless verification passed in the same run.

    OSV_RELEASE=s10 py scripts/pack_kartaview.py --data E:/data/kartaview_hr
    OSV_RELEASE=s10 py scripts/pack_kartaview.py --data ... --delete-originals
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import safeio                                    # noqa: E402
import shards                                    # noqa: E402


def read_manifest(path):
    recs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="E:/data/kartaview_hr")
    ap.add_argument("--size", type=int, default=shards.PACK_SIZE,
                    help="images per pack")
    ap.add_argument("--delete-originals", action="store_true",
                    help="remove the loose files after every blob has been "
                         "verified byte-identical out of its pack")
    a = ap.parse_args()

    data = Path(a.data)
    img = data / "img"
    mpath = data / "manifest.jsonl"
    recs = read_manifest(mpath)
    print("{:,} manifest records under {}".format(len(recs), data), flush=True)

    todo = [r for r in recs if "/" not in str(r.get("file", ""))]
    if todo:
        print("{:,} still loose".format(len(todo)), flush=True)
        pack_them(todo, recs, img, mpath, a.size)
    else:
        print("every record already names a pack; nothing to pack")

    if a.delete_originals:
        drop_originals(recs, img)
    else:
        print("loose files kept. Re-run with --delete-originals once the "
              "packed path has been exercised end to end.")


def pack_them(todo, recs, img, mpath, size):
    """Write the loose records into new packs and republish the manifest."""
    # Never reuse a pack index. Numbering restarted at 0000 on every run, so a
    # second pass over newly harvested images assigned them to `0000/...` and
    # overwrote the pack the first pass had written -- while the manifest
    # records of the images inside it still resolved to those member names.
    # With --delete-originals from the first run, that was the only copy.
    # A partly-filled final pack stays partly filled: a pack is a unit of
    # sequential reading, not a container that has to be topped up.
    start = shards.next_pack_index(img)
    if start:
        print("{:,} packs already written; new ones start at {}".format(
            start, shards.pack_stem(start)), flush=True)

    where = shards.assign_packs([r["id"] for r in todo], size, start=start)
    groups = defaultdict(list)
    for r in todo:
        groups[shards.shard_of(where[str(r["id"])])].append(r)

    t0 = time.time()
    done, verified = {}, 0
    for k, (pack, rows) in enumerate(sorted(groups.items())):
        items = []
        for r in rows:
            fp = img / str(r["file"])
            if not fp.exists():
                raise SystemExit(
                    "{} is in the manifest but not on disk. Packing a partial "
                    "set would produce an archive that reads as complete."
                    .format(fp))
            # The FULL "<pack>/<id>.jpg" path, matching how the release zips
            # store "<shard>/<id>.jpg". One convention for both corpora.
            items.append((where[str(r["id"])], fp.read_bytes()))
        out = shards.write_pack(img / (pack + ".zip"), items)

        # Read back through the production reader, not zipfile, so this checks
        # the path that will actually be used.
        got = shards.blobs([where[str(r["id"])] for r in rows], root=img,
                           quiet=True)
        for (name, blob), g, r in zip(items, got, rows):
            if blob != g:
                raise SystemExit(
                    "{} does not read back identically from {}".format(
                        name, out.name))
            verified += 1
            done[str(r["id"])] = where[str(r["id"])]
        print("  {:>4}/{:<4} {}  {:>5,} images  {:.1f} MB  {:.0f}s".format(
            k + 1, len(groups), pack, len(rows),
            out.stat().st_size / 1e6, time.time() - t0), flush=True)

    if verified != len(todo):
        raise SystemExit("verified {:,} of {:,}".format(verified, len(todo)))

    safeio.write_text(mpath.with_suffix(".jsonl.loose"),
                      mpath.read_text(encoding="utf-8"))
    for r in recs:
        if str(r["id"]) in done:
            r["file"] = done[str(r["id"])]
    safeio.write_text(mpath, "\n".join(json.dumps(r) for r in recs) + "\n")
    print("\nmanifest rewritten; previous kept as manifest.jsonl.loose",
          flush=True)


def drop_originals(recs, img):
    """Remove each loose original whose pack reproduces it byte for byte.

    Keyed on every packed record, not on what this run happened to write.
    After a normal packing run no record is loose, so the second invocation
    the module docstring advertises used to return before reaching here and
    delete nothing at all.

    Each original is compared against what its pack returns through the
    production reader before it is unlinked, so the deletion is gated on the
    replacement being readable and identical -- not merely on a pack existing.
    Reads are grouped by pack, so each is opened once.
    """
    packed = [r for r in recs if "/" in str(r.get("file", ""))]
    if not packed:
        print("no packed records; nothing to clean up")
        return
    names = [str(r["file"]) for r in packed]
    removed = absent = 0
    for sel, got in shards.iter_blobs(names, batch=500, root=img, quiet=True):
        for i, blob in zip(sel, got):
            fp = img / (str(packed[i]["id"]) + ".jpg")
            if not fp.exists():
                absent += 1
                continue
            if fp.read_bytes() != blob:
                raise SystemExit(
                    "{} differs from what {} returns. Refusing to delete an "
                    "original its pack does not reproduce."
                    .format(fp.name, names[i]))
            fp.unlink()
            removed += 1
    print("verified {:,} packed images; removed {:,} loose originals, "
          "{:,} already gone".format(len(packed), removed, absent))


if __name__ == "__main__":
    main()
