"""The reader must return the caller's order, and pack round-trips must be exact.

`blobs` returns images in the order the caller asked for, while reading each
archive in *stored* order underneath. Those two orders are deliberately
different, and confusing them pairs every image with another image's label --
the failure this project has paid for more than any other. So the ordering is
asserted directly, with archive order deliberately reversed against caller
order so a naive implementation cannot pass.
"""

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import shards  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """Two packs whose stored order is the reverse of their id order."""
    for pack, lo in (("0000", 0), ("0001", 10)):
        names = ["{}/{}.jpg".format(pack, lo + i) for i in range(5)]
        with zipfile.ZipFile(tmp_path / (pack + ".zip"), "w") as zf:
            for n in reversed(names):                 # stored backwards
                zf.writestr(n, ("blob-" + n.split("/")[1]).encode())
    return tmp_path


def test_blobs_returns_the_callers_order(store):
    want = ["0001/12.jpg", "0000/3.jpg", "0001/10.jpg", "0000/0.jpg"]
    got = shards.blobs(want, root=store, quiet=True)
    assert got == [("blob-" + m.split("/")[1]).encode() for m in want]


def test_the_slurp_path_returns_the_same_thing(store):
    """min_slurp=1 forces the sequential path, which also re-sorts into
    archive order internally. It must not leak that order out."""
    want = ["0000/4.jpg", "0000/1.jpg", "0000/2.jpg"]
    a = shards.blobs(want, root=store, min_slurp=1 << 30, quiet=True)
    b = shards.blobs(want, root=store, min_slurp=1, quiet=True)
    assert a == b


def test_a_missing_shard_is_refused_not_skipped(store):
    with pytest.raises(SystemExit, match="not on disk"):
        shards.blobs(["0009/1.jpg"], root=store, quiet=True)


def test_a_missing_member_is_refused(store):
    with pytest.raises(SystemExit):
        shards.blobs(["0000/999.jpg"], root=store, quiet=True)


def test_loose_files_are_read_through_the_same_entry_point(tmp_path):
    """The pre-pack layout. Supporting it here is what makes migration a data
    change rather than a code change in every caller."""
    (tmp_path / "7.jpg").write_bytes(b"loose-7")
    assert shards.blobs(["7.jpg"], root=tmp_path, quiet=True) == [b"loose-7"]


def test_loose_and_packed_can_be_mixed(store):
    (store / "99.jpg").write_bytes(b"loose-99")
    got = shards.blobs(["0000/1.jpg", "99.jpg"], root=store, quiet=True)
    assert got == [b"blob-1.jpg", b"loose-99"]


# --- packing -----------------------------------------------------------------

def test_assign_packs_groups_consecutive_ids(store):
    w = shards.assign_packs(["30", "10", "20", "40"], size=2)
    assert w["10"].startswith("0000/") and w["20"].startswith("0000/")
    assert w["30"].startswith("0001/") and w["40"].startswith("0001/")


def test_assign_packs_sorts_numerically_not_lexically():
    """Lexical order puts "1000" before "9", which would scatter one drive
    across two packs and defeat the point of grouping."""
    w = shards.assign_packs(["9", "1000", "10"], size=2)
    assert w["9"].startswith("0000/") and w["10"].startswith("0000/")
    assert w["1000"].startswith("0001/")


def test_write_pack_round_trips_through_the_reader(tmp_path):
    items = [("0000/{}.jpg".format(i), bytes([i]) * 100) for i in range(4)]
    shards.write_pack(tmp_path / "0000.zip", items)
    got = shards.blobs(["0000/{}.jpg".format(i) for i in range(4)],
                       root=tmp_path, quiet=True)
    assert got == [b for _, b in items]


def test_write_pack_leaves_no_partial_file_behind(tmp_path):
    shards.write_pack(tmp_path / "0000.zip", [("0000/a.jpg", b"x")])
    assert not list(tmp_path.glob("*.part"))


def test_packs_are_stored_not_deflated(tmp_path):
    """Deflate costs CPU on already-compressed JPEGs and puts the work back on
    the core the packing exists to free."""
    shards.write_pack(tmp_path / "0000.zip", [("0000/a.jpg", b"y" * 5000)])
    with zipfile.ZipFile(tmp_path / "0000.zip") as zf:
        assert zf.getinfo("0000/a.jpg").compress_type == zipfile.ZIP_STORED


# --- the three findings on PR #33 -------------------------------------------

def _pack(root, index, ids):
    items = [("{}/{}.jpg".format(shards.pack_stem(index), i),
              ("BYTES:%s" % i).encode()) for i in ids]
    shards.write_pack(root / (shards.pack_stem(index) + ".zip"), items)


def test_a_second_run_does_not_reuse_a_pack_index(tmp_path):
    """Numbering restarted at 0000 every run, so a later pass overwrote the
    earlier pass's pack while the manifest still resolved into it."""
    _pack(tmp_path, 0, ["1", "2"])
    assert shards.next_pack_index(tmp_path) == 1

    where = shards.assign_packs(["7", "8"], size=2,
                                start=shards.next_pack_index(tmp_path))
    assert set(where.values()) == {"0001/7.jpg", "0001/8.jpg"}


def test_an_existing_pack_is_never_overwritten(tmp_path):
    """The last line of defence: even asked directly, a pack does not move.
    Its members are what live manifest records resolve to."""
    _pack(tmp_path, 0, ["1"])
    with pytest.raises(SystemExit, match="already exists"):
        _pack(tmp_path, 0, ["9"])
    assert shards.blobs(["0000/1.jpg"], root=tmp_path, quiet=True) == [b"BYTES:1"]


def test_blobs_reads_loose_and_packed_together(tmp_path):
    """A manifest is mixed while a pack run is pending, and consumers took
    only one of the two forms."""
    _pack(tmp_path, 0, ["1", "2"])
    (tmp_path / "3.jpg").write_bytes(b"BYTES:3")
    got = shards.blobs(["3.jpg", "0000/2.jpg", "0000/1.jpg"],
                       root=tmp_path, quiet=True)
    assert got == [b"BYTES:3", b"BYTES:2", b"BYTES:1"]


def test_reader_takes_either_form_and_opens_a_pack_once(tmp_path, monkeypatch):
    _pack(tmp_path, 0, ["1", "2", "3"])
    (tmp_path / "9.jpg").write_bytes(b"BYTES:9")

    opens = []
    real = shards.archive

    def spy(shard, **kw):
        opens.append(shard)
        return real(shard, **kw)

    monkeypatch.setattr(shards, "archive", spy)
    with shards.Reader(tmp_path) as rd:
        got = [rd.read(n) for n in
               ["0000/1.jpg", "9.jpg", "0000/3.jpg", "0000/2.jpg"]]
    assert got == [b"BYTES:1", b"BYTES:9", b"BYTES:3", b"BYTES:2"]
    assert opens == ["0000"]


def test_source_of_names_the_file_that_holds_the_bytes(tmp_path):
    """A cache key stamped on the loose path stops existing once it is packed."""
    assert shards.source_of(tmp_path, "0003/5.jpg") == tmp_path / "0003.zip"
    assert shards.source_of(tmp_path, "5.jpg") == tmp_path / "5.jpg"


def test_a_string_root_and_a_loose_member_work(tmp_path):
    """`root` is normalised once, the way `archive` does it. It was used raw
    for the loose branch, so a string root -- or none at all -- raised
    TypeError on `root / member` before reaching any archive."""
    (tmp_path / "5.jpg").write_bytes(b"BYTES:5")
    assert shards.blobs(["5.jpg"], root=str(tmp_path), quiet=True) == [b"BYTES:5"]
    got = list(shards.iter_blobs(["5.jpg"], batch=8, root=str(tmp_path),
                                 quiet=True))
    assert [b for _, bs in got for b in bs] == [b"BYTES:5"]


def test_iter_blobs_agrees_with_blobs_about_loose_members(tmp_path):
    """`shard_of("5.jpg")` is the whole name, so the batched reader used to
    look for `5.jpg.zip` where `blobs` read the file."""
    _pack(tmp_path, 0, ["1", "2"])
    (tmp_path / "5.jpg").write_bytes(b"BYTES:5")
    names = ["0000/1.jpg", "5.jpg", "0000/2.jpg"]
    want = shards.blobs(names, root=tmp_path, quiet=True)
    seen = {}
    for sel, got in shards.iter_blobs(names, batch=2, root=tmp_path, quiet=True):
        seen.update(dict(zip(sel, got)))
    assert [seen[i] for i in range(len(names))] == want


def test_every_manifest_consumer_that_opens_images_uses_the_reader():
    """A KartaView image's location is whatever its manifest record says.

    Consumers reconstructed `"<id>.jpg"` or opened `rec["file"]` as a path.
    Both work while the corpus is loose and neither works once it is packed --
    the first fails only after the originals are deleted, which is later and
    worse. Four were migrated and a fifth, `eval_highres`, was missed; it
    raised FileNotFoundError through its own main().

    So the rule is checked rather than remembered: read the manifest and open
    images, and the images come through `shards`.
    """
    import re

    bad = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "manifest.jsonl" not in src:
            continue
        # Only opens of something stored. A harvester decoding bytes it has
        # just downloaded -- `Image.open(BytesIO(blob))` -- reads nothing from
        # the corpus and is not a consumer of it.
        opens = [m for m in re.findall(r"Image\.open\(([^)]*)", src)
                 if "BytesIO" not in m]
        if not opens:
            continue
        if not re.search(r"\bshards\.", src):
            bad.append(path.name)
    assert not bad, (
        "reads the KartaView manifest and opens images without going through "
        "shards, so it breaks once the corpus is packed: " + ", ".join(bad))


def test_no_manifest_consumer_treats_a_record_file_as_a_path():
    """`rec["file"]` is a location, not necessarily a file.

    Once packed it reads `"0000/123.jpg"`, a member inside an archive. Dividing
    a directory by it produces a path that does not exist, and the failure is
    whatever the caller does next -- `Image.open` for five consumers, and
    `.stat()` in the harvester's own summary line, which is why keying this
    check on `Image.open` alone was not enough.
    """
    import re

    # Both of these are correct: each is reached only for records that are
    # still loose -- the harvester guards on `"/" not in`, and the packer's
    # `pack_them` is handed exactly the loose records. Exempted by their text
    # rather than their line, so they survive edits above them while a
    # DIFFERENT construction in the same file is still caught.
    guarded = ('out / "img" / str(r["file"])', 'img / str(r["file"])')

    pat = re.compile(r'[^\n]*?/\s*(?:str\()?\s*\w+(?:\[[^\]]+\])?\["file"\]\)?')
    bad = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if "manifest.jsonl" not in src:
            continue
        for m in pat.finditer(src):
            if any(g in m.group(0) for g in guarded):
                continue
            line = src[:m.start()].count("\n") + 1
            bad.append("{}:{}".format(path.name, line))
    assert not bad, (
        "builds a filesystem path out of a manifest record's `file`, which is "
        "a pack member once the corpus is packed: " + ", ".join(bad))
