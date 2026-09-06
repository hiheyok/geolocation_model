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


def test_image_bytes_reads_loose_and_packed_together(tmp_path):
    """A manifest is mixed while a pack run is pending, and consumers took
    only one of the two forms."""
    _pack(tmp_path, 0, ["1", "2"])
    (tmp_path / "3.jpg").write_bytes(b"BYTES:3")
    got = shards.image_bytes(tmp_path, ["3.jpg", "0000/2.jpg", "0000/1.jpg"])
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
