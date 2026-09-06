"""`pack_kartaview.main()` end to end, on a real directory.

Three findings on PR #33, all of which needed the script actually run:

* rerunning restarted pack numbering at 0000 and overwrote the first run's
  pack, destroying images whose originals had been deleted;
* the manifest published pack member paths that no consumer could open;
* after a normal run every record names a pack, so `--delete-originals`
  returned before doing anything -- the second invocation the docstring
  advertises was a no-op.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pack_kartaview as pk  # noqa: E402
import shards  # noqa: E402


def make(tmp_path, ids):
    img = tmp_path / "img"
    img.mkdir(exist_ok=True)
    recs = []
    for i in ids:
        (img / ("%s.jpg" % i)).write_bytes(("BYTES:%s" % i).encode())
        recs.append({"id": i, "file": "%s.jpg" % i})
    with (tmp_path / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return tmp_path


def run(tmp_path, *extra):
    argv = ["pack_kartaview.py", "--data", str(tmp_path), "--size", "2"]
    sys.argv = argv + list(extra)
    pk.main()


def manifest(tmp_path):
    return {r["id"]: r["file"] for r in
            pk.read_manifest(tmp_path / "manifest.jsonl")}


def test_a_second_harvest_does_not_destroy_the_first(tmp_path):
    """The P1: with the originals gone, overwriting 0000.zip lost them."""
    make(tmp_path, ["1", "2"])
    run(tmp_path, "--delete-originals")
    assert manifest(tmp_path) == {"1": "0000/1.jpg", "2": "0000/2.jpg"}
    assert not (tmp_path / "img" / "1.jpg").exists()

    make(tmp_path, ["3", "4"])
    run(tmp_path)
    m = manifest(tmp_path)
    assert m["3"].startswith("0001/") and m["4"].startswith("0001/")
    # the first run's images are still readable, which is the whole point
    assert shards.blobs([m["1"], m["2"]], root=tmp_path / "img",
                        quiet=True) == [b"BYTES:1", b"BYTES:2"]


def test_the_rerun_cleanup_actually_removes_originals(tmp_path):
    """The P2: after packing, no record is loose, so the advertised second
    invocation used to return immediately."""
    make(tmp_path, ["1", "2", "3"])
    run(tmp_path)                                   # pack, keep originals
    assert (tmp_path / "img" / "1.jpg").exists()

    run(tmp_path, "--delete-originals")             # the documented rerun
    for i in ("1", "2", "3"):
        assert not (tmp_path / "img" / ("%s.jpg" % i)).exists()
    m = manifest(tmp_path)
    assert shards.blobs([m["1"]], root=tmp_path / "img",
                        quiet=True) == [b"BYTES:1"]


def test_cleanup_refuses_when_a_pack_disagrees(tmp_path):
    """Deletion is gated on the replacement reproducing the original."""
    make(tmp_path, ["1", "2"])
    run(tmp_path)
    (tmp_path / "img" / "1.jpg").write_bytes(b"TAMPERED")
    with pytest.raises(SystemExit, match="Refusing to delete"):
        run(tmp_path, "--delete-originals")
    assert (tmp_path / "img" / "1.jpg").exists()


def test_consumers_can_read_what_the_manifest_publishes(tmp_path):
    """The other P1: packed paths were published that no consumer could open."""
    make(tmp_path, ["1", "2"])
    run(tmp_path, "--delete-originals")
    recs = pk.read_manifest(tmp_path / "manifest.jsonl")
    img = tmp_path / "img"
    assert shards.blobs([r["file"] for r in recs], root=img,
                        quiet=True) == \
        [b"BYTES:1", b"BYTES:2"]
    with shards.Reader(img) as rd:
        assert [rd.read(r["file"]) for r in recs] == [b"BYTES:1", b"BYTES:2"]
