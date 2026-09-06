"""Replacing a JPEG under an unchanged manifest must re-embed, not reuse.

`multiquery`'s query cache stamped the data path, the manifest's file stamp,
the PCA basis, the encoder scale and the release -- everything except the
pixels. So correcting or re-downloading one image under the same manifest left
the build looking identical and handed back the old embedding, although a fresh
run would read different bytes (REVIEW4 #21).

Nothing else in the cache could catch it. The ids are unchanged, so the
id-equality check passes; the coordinates are unchanged, so the metadata
matches; and the vector is a well-formed embedding of *something*. It is simply
an embedding of the image that used to be there.

The narrow part is deliberate. `file_stamp` is size and mtime, which catches a
replaced or re-downloaded file -- the failure actually being guarded -- and
costs one stat. It would miss a byte-identical rewrite that also preserved the
mtime, and hashing 5,000 JPEGs on every cache lookup to close that gap is the
wrong trade for an accidental-rebuild guard. Both edges of that boundary are
pinned: `test_a_touched_file_invalidates` in the safe direction, and
`test_equal_size_and_mtime_is_a_known_blind_spot` in the direction it genuinely
cannot see.
"""

import os
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import multiquery as mq  # noqa: E402


def stamp(data, pick):
    """`query_stamp` now takes the manifest records, so that a packed image is
    stamped on the pack that holds it rather than on a loose path that stops
    existing. These fixtures predate the `file` field; `file_of` resolves that.
    """
    import multiquery as mq
    recs = {r["id"]: r for r in
            (json.loads(l) for l in
             (Path(data) / "manifest.jsonl").read_text(
                 encoding="utf-8").splitlines() if l.strip())}
    return mq.query_stamp(data, pick, recs)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A harvest root with a manifest and three images."""
    import config
    (tmp_path / "img").mkdir()
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join('{"id": "%s"}' % i for i in ("a", "b", "c")),
        encoding="utf-8")
    # Distinct lengths on purpose. Written in one loop they can land on the
    # same mtime tick, and `file_stamp` is size *and* mtime -- so equal-length
    # files created together are genuinely indistinguishable to it. That is a
    # real property of the guard, pinned by
    # `test_equal_size_and_mtime_is_a_known_blind_spot` rather than papered
    # over here.
    for n, i in enumerate(("a", "b", "c")):
        (tmp_path / "img" / (i + ".jpg")).write_bytes(b"j" * (10 + n))
    basis = tmp_path / "basis.npz"
    basis.write_bytes(b"basis")
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(mq, "BASIS", "basis.npz")
    return str(tmp_path)


def bump(p, body):
    """Rewrite a file and make sure its mtime actually moves.

    A test that rewrites within the filesystem's mtime granularity proves
    nothing about the guard -- it proves the clock did not tick.
    """
    p.write_bytes(body)
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))


def test_the_same_inputs_give_the_same_stamp(data):
    assert stamp(data, ["a", "b"]) == stamp(data, ["a", "b"])


def test_replacing_a_selected_image_invalidates(data):
    """The bug: same manifest, same ids, different pixels."""
    before = stamp(data, ["a", "b"])
    bump(Path(data) / "img" / "a.jpg", b"jpeg-a-corrected")
    assert stamp(data, ["a", "b"]) != before


def test_replacing_an_unselected_image_does_not_invalidate(data):
    """Only the images this cache actually embedded should matter.

    Otherwise every harvest touching any file re-embeds the whole selection.
    """
    before = stamp(data, ["a", "b"])
    bump(Path(data) / "img" / "c.jpg", b"jpeg-c-corrected")
    assert stamp(data, ["a", "b"]) == before


def test_a_touched_file_invalidates(data):
    """Size unchanged, mtime moved: the conservative direction.

    A re-download that produced identical bytes will re-embed needlessly. That
    costs time; the reverse would cost correctness.
    """
    before = stamp(data, ["a", "b"])
    bump(Path(data) / "img" / "a.jpg", b"jpeg-a")     # same bytes, new mtime
    assert stamp(data, ["a", "b"]) != before


def test_order_matters(data):
    """The vectors are written in `pick` order, so the stamp is ordered too."""
    assert stamp(data, ["a", "b"]) != stamp(data, ["b", "a"])


def test_a_different_selection_gives_a_different_stamp(data):
    assert stamp(data, ["a", "b"]) != stamp(data, ["a", "c"])


def test_equal_size_and_mtime_is_a_known_blind_spot(tmp_path, monkeypatch):
    """Two different files of the same size, written in the same tick, tie.

    `file_stamp` is size and mtime by design -- hashing bytes on every cache
    lookup is the trade it explicitly declines. This pins the limitation so it
    is a documented boundary rather than a surprise: the guard catches a file
    that was *replaced later*, which is the real failure, and not two distinct
    files that happen to coincide.
    """
    import config
    (tmp_path / "img").mkdir()
    (tmp_path / "manifest.jsonl").write_text('{"id": "x"}', encoding="utf-8")
    (tmp_path / "basis.npz").write_bytes(b"basis")
    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    monkeypatch.setattr(mq, "BASIS", "basis.npz")
    for i in ("p", "q"):
        f = tmp_path / "img" / (i + ".jpg")
        f.write_bytes(b"same-size!")
        os.utime(f, ns=(1_000_000_000, 1_000_000_000))
    assert (stamp(str(tmp_path), ["p"])
            == stamp(str(tmp_path), ["q"]))


def test_a_missing_image_does_not_raise_but_does_differ(data):
    """`file_stamp` reports absence rather than throwing, and absence is a
    different build -- the cache must not be reused across it."""
    before = stamp(data, ["a", "b"])
    (Path(data) / "img" / "a.jpg").unlink()
    assert stamp(data, ["a", "b"]) != before


def test_the_manifest_and_basis_still_count(data):
    """The checks that already worked must not have been lost in the change."""
    before = stamp(data, ["a", "b"])
    bump(Path(data) / "manifest.jsonl", b'{"id": "a"}\n{"id": "b"}\n')
    mid = stamp(data, ["a", "b"])
    assert mid != before
    bump(Path(data) / "basis.npz", b"basis-rebuilt")
    assert stamp(data, ["a", "b"]) != mid
