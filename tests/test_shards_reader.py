"""What `src/shards.py` promises about reading archives, tested by running it.

`test_one_reader.py` is a static ratchet: it enforces that only this module
opens an archive, which is a rule about imports and cannot see what the module
then does. That gap let a real defect through -- `iter_blobs` sliced its
shard-sorted members into fixed batches and called `blobs` per batch, so a
shard was reopened once per batch and the sequential-read threshold, judged
from each batch, could never be reached. Both claims were in the docstring.

These tests run the reader against real archives on disk.
"""

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import shards  # noqa: E402


@pytest.fixture
def zips(tmp_path):
    """Two shards: `aa` with 700 members, `bb` with 5. Contents encode their
    own name, so a mispaired blob is detectable rather than merely wrong."""
    plan = {"aa": 700, "bb": 5}
    members = []
    for shard, n in plan.items():
        with zipfile.ZipFile(tmp_path / (shard + ".zip"), "w") as z:
            for k in range(n):
                name = "{}/{:05d}.jpg".format(shard, k)
                z.writestr(name, ("BYTES:" + name).encode())
                members.append(name)
    return tmp_path, members


def counted(monkeypatch):
    """Wrap `shards.archive`, recording (shard, preload) per open."""
    opens = []
    real = shards.archive

    def spy(shard, root=None, preload=True, quiet=False):
        opens.append((shard, preload))
        return real(shard, root=root, preload=preload, quiet=quiet)

    monkeypatch.setattr(shards, "archive", spy)
    return opens


def test_iter_blobs_opens_each_shard_once(zips, monkeypatch):
    """700 members at batch=64 is 11 batches from one archive, not 11 opens."""
    root, members = zips
    opens = counted(monkeypatch)
    batches = list(shards.iter_blobs(members, batch=64, root=root, quiet=True))

    assert [s for s, _ in opens] == ["aa", "bb"]
    assert len(batches) == (700 + 63) // 64 + 1


def test_the_slurp_decision_sees_the_shard_not_the_batch(zips, monkeypatch):
    """The defect that made the preload unreachable: with min_slurp above the
    batch size but below the shard's contribution, the read must still be
    sequential."""
    root, members = zips
    opens = counted(monkeypatch)
    list(shards.iter_blobs(members, batch=64, min_slurp=500, root=root,
                           quiet=True))
    assert dict(opens) == {"aa": True, "bb": False}


def test_every_member_arrives_exactly_once_and_paired(zips):
    """Indices pair positionally with blobs; losing that pairs each image with
    another image's label."""
    root, members = zips
    seen = {}
    for sel, got in shards.iter_blobs(members, batch=64, root=root, quiet=True):
        assert len(sel) == len(got)
        for i, b in zip(sel, got):
            assert i not in seen
            seen[i] = b
    assert len(seen) == len(members)
    for i, m in enumerate(members):
        assert seen[i] == ("BYTES:" + m).encode()


def test_batches_do_not_span_shards(zips):
    root, members = zips
    for sel, _ in shards.iter_blobs(members, batch=64, root=root, quiet=True):
        assert len({shards.shard_of(members[i]) for i in sel}) == 1


def test_blobs_returns_caller_order(zips):
    """Even when the archive is read in its own order underneath."""
    root, members = zips
    pick = [members[9], members[703], members[1], members[400]]
    assert shards.blobs(pick, root=root, min_slurp=2, quiet=True) == \
        [("BYTES:" + m).encode() for m in pick]


def test_a_missing_member_is_named(zips):
    root, members = zips
    with pytest.raises(SystemExit, match="aa/99999.jpg is not in shard aa"):
        shards.blobs(["aa/99999.jpg"], root=root, quiet=True)


def test_a_missing_shard_is_refused(zips):
    root, _ = zips
    with pytest.raises(SystemExit, match="shard zz is not on disk"):
        shards.blobs(["zz/00000.jpg"], root=root, quiet=True)
