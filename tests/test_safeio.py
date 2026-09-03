"""A write that fails must leave the previous file untouched.

Checkpoints are written at the end of every epoch of an unattended overnight
run, and the runner is killed and relaunched by design. A torn write leaves a
file that exists, has a plausible size, and is wrong -- the failure shape this
project keeps paying for.

The test that matters is the third one. Round-tripping a file proves the happy
path and would pass just as well against a plain `torch.save`, which is exactly
the mistake made earlier in this project: a regression test that could not fail.
So the interrupted write is simulated directly, and the assertion is that the
old bytes survive and no debris is left in the directory.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import safeio  # noqa: E402


def test_torch_round_trip(tmp_path):
    p = tmp_path / "ck.pt"
    safeio.save_torch({"a": torch.arange(4), "b": "x"}, p)
    got = torch.load(p, map_location="cpu", weights_only=False)
    assert torch.equal(got["a"], torch.arange(4)) and got["b"] == "x"


def test_text_round_trip_is_utf8_whatever_the_platform_default(tmp_path):
    """cp1252 is the Windows default and has already produced mojibake here."""
    p = tmp_path / "r.md"
    text = "median −39.2 km — separated ± 0.4 · µ\n"
    safeio.write_text(p, text)
    assert p.read_text(encoding="utf-8") == text
    assert p.read_bytes().count(b"\r") == 0, "newline=\\n was not honoured"


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    """The point of the whole module: no half-written checkpoint."""
    p = tmp_path / "ck.pt"
    safeio.save_torch({"epoch": 1}, p)
    before = p.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(safeio.torch if hasattr(safeio, "torch") else torch,
                        "save", boom, raising=False)
    import torch as real_torch
    monkeypatch.setattr(real_torch, "save", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        safeio.save_torch({"epoch": 2}, p)

    assert p.read_bytes() == before, "the previous checkpoint was damaged"
    leftovers = [f.name for f in tmp_path.iterdir() if ".tmp" in f.name]
    assert not leftovers, "temp files left behind: {}".format(leftovers)


def test_a_failed_text_write_leaves_the_previous_file_intact(
        tmp_path, monkeypatch):
    """Fail at fsync, which is where a real full or failing disk gives out."""
    p = tmp_path / "r.md"
    safeio.write_text(p, "good\n")

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="disk full"):
        safeio.write_text(p, "replacement that must not land\n")
    assert p.read_text(encoding="utf-8") == "good\n"
    assert not [f for f in p.parent.iterdir() if ".tmp" in f.name]


def test_the_temp_file_shares_the_destination_directory(tmp_path):
    """os.replace is only atomic within one filesystem.

    Checkpoints live on C: and some caches on E:, so a temp file in the system
    temp directory would silently degrade to a copy -- which is the torn write
    this module exists to prevent.
    """
    p = tmp_path / "sub" / "ck.pt"
    p.parent.mkdir()
    assert safeio._tmp_for(p).parent == p.parent


def test_replace_cleans_up_when_the_rename_fails(tmp_path, monkeypatch):
    tmp = tmp_path / "x.tmp"
    tmp.write_bytes(b"junk")
    monkeypatch.setattr(os, "replace",
                        lambda *a: (_ for _ in ()).throw(OSError("cross-device")))
    with pytest.raises(OSError):
        safeio.replace_from(tmp, tmp_path / "x")
    assert not tmp.exists(), "temp file survived a failed rename"
