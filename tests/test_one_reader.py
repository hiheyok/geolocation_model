"""Only `src/shards.py` may open a release archive.

`embed_street.slurp` existed, was exported, and was adopted by four scripts.
`resmatch` still opened `zipfile.ZipFile(path)` on the raw file and read
members by name -- 43,000 seeks at 178 reads/s on a 5900 RPM drive, 6.6 MB/s
against 300 MB/s sequential, with the GPU idle throughout.

Nothing failed. That is the point of this file. An exported helper is a
suggestion, and this project's record says suggestions get left behind:
`fuse_flat` was shared while three callers wrote out the scoring composition;
`knnmeta.check` was shared while both consumers passed arguments that
activated none of its checks; `preprocess` was imported by `query_only`, which
then wrote its own loop and dropped the prefetch pool.

So the reader is enforced rather than offered. `LEGACY` lists the files that
predate `src/shards.py` and **may only shrink** -- delete an entry when that
file is migrated. A new file calling `ZipFile` fails here, which is the only
mechanism that has ever stopped this recurring.

Static, like `test_import_order`: several of these modules do real work at
import time, so they are parsed rather than run.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FILES = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").glob("*.py"))

# The one module allowed to open an archive.
READER = "shards.py"

# Files that predate the reader. This set may only get smaller. Adding to it
# is what this test exists to prevent, so a PR that grows it should be
# rejected rather than accepted with a note.
LEGACY = {
    "build_bank_ext.py", "build_dataset.py", "embed_native.py",
    "embed_street.py", "ocr_probe.py", "query_only.py", "resmatch.py",
    "tile_cache.py", "tile_probe.py", "tilebench.py",
}


def opens_archive(path):
    """Calls to `ZipFile(...)`, however the module was imported."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = (f.attr if isinstance(f, ast.Attribute)
                else f.id if isinstance(f, ast.Name) else "")
        if name == "ZipFile":
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_no_new_module_opens_an_archive(path):
    if path.name == READER or path.name in LEGACY:
        return
    hits = opens_archive(path)
    assert not hits, (
        "{} opens an archive directly at line(s) {}. Use `shards.blobs`: it "
        "groups members by shard and reads a shard sequentially when enough "
        "members come from it, which is ~46x faster than seeking per member "
        "on this machine's HDD. Reading by name is how resmatch spent four "
        "minutes per pass with the GPU idle."
        .format(path.name, ", ".join(str(h) for h in hits)))


def test_the_legacy_list_is_accurate():
    """A stale allowlist is worse than none -- it reads as approval.

    Every name in LEGACY must still open an archive. When one stops, the entry
    is dead and must be deleted, or the next file to regress could be added
    under cover of "it was already there".
    """
    stale = sorted(n for n in LEGACY
                   if not any(p.name == n and opens_archive(p) for p in FILES))
    assert not stale, (
        "these no longer open an archive and must be removed from LEGACY: {}"
        .format(", ".join(stale)))


def test_the_reader_itself_is_the_only_exception():
    assert (ROOT / "src" / READER).exists()
    assert READER not in LEGACY
