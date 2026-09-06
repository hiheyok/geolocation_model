"""Undefined names are crashes, not style.

Two P1 findings on this branch were one defect twice: a refactor removed a
local's assignment and left `del <name>` behind, so the script raised
`UnboundLocalError` at the end of the first shard -- after the expensive work
and before it was saved. Neither the suite nor a smoke test of the new helper
saw it, because both exercised the helper rather than the converted caller.

`pyflakes` reports the first of those and not the second, so both checks are
here. It sees a name that is never bound in its scope (`del source`), but not
one bound only on a branch that may not run: `tile_cache.main` binds an
unrelated `src` under `if grew:` and deletes it two lines later, so a
function-level binding exists and pyflakes stays quiet while every actual run
raises. `test_no_name_is_deleted_more_often_than_it_is_bound` covers that
shape -- deleting a name more times than anything assigns it cannot be right
on any path.

`KNOWN` may only shrink. Three of its four entries are false positives --
pyflakes does not follow a name bound by an enclosing `with` or an outer
function into a nested one. The fourth is a real bug, recorded rather than
silenced.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyflakes")

ROOT = Path(__file__).resolve().parent.parent

KNOWN = {
    # closure over a name bound by `with ... as zf` -- pyflakes limitation
    ("embed_native.py", "zf"),
    # closure over `net`, bound in the enclosing function
    ("pyramid_cache.py", "net"),
    # closure over `encs`, bound in the enclosing function
    ("tilebench.py", "encs"),
    # REAL: `prov.` is called four times and `provenance` is never imported,
    # so these paths raise NameError. Fixed separately -- it is not this
    # change, and it is listed here so it cannot be forgotten.
    ("multiquery.py", "prov"),
}


def undefined_names():
    out = subprocess.run(
        [sys.executable, "-m", "pyflakes", "scripts", "src"],
        cwd=ROOT, capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        if "undefined name" not in line:
            continue
        path, _, rest = line.partition(":")
        found.add((Path(path).name, rest.split("'")[1]))
    return found


def test_no_new_undefined_names():
    found = undefined_names()
    new = found - KNOWN
    assert not new, (
        "undefined names, which raise at runtime: "
        + ", ".join("{} in {}".format(n, f) for f, n in sorted(new)))


def test_the_known_list_is_accurate():
    """A fixed entry must leave, or the list stops meaning anything."""
    found = undefined_names()
    stale = KNOWN - found
    assert not stale, (
        "no longer reported, so remove from KNOWN: "
        + ", ".join("{} in {}".format(n, f) for f, n in sorted(stale)))


def deletes_exceeding_bindings():
    """(file, function, name) where a scope deletes a name more often than it
    binds it. The second `del` can only run on a name that is already gone."""
    import ast
    from collections import Counter

    bad = []
    for path in sorted((ROOT / "scripts").glob("*.py")) +             sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            stores, dels = Counter(), Counter()
            for node in ast.walk(fn):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store):
                        stores[node.id] += 1
                    elif isinstance(node.ctx, ast.Del):
                        dels[node.id] += 1
            for name, k in dels.items():
                if k > stores[name]:
                    bad.append((path.name, fn.name, name))
    return bad


# Verified correct: the two `del encs` sit on mutually exclusive paths -- the
# first is followed by `continue` -- and `encs` is rebound at the top of every
# iteration. The count cannot see either fact. One exemption is the price of
# catching the shape pyflakes misses.
DEL_OK = {("tilebench.py", "main", "encs")}


def test_no_name_is_deleted_more_often_than_it_is_bound():
    bad = [b for b in deletes_exceeding_bindings() if b not in DEL_OK]
    assert not bad, (
        "deleted more often than bound, so a del runs on a name that is "
        "already gone: "
        + ", ".join("{} in {}.{}".format(n, f, fn) for f, fn, n in bad))
