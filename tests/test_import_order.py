"""Modules in src/ are only importable after the sys.path insert.

Every script in scripts/ reaches the library by appending `src` to sys.path at
module level. An import of a src module placed *above* that line raises
ModuleNotFoundError the moment the script is run -- and nothing else catches it:
py_compile passes, pyflakes passes, and the whole test suite passes, because
none of them execute the script.

This is not hypothetical. Adding `import safeio` to seven files on 2026-09-03
put it in the stdlib block in three of them, alphabetically between `os` and
`subprocess`, which broke `overnight.py` -- the unattended runner. It was found
by running `--help` on every script, which is backlog item 15 and the reason
that item exists.

Running 57 subprocesses is too slow for this suite, and importing the modules is
not safe either: several do real work at module level (`probe_pool` loads a
checkpoint). So the check is static, which is enough to catch the ordering.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = {p.stem for p in (ROOT / "src").glob("*.py") if p.stem != "__init__"}
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def path_insert_line(tree):
    """Line of the `sys.path.insert(...)` that adds src, or None."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "insert"
                and isinstance(f.value, ast.Attribute) and f.value.attr == "path"):
            return node.lineno
    return None


def src_imports(tree):
    """(module, lineno) for every top-level import of a src/ module."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] in SRC:
                    out.append((n.name, node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] in SRC:
                out.append((node.module, node.lineno))
    return out


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.name for p in SCRIPTS])
def test_src_imports_come_after_the_path_insert(script):
    tree = ast.parse(script.read_text(encoding="utf-8", errors="replace"))
    imports = src_imports(tree)
    if not imports:
        return
    line = path_insert_line(tree)
    assert line is not None, (
        "{} imports {} from src/ but never extends sys.path"
        .format(script.name, ", ".join(m for m, _ in imports)))
    early = [(m, n) for m, n in imports if n < line]
    assert not early, (
        "{}: {} imported at line {} but sys.path gains src/ only at line {}; "
        "running this script raises ModuleNotFoundError".format(
            script.name, early[0][0], early[0][1], line))


def test_the_check_can_actually_fail(tmp_path):
    """A guard that cannot fire is not a guard."""
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nimport config\n"
                   'sys.path.insert(0, "src")\n', encoding="utf-8")
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    line = path_insert_line(tree)
    assert line == 3
    assert [(m, n) for m, n in src_imports(tree) if n < line] == [("config", 2)]
