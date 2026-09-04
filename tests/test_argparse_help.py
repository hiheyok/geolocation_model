"""`--help` must not crash, and a bare `%` in a help string makes it crash.

argparse runs every help string through `%`-formatting, so a percent sign meant
literally -- "the forward being 96% of the time" -- is read as a format spec
and `--help` dies with a TypeError from inside argparse.  Nothing else notices:
the script imports, compiles, and runs correctly for every invocation that does
not ask for help.

`scripts/tile_cache.py` carried exactly that for as long as it has existed, and
it was found only because a sweep ran `--help` over every script.  A sweep is
too slow for this suite (`test_import_order` explains why), but the defect is
visible in the source, so this checks it there.

Valid forms are `%%` for a literal and `%(default)s` for a substitution.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src").glob("*.py"))

SUBST = re.compile(r"%\([A-Za-z_]+\)[srdfgeoxX]")


def bad_percents(text):
    """Positions of every % argparse would try to interpolate and fail on.

    A scan, not a lookahead: in "96%% of" the second % is the escaped half of
    a valid pair, and a lookahead regex flags it. Consuming the pair is the
    only way to tell the two apart.
    """
    out, i = [], 0
    while i < len(text):
        if text[i] != "%":
            i += 1
            continue
        if text.startswith("%%", i):
            i += 2
            continue
        m = SUBST.match(text, i)
        if m:
            i = m.end()
            continue
        out.append(i)
        i += 1
    return out


def help_strings(tree):
    """(help text, lineno) for every add_argument(help=...) literal."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                val = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                continue          # an f-string or a name; argparse sees a str
            if isinstance(val, str):
                out.append((val, kw.value.lineno))
    return out


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_help_strings_escape_their_percent_signs(path):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    bad = [(lineno, txt) for txt, lineno in help_strings(tree)
           if bad_percents(txt)]
    assert not bad, (
        "{}: argparse formats every help string, so a literal percent must be "
        "written %% or --help raises TypeError from inside argparse:\n{}".format(
            path.name,
            "\n".join("  line {}: {}".format(n, t[:90]) for n, t in bad)))


def test_the_checker_recognises_the_forms_argparse_accepts():
    """The scanner is the whole test, so pin what it does and does not flag."""
    assert bad_percents("96% of the time")
    assert bad_percents("a % b")
    assert bad_percents("%(default)s and 40% more")
    assert not bad_percents("96%% of the time")
    assert not bad_percents("default: %(default)s")
    assert not bad_percents("%%%%")
    assert not bad_percents("no percent here at all")
