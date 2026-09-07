"""An exemption list may shrink. It may not grow, and it may not be swapped.

`LEGACY` shipped exempting all ten files that opened an archive, so the
ratchet it belonged to passed while enforcing nothing about the code that ran
-- it could only fail on a file that did not exist yet. That is the failure
mode every allowlist here has: it is written at the moment of least evidence,
when nothing has been migrated, and it is easiest to grow at exactly the
moment it should have stopped a change.

Each list carries a comment saying it may only get smaller. Two of the three
were then grown or left exempting everything anyway, because a comment is not
a check. This compares each list against the same list on the base branch:

  * it must not gain entries, and
  * every entry must already have been there.

The second half matters on its own -- a swap keeping the count identical is
the same defect wearing a different name, and this repository has now been
bitten three times by comparisons that only checked a count.

Skipped, not failed, where the base branch is unavailable: a working copy
without a remote is a legitimate way to run the suite, and a check that cannot
see its baseline must say so rather than pass quietly.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# (file, symbol) -> what the exemption means, for the failure message.
ALLOWLISTS = {
    ("tests/test_one_reader.py", "LEGACY"):
        "files that open a release archive without going through src/shards.py",
    ("tests/test_no_undefined_names.py", "KNOWN"):
        "names pyflakes reports as undefined",
    ("tests/test_no_undefined_names.py", "DEL_OK"):
        "scopes that delete a name more often than they bind it",
}


def base_ref():
    for ref in ("origin/main", "main"):
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            return ref
    return None


def literal(src, symbol, where):
    """The value of a module-level `symbol = <literal>` assignment."""
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == symbol for t in node.targets)):
            return set(ast.literal_eval(node.value))
    raise AssertionError("{} defines no {}".format(where, symbol))


@pytest.mark.parametrize(("path", "symbol"), sorted(ALLOWLISTS))
def test_the_allowlist_has_not_grown(path, symbol):
    ref = base_ref()
    if ref is None:
        pytest.skip("no base branch to compare against")
    r = subprocess.run(["git", "show", "{}:{}".format(ref, path)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("{} does not exist on {}".format(path, ref))

    was = literal(r.stdout, symbol, "{} on {}".format(path, ref))
    now = literal((ROOT / path).read_text(encoding="utf-8"), symbol, path)

    added = sorted(str(x) for x in now - was)
    assert not added, (
        "{} exempts {} more {} than {} does: {}.\n"
        "An exemption list is written when nothing has been migrated yet and "
        "is easiest to grow exactly when it should be refusing a change. If "
        "the new entry is genuinely correct rather than deferred, say so in "
        "the PR and move the baseline deliberately."
        .format(symbol, len(added), ALLOWLISTS[(path, symbol)], ref,
                ", ".join(added)))
