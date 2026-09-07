"""`check_merged.classify` must not let an unjudgeable commit pass as fine.

The script exists because a PR reading *merged* is not evidence its code
shipped. The way it could reproduce that same silence is a merge commit it
cannot resolve -- a deleted branch whose object is not local and cannot be
fetched -- being counted as "in main" because nothing said otherwise.

These fix the three outcomes against a stub oracle, so they need no repository
and no network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_merged as cm


def pr(n, sha, base="main"):
    return {"number": n, "title": "pr {}".format(n), "baseRefName": base,
            "mergeCommit": None if sha is None else {"oid": sha}}


def test_in_main_is_clean():
    stranded, unverified = cm.classify([pr(1, "a" * 40)], lambda s: True)
    assert stranded == []
    assert unverified == []


def test_not_an_ancestor_is_stranded():
    prs = [pr(1, "a" * 40), pr(2, "b" * 40, base="feature")]
    stranded, unverified = cm.classify(prs, lambda s: not s.startswith("b"))
    assert [p["number"] for p, _ in stranded] == [2]
    assert unverified == []


def test_unjudgeable_commit_is_not_silently_fine():
    """The whole point: None must not read as True."""
    stranded, unverified = cm.classify([pr(1, "c" * 40)], lambda s: None)
    assert stranded == []
    assert [p["number"] for p, _ in unverified] == [1]


def test_missing_merge_commit_is_not_silently_fine():
    def never_called(sha):
        raise AssertionError("resolver called for a PR with no merge commit")

    stranded, unverified = cm.classify([pr(1, None)], never_called)
    assert stranded == []
    assert [p["number"] for p, _ in unverified] == [1]
