"""`check_merged` must not turn "I cannot tell" into a finding.

The script exists because a PR reading *merged* is not evidence its code
shipped. It can reproduce that same silence in two ways, and the first version
did both:

  * an unjudgeable commit counted as "in main", the original worry; and
  * a git ERROR counted as "not in main", which is worse -- it prints
    STRANDED and "replay it onto main" for content that is already there.

The second is `git merge-base --is-ancestor` exiting 0 for yes, 1 for no, and
anything else for an error. `scripts/mutate.py` shipped with exactly this bug
and #43 fixed it; this repeated it two days later.

The subtler case has no error at all. In a shallow clone `--is-ancestor` walks
a graph that stops at the graft boundary and exits **1** -- indistinguishable
from a real stranding. Review reproduced a false STRANDED that way, so the
last test here builds real repositories and does the same.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_merged as cm


def pr(n, sha, base="main"):
    return {"number": n, "title": "pr {}".format(n), "baseRefName": base,
            "mergeCommit": None if sha is None else {"oid": sha}}


# ----------------------------------------------------------- classify ------

def test_in_main_is_clean():
    stranded, unverified = cm.classify([pr(1, "a" * 40)],
                                       lambda s: (True, ""))
    assert stranded == []
    assert unverified == []


def test_not_an_ancestor_is_stranded():
    prs = [pr(1, "a" * 40), pr(2, "b" * 40, base="feature")]
    stranded, unverified = cm.classify(
        prs, lambda s: (not s.startswith("b"), ""))
    assert [p["number"] for p, _ in stranded] == [2]
    assert unverified == []


def test_unjudgeable_commit_is_not_silently_fine():
    stranded, unverified = cm.classify([pr(1, "c" * 40)],
                                       lambda s: (None, "no idea"))
    assert stranded == []
    assert [(p["number"], w) for p, w in unverified] == [(1, "no idea")]


def test_unjudgeable_commit_is_not_reported_as_stranded_either():
    """The direction review found: None must not fall through to False."""
    stranded, _ = cm.classify([pr(1, "c" * 40)], lambda s: (None, "no idea"))
    assert stranded == []


def test_missing_merge_commit_is_not_silently_fine():
    def never_called(sha):
        raise AssertionError("resolver called for a PR with no merge commit")

    stranded, unverified = cm.classify([pr(1, None)], never_called)
    assert stranded == []
    assert [p["number"] for p, _ in unverified] == [1]


# ------------------------------------------------------------ in_main ------

def fake_sh(monkeypatch, rc_by_cmd):
    """Drive `in_main` by what each git subcommand returns."""
    class P:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout, self.stderr = rc, out, ""

    def sh(argv, check=True):
        for key, val in rc_by_cmd.items():
            if key in " ".join(argv):
                return P(*val) if isinstance(val, tuple) else P(val)
        return P(0)

    monkeypatch.setattr(cm, "sh", sh)


def test_a_git_error_is_not_a_stranding(monkeypatch):
    """Exit 128 is 'I could not look', not 'it is not there'."""
    fake_sh(monkeypatch, {"cat-file": 0, "merge-base": 128})
    verdict, why = cm.in_main("d" * 40)
    assert verdict is None
    assert "128" in why


def test_exit_one_in_a_complete_history_is_a_stranding(monkeypatch):
    fake_sh(monkeypatch, {"cat-file": 0, "merge-base": 1,
                          "is-shallow-repository": (0, "false\n")})
    assert cm.in_main("d" * 40)[0] is False


def test_exit_one_in_a_shallow_clone_is_not_a_stranding(monkeypatch):
    fake_sh(monkeypatch, {"cat-file": 0, "merge-base": 1,
                          "is-shallow-repository": (0, "true\n")})
    verdict, why = cm.in_main("d" * 40)
    assert verdict is None
    assert "shallow" in why


def test_a_yes_from_a_shallow_clone_is_still_trusted(monkeypatch):
    """A path git CAN see exists; truncation only hides paths."""
    fake_sh(monkeypatch, {"cat-file": 0, "merge-base": 0,
                          "is-shallow-repository": (0, "true\n")})
    assert cm.in_main("d" * 40)[0] is True


# -------------------------------------------------- against real git ------

GIT = shutil.which("git")


def git(cwd, *args):
    p = subprocess.run([GIT, *args], cwd=str(cwd), capture_output=True,
                       text=True)
    assert p.returncode == 0, " ".join(args) + ": " + (p.stderr or p.stdout)
    return p.stdout.strip()


@pytest.fixture
def merged_repo(tmp_path):
    """An origin whose main contains a merge commit, then more commits.

    The extra commits matter: with `--depth 1` the merge commit falls beyond
    the graft boundary, which is the state that produced the false STRANDED.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--quiet", "--initial-branch=main")
    git(origin, "config", "user.email", "t@example.com")
    git(origin, "config", "user.name", "t")
    (origin / "a.txt").write_text("one\n")
    git(origin, "add", "-A")
    git(origin, "commit", "--quiet", "-m", "one")

    git(origin, "switch", "--quiet", "-c", "feature")
    (origin / "b.txt").write_text("two\n")
    git(origin, "add", "-A")
    git(origin, "commit", "--quiet", "-m", "two")

    git(origin, "switch", "--quiet", "main")
    git(origin, "merge", "--quiet", "--no-ff", "-m", "merge feature", "feature")
    merge_sha = git(origin, "rev-parse", "HEAD")

    for i in range(3):
        (origin / "c{}.txt".format(i)).write_text("more\n")
        git(origin, "add", "-A")
        git(origin, "commit", "--quiet", "-m", "more {}".format(i))
    return origin, merge_sha


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_a_shallow_clone_does_not_report_merged_content_as_stranded(
        merged_repo, tmp_path, monkeypatch):
    """The reported defect, end to end."""
    origin, merge_sha = merged_repo
    clone = tmp_path / "shallow"
    git(tmp_path, "clone", "--quiet", "--depth", "1", origin.as_uri(),
        str(clone))
    monkeypatch.chdir(clone)
    assert git(clone, "rev-parse", "--is-shallow-repository") == "true"

    # The commit IS in main. Without the guard this returned False.
    verdict, why = cm.in_main(merge_sha)
    assert verdict is not False, (
        "reported content that is in main as stranded: " + why)


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_deepening_lets_the_same_clone_answer(merged_repo, tmp_path,
                                              monkeypatch):
    origin, merge_sha = merged_repo
    clone = tmp_path / "shallow2"
    git(tmp_path, "clone", "--quiet", "--depth", "1", origin.as_uri(),
        str(clone))
    monkeypatch.chdir(clone)
    assert cm.deepen() is True
    assert git(clone, "rev-parse", "--is-shallow-repository") == "false"
    assert cm.in_main(merge_sha)[0] is True


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_a_genuinely_stranded_commit_is_still_found(merged_repo, tmp_path,
                                                    monkeypatch):
    """The guard must not blunt the check it exists to protect."""
    origin, _ = merged_repo
    # A commit on a side branch that main never took.
    git(origin, "switch", "--quiet", "-c", "orphan")
    (origin / "d.txt").write_text("stranded\n")
    git(origin, "add", "-A")
    git(origin, "commit", "--quiet", "-m", "never merged")
    orphan = git(origin, "rev-parse", "HEAD")

    clone = tmp_path / "full"
    git(tmp_path, "clone", "--quiet", origin.as_uri(), str(clone))
    monkeypatch.chdir(clone)
    assert git(clone, "rev-parse", "--is-shallow-repository") == "false"
    assert cm.in_main(orphan)[0] is False
