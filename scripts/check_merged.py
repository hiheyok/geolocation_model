"""Check that every merged pull request's content actually reached main.

A PR can read *merged* and have shipped nothing. It happens when the PR
targets another branch -- the usual reason being that the work was stacked on
a change still in review -- and that base branch reaches `main` BEFORE the
child merges into it. The child then lands in a branch nothing merges again.
Github shows "Merged" in purple either way, and there is no warning.

This has happened three times in this repository:

    #41  based on #30's branch            replayed as #52
    #33  based on `central-shard-reader`  replayed as #53
    #34  based on `kartaview-packs`       replayed as #53

#33 is the clearest: #32 carried `central-shard-reader` into `main` at 22:08
and #33 merged into that same branch at 23:44, an hour and a half after it had
stopped being a path to anywhere. Two rounds of review went into it. Nobody
noticed for a day, and only then because a state file said the chain had never
been checked.

**The test is ancestry, not the base branch.** Stacking on another branch is
fine and normal -- it is what lets a dependent change be reviewed early. What
matters is whether the merge commit ended up on `main` in the end, which
`git merge-base --is-ancestor` answers in a second:

    git merge-base --is-ancestor <mergeCommit> origin/main

Checking `baseRefName == "main"` instead would flag every healthy stack and
miss nothing extra, and checking the PR's own "merged" flag is what got us
here.

**A commit that cannot be checked is reported, never passed.** When a PR's
branch has been deleted its merge commit may not be in the local object store,
and `--is-ancestor` cannot answer. Treating that as "fine" would reintroduce
exactly the silence this exists to break, so it is fetched on demand and, if
that fails, listed separately with a non-zero exit.

Exits non-zero if anything is stranded or unverifiable, so it can gate a
commit or a release.

    python scripts/check_merged.py               # the last 50 merged PRs
    python scripts/check_merged.py --limit 200
"""

import argparse
import json
import subprocess
import sys

MAIN = "origin/main"


def sh(argv, check=True):
    p = subprocess.run(argv, capture_output=True, text=True)
    if check and p.returncode:
        sys.exit("{} failed: {}".format(" ".join(argv[:3]),
                                        (p.stderr or p.stdout).strip()))
    return p


def merged_prs(limit):
    p = sh(["gh", "pr", "list", "--state", "merged", "--limit", str(limit),
            "--json", "number,title,baseRefName,mergeCommit"])
    return json.loads(p.stdout)


def have(sha):
    return sh(["git", "cat-file", "-e", sha + "^{commit}"],
              check=False).returncode == 0


def in_main(sha):
    """True, False, or None when the object is not available to judge."""
    if not have(sha):
        # The branch is usually deleted after a merge, so the commit may be
        # unreachable from any ref even though it is on the remote.
        sh(["git", "fetch", "--quiet", "origin", sha], check=False)
        if not have(sha):
            return None
    return sh(["git", "merge-base", "--is-ancestor", sha, MAIN],
              check=False).returncode == 0


def classify(prs, resolver):
    """Split merged PRs into (stranded, unverified) given an ancestry oracle.

    `resolver(sha) -> True | False | None`, where None means the object could
    not be judged. Separated from the subprocess plumbing so the part that can
    fail silently -- an unjudgeable commit quietly counting as fine -- is
    testable without a repository or a network.
    """
    stranded, unverified = [], []
    for pr in prs:
        sha = (pr.get("mergeCommit") or {}).get("oid")
        if not sha:
            unverified.append((pr, "no merge commit recorded"))
            continue
        got = resolver(sha)
        if got is None:
            unverified.append(
                (pr, "merge commit {} not available".format(sha[:8])))
        elif not got:
            stranded.append((pr, sha))
    return stranded, unverified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50,
                    help="how many recently merged PRs to check")
    a = ap.parse_args()

    sh(["git", "fetch", "--quiet", "origin", "main"])
    prs = merged_prs(a.limit)
    stranded, unknown = classify(prs, in_main)

    print("checked {} merged PRs against {}".format(len(prs), MAIN))
    for pr, sha in stranded:
        print("\nSTRANDED  #{}  {}".format(pr["number"], pr["title"]))
        print("  merged into {!r}, and {} is not an ancestor of {}."
              .format(pr["baseRefName"], sha[:8], MAIN))
        print("  Its content is not in main. Replay it onto main.")
    for pr, why in unknown:
        print("\nUNVERIFIED  #{}  {}\n  {}".format(
            pr["number"], pr["title"], why))

    if not stranded and not unknown:
        print("all in main")
        return 0
    # Unverified is a failure too. A check that goes quiet when it cannot see
    # is the same silence this exists to break.
    print("\n{} stranded, {} unverified".format(len(stranded), len(unknown)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
