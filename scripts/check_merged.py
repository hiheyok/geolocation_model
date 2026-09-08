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

**A commit that cannot be checked is reported, never guessed -- in either
direction.** There are three outcomes, not two. `--is-ancestor` exits 0 for
yes, 1 for no, and anything else for an error, and the first version of this
collapsed every nonzero into "no" -- the same defect `scripts/mutate.py`
shipped with and #43 fixed, repeated here two days later, where it turns "I
cannot tell" into "replay it onto main".

Two ways the answer is unavailable:

  * **The object is missing.** A merged branch is usually deleted, so its
    merge commit may not be in the local store. It is fetched on demand.
  * **The history is truncated.** In a shallow clone `--is-ancestor` walks a
    graph that stops at the graft boundary and reports exit 1 -- indistinguish-
    able from a real stranding. Review reproduced STRANDED, with "replay it
    onto main", for content already in `main`. So the clone is deepened first,
    and if it cannot be, a "no" is downgraded to UNVERIFIED. A "yes" from a
    shallow clone is still trustworthy: a path git can see does exist.

Anything unverifiable exits non-zero, so it is never quietly fine.

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


def shallow():
    """Is the local history truncated?

    A shallow clone can answer "not an ancestor" about a commit that IS one:
    `--is-ancestor` walks the graph, and beyond the graft boundary there is no
    graph to walk. The answer is exit 1, identical to a real non-ancestry.
    """
    return sh(["git", "rev-parse", "--is-shallow-repository"],
              check=False).stdout.strip() == "true"


def deepen():
    """Complete a shallow history so ancestry can be decided, if we can.

    Done once, before any single-commit fetch: fetching a merge commit into a
    shallow repository gets the object without connecting it to `main`, which
    is the exact state that produces a confident wrong answer.
    """
    if not shallow():
        return True
    sh(["git", "fetch", "--quiet", "--unshallow", "origin", "main"],
       check=False)
    return not shallow()


def in_main(sha):
    """`(verdict, why)` -- True in main, False stranded, None cannot tell.

    Three outcomes, not two, and the distinction is the whole point of the
    script. `git merge-base --is-ancestor` exits **0** for yes, **1** for no,
    and **anything else for an error** -- a missing object, a bad ref, a
    corrupt repository. Collapsing every nonzero into "no" is what
    scripts/mutate.py shipped with and #43 fixed; this had the same bug two
    days later, and here it turns "I cannot tell" into "replay it onto main",
    which is worse than silence.
    """
    if not have(sha):
        # The branch is usually deleted after a merge, so the commit may be
        # unreachable from any ref even though it is on the remote.
        sh(["git", "fetch", "--quiet", "origin", sha], check=False)
        if not have(sha):
            return None, "merge commit {} could not be fetched".format(sha[:8])
    rc = sh(["git", "merge-base", "--is-ancestor", sha, MAIN],
            check=False).returncode
    if rc == 0:
        return True, ""
    if rc != 1:
        return None, ("git merge-base exited {} for {}, which is an error, "
                      "not an answer".format(rc, sha[:8]))
    # rc == 1 means "not reachable". In a complete history that is the
    # finding. In a truncated one it may only mean the path was never
    # fetched, so it is not reported as a stranding.
    if shallow():
        return None, ("{} is not reachable from {}, but this clone is still "
                      "shallow, so the path between them may simply be "
                      "missing".format(sha[:8], MAIN))
    return False, ""


def classify(prs, resolver):
    """Split merged PRs into (stranded, unverified) given an ancestry oracle.

    `resolver(sha) -> (True | False | None, why)`, where None means the
    question could not be answered and `why` says which way it failed.
    Separated from the subprocess plumbing so the part that can fail silently
    -- an unjudgeable commit quietly counting as fine, or as stranded -- is
    testable without a repository or a network.
    """
    stranded, unverified = [], []
    for pr in prs:
        sha = (pr.get("mergeCommit") or {}).get("oid")
        if not sha:
            unverified.append((pr, "no merge commit recorded"))
            continue
        verdict, why = resolver(sha)
        if verdict is None:
            unverified.append((pr, why))
        elif not verdict:
            stranded.append((pr, sha))
    return stranded, unverified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50,
                    help="how many recently merged PRs to check")
    a = ap.parse_args()

    sh(["git", "fetch", "--quiet", "origin", "main"])
    if not deepen():
        print("warning: this clone is shallow and could not be deepened; "
              "commits it cannot connect to {} are reported UNVERIFIED "
              "rather than stranded".format(MAIN))
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
