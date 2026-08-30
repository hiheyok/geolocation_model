"""Check every command in README.md against the scripts' real argparse surfaces.

A README that documents a flag which no longer exists is worse than one that
documents nothing, because it costs a reader more to discover the lie than to
have read the source in the first place. This extracts every command from the
fenced bash blocks, runs each script's --help, and checks that every flag used
actually exists -- plus that every file the prose names by path is really there.

Exits non-zero on any problem, so it can gate a commit.

    python scripts/check_readme.py
"""

import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def commands(text):
    """Every `python ...` line in a fenced bash block, continuations joined."""
    out = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").split("\n"):
            line = line.strip()
            if line.startswith("python "):
                out.append(line)
    return out


def main():
    text = io.open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    # a release must be set or config builds paths for the default one; the
    # scripts only need to reach --help, so any valid value does
    env = dict(os.environ, OSV_RELEASE=os.environ.get("OSV_RELEASE", "s10"))
    bad = 0

    for cmd in commands(text):
        parts = cmd.split()
        script, flags = parts[1], [p for p in parts[2:] if p.startswith("--")]
        path = os.path.join(ROOT, script.replace("/", os.sep))
        if not os.path.exists(path):
            print("MISSING SCRIPT  {}".format(script))
            bad += 1
            continue
        got = subprocess.run([PY, path, "--help"], capture_output=True,
                             text=True, cwd=ROOT, env=env)
        if got.returncode != 0:
            print("--help FAILED   {}\n{}".format(script, got.stderr[-300:]))
            bad += 1
            continue
        unknown = [f for f in flags if f not in got.stdout]
        if unknown:
            print("UNKNOWN FLAGS   {}  ->  {}".format(script, unknown))
            bad += 1
        else:
            print("ok  {:<32} {}".format(script, " ".join(flags) or "(no flags)"))

    for ref in sorted(set(re.findall(r"`([a-zA-Z0-9_]+/[a-zA-Z0-9_/]+\.(?:py|md|txt|pdf))`",
                                     text))):
        if not os.path.exists(os.path.join(ROOT, ref.replace("/", os.sep))):
            print("README names a path that does not exist: {}".format(ref))
            bad += 1

    print("\n{} problem(s)".format(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
