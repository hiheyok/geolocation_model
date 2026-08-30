"""Check every command in README.md against the scripts' real argparse surfaces.

A README that documents a flag which no longer exists is worse than one that
documents nothing, so this is worth automating rather than eyeballing.
"""
import io
import re
import subprocess
import sys
import os

ROOT = r"C:\Users\longd\Programming\geolocation_model"
PY = sys.executable

s = io.open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()

# pull python commands out of the fenced blocks, joining backslash continuations
cmds = []
for block in re.findall(r"```bash\n(.*?)```", s, re.S):
    joined = block.replace("\\\n", " ")
    for line in joined.split("\n"):
        line = line.strip()
        if line.startswith("python "):
            cmds.append(line)

env = dict(os.environ, OSV_RELEASE="s10")
bad = 0
for c in cmds:
    parts = c.split()
    script = parts[1]
    flags = [p for p in parts[2:] if p.startswith("--")]
    path = os.path.join(ROOT, script.replace("/", os.sep))
    if not os.path.exists(path):
        print("MISSING SCRIPT  {}".format(script))
        bad += 1
        continue
    out = subprocess.run([PY, path, "--help"], capture_output=True, text=True,
                         cwd=ROOT, env=env)
    if out.returncode != 0:
        print("--help FAILED   {}\n{}".format(script, out.stderr[-300:]))
        bad += 1
        continue
    help_text = out.stdout
    unknown = [f for f in flags if f not in help_text]
    if unknown:
        print("UNKNOWN FLAGS   {}  ->  {}".format(script, unknown))
        bad += 1
    else:
        print("ok  {:<34} {}".format(script, " ".join(flags) or "(no flags)"))

# every file the README names should exist
for ref in sorted(set(re.findall(r"`([a-zA-Z0-9_/]+\.(?:py|md|txt|pdf))`", s))):
    p = os.path.join(ROOT, ref.replace("/", os.sep))
    if not os.path.exists(p):
        print("README names a file that does not exist: {}".format(ref))
        bad += 1

print("\n{} problem(s)".format(bad))
