"""Put a defect back and check the test notices.

A test written after a bug is fixed usually passes. That says nothing: it also
passes if it asserts the wrong thing, stubs the code under test, or checks a
property next to the one that broke. Every such test in this repository was
green when it was written.

The ones that shipped anyway, all found by review:

  * `test_bank_for_checkpoint`'s fixture stubs `check_bytes` for the whole
    file, so the test named "compares against its retrieval prefix" asserted
    the filename half and was structurally blind to the byte half -- which was
    the broken half.
  * the forward-parity test compared a model with retrieval against one
    without, on a network whose retrieval gates are all zero-initialised. The
    two are identical by construction and the test could not fail.
  * a test for recomputing a bank stubbed the function that recomputed it, and
    stayed green while every call on real data raised NameError.

So the question is not "does the test pass" but "can it fail". This applies
the defect, runs the test, and requires a failure:

    py scripts/mutate.py --file src/knnmeta.py --test tests/test_knnmeta.py \\
        --old "bytes_path or street_path" --new "street_path"

    py scripts/mutate.py --file scripts/knn_gap.py --test tests/x.py \\
        --old-file old.txt --new-file new.txt          # multi-line

The file is restored from the bytes read at the start, in a `finally`, and the
restoration is verified before exit -- a helper that edits your source has to
be more careful about putting it back than about anything else it does.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# https://docs.pytest.org/en/stable/reference/exit-codes.html
PYTEST_EXIT = {
    0: "all tests passed",
    1: "tests failed",
    2: "interrupted",
    3: "internal error",
    4: "usage error -- a path or test id that does not exist",
    5: "no tests were collected",
}


def digest(b):
    return hashlib.sha256(b).hexdigest()[:12]


def read_arg(inline, path, what):
    if (inline is None) == (path is None):
        raise SystemExit(
            "give exactly one of --{0} or --{0}-file".format(what))
    if inline is not None:
        return inline
    return Path(path).read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="source file to mutate")
    ap.add_argument("--test", required=True, nargs="+",
                    help="pytest targets that must fail")
    ap.add_argument("--old", help="text to replace; must occur exactly once")
    ap.add_argument("--old-file", help="--old, read from a file")
    ap.add_argument("--new", help="what to put in its place")
    ap.add_argument("--new-file", help="--new, read from a file")
    ap.add_argument("--expect", default="fail", choices=("fail", "pass"),
                    help="'pass' asserts the mutation is NOT caught, which is "
                         "how a known-harmless edit gets recorded as such "
                         "rather than argued about")
    a = ap.parse_args()

    old = read_arg(a.old, a.old_file, "old")
    new = read_arg(a.new, a.new_file, "new")
    path = Path(a.file)
    original = path.read_bytes()
    before = digest(original)
    # Match on LF regardless of what is on disk. Every file in this repo is
    # CRLF, and a snippet copied out of an editor or read with `read_text` is
    # LF, so a byte-exact comparison finds nothing and the tool refuses a
    # mutation that is in fact present. The mutated file is written back in
    # the line endings it had; the restore uses the original bytes either way.
    CRLF, LF = chr(13) + chr(10), chr(10)
    crlf = CRLF in original.decode("utf-8")
    text = original.decode("utf-8").replace(CRLF, LF)
    old = old.replace(CRLF, LF)
    new = new.replace(CRLF, LF)

    n = text.count(old)
    if n != 1:
        raise SystemExit(
            "--old occurs {} times in {}; it must occur exactly once so the "
            "mutation is the one you meant".format(n, path))
    if old == new:
        raise SystemExit("--old and --new are the same; nothing would change")

    print("mutating   {}  ({} -> applying defect)".format(path, before),
          flush=True)
    try:
        out = text.replace(old, new)
        if crlf:
            out = out.replace(LF, CRLF)
        path.write_bytes(out.encode("utf-8"))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", *a.test],
                           cwd=ROOT)
        code = r.returncode
    finally:
        path.write_bytes(original)
        after = digest(path.read_bytes())
        if after != before:
            raise SystemExit(
                "FAILED TO RESTORE {}: was {}, now {}. The original bytes are "
                "lost from this process; recover with git."
                .format(path, before, after))
        print("restored   {}  ({})".format(path, after), flush=True)

    # Only exit 1 means a test failed. Treating every nonzero code as "the
    # defect was caught" made a mistyped path -- exit 4, "file or directory not
    # found" -- report success, and 5, "no tests ran", do the same. A tool that
    # exists to catch tests which cannot fail must not itself pass on a suite
    # that never ran.
    if code not in (0, 1):
        raise SystemExit(
            "pytest exited {} ({}), so nothing was measured. The file was "
            "restored; fix the invocation and run it again."
            .format(code, PYTEST_EXIT.get(code, "unknown")))
    caught = code == 1

    want_caught = a.expect == "fail"
    if caught == want_caught:
        print("\nOK: the defect was {} by {}".format(
            "caught" if caught else "not caught (as expected)",
            " ".join(a.test)))
        return 0
    if want_caught:
        raise SystemExit(
            "\nTHE TEST DID NOT FAIL.\n\n{} passes with the defect applied, so "
            "it does not test what it is named after. A test written after a "
            "fix has to be shown to fail without it."
            .format(" ".join(a.test)))
    raise SystemExit(
        "\nThe test failed on an edit expected to be harmless, so either the "
        "edit is not harmless or the test is over-specified.")


if __name__ == "__main__":
    sys.exit(main())
