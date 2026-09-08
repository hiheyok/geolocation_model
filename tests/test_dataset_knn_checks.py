"""The dataset must ask both k-NN questions, and ask each of the right file.

`GeoStepDataset` ran `check_bytes` and never `check_ext` (REVIEW4 #1). It reads
the extension's z16 addresses to extend its own address tables, and never
checked that this is the extension the street file was built over. Every
extension on disk holds exactly 750,000 rows, so a mismatch leaves every
length equal and every neighbour index in range, and hands each matched
embedding another photograph's address -- as a training target.

Which file each check receives is the substance, not a detail. They are
different questions:

    check_ext    which photographs?   the JOINED bank records the stems
    check_bytes  which vectors?       the RETRIEVAL prefix carries the digest

Pointing both at one file is how `bank_for_checkpoint` broke -- redirecting to
the retrieval prefix to fix the digest stopped the extension check seeing the
joined bank, and a bank stacked from `bank_ext2` passed against a cache
addressing `bank_ext`. So this asserts the arguments, not that the calls
appear.

Static, because constructing a `GeoStepDataset` needs a release parquet,
targets, a street cache, a map cache and a k-NN cache; the repository's other
dataset tests exercise its pieces for the same reason. The behaviour of both
checks is covered in `test_knnmeta.py`; what is unproven anywhere else is that
the dataset wires them up.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SRC = (ROOT / "src" / "dataset.py").read_text(encoding="utf-8")


def knnmeta_calls():
    """{name: [rendered first argument, ...]} for every knnmeta.* call."""
    out = {}
    for node in ast.walk(ast.parse(SRC)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == "knnmeta" and node.args):
            out.setdefault(f.attr, []).append(ast.unparse(node.args[1])
                                              if len(node.args) > 1 else "")
    return out


def test_the_dataset_checks_extension_identity_at_all():
    calls = knnmeta_calls()
    assert "check_ext" in calls, (
        "dataset.py never calls check_ext, so a k-NN cache addressing one "
        "extension can be used with a street file stacked from another -- "
        "equal lengths, in-range indices, another photograph's coordinates")


def test_extension_identity_is_asked_of_the_joined_bank():
    """The joined file is what records which extensions it was stacked from."""
    args = knnmeta_calls()["check_ext"]
    assert any("_street_path" in a for a in args), (
        "check_ext is passed {}; it must receive the joined bank, which is "
        "self._street_path -- the retrieval prefix records no stems"
        .format(args))


def test_the_byte_digest_is_asked_of_the_retrieval_prefix():
    """`build_knn` digests the retrieval file, so that is what must match."""
    args = knnmeta_calls()["check_bytes"]
    assert any("want_sf" in a for a in args), (
        "check_bytes is passed {}; it must receive the file build_knn "
        "digested, which for a conditioned bank is the retrieval prefix"
        .format(args))


def test_the_two_checks_are_not_given_the_same_file():
    """The failure mode that produced this pair: one path serving two
    questions, so fixing one caller silently breaks the other."""
    ext = knnmeta_calls()["check_ext"]
    byt = knnmeta_calls()["check_bytes"]
    assert set(ext).isdisjoint(byt), (
        "check_ext and check_bytes receive the same argument {}; they ask "
        "different questions of different files".format(set(ext) & set(byt)))
