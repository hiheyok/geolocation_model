"""An extension embedding file must be named so its metadata can be found.

`provenance.ext_stem_for` locates the metadata by stripping underscore-parts
from the RIGHT: `bank_ext_dual` -> `bank_ext`. So the name has to read
`<ext_stem>_<scheme>`, stem first.

`tilebig.py` named its two arms `pyr_l0_ext` and `pyr_l0l1_ext` -- stem last.
`ext_stem_for` returned None, `stack_bank` refused both arms with "no metadata
found for extension", and 24 stages failed *after* a 233-minute tile pass had
already succeeded. Nothing before this test could see it, because the failure
needs a real run to reach the stacking stage.

Static, like `test_one_reader`: `tilebig` does work at import time.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import provenance as prov            # noqa: E402

# Every bank extension in the project is a `bank_ext*` stem; the metadata file
# is `<stem>_meta.npz`. A name that does not begin with one cannot resolve.
PREFIX = "bank_ext"


def arms_of(path):
    """{arm: extension embedding stem}, read without importing the module.

    Not `literal_eval`: the ARMS values reference module constants such as
    EXT_TILES, so only the second element -- the one this test is about -- is
    pulled out, and only when it is a plain string.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "ARMS" for t in node.targets)):
            continue
        out = {}
        for k, v in zip(node.value.keys, node.value.values):
            second = v.elts[1]
            assert isinstance(second, ast.Constant), (
                "ARMS[{!r}] names its extension file indirectly; this test "
                "reads it statically".format(k.value))
            out[k.value] = second.value
        return out
    return None


def test_tilebig_extension_names_are_stem_first():
    arms = arms_of(ROOT / "scripts" / "tilebig.py")
    assert arms, "tilebig.py no longer defines an ARMS literal"
    for arm, ext_file in arms.items():
        assert ext_file.startswith(PREFIX), (
            "{}: extension embeddings are named {!r}, which ext_stem_for "
            "cannot resolve -- it strips suffixes from the right, so the "
            "stem must come first, e.g. {}_{}".format(
                arm, ext_file, PREFIX, arm.lower()))


def test_ext_stem_for_strips_from_the_right(tmp_path, monkeypatch):
    """The rule the naming convention exists to satisfy, stated directly."""
    import config

    monkeypatch.setattr(config, "STREET_CACHE", tmp_path)
    (tmp_path / "bank_ext_meta.npz").write_bytes(b"")

    assert prov.ext_stem_for("bank_ext_dual") == "bank_ext"
    assert prov.ext_stem_for("bank_ext_pyrl0l1") == "bank_ext"
    assert prov.ext_stem_for("bank_ext") == "bank_ext"
    # stem last: unresolvable, which is exactly what tilebig shipped
    assert prov.ext_stem_for("pyr_l0_ext") is None
