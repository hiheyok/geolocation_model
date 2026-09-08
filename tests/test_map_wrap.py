"""The demo map's overlays must follow the tiles across the antimeridian.

Web Mercator repeats east-west. `drawMap`'s tile loop already draws that -- it
positions tile `tx` at `tx*T - ox` for `tx` outside `[0, n)` and wraps only the
URL. The overlays did not: `wpx` returns a coordinate in the canonical
`[0, world)` copy, so a candidate at -179.999 beside a prediction at 179.999
was placed a whole world away, **8,388,608 px at z14**, off-screen. Panning
east past the antimeridian did the same to the pin and the confidence ring,
because `cx` grew without bound while the overlays stayed in copy zero.

Found by review on PR #54, and it had no test because nothing in this
repository tested browser JavaScript at all. That is the gap this closes: the
arithmetic in `index.html` is pure, so it can be extracted and run under node
without a DOM.

**The functions are read out of the shipped file, not copied here.** A copy
would pass forever after `index.html` drifted away from it -- which is the
"tests that could not fail" shape this project keeps hitting. The block is
delimited by sentinel comments in `index.html`; if they are renamed this
test fails loudly rather than silently testing nothing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "scripts" / "static" / "index.html"
START = "/* ==== pure map arithmetic; tests/test_map_wrap.py extracts this block ==== */"
END = "/* ==== end pure map arithmetic ==== */"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is not installed")


def helpers():
    """The pure block of index.html, or a failure naming what changed."""
    src = PAGE.read_text(encoding="utf-8")
    if START not in src or END not in src:
        pytest.fail(
            "the sentinel comments around the pure map arithmetic are gone "
            "from {}; this test extracts that block rather than keeping a "
            "copy, so it cannot run".format(PAGE.name))
    return src[src.index(START) + len(START):src.index(END)]


def run(body, expr):
    """Evaluate `expr` in node with the page's helpers and a MAP stub."""
    js = "const MAP = {{ T: 512 }};\n{}\nconsole.log(JSON.stringify({}));".format(
        body, expr)
    p = subprocess.run([shutil.which("node"), "-e", js],
                       capture_output=True, text=True)
    if p.returncode:
        pytest.fail("node failed: " + (p.stderr or p.stdout).strip())
    return json.loads(p.stdout)


@pytest.fixture(scope="module")
def js():
    return helpers()


def test_the_block_is_actually_extracted(js):
    """Guards the extraction itself, not the arithmetic."""
    assert "function nearestX" in js
    assert "const wpx" in js


def test_nearest_copy_is_a_no_op_when_already_nearest(js):
    world = 2 ** 14 * 512
    centre = world / 2
    assert run(js, "nearestX({}, {}, {})".format(centre + 10, centre, world)) \
        == centre + 10


def test_a_candidate_across_the_antimeridian_lands_on_screen(js):
    """The reported defect: 179.999 and -179.999 are 220 m apart on Earth."""
    z, w, world = 14, 1200, 2 ** 14 * 512
    cx = run(js, "wpx(179.999, {})".format(z))          # centred on the pin
    ox = cx - w / 2
    naive = run(js, "wpx(-179.999, {})".format(z)) - ox
    fixed = run(js, "nearestX(wpx(-179.999, {}), {}, {}) - {}".format(
        z, cx, world, ox))
    assert abs(naive) > world / 2          # the bug: a whole world away
    assert 0 <= fixed <= w                 # the fix: inside the viewport
    # And at the right distance, not merely on screen: 0.002 deg of longitude
    # is world * 0.002/360 = 46.6 px at z14 (about 222 m at 4.78 m/px).
    assert abs(abs(fixed - w / 2) - world * 0.002 / 360) < 0.5


def test_the_pin_survives_panning_around_the_world(js):
    """A pin stays under an arbitrarily far-panned viewport.

    `drawMap` also wraps `cx` into `[0, world)`, but that is hygiene against
    unbounded growth and float drift, NOT what makes this work: `nearestX`
    picks the copy nearest whatever `cx` it is given. So this deliberately
    passes an UNWRAPPED `cx` -- three and a half worlds east, as a long pan
    gives -- and the overlay must still land within half a world of centre.
    Removing the `cx` wrap changes nothing here, and that was confirmed with
    scripts/mutate.py --expect pass rather than assumed.
    """
    z, world = 14, 2 ** 14 * 512
    for worlds in (3.5, -2.25, 17.0):
        cx = run(js, "wpx(0, {})".format(z)) + worlds * world
        px = run(js, "nearestX(wpx(0, {}), {}, {})".format(z, cx, world))
        assert abs(px - cx) <= world / 2


@pytest.mark.parametrize("lon", [-180.0, -90.0, 0.0, 90.0, 179.999])
def test_wrapping_never_moves_a_point_that_is_already_centred(js, lon):
    z, world = 14, 2 ** 14 * 512
    x = run(js, "wpx({}, {})".format(lon, z))
    assert run(js, "nearestX({}, {}, {})".format(x, x, world)) == x


def test_the_edges_of_the_world_map_to_the_edges(js):
    z, world = 14, 2 ** 14 * 512
    assert run(js, "wpx(-180, {})".format(z)) == 0
    assert run(js, "wpx(180, {})".format(z)) == world
