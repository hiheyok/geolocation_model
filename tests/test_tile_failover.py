"""A second tile server is a second renderer, and that is the risk.

`connect` falls back to a backup when the primary does not answer. The useful
part is not the failover -- it is that the failover is loud and that there is a
way to ask whether the two servers actually agree.

Nothing in the map token cache records which server produced it. So a run that
silently drops to a backup can extend an existing cache with tiles from a
different renderer, and every shape, count and completion mask still agrees.
That is the same failure as a bank with two halves in different embedding
spaces, and it is why `connect` prints the substitution.

The comparison is deliberately at the token level, not the byte level, because
measuring the real pair showed a byte check would give the wrong answer. The
two servers here return masks differing on 0.03%-0.28% of pixels at every zoom
-- identical class tables, identical value sets, every difference on a feature
boundary. Rasterisation, not data. At the 12-d class fractions the model
actually reads, the same disagreement is max 7.8e-3 on a quantity in [0, 1].
A byte check would have condemned an interchangeable backup.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import tiles as T  # noqa: E402


class Stub(T.TileClient):
    """A client that answers from memory, so these tests need no network."""

    def __init__(self, base, up=True, classes=None, masks=None):
        super().__init__(base, timeout=0.01, retries=1)
        self._up = up
        self._classes = classes or {"background": 0, "water": 23}
        self._masks = masks or {}

    def health(self):
        if not self._up:
            raise T.TileError("{} refused".format(self.base))
        return {"ok": True}

    def classes(self):
        return self._classes

    def mask(self, z, x, y, mode="class"):
        key = (z, x, y)
        if key in self._masks:
            return self._masks[key]
        m = np.zeros((T.TILE_PX, T.TILE_PX), np.uint8)
        m[: 256 + z] = 23
        return m


def test_connect_prefers_the_first_that_answers(monkeypatch, capsys):
    made = []

    def fake(base, timeout=None, retries=None):
        c = Stub(base, up=(base != "http://down"))
        made.append(base)
        return c

    monkeypatch.setattr(T, "TileClient", fake)
    c = T.connect(["http://up", "http://other"])
    assert c.base == "http://up"
    assert "BACKUP" not in capsys.readouterr().out


def test_the_fallback_is_announced(monkeypatch, capsys):
    """Silence here is how a cache ends up spanning two renderers."""
    monkeypatch.setattr(T, "TileClient",
                        lambda base, **kw: Stub(base, up=(base != "http://down")))
    c = T.connect(["http://down", "http://up"])
    assert c.base == "http://up"
    out = capsys.readouterr().out
    assert "BACKUP" in out and "http://up" in out


def test_quiet_suppresses_only_the_announcement(monkeypatch, capsys):
    monkeypatch.setattr(T, "TileClient",
                        lambda base, **kw: Stub(base, up=(base != "http://down")))
    c = T.connect(["http://down", "http://up"], quiet=True)
    assert c.base == "http://up"
    assert "BACKUP" not in capsys.readouterr().out


def test_every_candidate_error_is_reported(monkeypatch):
    """"Down" and "moved" want different responses from whoever reads this."""
    monkeypatch.setattr(T, "TileClient", lambda base, **kw: Stub(base, up=False))
    with pytest.raises(SystemExit) as e:
        T.connect(["http://one", "http://two"])
    assert "http://one" in str(e.value) and "http://two" in str(e.value)


def test_bases_is_required():
    """Same rule as TileClient.base: no default address to drift from."""
    with pytest.raises(TypeError):
        T.connect()


# --- agreement ---------------------------------------------------------------

def test_identical_servers_agree():
    a, b = Stub("http://a"), Stub("http://b")
    fatal, detail = a.compare_with(b, probes=((4, 8, 5),))
    assert not fatal
    assert detail[0][3] == 0.0 and detail[0][4] == 0.0


def test_a_boundary_difference_is_reported_but_not_fatal():
    """The measured real case: a handful of edge pixels, identical classes."""
    m = np.zeros((T.TILE_PX, T.TILE_PX), np.uint8)
    m[:256] = 23
    m2 = m.copy()
    m2[255, :40] = 0                      # 40 pixels on one boundary row
    a = Stub("http://a", masks={(4, 8, 5): m})
    b = Stub("http://b", masks={(4, 8, 5): m2})
    fatal, detail = a.compare_with(b, probes=((4, 8, 5),))
    assert not fatal, "edge rasterisation must not condemn a usable backup"
    assert 0 < detail[0][3] < 0.001       # pixels do differ
    assert 0 < detail[0][4] < T.TileClient.TOKEN_TOL


def test_a_wholesale_rendering_difference_is_fatal():
    m = np.zeros((T.TILE_PX, T.TILE_PX), np.uint8)
    m[:256] = 23
    a = Stub("http://a", masks={(4, 8, 5): m})
    b = Stub("http://b", masks={(4, 8, 5): np.full_like(m, 92)})
    fatal, _ = a.compare_with(b, probes=((4, 8, 5),))
    assert fatal and "tokens differ" in fatal[0]
    assert not a.agrees_with(b, probes=((4, 8, 5),))


def test_a_different_class_table_is_fatal_whatever_the_pixels():
    """Masks are class ids: a remapped table changes every token's meaning
    while the pixel values themselves may be identical."""
    a = Stub("http://a", classes={"background": 0, "water": 23})
    b = Stub("http://b", classes={"background": 0, "wood": 23})
    fatal, _ = a.compare_with(b, probes=())
    assert fatal == ["class tables differ"]
