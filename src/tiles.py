"""The only module that speaks HTTP to the tile server.

A client library for an external service, not part of this application's
wiring: it takes the server address as an argument and imports no config, so
it can be pointed at a different server or a stub without touching anything
else.

Fetches single-channel class masks and turns them into the per-patch class
histograms the map encoder consumes.  A class-id mask is never interpolated --
only aggregated -- so the illegal-id failure mode cannot occur downstream.
"""

import io
import threading

import time

import numpy as np
import requests
from PIL import Image

# The tile server's wire format. These belong here rather than in config
# because they are the external service's contract, not this application's
# settings: they change when the server changes, and LEGAL_IDS is derived
# from the two above it. The server *address* is the opposite -- that is
# deployment wiring, so it lives in config.TILE_SERVER and is passed in.
CLASS_STEP = 23          # ids are spaced 23 apart: 0, 23, ..., 253
N_CLASSES = 12
TILE_PX = 512
LEGAL_IDS = frozenset(range(0, CLASS_STEP * N_CLASSES, CLASS_STEP))


class TileError(RuntimeError):
    pass


def connect(bases, timeout=8.0, retries=1, quiet=False):
    """A client for the first candidate that answers /health.

    The fallback is deliberately **loud**. The map token cache is built from
    whatever this returns, and nothing in that cache records which server
    produced it -- so a run that silently drops to the backup can extend a
    cache with tiles from a different renderer, and every shape, count and
    completion mask still agrees. That is the same failure as a bank with two
    halves in different embedding spaces, and it is the reason this prints the
    choice rather than just making it.

    Raises with every candidate's error if none answer, because "the tile
    server is down" and "the tile server is at a new address" want different
    responses from whoever is reading the log.

    `bases` is required, like `TileClient.base` and for the same reason: this
    module is a client for an external service and imports no config, so it can
    be pointed at a stub without touching anything else. Callers pass
    `config.TILE_SERVERS`.
    """
    errs = []
    for i, base in enumerate(bases):
        c = TileClient(base, timeout=timeout, retries=retries)
        try:
            c.health()
        except Exception as exc:                          # noqa: BLE001
            errs.append("{}: {}".format(base, str(exc).strip()[:80]))
            continue
        if i and not quiet:
            print("tile server: {} did not answer; USING BACKUP {}. Nothing "
                  "in the map cache records which server produced it, so do "
                  "not extend an existing cache across this boundary without "
                  "checking they agree (TileClient.agrees_with)."
                  .format(", ".join(bases[:i]), base), flush=True)
        return c
    raise SystemExit(
        "no tile server answered /health:" + "".join(
            chr(10) + "  " + e for e in errs))


class TileClient:
    """Thread-safe: each thread gets its own pooled session."""

    def __init__(self, base, timeout=20.0, retries=3):
        """base: the tile server root, e.g. config.TILE_SERVER.

        Required rather than defaulted. It used to default to a hardcoded
        LAN address that duplicated config.TILE_SERVER, so a change to the
        config would leave any caller relying on the default pointing at
        the old server with nothing to indicate it.
        """
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    @property
    def session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            s.mount("http://", requests.adapters.HTTPAdapter(
                pool_connections=32, pool_maxsize=32, max_retries=0))
            self._local.s = s
        return s

    def _get(self, path):
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.base + path, timeout=self.timeout)
                if r.status_code == 200:
                    return r.content
                if 400 <= r.status_code < 500:
                    raise TileError(f"{r.status_code} for {path}: {r.text[:200]}")
                last = TileError(f"{r.status_code} for {path}")
            except requests.RequestException as e:
                last = e
            time.sleep(0.15 * (attempt + 1))
        raise TileError(f"{path} failed after {self.retries} attempts: {last}")

    # -- endpoints ---------------------------------------------------------

    def health(self):
        import json
        return json.loads(self._get("/health"))

    # A byte comparison of two masks is the wrong test, and measuring it showed
    # why. The primary and the backup here return masks differing on 0.03% to
    # 0.28% of pixels at every zoom -- but the class tables are identical, the
    # value sets are identical, and every difference sits on a feature boundary
    # (23<->138, 0<->230). It is rasterisation, not different data.
    #
    # What the model reads is not pixels but a 12-d class fraction per 32x32
    # patch, and there the same disagreement is max 7.8e-3, mean 3.3e-4, on a
    # quantity in [0, 1]. A byte check would have condemned a backup that is in
    # fact interchangeable, and a bare "they differ" would have said nothing
    # about whether it mattered.
    TOKEN_TOL = 0.05

    def compare_with(self, other, probes=((0, 0, 0), (4, 8, 5), (8, 137, 91),
                                          (12, 2047, 1362))):
        """Quantify how two servers disagree, at the level the model reads.

        Returns (fatal, detail). `fatal` lists differences that would corrupt a
        shared cache; `detail` carries (z, x, y, pixel_frac, token_max) so a
        caller can judge rather than accept a verdict.
        """
        fatal, detail = [], []
        try:
            if self.classes() != other.classes():
                # Masks are class ids, so a different table remaps every token
                # while every shape and value range stays plausible.
                fatal.append("class tables differ")
        except Exception as exc:                          # noqa: BLE001
            fatal.append("classes unavailable: {}".format(str(exc)[:60]))
        for z, x, y in probes:
            try:
                ma, mb = self.mask(z, x, y), other.mask(z, x, y)
                ta = to_tokens(ma).astype("float64")
                tb = to_tokens(mb).astype("float64")
                dp = float((ma != mb).mean())
                dt = float(abs(ta - tb).max())
                detail.append((z, x, y, dp, dt))
                if dt > self.TOKEN_TOL:
                    fatal.append(
                        "tile {}/{}/{} tokens differ by {:.3f}, above the "
                        "{:.2f} tolerance".format(z, x, y, dt, self.TOKEN_TOL))
            except Exception as exc:                      # noqa: BLE001
                fatal.append("tile {}/{}/{}: {}".format(z, x, y, str(exc)[:50]))
        return fatal, detail

    def agrees_with(self, other, **kw):
        """True when nothing that would corrupt a shared cache differs."""
        return not self.compare_with(other, **kw)[0]

    def classes(self):
        import json
        return json.loads(self._get("/classes"))

    def mask(self, z, x, y, mode="class"):
        """Class-id mask as (512, 512) uint8."""
        q = "" if mode == "class" else f"?mode={mode}"
        raw = self._get(f"/tile/{z}/{x}/{y}/mask.png{q}")
        img = Image.open(io.BytesIO(raw))
        if img.mode != "L":
            raise TileError(f"expected colour type 0 greyscale, got mode {img.mode}")
        return np.asarray(img, dtype=np.uint8)

    def image(self, z, x, y):
        """Greyscale cartographic render as (512, 512) uint8 -- ablation only."""
        raw = self._get(f"/tile/{z}/{x}/{y}.png")
        return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.uint8)

    def png(self, z, x, y):
        """The cartographic render as raw PNG bytes, for passing straight to a
        browser. image() decodes it to an array instead, which a proxy would
        only have to re-encode."""
        return self._get(f"/tile/{z}/{x}/{y}.png")

    def labels(self, z, x, y, layers=None, limit=None, lang="latin"):
        import json
        q = [f"lang={lang}"]
        if layers:
            q.append("layers=" + ",".join(layers))
        if limit:
            q.append(f"limit={int(limit)}")
        return json.loads(self._get(f"/tile/{z}/{x}/{y}/labels.json?" + "&".join(q)))


# -- tokenisation ----------------------------------------------------------

def check_legal(mask):
    """Assert the mask holds only legal class ids.  The one place this can enter."""
    if mask.dtype != np.uint8:
        raise TileError(f"mask must be uint8, got {mask.dtype}")
    bad = np.unique(mask[(mask % CLASS_STEP) != 0])
    if bad.size:
        raise TileError(f"illegal class ids present: {bad.tolist()[:8]}")
    hi = int(mask.max()) // CLASS_STEP
    if hi >= N_CLASSES:
        raise TileError(f"class id {hi} out of range 0..{N_CLASSES - 1}")


def to_tokens(mask, grid=16, sub=1):
    """(512, 512) class mask -> (grid*grid, 12*sub*sub) float32 class fractions.

    Aggregation, never interpolation: every pixel contributes, and the result is
    a strict superset of majority vote (majority is its argmax).

    `sub` splits each patch into sub x sub cells and histograms each, so
    within-patch layout survives. sub=1 is the original token and is what every
    arm before 2026-09-02 was trained on.

    **Why it matters.** A 32x32 patch is 1024 pixels and sub=1 turns it into 12
    fractions, which is orientation-blind by construction: a north-south road
    and an east-west road give byte-identical tokens, as do a T-junction and a
    straight road with the same pixel count. Measured on 16,600 z12 patches with
    a coherent road direction, an MLP recovers that direction 52.7% of the time
    from sub=1 -- the majority rate, i.e. not at all -- against 77.6% from sub=2
    and 82.9% from sub=4. The structure is in the tile; the tokenizer discards
    it. See scripts/map_structure_probe.py.
    """
    if mask.shape != (TILE_PX, TILE_PX):
        raise TileError(f"expected {TILE_PX}x{TILE_PX}, got {mask.shape}")
    if TILE_PX % grid:
        raise TileError(f"grid {grid} does not divide {TILE_PX}")
    check_legal(mask)

    p = TILE_PX // grid
    if p % sub:
        raise TileError(f"sub {sub} does not divide the {p}px patch")
    ids = (mask // CLASS_STEP).astype(np.uint8)
    c = p // sub
    # (grid, sub, c, grid, sub, c) -> one row per patch, cells row-major within
    a = ids.reshape(grid, sub, c, grid, sub, c)
    a = a.transpose(0, 3, 1, 4, 2, 5).reshape(grid * grid, sub * sub, c * c)

    out = np.zeros((grid * grid, sub * sub, N_CLASSES), dtype=np.float32)
    for k in range(N_CLASSES):
        out[:, :, k] = (a == k).sum(axis=-1)
    out /= float(c * c)
    return out.reshape(grid * grid, sub * sub * N_CLASSES)


def token_row_order(grid=16):
    """Token i corresponds to action i: row-major, matching row*g + col."""
    return [(i // grid, i % grid) for i in range(grid * grid)]
