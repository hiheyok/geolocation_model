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
