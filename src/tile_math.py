"""Web Mercator tile addressing for the map-guided geolocation agent.

The whole system rests on one identity: a recursive g x g subdivision of the
world is exactly the XYZ tile grid when g is a power of two.  One zoom step
adds log2(g) XYZ zoom levels, and step 0 is exactly z0/0/0.  A "tile path" is
therefore an XYZ address written in base-g digits, and the descent from one
step to the next is a digit append.

Pure math, no I/O.  `g` is a parameter everywhere so the 4x4 debug ladder and
the 16x16 shipping configuration share one implementation.
"""

import math

# Shipping configuration.  Nothing outside this module hardcodes 16 or 256.
G = 16
STEPS = 4

# Web Mercator clips here; the poles are not representable at any zoom.
MAX_LAT = 85.05112877980659


def bits(g: int = G) -> int:
    """Zoom levels consumed per step.  Requires g to be a power of two."""
    if g < 2 or (g & (g - 1)):
        raise ValueError(f"g must be a power of two, got {g}")
    return g.bit_length() - 1


def final_zoom(g: int = G, steps: int = STEPS) -> int:
    return bits(g) * steps


def actions(g: int = G) -> int:
    return g * g


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------

def project(lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) degrees -> normalised Mercator (X, Y), both in [0, 1]."""
    lat = max(-MAX_LAT, min(MAX_LAT, lat))
    x = (lon + 180.0) / 360.0
    s = math.sin(math.radians(lat))
    y = 0.5 - math.log((1.0 + s) / (1.0 - s)) / (4.0 * math.pi)
    return x, y


def unproject(x: float, y: float) -> tuple[float, float]:
    """Normalised Mercator (X, Y) -> (lat, lon) degrees."""
    lon = x * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y))))
    return lat, lon


def tile_for(lat: float, lon: float, z: int) -> tuple[int, int, int]:
    """(lat, lon) -> the XYZ tile containing it at zoom z."""
    n = 1 << z
    px, py = project(lat, lon)
    x = min(n - 1, max(0, int(px * n)))
    y = min(n - 1, max(0, int(py * n)))
    return z, x, y


def tile_to_latlon(z: int, x: int, y: int, u: float = 0.0, v: float = 0.0):
    """Point at fractional offset (u, v) inside tile (z, x, y) -> (lat, lon).

    (0, 0) is the tile's top-left corner, matching image and mask indexing.
    """
    n = 1 << z
    return unproject((x + u) / n, (y + v) / n)


def norm_corner(z: int, x: int, y: int) -> tuple[float, float]:
    """Tile's top-left corner in normalised Mercator -- the state encoder input."""
    n = 1 << z
    return x / n, y / n


# --------------------------------------------------------------------------
# path <-> address
# --------------------------------------------------------------------------


# Bit layout for packing a tile address into one integer, used as the key of
# every (z, x, y) -> row lookup table in the project.
#
#     bit  63        62 .. 58     57 .. 29     28 .. 0
#          unused    z (5 bits)   x (29 bits)  y (29 bits)
#
# Chosen so the result is always a *positive* signed 64-bit integer, because
# these keys are built vectorised as numpy int64 (dataset.py, beam.py) and a
# value that ran into bit 63 would come back negative and silently fail to match
# the same tile packed as a Python int.
#
# The 29-bit coordinate fields are the binding constraint, not the zoom field:
# at zoom z a coordinate is < 2**z, so 29 bits holds every tile up to z28 --
# comfortably past the z16 this agent reaches, and past the z22 or so any XYZ
# server serves. The zoom field is 5 bits, so z <= 31.
Z_SHIFT = 58
X_SHIFT = 29
COORD_BITS = X_SHIFT
MAX_KEY_ZOOM = X_SHIFT - 1          # 28


def tile_key(z, x, y):
    """Pack a tile address into one integer for use as a dict or lookup key.

    Takes scalars or numpy integer arrays and returns the same kind, so the
    vectorised index build and the per-item lookup produce identical keys --
    which is the entire point of this living in one place.

    Scalars are normalised to Python int, including numpy integers, so a key
    hashes the same however the caller happened to be holding its coordinates.
    Arrays are shifted elementwise and keep their dtype. The array test is
    duck-typed rather than an isinstance check because this module imports only
    `math` and is not going to take a numpy dependency for one branch.

    There is no unpack function on purpose: nothing needs to read a key back,
    and adding one would invite the layout being reimplemented at a call site.
    """
    if getattr(z, "shape", ()) != ():
        return (z << Z_SHIFT) | (x << X_SHIFT) | y
    return (int(z) << Z_SHIFT) | (int(x) << X_SHIFT) | int(y)


def descend(z: int, x: int, y: int, action: int, g: int = G):
    """Apply one action.  This is the entire step recurrence: a digit append."""
    return z + bits(g), x * g + (action % g), y * g + (action // g)


def path_to_xyz(path, g: int = G) -> tuple[int, int, int]:
    z = x = y = 0
    for a in path:
        z, x, y = descend(z, x, y, a, g)
    return z, x, y


def xyz_to_path(z: int, x: int, y: int, g: int = G) -> list[int]:
    b = bits(g)
    if z % b:
        raise ValueError(f"z={z} is not reachable with g={g} (needs a multiple of {b})")
    n_steps, out, mask = z // b, [], g - 1
    for t in range(n_steps):
        sh = b * (n_steps - 1 - t)
        out.append((((y >> sh) & mask) * g) + ((x >> sh) & mask))
    return out


# --------------------------------------------------------------------------
# training targets
# --------------------------------------------------------------------------

def target_actions(lat: float, lon: float, g: int = G, steps: int = STEPS) -> list[int]:
    """The correct action at every step, read off the final tile's digits."""
    b = bits(g)
    _, xf, yf = tile_for(lat, lon, b * steps)
    return xyz_to_path(b * steps, xf, yf, g)


def prefix_tile(lat: float, lon: float, t: int, g: int = G, steps: int = STEPS):
    """The teacher-forced view at step t: the true tile's ancestor at zoom b*t."""
    b = bits(g)
    _, xf, yf = tile_for(lat, lon, b * steps)
    sh = b * (steps - t)
    return b * t, xf >> sh, yf >> sh


def target_uv(lat: float, lon: float, z: int, x: int, y: int) -> tuple[float, float]:
    """Fractional position of (lat, lon) inside tile (z, x, y) -- the click target."""
    n = 1 << z
    px, py = project(lat, lon)
    return px * n - x, py * n - y


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance -- the metric every result is reported in."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))
