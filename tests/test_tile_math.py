"""Property tests for tile addressing.  Run directly: python tests/test_tile_math.py"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import tile_math as tm  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def main():
    rng = random.Random(0)

    print("path <-> xyz round-trip, every grid size")
    for g in (2, 4, 8, 16):
        ok = True
        bad = ""
        for _ in range(2000):
            n = rng.randint(1, 6)
            path = [rng.randrange(g * g) for _ in range(n)]
            z, x, y = tm.path_to_xyz(path, g)
            back = tm.xyz_to_path(z, x, y, g)
            if back != path:
                ok, bad = False, f"{path} -> ({z},{x},{y}) -> {back}"
                break
        check(f"g={g:<2} round-trip", ok, bad)

    print("\nagainst the tile server's documented example")
    check("tile_for(51.5024,-0.0908,14) == (14,8187,5448)",
          tm.tile_for(51.5024, -0.0908, 14) == (14, 8187, 5448),
          str(tm.tile_for(51.5024, -0.0908, 14)))
    check("z16 path == [87,95,46,31]",
          tm.target_actions(51.5024, -0.0908) == [87, 95, 46, 31],
          str(tm.target_actions(51.5024, -0.0908)))

    print("\ntargets agree with addressing, on random coordinates")
    ok_desc = ok_pref = ok_final = True
    d1 = d2 = d3 = ""
    for _ in range(5000):
        lat = rng.uniform(-84.0, 84.0)
        lon = rng.uniform(-179.9, 179.9)
        acts = tm.target_actions(lat, lon)
        zf = tm.final_zoom()
        want = tm.tile_for(lat, lon, zf)

        # descending through the targets from the world reaches the true tile
        got = (0, 0, 0)
        for a in acts:
            got = tm.descend(*got, a)
        if got != want:
            ok_final, d3 = False, f"{lat:.4f},{lon:.4f} {got} != {want}"

        # prefix_tile agrees with the descent at every step
        for t in range(tm.STEPS + 1):
            pref = tm.prefix_tile(lat, lon, t)
            walk = (0, 0, 0)
            for a in acts[:t]:
                walk = tm.descend(*walk, a)
            if pref != walk:
                ok_pref, d2 = False, f"t={t} {pref} != {walk}"
        if tm.prefix_tile(lat, lon, 0) != (0, 0, 0):
            ok_desc, d1 = False, "step 0 is not the world tile"

    check("step 0 is always z0/0/0", ok_desc, d1)
    check("prefix_tile == descent prefix", ok_pref, d2)
    check("target actions reach the true tile", ok_final, d3)

    print("\nclick target inverts to the original coordinate")
    worst = 0.0
    for _ in range(5000):
        lat = rng.uniform(-84.0, 84.0)
        lon = rng.uniform(-179.9, 179.9)
        z, x, y = tm.tile_for(lat, lon, tm.final_zoom())
        u, v = tm.target_uv(lat, lon, z, x, y)
        if not (0.0 <= u < 1.0 and 0.0 <= v < 1.0):
            worst = float("inf")
            break
        rlat, rlon = tm.tile_to_latlon(z, x, y, u, v)
        worst = max(worst, tm.haversine_km(lat, lon, rlat, rlon))
    check("u,v in [0,1) and inverts to < 1 m", worst < 0.001, f"worst {worst*1000:.4f} m")

    print("\nsanity")
    check("final zoom is 16", tm.final_zoom() == 16, str(tm.final_zoom()))
    check("g=4 ladder reaches z12 in 6 steps",
          tm.final_zoom(4, 6) == 12, str(tm.final_zoom(4, 6)))
    # 1 deg lon at 51.5N is 111.32*cos(51.5) = 69.3 km, so 0.1 deg is 6.93 km
    lon_km = tm.haversine_km(51.5, -0.1, 51.5, 0.0)
    check("haversine 0.1 deg lon at London is ~6.93 km",
          6.8 < lon_km < 7.05, f"{lon_km:.3f} km")
    eq_km = tm.haversine_km(0.0, 0.0, 0.0, 1.0)
    check("haversine 1 deg lon at equator is ~111.3 km",
          111.0 < eq_km < 111.6, f"{eq_km:.3f} km")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("all tile_math property tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
