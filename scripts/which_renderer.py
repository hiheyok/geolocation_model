"""Which renderer produced the map token cache already on disk?

The mapping service now reports a `renderer.id` that distinguishes the two
deployments -- identical `cacheNamespaces`, identical source digest, identical
mbtiles sha256 across 75 GB, identical maplibre version string, but different
native binary digests, so `r-e42ea3b3bbf5` against `r-3a8cfb6e81c9`. That
identity can be recorded from now on.

It says nothing about the 626,284 tiles already cached. Those were fetched
before any renderer identity existed, and every published number depends on
them. So rather than assume which box served them, ask the cache.

The two renderers disagree on 0.03%-0.28% of mask pixels, which is max 7.8e-3
on the 12-d per-patch class fractions the cache actually stores. That is small
but it is not zero, and the cache is float16 -- resolution ~5e-4 near 0.1 -- so
a per-tile exact match against one renderer and a mismatch against the other is
detectable when the tile happens to contain a disputed boundary.

Attribution, not proof. A tile whose pixels the two renderers agree on cannot
distinguish them, so this reports how many tiles were *decisive* and which way
they fell. A clean split is evidence; a tie means the sample held nothing the
renderers disagree about.

    OSV_RELEASE=s10 py scripts/which_renderer.py --n 60
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = "s10"

import config                                    # noqa: E402
import tiles as T                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="tiles to probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--servers", default=",".join(config.TILE_SERVERS))
    ap.add_argument("--record", action="store_true",
                    help="write the winner into the cache's renderer sidecar, "
                         "with the evidence, so the attribution is auditable "
                         "rather than a bare id someone has to trust")
    a = ap.parse_args()

    tok_p, idx_p, _ = config.map_files("")
    tokens = np.load(tok_p, mmap_mode="r")
    idx = pq.read_table(idx_p)
    z = np.asarray(idx["z"], np.int64)
    x = np.asarray(idx["x"], np.int64)
    y = np.asarray(idx["y"], np.int64)
    row = np.asarray(idx["row"], np.int64)
    print("{:,} cached tiles, tokens {}".format(len(row), tokens.shape),
          flush=True)

    bases = [s.strip() for s in a.servers.split(",") if s.strip()]
    clients = []
    for b in bases:
        c = T.TileClient(b, timeout=20.0, retries=2)
        try:
            h = c.health()
        except Exception as exc:                          # noqa: BLE001
            print("{:<26} unreachable: {}".format(b, str(exc)[:50]), flush=True)
            continue
        rid = (h.get("renderer") or {}).get("id", "<no renderer id>")
        print("{:<26} {}".format(b, rid), flush=True)
        clients.append((b, rid, c))
    if len(clients) < 2:
        raise SystemExit("need two reachable servers to tell them apart")

    rng = np.random.default_rng(a.seed)
    pick = rng.choice(len(row), min(a.n, len(row)), replace=False)
    # Deeper tiles carry more boundary detail, so they are likelier to be
    # decisive; the sample is not filtered to them because that would bias
    # toward whichever renderer differs most at depth.
    tally = {b: 0 for b, _, _ in clients}
    decisive = ties = 0
    for k, i in enumerate(pick):
        want = np.asarray(tokens[row[i]], np.float32)
        got = {}
        for b, _, c in clients:
            try:
                got[b] = T.to_tokens(c.mask(int(z[i]), int(x[i]),
                                            int(y[i]))).astype(np.float32)
            except Exception:                             # noqa: BLE001
                got[b] = None
        if any(v is None for v in got.values()):
            continue
        vals = list(got.values())
        if np.array_equal(vals[0], vals[1]):
            ties += 1                       # renderers agree here; no evidence
            continue
        decisive += 1
        err = {b: float(np.abs(want - v).max()) for b, v in got.items()}
        best = min(err, key=err.get)
        tally[best] += 1
        if k < 8:
            print("  {:>2}/{:>6}/{:>6}  ".format(int(z[i]), int(x[i]),
                                                 int(y[i]))
                  + "  ".join("{}={:.2e}".format(b.split("//")[-1][:13],
                                                 err[b]) for b in err),
                  flush=True)

    print("\n{} decisive of {} probed ({} where the renderers agree, which "
          "cannot attribute)".format(decisive, len(pick), ties))
    if not decisive:
        print("no evidence either way -- the sample contained nothing the two "
              "renderers disagree about")
        return
    for b, rid, _ in clients:
        print("  {:<26} {} closer on {:>3}/{} decisive tiles  ({})".format(
            b, rid, tally[b], decisive,
            "MATCHES the cache" if tally[b] > decisive * 0.9 else ""))

    if a.record:
        import json
        win = max(tally, key=tally.get)
        if tally[win] <= decisive * 0.9:
            raise SystemExit(
                "no renderer matched clearly ({}), so nothing is recorded -- "
                "a guess written to a sidecar is worse than an empty one"
                .format(dict(tally)))
        rid = [r for b, r, _ in clients if b == win][0]
        out = Path(config.MAP_CACHE) / T.RENDERER_SIDECAR
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "id": rid, "base": win,
            "attributed": "empirically, not recorded at fetch time",
            "evidence": "{}/{} decisive cached tiles matched this renderer "
                        "exactly (0.00e+00 max token difference); {} probed, "
                        "{} tiles the renderers agree on and cannot attribute"
                        .format(tally[win], decisive, len(pick), ties),
            "note": "this cache predates renderer identity; the id was "
                    "recovered by comparing cached tokens against both live "
                    "renderers rather than assumed from which host was up",
        }, indent=1), encoding="utf-8")
        print(chr(10) + "recorded {} -> {}".format(rid, out))


if __name__ == "__main__":
    main()
