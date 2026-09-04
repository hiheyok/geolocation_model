"""Rollout: greedy and beam search, with live tile fetches.

Training is offline, but inference needs arbitrary z12/z16 tiles that no cache
can cover, so this is the one part of the model path that talks to the tile
server.  Cache hits are served from the memmap; misses are fetched in parallel.

Beams are ranked on cumulative tile log-probability alone.  The click head runs
per beam to produce coordinates and its spread feeds confidence_radius_km, but
it does not reorder the beam.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import tile_math as tm
import tiles as T


def source_for(ck, grid=tm.G, **kw):
    """A TokenSource matching a checkpoint's map representation.

    The cache and the `sub` it was tokenised at are properties of the arm, not
    of the release. Building one with the defaults scores a sub=2 model against
    12-d tokens, which is a different model than the one that was trained.
    """
    return TokenSource(grid, cache=ck.get("map_cache"),
                       sub=ck.get("map_sub", 1), **kw)


class TokenSource:
    """Cached token grids with a live fallback to the tile server."""

    def __init__(self, grid=tm.G, cache=None, client=None, threads=16, sub=1):
        tokens_p, index_p, done_p = config.map_files(cache)
        self.grid = grid
        self.sub = int(sub)
        self.tokens = np.load(tokens_p, mmap_mode="r")
        # A live-fetched tile must be tokenised exactly as the cache was, or
        # the two paths silently disagree in width and the beam scores a
        # different representation than the model was trained on.
        want = T.N_CLASSES * self.sub * self.sub
        if self.tokens.shape[-1] != want:
            raise ValueError(
                "map cache is {}-d but sub={} implies {}-d".format(
                    self.tokens.shape[-1], self.sub, want))
        idx = pq.read_table(index_p)
        k = tm.tile_key(np.asarray(idx["z"]).astype(np.int64),
                        np.asarray(idx["x"]).astype(np.int64),
                        np.asarray(idx["y"]).astype(np.int64))
        rows = np.asarray(idx["row"]).astype(np.int64)
        # Drop rows the cache never actually fetched. An unfetched row is the
        # memmap's zero fill, which is a legal token histogram, so keeping it
        # in the lookup means beam search silently scores a blank map instead
        # of a tile -- the same defect dataset.py refuses outright. Here the
        # better answer is a miss, not an error: there is already a live
        # fallback, so an incomplete cache degrades to slower and correct.
        if done_p.exists():
            done = np.load(done_p)
            # Fail closed. The first version skipped the whole check when the
            # mask was too short -- exactly the case where it is least
            # trustworthy -- so a truncated mask let unwritten zero rows back
            # into inference under a guard that read as protective.
            need = int(rows.max(initial=-1)) + 1
            if len(done) < need:
                raise SystemExit(
                    "map cache mask has {:,} entries but the index addresses "
                    "row {:,}; it belongs to a different build. Delete it and "
                    "re-run fetch_tiles.".format(len(done), need - 1))
            ok = done[rows] == 1
            if True:
                if not ok.all():
                    print("map cache: {:,} of {:,} indexed tiles were never "
                          "fetched; they will be fetched live"
                          .format(int((~ok).sum()), len(rows)), flush=True)
                k, rows = k[ok], rows[ok]
        self.lut = dict(zip(k.tolist(), rows.tolist()))
        self.client = client or T.TileClient(config.TILE_SERVER)
        self.pool = ThreadPoolExecutor(max_workers=threads)
        self.live = {}
        self.n_hit = self.n_miss = 0

    def get(self, keys):
        """keys: list of (z, x, y) -> (N, A, C) float32."""
        out = np.zeros((len(keys), self.grid * self.grid,
                        self.tokens.shape[-1]), np.float32)
        misses = []
        for i, (z, x, y) in enumerate(keys):
            kk = tm.tile_key(z, x, y)
            row = self.lut.get(kk)
            if row is not None:
                out[i] = self.tokens[row]
                self.n_hit += 1
            elif kk in self.live:
                out[i] = self.live[kk]
                self.n_hit += 1
            else:
                misses.append((i, z, x, y, kk))

        if misses:
            self.n_miss += len(misses)

            def fetch(m):
                _, z, x, y, _ = m
                return T.to_tokens(self.client.mask(z, x, y), self.grid,
                                   self.sub)

            for m, arr in zip(misses, self.pool.map(fetch, misses)):
                out[m[0]] = arr
                if len(self.live) < 400000:
                    self.live[m[4]] = arr.astype(np.float16)
        return out



def _views(tiles_, street, source, dev, step):
    """Every live beam's current tile, flattened into one batch for the model.

    Returns (tokens, street, x0, y0, step, beams_per_image).

    Both callers -- the search loop and the click head that runs after it --
    need exactly this, and they used to build it separately. Keeping one copy
    matters more here than the seven lines it saves: when beam.py last held its
    own version of something model.py also did, the two diverged silently and
    only the rollout path broke, because training never goes through here.
    """
    B = street.shape[0]
    flat = [tl for img in tiles_ for tl in img]
    nb = len(flat) // B
    corners = [tm.norm_corner(*k) for k in flat]      # once per tile, not twice
    return (torch.from_numpy(source.get(flat)).to(dev),
            street.unsqueeze(1).expand(B, nb, -1).reshape(B * nb, -1),
            torch.tensor([c[0] for c in corners], dtype=torch.float32, device=dev),
            torch.tensor([c[1] for c in corners], dtype=torch.float32, device=dev),
            torch.full((len(flat),), step, dtype=torch.long, device=dev),
            nb)


@torch.no_grad()
def search(model, street, source, dev, beam_k=16, top_m=16,
           g=tm.G, steps=tm.STEPS, greedy=False, sink_prune=1.0,
           score_steps=None, nbrs=None):
    """street: (B, D) float32 tensor.  Returns per-image candidate dicts."""
    B = street.shape[0]
    K = 1 if greedy else beam_k
    A = g * g

    tiles_ = [[(0, 0, 0)] for _ in range(B)]      # per image, per beam
    scores = np.zeros((B, 1), dtype=np.float64)
    paths = [[[]] for _ in range(B)]

    for t in range(steps):
        tok, st, x0, y0, sp, nb = _views(tiles_, street, source, dev, t)

        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            f, keys = model.fuse_flat(st, tok, x0, y0, sp)
            # one row per live beam per image, and the learned keys come
            # with it -- see GeoAgent.retr_prior
            n_logits = keys.shape[1] + (1 if model.sink is not None else 0)
            prior = model.retr_prior(nbrs, street, x0, y0, sp, nb, n_logits)
            # and the tile memory, through the same slot -- see _add_geo
            prior = model._add_geo(prior, f, x0, y0, sp, n_logits)
            # `sp` is required, not optional: the extra sink keys are indexed
            # by step, and omitting it silently evaluates a sink-k 1 model --
            # training one network and scoring another.
            logits = model.policy_logits(f, keys, prior, sp).float()
        # With a sink class the softmax spans A+1: log p(a) already decomposes
        # into log p(not-sink) + log p(a | not-sink), so a beam the model
        # believes is dead is penalised in its own cumulative score.  The sink
        # itself is never expanded -- it is a verdict, not a place.
        has_sink = logits.shape[-1] == A + 1
        lp_all = torch.log_softmax(logits, -1)
        p_sink = (lp_all[:, A].exp().view(B, nb).cpu().numpy()
                  if has_sink else np.zeros((B, nb)))
        logp = lp_all[:, :A].view(B, nb, A).cpu().numpy()

        # Steps beyond score_steps contribute nothing to the ranking and
        # expand greedily.  Measured on the dual-encoder model, s2 and s3 supply
        # ~70% of every path's score from distributions that are 5.9% and 2.6%
        # accurate, so their spread across branches swamps the s0/s1 signal that
        # actually decides where the answer is.
        rank_here = score_steps is None or t < score_steps
        m = 1 if (greedy or not rank_here) else min(top_m, A)
        w = 1.0 if rank_here else 0.0
        new_tiles, new_paths = [], []
        # Every image must leave this step with the same number of beams.
        # _views flattens (image, beam) to `b * nb + j`, so a short row would
        # shift every later image onto another image's street embedding -- and
        # the score array is dense, so an unfilled slot would read as 0.0, a
        # better cumulative log-probability than any real path.
        width = min(nb * m, K)
        new_scores = np.full((B, width), -np.inf)

        def expand(b, js):
            out = []
            for j in js:
                order = np.argpartition(-logp[b, j], m - 1)[:m]
                for aidx in order:
                    out.append((scores[b, j] + w * logp[b, j, aidx], j, int(aidx)))
            return out

        for b in range(B):
            live = [j for j in range(nb) if p_sink[b, j] < sink_prune]
            cand = sorted(expand(b, live), key=lambda c: -c[0])[:width]
            if len(cand) < width:
                # Pruning must not cost this image beam width, or its beams
                # stop lining up with everyone else's.  Top up from the beams
                # pruning rejected, worst-rejected last; `width` is the same
                # for every image, so this always fills.
                rest = [j for j in range(nb) if j not in set(live)]
                extra = sorted(expand(b, rest), key=lambda c: -c[0])
                cand += extra[:width - len(cand)]
            new_tiles.append([tm.descend(*tiles_[b][j], a, g) for _, j, a in cand])
            new_paths.append([paths[b][j] + [a] for _, j, a in cand])
            for i, (sc, _, _) in enumerate(cand):
                new_scores[b, i] = sc
        assert all(len(r) == width for r in new_tiles), "ragged beam width"
        tiles_, paths, scores = new_tiles, new_paths, new_scores

    # click head on the final view of every surviving beam
    tok, st, x0, y0, sp, nb = _views(tiles_, street, source, dev, steps)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        f, _ = model.fuse_flat(st, tok, x0, y0, sp)
        uv = model.click_uv(f).float().cpu().numpy()

    results = []
    for b in range(B):
        cands = []
        for j in range(nb):
            z, x, y = tiles_[b][j]
            u, v = uv[b * nb + j]
            lat, lon = tm.tile_to_latlon(z, x, y, float(u), float(v))
            cands.append({"lat": lat, "lon": lon, "score": float(scores[b, j]),
                          "path": paths[b][j], "tile": (z, x, y)})
        results.append({"candidates": cands,
                        "best": cands[0],
                        "confidence_radius_km": confidence_radius(cands)})
    return results


def confidence_radius(cands):
    """Softmax-weighted RMS great-circle spread of the beam about its centroid."""
    if len(cands) == 1:
        return 0.0
    s = np.array([c["score"] for c in cands], dtype=np.float64)
    w = np.exp(s - s.max())
    w /= w.sum()
    lat = np.radians([c["lat"] for c in cands])
    lon = np.radians([c["lon"] for c in cands])
    v = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
    c = (v * w[:, None]).sum(0)
    c /= max(np.linalg.norm(c), 1e-12)
    clat = np.degrees(np.arcsin(np.clip(c[2], -1, 1)))
    clon = np.degrees(np.arctan2(c[1], c[0]))
    d = np.array([tm.haversine_km(la, lo, clat, clon)
                  for la, lo in zip(np.degrees(lat), np.degrees(lon))])
    return float(np.sqrt((w * d ** 2).sum()))
