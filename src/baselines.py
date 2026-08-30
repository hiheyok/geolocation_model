"""Non-agent baselines -- run these BEFORE the agent.

The project's thesis is that iterative map-guided search beats the obvious
approach.  Without these rows that claim cannot be made.  Every baseline uses
the same frozen embeddings, the same splits and the same metrics, so the
comparison measures the approach and not the encoder.

Coordinates are always aggregated as 3D unit vectors.  Averaging lat/lon across
hemispheres lands you in the ocean.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tile_math as tm
from dataset import GeoStepDataset

BUCKETS = (1.0, 25.0, 100.0, 750.0)


def to_unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def from_unit(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, lon


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def report(name, err_km):
    row = {"name": name, "median": float(np.median(err_km)),
           "mean": float(err_km.mean())}
    for b in BUCKETS:
        row["at{:g}".format(b)] = float((err_km <= b).mean())
    return row


def print_table(rows):
    head = "{:<26} {:>10} {:>10}".format("baseline", "median km", "mean km")
    head += "".join("{:>9}".format("<{:g}km".format(b)) for b in BUCKETS)
    print(head)
    print("-" * len(head))
    for r in rows:
        line = "{:<26} {:>10.1f} {:>10.1f}".format(r["name"], r["median"], r["mean"])
        line += "".join("{:>8.1%} ".format(r["at{:g}".format(b)]) for b in BUCKETS)
        print(line)


def load(split, street_file="embeddings.f16.npy"):
    d = GeoStepDataset(split, street_file=street_file)
    emb = np.asarray(d.street[d.rows], dtype=np.float32)
    return d, emb


def baseline_centroid(tr, va):
    c = to_unit(tr.lat, tr.lon).mean(0)
    lat, lon = from_unit(c)
    err = great_circle_km(np.full(len(va.lat), lat), np.full(len(va.lat), lon),
                          va.lat, va.lon)
    return report("train centroid", err)


def baseline_biggest_cell(tr, va):
    """Always guess the most populated z4 cell -- the floor for step 0."""
    acts = np.array([tm.target_actions(la, lo)[0] for la, lo in zip(tr.lat, tr.lon)])
    top = np.bincount(acts, minlength=tm.actions()).argmax()
    members = to_unit(tr.lat[acts == top], tr.lon[acts == top]).mean(0)
    lat, lon = from_unit(members)
    err = great_circle_km(np.full(len(va.lat), lat), np.full(len(va.lat), lon),
                          va.lat, va.lon)
    r = report("biggest z4 cell", err)
    r["note"] = "cell {} holds {:.1%} of train".format(top, (acts == top).mean())
    return r


def baseline_knn(tr, va, etr, eva, ks=(1, 5, 15), dev="cuda", chunk=2048):
    """Cosine retrieval over frozen DINOv2.  The real bar."""
    A = torch.from_numpy(etr).to(dev)
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-6)
    U = torch.from_numpy(to_unit(tr.lat, tr.lon).astype(np.float32)).to(dev)
    out = {}
    kmax = max(ks)
    idx_all, sim_all = [], []
    for lo in range(0, len(eva), chunk):
        B = torch.from_numpy(eva[lo:lo + chunk]).to(dev)
        B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-6)
        sim = B @ A.T
        s, i = sim.topk(kmax, dim=1)
        idx_all.append(i)
        sim_all.append(s)
    idx = torch.cat(idx_all)
    sims = torch.cat(sim_all)
    for k in ks:
        w = torch.softmax(sims[:, :k] * 20.0, dim=1).unsqueeze(-1)
        pred = (U[idx[:, :k]] * w).sum(1).cpu().numpy()
        lat, lon = from_unit(pred)
        out[k] = report("kNN DINOv2 k={}".format(k),
                        great_circle_km(lat, lon, va.lat, va.lon))
    return [out[k] for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--street-file", default="embeddings.f16.npy")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, etr = load("train", a.street_file)
    va, eva = load(a.split, a.street_file)
    print("train {:,}   {} {:,}   embedding dim {}\n"
          .format(len(tr), a.split, len(va), etr.shape[1]))

    rows = [baseline_centroid(tr, va), baseline_biggest_cell(tr, va)]
    rows += baseline_knn(tr, va, etr, eva, dev=dev)
    print_table(rows)
    for r in rows:
        if "note" in r:
            print("\nnote: {} -- {}".format(r["name"], r["note"]))


if __name__ == "__main__":
    main()
