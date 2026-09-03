"""Assemble everything the overnight run produced into one table.

Reads three sources and does not trust any single one of them:

  * the checkpoints, for what each arm actually selected and when;
  * the evaluation logs, for held-out numbers at each beam width;
  * runs/util.csv, for what the machine was doing while it happened.

Written so the morning's first question -- did more data help, or did more
epochs on less data get there too -- is answered by one file.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import safeio

RUNS = ROOT / "runs"
LOGS = RUNS / "logs"

ROW = re.compile(r"agent beam_k=(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%")
STEPACC = re.compile(r"beam_k (\d+)\s+[\d.]+s\s+step acc\s+(.*)")
SEL = re.compile(r"sel\s+median\s+([\d.]+) km\s+<25km\s+([\d.]+)%")
EPOCH = re.compile(r"^ep\s+(\d+)\s+([\d.]+)s")


def read_evals(tag):
    """{split: {k: (median, mean, <1km, <25km)}} from that arm's eval logs."""
    out = {}
    for split in ("val", "test"):
        p = LOGS / "{}_eval_{}.log".format(tag, split)
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        got = {}
        for m in ROW.finditer(txt):
            got[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
        if got:
            out[split] = got
    return out


def read_train(tag):
    """Per-epoch selection curve, and the wall clock, from the training log."""
    p = LOGS / "{}.log".format(tag)
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8", errors="replace")
    sel = [(float(a), float(b)) for a, b in SEL.findall(txt)]
    eps = [(int(a), float(b)) for a, b in EPOCH.findall(txt)]
    return {"sel": sel, "epochs": eps,
            "best_km": min((s[0] for s in sel), default=float("nan")),
            "best_ep": (int(np.argmin([s[0] for s in sel])) + 1) if sel else None,
            "secs_per_epoch": (float(np.median([e[1] for e in eps]))
                               if eps else float("nan"))}


def read_ckpt(tag):
    try:
        import torch
        ck = torch.load(ROOT / "checkpoints" / (tag + ".pt"), map_location="cpu",
                        weights_only=False)
        return {k: v for k, v in ck.items() if k != "model"}
    except Exception:
        return {}


def util_summary():
    p = RUNS / "util.csv"
    if not p.exists():
        return {}
    rows = [l.split(",") for l in p.read_text(encoding="utf-8").strip().split("\n")[1:]]
    by = {}
    for r in rows:
        if len(r) < 9:
            continue
        by.setdefault(r[1], []).append([float(x) for x in r[2:]])
    out = {}
    for stage, v in by.items():
        a = np.array(v)
        keep = lambda j: a[a[:, j] >= 0, j]
        m = lambda j: float(keep(j).mean()) if len(keep(j)) else float("nan")
        out[stage] = {"n": len(a), "gpu": m(0), "gpu_peak": float(a[:, 0].max()),
                      "vram": m(1), "shared": m(2), "watts": m(3),
                      "temp": m(4), "cpu": m(5), "ram": m(6)}
    return out


def fmt_km(v):
    return "n/a" if v is None or v != v else "{:.1f}".format(v)


def main():
    ap = argparse.ArgumentParser()
    # NOT REPORT.md. This regenerates its whole output from state.json,
    # and overnight.finish() calls it at the end of every run -- so
    # pointing it at REPORT.md silently destroyed the hand-written
    # analysis there once per run. REPORT.md is curated; this is the
    # generated table it can cite.
    ap.add_argument("--out", default=str(RUNS / "ARMS.md"))
    a = ap.parse_args()

    state = {}
    if (RUNS / "state.json").exists():
        state = json.loads((RUNS / "state.json").read_text(encoding="utf-8"))

    tags = sorted({p.stem for p in LOGS.glob("s*_*.log")
                   if "_eval_" not in p.stem} | {"s01_km"})
    tags = [t for t in tags if (ROOT / "checkpoints" / (t + ".pt")).exists()]

    L = []
    L.append("# Overnight run: data scale vs epochs\n")
    L.append("Selection criterion is greedy-decode **val median km**, not val "
             "loss. Beam ranking is on s0-s2 (`--score-steps 3`), the shipping "
             "depth.\n")

    L.append("\n## Arms\n")
    L.append("| arm | release | images | epochs | steps | selected ep | "
             "val km (greedy, sel) | test km k=2 | test <25km | s/epoch |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = []
    for t in tags:
        ck = read_ckpt(t)
        tr = read_train(t)
        ev = read_evals(t)
        n = ck.get("limit") or ""
        m = re.search(r"n(\d+)k_e(\d+)", t)
        imgs = int(m.group(1)) * 1000 if m else ""
        eps = int(m.group(2)) if m else ck.get("epoch", "")
        steps = int(imgs * eps / 64) if m else ""
        test2 = ev.get("test", {}).get(2)
        rows.append((t, ck, tr, ev))
        L.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {:.0f} |".format(
            t, ck.get("release", "?"),
            "{:,}".format(imgs) if imgs else "all",
            eps, "{:,}".format(steps) if steps else "-",
            ck.get("epoch", "?"), fmt_km(tr.get("best_km")),
            fmt_km(test2[0]) if test2 else "n/a",
            "{:.1f}%".format(test2[3]) if test2 else "n/a",
            tr.get("secs_per_epoch", float("nan"))))

    L.append("\n## Beam width sweep (val)\n")
    L.append("| arm | " + " | ".join("k={}".format(k) for k in (1, 2, 4)) + " |")
    L.append("|---|---:|---:|---:|")
    for t, ck, tr, ev in rows:
        v = ev.get("val", {})
        if not v:
            continue
        L.append("| `{}` | ".format(t) + " | ".join(
            fmt_km(v[k][0]) if k in v else "-" for k in (1, 2, 4)) + " |")

    L.append("\n## System utilisation by stage\n")
    L.append("| stage | samples | GPU % (peak) | VRAM MB | shared MB | W | degC | CPU % | RAM GB |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stage, u in sorted(util_summary().items()):
        if stage == "idle":
            continue
        L.append("| {} | {} | {:.0f} ({:.0f}) | {:.0f} | {:.0f} | {:.0f} | "
                 "{:.0f} | {:.0f} | {:.1f} |".format(
                     stage, u["n"], u["gpu"], u["gpu_peak"], u["vram"],
                     u["shared"], u["watts"], u["temp"], u["cpu"], u["ram"]))
    L.append("\n`shared MB` is GPU memory that spilled into system DRAM. "
             "The desktop baseline here is 230-290 MB; materially above that "
             "means a process is over-committed and every access is crossing "
             "PCIe while nvidia-smi still reports 100% utilisation.\n")

    L.append("\n## Stage timings\n")
    L.append("| stage | minutes |")
    L.append("|---|---:|")
    for k, v in state.get("done", {}).items():
        L.append("| {} | {:.1f} |".format(k, v.get("secs", 0) / 60))
    if state.get("failed"):
        L.append("\n**Not completed:** " + ", ".join(
            "{} ({})".format(k, v) for k, v in state["failed"].items()) + "\n")

    txt = "\n".join(L) + "\n"
    safeio.write_text(a.out, txt)
    print(txt)


if __name__ == "__main__":
    main()
