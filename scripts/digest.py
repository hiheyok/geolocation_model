"""One readable summary of everything measured, written without a human present.

The runners leave their results scattered across per-image error caches, a dozen
BOOTSTRAP_*.md files and the training logs. That is fine when someone is
watching each stage land, and useless when they come back to it hours later.

This collects the lot into runs/FINAL.md: every arm that has cached errors, its
provenance read off the checkpoint, and every paired comparison that was run. It
touches nothing but files that already exist, so it cannot fail the run and is
safe to call even when every stage above it failed.
"""

import io
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config

ERRS = ROOT / "runs" / "errs"
RUNS = ROOT / "runs"
OUT = RUNS / "FINAL.md"


def arm_rows():
    rows = []
    for p in sorted(ERRS.glob("*_test_5000r_k2_d3.npy")):
        tag = p.name.replace("_test_5000r_k2_d3.npy", "")
        e = np.load(p)
        meta = {}
        ck = config.CHECKPOINTS / (tag + ".pt")
        if ck.exists():
            try:
                import torch
                c = torch.load(ck, map_location="cpu", weights_only=False)
                meta = {k: c.get(k) for k in
                        ("split_mode", "retr_mode", "epoch", "epochs_total",
                         "street_file", "init_from", "val_hit", "release")}
                # --retr-mode keeps its argparse default even when --retr
                # is off, so the arg alone would label the control "scalar"
                if not c.get("retr"):
                    meta["retr_mode"] = "none"
                meta["release"] = c.get("release") or "s01"
            except Exception:
                pass
        rows.append((tag, e, meta, p.stat().st_mtime))
    return rows


def main():
    rows = arm_rows()
    L = []
    L.append("# Results digest")
    L.append("")
    L.append("Generated {} by `scripts/digest.py`. Every arm below is the test "
             "split, 5,000 seeded-random images, beam k=2, ranked on s0-s2 -- "
             "the shipping protocol. Read the paired intervals further down "
             "before believing any ordering here: the median carries a ~16 km "
             "95% interval at this sample size."
             .format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    L.append("")

    L.append("## Arms, best hit rate first")
    L.append("")
    L.append("| arm | rel | split | mode | ep | median km | mean km | "
             "`<1km` | `<25km` | street file |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for tag, e, m, _ in sorted(rows, key=lambda r: -(r[1] < 25).mean()):
        L.append("| `{}` | {} | {} | {} | {} | {:.1f} | {:.1f} | {:.1f}% | "
                 "{:.1f}% | {} |"
                 .format(tag, m.get("release", "?"),
                         m.get("split_mode", "?"), m.get("retr_mode", "?"),
                         m.get("epochs_total") or m.get("epoch") or "?",
                         float(np.median(e)), float(e.mean()),
                         100 * float((e < 1).mean()), 100 * float((e < 25).mean()),
                         (m.get("street_file") or "?").replace(".f16.npy", "")))
    L.append("")

    best = max(rows, key=lambda r: (r[1] < 25).mean()) if rows else None
    if best is not None:
        tag, e, m, _ = best
        seq = [r for r in rows if r[2].get("split_mode") == "sequence"]
        c8 = [r for r in rows if r[2].get("split_mode") == "cell8"]
        L.append("**Headline.** Best by hit rate is `{}`: {:.1f} km median, "
                 "{:.1f}% under 25 km.".format(tag, float(np.median(e)),
                                               100 * float((e < 25).mean())))
        if c8:
            b8 = max(c8, key=lambda r: (r[1] < 25).mean())
            L.append("")
            L.append("**Quote this one externally instead.** The best `cell8` "
                     "arm is `{}`: {:.1f} km median, {:.1f}% under 25 km. "
                     "`cell8` holds whole z8 cells out of training *and* filters "
                     "the bank to the same cells, so neither the weights nor the "
                     "corpus has seen the region -- which is the condition the "
                     "OSV-5M protocol enforces and the `sequence` split does not."
                     .format(b8[0], float(np.median(b8[1])),
                             100 * float((b8[1] < 25).mean())))
        L.append("")

    L.append("## Paired comparisons")
    L.append("")
    L.append("A difference without one of these next to it is not a claim.")
    L.append("")
    for p in sorted(RUNS.glob("BOOTSTRAP_*.md"), key=lambda q: q.stat().st_mtime):
        body = io.open(p, encoding="utf-8").read().strip()
        keep = [ln for ln in body.split("\n")
                if ln.startswith("|") and "---" not in ln]
        if not keep:
            continue
        L.append("### `{}`".format(p.name))
        L.append("")
        L.append("\n".join(keep[:1] + ["|---|---|---|---|---|"] + keep[1:]))
        L.append("")

    L.append("## Stages")
    L.append("")
    marks = sorted((RUNS / "marks").glob("*.done"))
    L.append("{} stages completed: {}".format(
        len(marks), ", ".join("`{}`".format(m.stem) for m in marks)) or "none")
    L.append("")
    for name in ("runner.log",):
        f = RUNS / name
        if not f.exists():
            continue
        tail = io.open(f, encoding="utf-8", errors="replace").read().split("\n")
        skipped = [ln for ln in tail if " skip " in ln or "FAIL" in ln]
        if skipped:
            L.append("Skipped or failed:")
            L.append("")
            L.append("```")
            L.extend(skipped[-25:])
            L.append("```")

    io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("wrote {}  ({} arms, {} comparisons)".format(
        OUT, len(rows), len(list(RUNS.glob("BOOTSTRAP_*.md")))))


if __name__ == "__main__":
    main()
