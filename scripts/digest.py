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


MARK = {"verified by digest": "clean",
        "unrecorded; cache older than the checkpoint": "clean?",
        "unverified: cache is NEWER than the checkpoint": "**suspect**",
        "unverified: no digest and no timestamp": "unknown",
        "REBUILT SINCE TRAINING (proven by digest)": "**STALE**",
        "cache file is gone": "gone"}


NO_CKPT = "no checkpoint"


def cache_mark(state):
    """`no checkpoint` is not the same as `unknown`, and neither is `?`.

    20 arms in this workspace have a cached error file and no checkpoint left
    on disk. Nothing about them can be verified -- not the split, not the
    street file, not the cache they trained against -- and the first version
    of this column printed `?`, which reads like a formatting gap rather than
    "this row cannot be checked at all".
    """
    return MARK.get(state, NO_CKPT if state is None else state)


def verifiable(meta):
    """Can this arm be compared with anything? Provenance, not quality."""
    st = meta.get("cache")
    return st is not None and st not in (
        "REBUILT SINCE TRAINING (proven by digest)", "cache file is gone")


def headline_pool(rows):
    """`(comparable rows, how many were excluded)` for the headline.

    Split out of `main` because this is the part that decides what the file
    claims, and it was chosen by raw hit rate alone. That put `s10_b55_c6`
    first at 73.8% -- no checkpoint on disk, error file from 2026-09-01,
    three days before the same-sequence leak was fixed. Arms nothing can be
    checked about sort to the top precisely because they were measured before
    the fixes.
    """
    ok = [r for r in rows if verifiable(r[2])]
    return ok, len(rows) - len(ok)


def cache_note(rows):
    """Say what the cache column means, when any arm is not verified.

    This table sorts arms by hit rate. Training against the pre-2026-09-04
    leaky neighbour cache is worth 2 to 4 pp (`runs/LEAKTRAIN.md`), which is
    larger than most differences it ranks -- so without this the ranking is
    partly a ranking of which cache an arm happened to train against, and
    reads as a ranking of ideas.
    """
    # Only a verified cache is "nothing to say". The first version also
    # admitted None -- an arm with no checkpoint -- so a table of entirely
    # unverifiable arms printed no note at all. That is the third time this
    # week that treating unverifiable as fine has produced silence exactly
    # where the warning was needed (bootstrap.cache_lines, check_merged on
    # shallow clones). Absence of evidence is the thing being reported.
    states = {r[2].get("cache") for r in rows}
    if states <= {"verified by digest"}:
        return []
    return [
        "**Read the `cache` column before comparing two rows.** Every `knn_*` "
        "neighbour cache was rebuilt on 2026-09-04 04:05 after the "
        "same-sequence bank leak, and a checkpoint records the cache's path "
        "rather than its bytes. Training against the leaky one is worth "
        "**2 to 4 pp** -- more than most gaps in this table.",
        "",
        "* `clean` -- the cache on disk is the one it trained against, proven "
        "by digest.",
        "* `clean?` -- no digest recorded, but the cache is older than the "
        "checkpoint, so nothing contradicts it. Most pre-2026-09-08 arms.",
        "* `**suspect**` -- the cache is NEWER than the checkpoint. That is "
        "not proof (a copy or a `touch` moves mtime), but it cannot be ruled "
        "out.",
        "* `**STALE**` -- the digest disagrees. Proven trained against "
        "different bytes.",
        "",
        "Arms sharing a mark are comparable to each other. Comparing across "
        "marks is what produced a merged, wrong conclusion in PR #59.",
        "",
    ]


def arm_rows():
    rows = []
    # One row per arm, not one per file. Legacy and stamped caches coexist for
    # five tags in this workspace, and globbing every file listed each arm
    # twice -- with the current checkpoint's metadata attached to whichever
    # errors happened to be read.
    from bootstrap import find_err_cache, tag_of_err_cache
    seen = {}
    for f in sorted(ERRS.glob("*_test_5000r_k2_d3.npy")):
        seen.setdefault(tag_of_err_cache(f.name), None)
    for tag in sorted(seen):
        p = find_err_cache(tag)
        if p is None:
            continue        # no cache matching the checkpoint now on disk
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
                # Which neighbour cache this arm TRAINED against, by content.
                # Every `knn_*` was rebuilt on 2026-09-04 04:05 after the
                # same-sequence leak, and a checkpoint records the path, not
                # the bytes. Arms from either side of that are not comparable
                # to each other: the same street file scores 57.5% trained on
                # the leaky cache and 54.6% on the clean one, separated
                # (`runs/LEAKTRAIN.md`). This file ranks arms by hit rate, so
                # without the column the ranking silently sorts by cache.
                import knnmeta
                meta["cache"] = knnmeta.cache_provenance(c, tag)[0]
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
             "`<1km` | `<25km` | street file | cache |")
    L.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for tag, e, m, _ in sorted(rows, key=lambda r: -(r[1] < 25).mean()):
        L.append("| `{}` | {} | {} | {} | {} | {:.1f} | {:.1f} | {:.1f}% | "
                 "{:.1f}% | {} | {} |"
                 .format(tag, m.get("release", "?"),
                         m.get("split_mode", "?"), m.get("retr_mode", "?"),
                         m.get("epochs_total") or m.get("epoch") or "?",
                         float(np.median(e)), float(e.mean()),
                         100 * float((e < 1).mean()), 100 * float((e < 25).mean()),
                         (m.get("street_file") or "?").replace(".f16.npy", ""),
                         cache_mark(m.get("cache"))))
    L.append("")
    L += cache_note(rows)

    # An arm whose provenance cannot be established must not headline. The
    # previous best by raw hit rate was `s10_b55_c6` at 73.8%: no checkpoint
    # on disk, so no split, no street file and no cache, and an error file
    # from 2026-09-01 -- three days before the same-sequence leak was fixed.
    # Ranking by hit rate alone puts exactly the arms nothing can be checked
    # about at the top, because they are the ones measured before the fixes.
    ok, hidden = headline_pool(rows)
    best = max(ok, key=lambda r: (r[1] < 25).mean()) if ok else None
    if hidden:
        L.append("**{} of {} arms are excluded from the headline below** "
                 "because their provenance cannot be established -- no "
                 "checkpoint on disk, or a cache proven to have been rebuilt "
                 "since. They remain in the table with their marks. The best "
                 "raw hit rate in this file belongs to one of them, measured "
                 "before the same-sequence bank leak was fixed on 2026-09-04, "
                 "and it is not a result."
                 .format(hidden, len(rows)))
        L.append("")
    if best is not None:
        tag, e, m, _ = best
        seq = [r for r in ok if r[2].get("split_mode") == "sequence"]
        c8 = [r for r in ok if r[2].get("split_mode") == "cell8"]
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
