"""Paired bootstrap over arms, so an ordering is a claim rather than a ranking.

The median of a heavy-tailed great-circle error carries a ~16 km 95% interval at
n ~ 5,000.  Most differences that have separated arms in this project are 5-10
km, which is inside that.  The hit rate `<25 km` carries ~0.6 pp at the same n
and usually can separate them.

Arms see the same images, so the difference is paired and the interval is much
tighter than two independent intervals would suggest -- which is the whole point
of doing it this way rather than eyeballing two medians.

Per-image errors are cached, so re-running a comparison costs nothing.

    python scripts/bootstrap.py --tags a,b,c --split test
"""

import argparse
import hashlib
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config
import knnmeta
import names
import safeio
import splits as sp
from beam import source_for
from dataset import GeoStepDataset, street_table
from evaluate import check_split, evaluate, load_model, street_file_for

CACHE = ROOT / "runs" / "errs"


def ckpt_stamp(tag):
    """Identify the *file*, not the tag.

    A tag is a name the runner reuses: re-training `d768-b350-e6` writes a new
    checkpoint over the old one, and a cache keyed only by the tag then serves
    the previous model's errors under the new model's name -- silently, and with
    entirely plausible numbers.  Size and mtime are enough to notice a rewrite
    and cost nothing; hashing a 200 MB checkpoint on every lookup would not.
    """
    return safeio.file_stamp(config.CHECKPOINTS / (tag + ".pt"))


def cache_lines(tags, caches):
    """Report lines for the caches these arms trained against, or [].

    Split out of `main` because the condition is the part that went wrong.
    The first version required the arms to DISAGREE, copying the idiom from
    the weight-decay block -- so two equally unverifiable arms produced no
    warning at all, which is the silence this check exists to break,
    reproduced one layer above the check itself.

    Every arm that is not verified is reported, whether or not they differ.
    """
    bad = [t for t in tags if caches[t][0] != knnmeta.CACHE_OK]
    if not bad:
        return []
    proven = [t for t in bad if caches[t][0] == knnmeta.CACHE_REBUILT]
    L = ["**{} of these {} arms cannot be shown to have trained against the "
         "cache now on disk.** Every `knn_*` cache was rebuilt on 2026-09-04 "
         "after the same-sequence bank leak; a checkpoint records the cache's "
         "PATH, not its content. Measured, training against the leaky one is "
         "worth **2 to 4 pp** -- larger than most contrasts in this table.{}"
         .format(len(bad), len(tags),
                 "" if not proven else
                 " For {} the mismatch is PROVEN by digest.".format(
                     ", ".join("`{}`".format(x) for x in proven))),
         "",
         "| arm | cache it trained against |",
         "|---|---|"]
    for t in tags:
        st, why = caches[t]
        L.append("| `{}` | {} -- {} |".format(t, st, why))
    L.append("")
    return L


def provenance(tag, split, dev="cpu"):
    """(release, split_mode, split_hash) for a tag, validated against the data.

    Read before the cache is consulted, not after. check_split used to run only
    on a cache miss, so a cached arm was returned with no proof that it came
    from the same release or the same split as the arms it is about to be
    paired against -- and a paired bootstrap over two different row sets is not
    a comparison, it is noise with a confidence interval on it.

    Only the checkpoint dict is read here, not the model: provenance costs a
    torch.load, and building a GeoAgent to answer it would make every cache hit
    pay for the thing the cache exists to avoid.
    """
    ck = torch.load(config.CHECKPOINTS / (tag + ".pt"), map_location="cpu",
                    weights_only=False)
    mode = ck.get("split_mode", sp.PRIMARY)
    # The canonical hash, so an arm carrying the pre-2026-09-03 weak digest
    # and one carrying the strong digest over the same rows compare equal.
    # check_split has already printed what the weak digest cannot prove.
    canon = check_split(ck, mode, split)
    return (ck.get("release"), mode, canon or ck.get("split_hash"))


def tag_of_err_cache(name):
    """Recover the tag from an error-cache filename, stamp or not."""
    import re
    # <tag>[_<12-hex ckpt stamp>][_i<8-hex inputs stamp>]_<split>_...
    m = re.match(r"^(.*?)(?:_[0-9a-f]{12})?(?:_i[0-9a-f]{8})?"
                 r"_(?:test|val|train)_\d+r_k\d+_d\d+$",
                 name.replace(".npy", ""))
    return m.group(1) if m else name


def find_err_cache(tag, split="test", n=5000, beam_k=2, score_steps=3):
    """The newest cached error array for a tag, stamped or not.

    The checkpoint stamp added on 2026-09-03 changed these filenames from
    `<tag>_<split>_...` to `<tag>_<stamp>_<split>_...`, which silently broke
    every reader that built the old name by hand -- marathon fell back to
    validation data, and digest parsed the stamp into the tag. One resolver so
    the format is written down once.
    """
    tail = "{}_{}r_k{}_d{}.npy".format(split, n, beam_k, score_steps)
    hits = [h for h in CACHE.glob("{}_*{}".format(tag, tail))
            if tag_of_err_cache(h.name) == tag]
    if not hits:
        return None

    # Newest-by-mtime is not good enough: after retraining, the previous
    # model's cache is still the newest file for this tag until the new one is
    # computed, so "newest" hands back the old model's errors under the new
    # model's name. Require the current checkpoint's stamp.
    ckp = config.CHECKPOINTS / (tag + ".pt")
    if ckp.exists():
        want = "{}_{}{}_{}".format(tag, ckpt_stamp(tag), inputs_stamp(tag),
                                   tail)
        for h in hits:
            if h.name == want:
                return h
        # An unstamped file is only trustworthy if it postdates the checkpoint
        # -- the same rule the migration in errors_for applies.
        legacy = CACHE / "{}_{}".format(tag, tail)
        if legacy.exists() and not inputs_stamp(tag)                 and legacy.stat().st_mtime_ns > ckp.stat().st_mtime_ns:
            return legacy
        return None
    # No checkpoint on disk (a deleted arm): nothing to match against, so the
    # newest is the best available answer.
    return max(hits, key=lambda h: h.stat().st_mtime)


def inputs_stamp(tag):
    """Fingerprint everything outside the checkpoint that decides the errors.

    `ckpt_stamp` identifies the weights. The errors also depend on the
    retrieval cache and the bank those weights are scored against, and nothing
    in the key covered them -- so rebuilding the kNN cache left every cached
    error array valid-looking and stale.

    That is not hypothetical: on 2026-09-04 the cache was rebuilt to remove
    same-sequence leakage, and the re-measurement returned in six seconds with
    the pre-rebuild numbers, to four significant figures. Same failure shape as
    the query cache that fingerprinted filenames rather than contents, one
    artifact over.

    Returns "" when the checkpoint names neither, so a non-retrieval arm keeps
    its existing key rather than being invalidated for no reason.
    """
    try:
        ck = torch.load(config.CHECKPOINTS / (tag + ".pt"), map_location="cpu",
                        weights_only=False)
    except Exception:
        # A file that is not a loadable checkpoint has no inputs to stamp.
        # Return "" rather than raising: this is a key helper, and whatever is
        # wrong with the file will surface loudly at load_model.
        return ""
    parts = [safeio.file_stamp(config.STREET_CACHE / ck[f])
             for f in ("knn_file", "street_file") if ck.get(f)]
    if not parts:
        return ""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()[:8]
    return "_i" + h


def errors_for(tag, split, n, beam_k, score_steps, dev, source,
               retr_off=False):
    """Per-image error, keyed by checkpoint, its retrieval inputs, and protocol."""
    CACHE.mkdir(parents=True, exist_ok=True)
    # "r" marks the seeded random sample; the old caches were the first n rows
    # `_noretr` in the key, not alongside it: a retrieval-off run and a normal
    # one are different measurements of the same checkpoint, and a key that
    # cannot tell them apart would serve one as the other.
    key = "{}_{}{}_{}_{}r_k{}_d{}{}.npy".format(
        tag, ckpt_stamp(tag), inputs_stamp(tag), split, n, beam_k, score_steps,
        "_noretr" if retr_off else "")
    p = CACHE / key
    if p.exists():
        return np.load(p)

    # Adopt a pre-stamp cache only when its own mtime proves it was written
    # after the checkpoint now on disk.  That is the exact condition the stamp
    # enforces going forward, so this migrates the honest files and recomputes
    # the ones that cannot be shown to match -- rather than trusting the name.
    # Migration is deliberately NOT extended to the checkpoint-stamped keys
    # written before the inputs stamp existed. Those were computed against
    # whatever retrieval cache happened to be on disk at the time, which for
    # every arm on this machine is the leaky one; adopting them by mtime would
    # reintroduce exactly the staleness this key exists to prevent. The
    # pre-stamp files below are older still and are migrated only when their
    # mtime proves they postdate the checkpoint.
    legacy = (CACHE / "{}_{}_{}r_k{}_d{}.npy".format(
        tag, split, n, beam_k, score_steps)) if not retr_off else None
    ckp = config.CHECKPOINTS / (tag + ".pt")
    if legacy is not None and legacy.exists() and \
            legacy.stat().st_mtime_ns > ckp.stat().st_mtime_ns \
            and not inputs_stamp(tag):
        e = np.load(legacy)
        np.save(p, e)
        return e

    model, ck, d_street = load_model(tag, dev)
    mode = ck.get("split_mode", sp.PRIMARY)
    check_split(ck, mode, split)
    sf = street_file_for(ck, d_street)
    tbl = None
    if ck.get("retr_mode") in ("pos", "dual"):
        tbl = street_table(config.STREET_CACHE / sf, dev)
    ds = GeoStepDataset(split, street_file=sf, split_mode=mode,
                        knn_file=ck.get("knn_file"),
                        knn_k=ck.get("retr_k", 0) if ck.get("retr") else 0)
    m = evaluate(model, ds, source, dev, n, beam_k=beam_k,
                 top_m=max(4, beam_k), greedy=(beam_k == 1),
                 score_steps=score_steps, street_gpu=tbl, retr_off=retr_off)
    np.save(p, m["err"])
    del model, tbl
    torch.cuda.empty_cache()
    return m["err"]


def paired(a, b, reps, rng, thresh=25.0):
    """95% CI on (a - b) for the median and for the <thresh km hit rate."""
    n = len(a)
    dm = np.empty(reps)
    dh = np.empty(reps)
    for i in range(reps):
        s = rng.integers(0, n, n)
        dm[i] = np.median(a[s]) - np.median(b[s])
        dh[i] = (a[s] < thresh).mean() - (b[s] < thresh).mean()
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return q(dm), q(dh)


def verdict(lo, hi):
    """Separated only if the whole interval is strictly one side of zero.

    `(lo > 0) == (hi > 0)` calls [-0.5, 0.0] separated, because both
    comparisons are false -- an interval that contains zero reported as a real
    effect. Hit-rate differences are multiples of 1/n, so an endpoint landing
    exactly on zero is not hypothetical.
    """
    return "separated" if (lo > 0 or hi < 0) else "inside noise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma separated checkpoint tags")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--score-steps", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--retr-off", action="store_true",
                    help="score with the retrieval prior removed at inference, "
                         "which measures corpus dependency directly")
    a = ap.parse_args()

    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # One source per arm, not one shared: two arms in the same table may be
    # tokenised at different `sub`, and a shared source would score one of them
    # against the other's map representation.
    rng = np.random.default_rng(config.SPLIT_SEED)
    # Every arm must be measured over the same images before any of them are
    # paired. Same release, same split mode, same split hash, same n: with the
    # seeded sample in evaluate() those four fix the row set exactly. Without
    # this a `sequence` arm and a `cell8` arm pair positionally and produce a
    # tight interval over unrelated images.
    prov = {t: provenance(t, a.split) for t in tags}
    if len(set(prov.values())) > 1:
        lines = "\n".join("  {:<28} release={} split={} hash={}".format(t, *v)
                           for t, v in prov.items())
        raise SystemExit(
            "these arms were not measured over the same rows, so a paired "
            "bootstrap between them is meaningless:\n" + lines)

    src_for_tag = {}

    errs = {}
    for t in tags:
        ckt = torch.load(config.CHECKPOINTS / (t + ".pt"), map_location="cpu",
                         weights_only=False)
        key = (ckt.get("map_cache"), ckt.get("map_sub", 1))
        if key not in src_for_tag:
            src_for_tag[key] = source_for(ckt)
        errs[t] = errors_for(t, a.split, a.n, a.beam, a.score_steps, dev,
                             src_for_tag[key], a.retr_off)
        print("{:<24} n={:,}  median {:7.1f} km  mean {:8.1f}  <25km {:5.1%}"
              .format(t, len(errs[t]), float(np.median(errs[t])),
                      float(errs[t].mean()), float((errs[t] < 25).mean())),
              flush=True)

    lens = {len(v) for v in errs.values()}
    if len(lens) != 1:
        raise SystemExit("arms cover different image counts {} -- a paired test "
                         "needs the same images in the same order".format(lens))

    L = ["", "paired bootstrap, {} split, {:,} images, {:,} resamples, k={}, "
         "ranked on s0-s{}".format(a.split, len(errs[tags[0]]), a.reps,
                                   a.beam, a.score_steps - 1), ""]
    # A legend, because the tags alone do not say what differs between arms:
    # d1536-b265-e6 and d768-b265-e4 differ in width AND bank AND epochs, and
    # nothing in either name says so. names.describe reads from an explicit
    # table, so an arm it cannot name honestly is simply left out.
    known = [(t, names.describe(t)) for t in tags if names.describe(t)]
    if known:
        L.append("| arm | what it is |")
        L.append("|---|---|")
        for t, d in known:
            L.append("| `{}` | {} |".format(t, d))
        L.append("")
    # Which weight-decay grouping each arm was trained under. Not a refusal:
    # comparing the two rules IS the experiment for review item 29. But a table
    # that pools arms trained under different rules without saying so is how a
    # protocol change gets read as a result, so it is stated whenever the arms
    # disagree.
    # A checkpoint with no `wd_legacy` field was trained BEFORE the flag
    # existed, which means it was trained under the legacy substring rule --
    # so a missing field is "legacy", not "by module type". Reading it as
    # falsy labelled all 122 existing arms as fixed, in the very table added
    # to stop the rule being misread.
    def _rule(t):
        v = torch.load(config.CHECKPOINTS / (t + ".pt"), map_location="cpu",
                       weights_only=False).get("wd_legacy")
        if v is None:
            return "legacy substring (predates the flag)"
        return "legacy substring" if v else "by module type"

    # What the recorded seed actually covers. Arms trained before `neg_random`
    # existed drew their sink negatives from OS entropy, so their seed fixed
    # weight init, batch order and dropout but not the off-path tiles -- two
    # runs of the same command saw different negatives. Saying so beside the
    # numbers is the point: a table that reports a seed per arm invites the
    # reader to assume the arms are reproducible to it.
    _seen = {}

    def _load(t):
        if t not in _seen:
            _seen[t] = torch.load(config.CHECKPOINTS / (t + ".pt"),
                                  map_location="cpu", weights_only=False)
        return _seen[t]

    def _negs(t):
        v = _load(t).get("neg_random")
        if v is None:
            return "OS entropy (predates the flag)"
        return "OS entropy" if v else "seeded"

    negs = {t: _negs(t) for t in tags}
    if len(set(negs.values())) > 1:
        L.append("**Sink negatives are not drawn the same way across these "
                 "arms.** A seeded arm sees the same off-path tiles on a "
                 "re-run; an OS-entropy arm does not, whatever seed it "
                 "records.")
        L.append("")
        L.append("| arm | sink negatives |")
        L.append("|---|---|")
        for t in tags:
            L.append("| `{}` | {} |".format(t, negs[t]))
        L.append("")

    rules = {t: _rule(t) for t in tags}
    if len(set(rules.values())) > 1:
        L.append("**Weight-decay grouping differs between these arms.** "
                 "The substring rule exempted the learned retrieval keys "
                 "(196,608 parameters, 3.7% of the model) from decay.")
        L.append("")
        L.append("| arm | weight-decay grouping |")
        L.append("|---|---|")
        for t in tags:
            L.append("| `{}` | {} |".format(t, rules[t]))
        L.append("")
    # The cache each arm was TRAINED against, by content rather than by name.
    # Every `knn_*` was rebuilt on 2026-09-04 04:05 after the same-sequence
    # bank leak was found, and checkpoints on either side of that record the
    # same filename. Comparing across it measures the cache: the same street
    # file trained on the leaky bytes scores 57.5% and on the clean bytes
    # 54.6%, separated (`runs/LEAKTRAIN.md`). This file already warns about
    # weight-decay grouping, which is worth nothing by comparison.
    L += cache_lines(tags, {t: knnmeta.cache_provenance(_load(t), t)
                            for t in tags})
    L.append("| contrast | median diff, 95% CI | | <25km diff, 95% CI | |")
    L.append("|---|---|---|---|---|")
    for x, y in itertools.combinations(tags, 2):
        (ml, mh), (hl, hh) = paired(errs[x], errs[y], a.reps, rng)
        L.append("| {} vs {} | [{:+.1f}, {:+.1f}] km | {} | [{:+.2f}, {:+.2f}] pp | {} |"
                 .format(x, y, ml, mh, verdict(ml, mh),
                         100 * hl, 100 * hh, verdict(hl, hh)))
    L.append("")
    L.append("A positive median difference means the first arm is worse "
             "(more km); a positive <25km difference means it is better.")
    txt = "\n".join(L)
    print(txt)
    if a.out:
        safeio.write_text(a.out, txt + "\n")


if __name__ == "__main__":
    main()
