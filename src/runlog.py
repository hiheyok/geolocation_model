"""The runner's log and marker conventions, in one place.

Four review items (70-73) are one design: a stage is identified by its name
alone, and its log is append-only across days.  Both make a file say something
about work that a *different* run did.

This cost an hour on 2026-09-02.  A wait loop grepped a training log for
`FAIL`, matched a failure appended the previous day, and killed
`fuse-attn-pyr47` at epoch 11 of 12.  Nothing was wrong with the grep; the file
simply held two runs and did not distinguish them.

Two conventions follow, and both are about scoping a question to one run:

* **Read the last attempt, never the file.**  `last_attempt` slices off
  everything before the final header.  Every parser here wants the current
  attempt -- a per-epoch curve mixing two attempts is not a curve, and the
  calibration parser took the *first* match in the file, which after a retry
  is the timing of the run that failed.
* **A marker records what it marked.**  A stage's marker is a filename, so a
  changed command, a changed release or a rebuilt input all leave it looking
  finished.  Writing the identity into the marker and comparing on read is
  what makes it a record rather than a flag.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HEADER = "==== attempt"
ATTEMPT = re.compile(r"^==== attempt \d+ at .*$", re.M)


def last_attempt(text):
    """Everything after the final attempt header, or all of it if there is none.

    A log with no header is a single run and needs no slicing -- that is the
    common case for anything not launched by the runner, and returning nothing
    for it would be a worse failure than the one this fixes.
    """
    hits = list(ATTEMPT.finditer(text))
    return text[hits[-1].end():] if hits else text


def attempts(text):
    """How many attempt headers the log holds. 0 means it was never retried."""
    return len(ATTEMPT.findall(text))


def marker_identity(name, argv, release=None, **extra):
    """What a completion marker is a completion *of*.

    Keyed on the command rather than the stage name: renaming is handled
    separately by `legacy_check`, but re-pointing a stage at different
    arguments -- a different bank, a different width, a different number of
    epochs -- is the case that silently reused a finished marker and skipped
    the work.
    """
    rec = {"stage": name, "argv": list(argv), "release": release}
    rec.update(extra)
    return json.dumps(rec, sort_keys=True)


# Recorded in the marker but never compared: the outputs are stamped *after*
# the stage produced them, so they cannot be part of an identity computed
# before it runs. They are verified separately, by `outputs_intact`.
NOT_IDENTITY = ("outputs",)

# Flags whose value names a file the stage READS, with the directory that
# resolves it. Deliberately only the unambiguous ones: `--tag` means the
# checkpoint to write under train.py and the checkpoint to read under
# bootstrap.py, so guessing it wrong would either strand a stage forever or
# certify one that never ran.
INPUT_FLAGS = {"--street-file": "street", "--knn-file": "street",
               "--basis": "street", "--bank": "street", "--init": "ckpt",
               # `knn_gap --a X --b Y` and `concat_street --a X --b Y`. Both
               # name street-cache files, and a comparison stage is exactly
               # where an unstamped input does the most damage: the argv is
               # character for character the same whichever build of X and Y
               # is on disk, so a rebuilt arm left its old gap "satisfied"
               # and the runner reported a comparison it had not made.
               "--a": "street", "--b": "street",
               # The whole pool -> stack -> project chain. Every one of these
               # names a file some earlier stage wrote, under a name that does
               # not change when its contents do -- so without them a rebuilt
               # pool left the fit, the stacks, the projection, the index and
               # the comparison all "satisfied", and the runner exited 0 on an
               # experiment whose first stage had been replaced. The sweep hid
               # it in the default path (a deleted intermediate is caught by
               # `outputs_intact`) and `--keep` did not.
               "--src": "street", "--tiles": "street",
               "--base": "street", "--ext": "street",
               # `bootstrap --tags a,b,c` names the checkpoints a comparison
               # is a comparison OF, and its argv does not move when any of
               # them is retrained. So a rebuilt arm re-ran its training,
               # found the comparison satisfied, kept the previous report and
               # exited 0. Same shape as `--a`/`--b`, one list wide.
               "--tags": "ckptlist"}


_REFS = {}


def checkpoint_refs(p):
    """The cache files a checkpoint names, or None if it cannot be read.

    A comparison's identity has to follow these. `--tags` stamps the
    checkpoints, which catches a retrained arm -- but an arm can stay exactly
    as it is while the bank or the neighbour table underneath it is rebuilt,
    and beam search reads both at evaluation time. Then nothing in the argv
    or the checkpoints moved, the comparison stayed satisfied, and the
    previous report survived a rebuild it no longer describes.

    Memoised on (size, mtime): `identity` is computed more than once per
    stage and this opens every checkpoint a bootstrap names.
    """
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_size, st.st_mtime)
    if key in _REFS:
        return _REFS[key]
    try:
        import torch
        c = torch.load(p, map_location="cpu", weights_only=False)
        refs = [c.get("street_file"), c.get("knn_file")]
    except Exception:
        refs = []                      # present but unreadable, which is not
    _REFS[key] = refs                  # the same as absent
    return refs


def street_path(cache, val):
    """The file a street-cache flag names, under whichever convention it uses.

    Returns the first candidate that exists, else the bare name so a genuinely
    missing input still stamps `absent` rather than resolving to some other
    file that happens to be there.
    """
    for cand in (cache / val, cache / (val + ".f16.npy"),
                 cache / (val + "_meta.npz")):
        if cand.exists():
            return cand
    return cache / val


def stage_inputs(argv):
    """Paths a stage reads, derived from its own command line.

    Item 70 was closed with an identity of name, argv and release, and that
    covers a re-pointed stage but not a rebuilt one: the argv naming
    `knn_pca768_bank70_...npz` is character for character the same argv
    whether that file is the leaky cache or the clean rebuild. For the
    same-sequence fix specifically, an older cache could be dropped in under
    the same name and the rebuild stage would go on reporting satisfied.

    So the identity stamps the *contents* of what the command names. Included
    unconditionally: `dataset.parquet`, which defines row order for every
    artifact downstream, and the entry script itself, so editing the
    implementation invalidates results produced by the previous one.

    The boundary worth stating: the entry script is stamped, its imports are
    not. Stamping all of `src/` would invalidate every marker in the queue on
    any edit, which in a research runner means re-running finished work several
    times a day -- a cure that would simply be turned off.
    """
    import config
    import safeio

    paths, literal = {}, {}
    if argv:
        paths["code:" + str(argv[0])] = ROOT / str(argv[0])
    paths["dataset.parquet"] = config.DATASET_PARQUET
    for i, tok in enumerate(argv[:-1]):
        where = INPUT_FLAGS.get(str(tok))
        if where is None:
            continue
        val = str(argv[i + 1])
        if where == "ckpt":
            paths[str(tok) + " " + val] = config.CHECKPOINTS / (val + ".pt")
            continue
        if where == "ckptlist":
            # Stamped per tag rather than as one blob, so the record says
            # which arm moved rather than only that something did -- and
            # transitively, because what a comparison actually reads is each
            # arm's weights AND the bank and neighbour table those weights
            # name.
            for t in (x.strip() for x in val.split(",")):
                if not t:
                    continue
                ck = config.CHECKPOINTS / (t + ".pt")
                paths["{} {}".format(tok, t)] = ck
                refs = checkpoint_refs(ck)
                if refs is None:
                    continue           # absent; its own stamp says so
                if not refs:
                    literal["{} {} (unreadable)".format(tok, t)] = "unreadable"
                for r in refs:
                    if r:
                        paths["{} {} -> {}".format(tok, t, r)] = street_path(
                            config.STREET_CACHE, str(r))
            continue
        # Three naming conventions live in this cache and the flags do not
        # distinguish them: `--knn-file` names `knn_x.npz`, `--base` names the
        # stem `pyr_l0l1_b115`, and `--ext bank_ext70` names a corpus whose
        # file is `bank_ext70_meta.npz`. Resolving only the bare name would
        # leave two of the three recorded as `absent` in perpetuity, which
        # reads like a stamp and is not one.
        paths[str(tok) + " " + val] = street_path(config.STREET_CACHE, val)
    out = {k: safeio.file_stamp(v) for k, v in sorted(paths.items())}
    out.update(literal)
    return out


def stage_outputs(argv):
    """Paths a stage is expected to have produced, stamped after it ran.

    Only what actually exists is recorded, which is what keeps this safe: a
    flag resolved to the wrong path simply goes unrecorded, rather than
    becoming an output that can never be found and a stage that re-runs
    forever.
    """
    import config
    import safeio

    out = {}
    cand = []
    for i, tok in enumerate(argv[:-1]):
        val = str(argv[i + 1])
        if str(tok) in ("--out", "--export"):
            cand += [ROOT / val, config.STREET_CACHE / val,
                     config.STREET_CACHE / (val + ".f16.npy")]
            # `project_street` writes `<out>_pca.npz` beside the projection,
            # and that basis is what every later extension is projected with.
            # Recorded only if it is there, so this costs nothing for the
            # stages that write no basis -- but without it, deleting a basis
            # leaves its fitting stage satisfied and the next projection fails
            # with no indication of which stage should have re-run.
            cand.append(config.STREET_CACHE /
                        (val.replace(".f16.npy", "") + "_pca.npz"))
        elif str(tok) == "--tag" and str(argv[0]).endswith("train.py"):
            cand.append(config.CHECKPOINTS / (val + ".pt"))
    for p in cand:
        try:
            if p.exists() and p.is_file():
                out[p.name] = safeio.file_stamp(p)
        except OSError:
            pass
    return out


def outputs_intact(text):
    """Whether every output a marker recorded is still there, unchanged.

    Deleting a checkpoint and leaving its marker used to make the training
    stage skip, so the run "succeeded" with no model at the end of it. The
    same check catches an output replaced under its own name, which is the
    dangerous direction for a rebuilt k-NN cache.

    Returns (ok, why). A marker that recorded no outputs is not evidence of
    anything, so it passes -- there is nothing to contradict.
    """
    import safeio

    try:
        rec = json.loads((text or "").strip())
    except ValueError:
        return True, ""
    got = rec.get("outputs") or {}
    if not isinstance(got, dict):
        return True, ""
    import config

    for name, want in sorted(got.items()):
        for root in (ROOT, config.CHECKPOINTS, config.STREET_CACHE, ROOT / "runs"):
            p = root / name
            if p.exists():
                now = safeio.file_stamp(p)
                if now != want:
                    return False, "{} changed since it was written".format(name)
                break
        else:
            return False, "{} is gone".format(name)
    return True, ""


def marker_matches(text, identity):
    """Whether a marker's contents describe this stage: True, False or None.

    None means the marker predates the identity convention, which every marker
    already on disk does. That is a third answer on purpose: treating those as
    a mismatch would re-run every finished stage in the queue, and treating
    them as a match without saying so is the hole being closed.

    A marker that *looks* like the new format and does not parse is a
    mismatch, not a legacy file. Marker writes were not atomic, so an
    interrupted write leaves a truncated `{...` on disk -- and since the
    caller reads None as satisfied, the stage that was killed mid-write would
    be skipped as complete on the next run. Legacy markers do not begin with
    a brace, so the two cases are separable and only one of them is old.
    """
    text = (text or "").strip()
    if not text.startswith("{"):
        return None
    try:
        got, want = json.loads(text), json.loads(identity)
    except ValueError:
        return False
    if not (isinstance(got, dict) and isinstance(want, dict)):
        return False
    drop = set(NOT_IDENTITY)
    # A marker written before input stamping records name, argv and release
    # and nothing else. Comparing it against an identity that also carries
    # input stamps would mark every finished stage in the queue stale and
    # re-run it, so the older record is compared on the fields it actually
    # has -- the same migration the plain-text markers got, one convention
    # later. It answers the weaker question, which is why it warns.
    if "inputs" not in got:
        drop.add("inputs")
    return ({k: v for k, v in got.items() if k not in drop}
            == {k: v for k, v in want.items() if k not in drop})


def marker_is_pre_inputs(text):
    """Whether a marker predates input stamping, so its match is the weak one."""
    try:
        rec = json.loads((text or "").strip())
    except ValueError:
        return False
    return isinstance(rec, dict) and "inputs" not in rec
