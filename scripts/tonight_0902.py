"""Overnight queue for 2026-09-02, ordered around a hard tile-server deadline.

**The tile server goes offline at 09:30.** Beam rollout fetches z12/z16 tiles
live, so every stage that evaluates a policy needs it and every stage that only
touches cached embeddings does not. That single fact fixes the ordering: the
tile-dependent work runs first against a 09:15 cutoff, and the long GPU job that
needs no network runs after, where it can overrun into the day harmlessly.

Queue, in order:

  train-d768-b350-e2/e4/e6, boot-d768-b350
                       finish the 768-d ladder on the 3.40M bank.  Shared
                       markers with w768b70.py, so anything that driver already
                       completed is skipped rather than repeated.  Retries are
                       raised to 3: the first attempt died on a transient
                       `CUDA error: unknown error` 20 minutes in, with no TDR
                       in the Windows event log and the same code path having
                       trained d1536-b350-e6 an hour earlier.  One retry is not
                       enough budget to absorb that unattended.
  multiq-*-diverse/near
                       the multi-photograph curve -- the headline unrun
                       experiment, and the direct test of whether query-side
                       density substitutes for bank-side density.  Two
                       orderings of the extra photographs: round-robin across
                       drives, and nearest-first as a control.
  kartaview-d768-b350-e6
                       the held-out KartaView set through the new arm.  Asks
                       whether 2.65M -> 3.40M moves the 442 km external number
                       at all, which is the sharpest available test of the
                       coverage explanation.
  pyrcache-hr47k       pyramid cache over the *full* 47,646-image harvest.  No
                       tiles.  ~10 h at the measured 2.6 img/s per encoder, and
                       resumable through its own `_done` mask, so an unfinished
                       run is progress rather than waste.
  fuse-attn-pyr47      the deconfounded attention arm.  The -15.75 pp result
                       trained on 14,938 images; this one gets 3.2x that, into
                       its own cache so the comparison target is not overwritten.

Everything is marker-gated and idempotent: re-running resumes.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

REL = "s10"
if not os.environ.get("OSV_RELEASE"):
    os.environ["OSV_RELEASE"] = REL

import config
import overnight as O
from overnight import (Stage, legacy_check, load_state, log,
                       run_stage)
from marathon import ARCH, tiles_up
# Reuse w768b70's own command builder rather than restating it: the ladder must
# be byte-identical to what that driver would have run, or the markers lie.
from w768b70 import train as w70_train

DROP = "0.3"

# The 1536-d bank, for asking whether dropout fixes the best benchmark arm too.
SRC1536 = "pool_bal_bank70.f16.npy"
KNN1536 = config.knn_name(SRC1536, "sequence", ext="bank_ext70")


def train1536(tag, init=None, drop=None, epochs=2):
    """The d1536 ladder, optionally with neighbour dropout. Same recipe as the
    768-d one but against the pooled bank, which is 10.75 GB and therefore runs
    on the memmap tier -- about 4% slower, not enough to matter."""
    cmd = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
           "--batch", "64", "--limit", "400000",
           "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
           "--split-mode", "sequence", "--street-file", SRC1536,
           "--knn-file", KNN1536] + ARCH
    if init:
        cmd += ["--init", init]
    if drop:
        cmd += ["--retr-drop", drop]
    return cmd


def drop_sink(tag, init, p, k):
    """Dropout and extra sink keys together, so the sink arm is one variable
    away from the best arm on the curve rather than from the no-drop base."""
    return w70_train(tag, init) + ["--retr-drop", p, "--sink-k", str(k)]


SUB2 = "cache/map/s10_sub2"


def drop_sub2(tag, init, p):
    """Best-p dropout with 2x2 sub-patch map tokens -- one variable from the
    best arm. Each 32x32 patch becomes four class histograms instead of one, so
    within-patch layout survives. Measured on 16,600 z12 patches, road
    orientation is recoverable at 74.8% from sub=2 against 52.7% from the
    current token, where 52.6% is the majority rate."""
    return w70_train(tag, init) + ["--retr-drop", p, "--map-cache", SUB2]


def drop_p(tag, init, p):
    """A d768 ladder at an arbitrary dropout probability, for the p-curve."""
    return w70_train(tag, init) + ["--retr-drop", p]


def drop_train(tag, init=None):
    """The same ladder, with the retrieval prior's neighbours randomly hidden.

    The 2x2 on 5,000 held-out KartaView images showed the external loss follows
    the MODEL, not the bank: d768-b350-e6 is 0.76-1.20 pp worse than
    d768-b265-e6 whichever bank it is given at inference, while the bank effect
    at fixed model is -0.12 pp (inside noise) to -0.56 pp. So training against
    dense retrieval teaches a dependency that does not survive off-domain,
    where top-1 similarity is 0.71 rather than 0.90. Hiding neighbours during
    training is the direct test of that reading."""
    return w70_train(tag, init) + ["--retr-drop", DROP]

TAG = "d768-b350-e6"


def alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq {}".format(pid)],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def at_today(hh, mm):
    n = datetime.now()
    t = n.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t < n:                      # already past -> that time tomorrow
        t += timedelta(days=1)
    return t.timestamp()


def main():
    wait_pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 14.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()

    # The tile server is available past 09:30 after all, so the cutoff no
    # longer orders the queue: stages run in value order instead. Kept as a
    # far-future value rather than deleted, since the gate is what makes an
    # unattended run safe when a cutoff does exist.
    tile_off = float(os.environ.get("TILE_OFF_TS", O.now() + 30 * 86400))
    deadline = O.now() + hours * 3600

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("tonight_0902: tile work before {}, then the pyramid cache"
        .format(O.hhmm(tile_off)))
    log("deadline {}  ({:.1f} h)   tile server {}"
        .format(O.hhmm(deadline), hours, "up" if tiles_up() else "DOWN"))

    if wait_pid:
        log("waiting for pid {} (w768b70) to exit".format(wait_pid))
        cap = O.now() + 6 * 3600
        while alive(wait_pid) and O.now() < cap:
            time.sleep(60)
        log("pid {} {}".format(wait_pid,
                               "still alive, proceeding anyway"
                               if alive(wait_pid) else "gone"))

    plan = [
        # (stage, needs the tile server)
        #
        # Stage names carry the arm they produce, so a log line says what was
        # trained. `legacy_check` maps each onto the opaque name w768b70.py
        # used, so rungs already finished under the old name are not repeated.
        (Stage("train-d768-b350-e2", w70_train("d768-b350-e2"),
               release=REL, est=25 * 60, retries=3,
               check=legacy_check("w70_2")), True),
        (Stage("train-d768-b350-e4", w70_train("d768-b350-e4",
                                               "d768-b350-e2"),
               release=REL, est=25 * 60, retries=3,
               check=legacy_check("w70_4")), True),
        (Stage("train-d768-b350-e6", w70_train("d768-b350-e6",
                                               "d768-b350-e4"),
               release=REL, est=25 * 60, retries=3,
               check=legacy_check("w70_6")), True),
        (Stage("boot-d768-b350",
               ["scripts/boot_existing.py", "--tags",
                "d1536-b350-e6,d768-b265-e6,d768-b350-e2,d768-b350-e4,"
                "d768-b350-e6",
                "--out", str(O.RUNS / "BOOTSTRAP_w768_b70.md")],
               release=REL, est=8 * 60, retries=2,
               check=legacy_check("w70_eval")), True),
        # 390 full groups of 8 within 100 m, 3,120 images. Sized by dry run:
        # at the original 30 m / random-4,000 only one group of 200 had four
        # members, so the 4- and 8-photo columns would have silently repeated
        # the 2-photo one and read as a plateau.
        # 390 full groups of 8 within 100 m, 3,120 images. Sized by dry run:
        # at the original 30 m / random-4,000 only one group of 200 had four
        # members, so the 4- and 8-photo columns would have silently repeated
        # the 2-photo one and read as a plateau.
        #
        # Two merges, because how the photographs' candidates are combined is
        # itself the experiment. `global` ranks all 8x32 together and lets the
        # strongest-matching photograph fill the slots; `rr` gives each an
        # equal share and deduplicates, which is what serve.py ships. The
        # -global pair ran before --merge existed and is aliased, not repeated.
        (Stage("multiq-d768-b350-e6-diverse-global",
               ["scripts/multiquery.py", "--tag", TAG, "--radius", "100",
                "--groups", "400", "--sizes", "1,2,4,8",
                "--order", "diverse", "--merge", "global"],
               release=REL, est=15 * 60, retries=2,
               check=legacy_check("multiq-d768-b350-e6-diverse")), True),
        (Stage("multiq-d768-b350-e6-near-global",
               ["scripts/multiquery.py", "--tag", TAG, "--radius", "100",
                "--groups", "400", "--sizes", "1,2,4,8",
                "--order", "near", "--merge", "global"],
               release=REL, est=15 * 60, retries=2,
               check=legacy_check("multiq-d768-b350-e6-near")), True),
        (Stage("multiq-d768-b350-e6-diverse-rr",
               ["scripts/multiquery.py", "--tag", TAG, "--radius", "100",
                "--groups", "400", "--sizes", "1,2,4,8",
                "--order", "diverse", "--merge", "rr"],
               release=REL, est=15 * 60, retries=2), True),
        (Stage("multiq-d768-b350-e6-near-rr",
               ["scripts/multiquery.py", "--tag", TAG, "--radius", "100",
                "--groups", "400", "--sizes", "1,2,4,8",
                "--order", "near", "--merge", "rr"],
               release=REL, est=15 * 60, retries=2), True),
        # Power. The 8-photo runs have 390 groups, where a 1 pp step is about
        # five images changing status and the `near` control duly came out
        # non-monotone (8.0 -> 7.5 -> 6.4 -> 9.5). Dropping to groups of 4
        # yields 2,496 of them -- 6.4x the sample, paired CI half-width ~0.96
        # pp -- which is the smallest run that can actually resolve the effect.
        # Runs last because it re-embeds 9,984 images over the shared cache.
        (Stage("multiq-d768-b350-e6-pow4-rr",
               ["scripts/multiquery.py", "--tag", TAG, "--radius", "100",
                "--groups", "4000", "--sizes", "1,2,4",
                "--order", "diverse", "--merge", "rr",
                "--export", str(O.RUNS / "multiq_pow4.npz")],
               release=REL, est=50 * 60, retries=2), True),
        # Both arms on the SAME 1,000 hash-selected images, each against its
        # own bank. That is the only clean test of whether the corpus step
        # that bought +2.8 pp on the benchmark buys anything on photographs
        # the bank does not cover.
        (Stage("kartaview-d768-b350-e6",
               ["scripts/eval_highres.py", "--tag", TAG, "--n", "1000",
                "--export", str(O.RUNS / "hr_b350.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("kartaview-d768-b265-e6",
               ["scripts/eval_highres.py", "--tag", "d768-b265-e6",
                "--n", "1000", "--export", str(O.RUNS / "hr_b265.npz")],
               release=REL, est=25 * 60, retries=2), True),
        # n=1,000 left the corpus contrast at -1.40 pp [-3.00, +0.20]: inside
        # noise, but wide enough to hide anything from a real loss to a small
        # gain. This is the question the whole corpus axis rests on, so it is
        # worth 5x the sample: the interval should tighten to about +/-0.7 pp.
        # Hash sampling makes these 5,000 a superset of those 1,000.
        (Stage("kartaview-d768-b350-e6-n5k",
               ["scripts/eval_highres.py", "--tag", TAG, "--n", "5000",
                "--export", str(O.RUNS / "hr_b350_n5k.npz")],
               release=REL, est=35 * 60, retries=2), True),
        (Stage("kartaview-d768-b265-e6-n5k",
               ["scripts/eval_highres.py", "--tag", "d768-b265-e6",
                "--n", "5000", "--export", str(O.RUNS / "hr_b265_n5k.npz")],
               release=REL, est=35 * 60, retries=2), True),
        # 2x2: the corpus step changes the model AND the bank together, so the
        # -1.32 pp external loss cannot yet be attributed. These are the two
        # cross cells -- each model against the other's bank -- which separate
        # "the denser bank retrieves worse off-domain" from "the model trained
        # against a denser bank leans on retrieval harder and generalises
        # worse". Both banks live in the same PCA space, so the swap is
        # coherent rather than a units mismatch.
        (Stage("kartaview-x-m265-b350",
               ["scripts/eval_highres.py", "--tag", "d768-b265-e6",
                "--bank", "pca768_bank70.f16.npy", "--n", "5000",
                "--export", str(O.RUNS / "hr_m265_b350.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("kartaview-x-m350-b265",
               ["scripts/eval_highres.py", "--tag", TAG,
                "--bank", "pca768_bank55.f16.npy", "--n", "5000",
                "--export", str(O.RUNS / "hr_m350_b265.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("train-d768-b350-e2-drop30",
               drop_train("d768-b350-e2-drop30"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e4-drop30",
               drop_train("d768-b350-e4-drop30", "d768-b350-e2-drop30"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e6-drop30",
               drop_train("d768-b350-e6-drop30", "d768-b350-e4-drop30"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("boot-d768-b350-drop30",
               ["scripts/boot_existing.py", "--tags",
                "d768-b265-e6,d768-b350-e6,d768-b350-e6-drop30",
                "--out", str(O.RUNS / "BOOTSTRAP_drop30.md")],
               release=REL, est=8 * 60, retries=2), True),
        (Stage("kartaview-drop30-n5k",
               ["scripts/eval_highres.py", "--tag", "d768-b350-e6-drop30",
                "--n", "5000",
                "--export", str(O.RUNS / "hr_drop30_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        # d1536-b350-e6 holds the best benchmark number and has never been
        # measured off-benchmark. If it carries the same retrieval dependency,
        # the shipping recommendation changes again.
        (Stage("kartaview-d1536-b350-e6-n5k",
               ["scripts/eval_highres.py", "--tag", "d1536-b350-e6",
                "--n", "5000",
                "--export", str(O.RUNS / "hr_d1536_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        # Does a model that leans less on retrieval still gain from extra
        # photographs? If the +1.2 pp shrinks, both levers were pulling the
        # same rope and they will not add.
        (Stage("multiq-drop30-pow4-rr",
               ["scripts/multiquery.py", "--tag", "d768-b350-e6-drop30",
                "--radius", "100", "--groups", "4000", "--sizes", "1,2,4",
                "--order", "diverse", "--merge", "rr",
                "--export", str(O.RUNS / "multiq_drop30.npz")],
               release=REL, est=20 * 60, retries=2), True),
        # ---- 1. does dropout fix the best benchmark arm too? -------------
        # d1536-b350-e6 is 76.2% on the benchmark and 11.7% externally, second
        # worst of four. If dropout lifts it the way it lifted d768, the
        # shipping arm becomes d1536-b350-e6-drop30 and we keep both numbers.
        (Stage("train-d1536-b350-e2-drop30",
               train1536("d1536-b350-e2-drop30", drop=DROP),
               release=REL, est=30 * 60, retries=3), True),
        (Stage("train-d1536-b350-e4-drop30",
               train1536("d1536-b350-e4-drop30", "d1536-b350-e2-drop30", DROP),
               release=REL, est=30 * 60, retries=3), True),
        (Stage("train-d1536-b350-e6-drop30",
               train1536("d1536-b350-e6-drop30", "d1536-b350-e4-drop30", DROP),
               release=REL, est=30 * 60, retries=3), True),
        (Stage("boot-d1536-drop30",
               ["scripts/boot_existing.py", "--tags",
                "d1536-b350-e6,d768-b350-e6-drop30,d1536-b350-e6-drop30",
                "--out", str(O.RUNS / "BOOTSTRAP_d1536_drop30.md")],
               release=REL, est=10 * 60, retries=2), True),
        (Stage("kartaview-d1536-drop30-n5k",
               ["scripts/eval_highres.py", "--tag", "d1536-b350-e6-drop30",
                "--n", "5000",
                "--export", str(O.RUNS / "hr_d1536_drop30_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),

        # ---- 2. the p-curve: 0.3 was a guess ------------------------------
        # Two more points say whether 0.3 sits on a plateau or a peak. 0.1 and
        # 0.5 bracket it widely enough that a peak would show.
        (Stage("train-d768-b350-e2-drop10", drop_p("d768-b350-e2-drop10",
                                                   None, "0.1"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e4-drop10", drop_p("d768-b350-e4-drop10",
                                                   "d768-b350-e2-drop10", "0.1"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e6-drop10", drop_p("d768-b350-e6-drop10",
                                                   "d768-b350-e4-drop10", "0.1"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("kartaview-drop10-n5k",
               ["scripts/eval_highres.py", "--tag", "d768-b350-e6-drop10",
                "--n", "5000",
                "--export", str(O.RUNS / "hr_drop10_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("train-d768-b350-e2-drop50", drop_p("d768-b350-e2-drop50",
                                                   None, "0.5"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e4-drop50", drop_p("d768-b350-e4-drop50",
                                                   "d768-b350-e2-drop50", "0.5"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e6-drop50", drop_p("d768-b350-e6-drop50",
                                                   "d768-b350-e4-drop50", "0.5"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("kartaview-drop50-n5k",
               ["scripts/eval_highres.py", "--tag", "d768-b350-e6-drop50",
                "--n", "5000",
                "--export", str(O.RUNS / "hr_drop50_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("boot-pcurve",
               ["scripts/boot_existing.py", "--tags",
                "d768-b350-e6,d768-b350-e6-drop10,d768-b350-e6-drop30,"
                "d768-b350-e6-drop50",
                "--out", str(O.RUNS / "BOOTSTRAP_pcurve.md")],
               release=REL, est=12 * 60, retries=2), True),

        # The curve is still climbing at 0.5: external <25km goes 11.6 -> 11.7
        # -> 12.7 -> 13.3% for p = 0, 0.1, 0.3, 0.5, and 0.5 already beats the
        # 2.65M arm's 12.9%. So the optimum is at least 0.5 and the sweep has
        # to continue rather than stop at a number that merely looks brave.
        # 0.7 leaves ~5 of 16 neighbours, which is where a real cost should
        # start to show if one exists.
        (Stage("train-d768-b350-e2-drop70", drop_p("d768-b350-e2-drop70",
                                                   None, "0.7"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e4-drop70", drop_p("d768-b350-e4-drop70",
                                                   "d768-b350-e2-drop70", "0.7"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e6-drop70", drop_p("d768-b350-e6-drop70",
                                                   "d768-b350-e4-drop70", "0.7"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("kartaview-drop70-n5k",
               ["scripts/eval_highres.py", "--tag", "d768-b350-e6-drop70",
                "--n", "5000",
                "--export", str(O.RUNS / "hr_drop70_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("boot-pcurve2",
               ["scripts/boot_existing.py", "--tags",
                "d768-b350-e6,d768-b350-e6-drop30,d768-b350-e6-drop50,"
                "d768-b350-e6-drop70,d1536-b350-e6-drop30",
                "--out", str(O.RUNS / "BOOTSTRAP_pcurve2.md")],
               release=REL, est=12 * 60, retries=2), True),
        # Monotone from 0.1 to 0.7 and still climbing: 11.7 -> 12.7 -> 13.3 ->
        # 14.1% external, +2.48 pp over no-drop. 0.9 leaves ~1.6 of 16
        # neighbours, which is close to the floor the implementation allows
        # (an all-dropped row rescues one). If that is better again, the
        # reading is not "tune p" but "the policy should barely see retrieval
        # while training" -- and the gates still need *some* signal or they
        # never train at all, which is the zero-gate deadlock already recorded.
        (Stage("train-d768-b350-e2-drop90", drop_p("d768-b350-e2-drop90",
                                                   None, "0.9"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e4-drop90", drop_p("d768-b350-e4-drop90",
                                                   "d768-b350-e2-drop90", "0.9"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e6-drop90", drop_p("d768-b350-e6-drop90",
                                                   "d768-b350-e4-drop90", "0.9"),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("kartaview-drop90-n5k",
               ["scripts/eval_highres.py", "--tag", "d768-b350-e6-drop90",
                "--n", "5000",
                "--export", str(O.RUNS / "hr_drop90_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("boot-pcurve3",
               ["scripts/boot_existing.py", "--tags",
                "d768-b350-e6,d768-b350-e6-drop50,d768-b350-e6-drop70,"
                "d768-b350-e6-drop90",
                "--out", str(O.RUNS / "BOOTSTRAP_pcurve3.md")],
               release=REL, est=12 * 60, retries=2), True),
        # Sink capacity, tested on top of the best arm on the p-curve so the
        # contrast is one variable. The base sink is a single fixed direction
        # -- one linear probe on the fused state, blind to the map, shared
        # across four zoom steps where "not in this tile" means a quarter of
        # the planet at s0 and 611 m at s3. K keys per step with a logsumexp
        # reads as "reject if any of K reasons fires". +3,084 parameters, and
        # neutral at init (extra biases start at -20, verified to 0.000e+00).
        (Stage("train-d768-b350-e2-drop70-sub2",
               drop_sub2("d768-b350-e2-drop70-sub2", None, "0.7"),
               release=REL, est=30 * 60, retries=3), True),
        (Stage("train-d768-b350-e4-drop70-sub2",
               drop_sub2("d768-b350-e4-drop70-sub2",
                         "d768-b350-e2-drop70-sub2", "0.7"),
               release=REL, est=30 * 60, retries=3), True),
        (Stage("train-d768-b350-e6-drop70-sub2",
               drop_sub2("d768-b350-e6-drop70-sub2",
                         "d768-b350-e4-drop70-sub2", "0.7"),
               release=REL, est=30 * 60, retries=3), True),
        (Stage("kartaview-sub2-n5k",
               ["scripts/eval_highres.py", "--tag",
                "d768-b350-e6-drop70-sub2", "--n", "5000",
                "--export", str(O.RUNS / "hr_sub2_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("boot-sub2",
               ["scripts/boot_existing.py", "--tags",
                "d768-b350-e6-drop70,d768-b350-e6-drop70-sub2",
                "--out", str(O.RUNS / "BOOTSTRAP_sub2.md")],
               release=REL, est=12 * 60, retries=2), True),
        (Stage("train-d768-b350-e2-drop70-sink4",
               drop_sink("d768-b350-e2-drop70-sink4", None, "0.7", 4),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e4-drop70-sink4",
               drop_sink("d768-b350-e4-drop70-sink4",
                         "d768-b350-e2-drop70-sink4", "0.7", 4),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("train-d768-b350-e6-drop70-sink4",
               drop_sink("d768-b350-e6-drop70-sink4",
                         "d768-b350-e4-drop70-sink4", "0.7", 4),
               release=REL, est=28 * 60, retries=3), True),
        (Stage("kartaview-sink4-n5k",
               ["scripts/eval_highres.py", "--tag",
                "d768-b350-e6-drop70-sink4", "--n", "5000",
                "--export", str(O.RUNS / "hr_sink4_n5k.npz")],
               release=REL, est=25 * 60, retries=2), True),
        (Stage("boot-sink4",
               ["scripts/boot_existing.py", "--tags",
                "d768-b350-e6-drop70,d768-b350-e6-drop70-sink4,"
                "d1536-b350-e6-drop30",
                "--out", str(O.RUNS / "BOOTSTRAP_sink4.md")],
               release=REL, est=12 * 60, retries=2), True),
        (Stage("pyrcache-hr47k",
               ["scripts/pyramid_cache.py", "--out", "pyr47", "--n", "0"],
               release=REL, est=620 * 60, retries=2), False),
        # Guarded: this reads the cache the previous stage builds, and when
        # that stage is dropped for lack of time this one fails three times in
        # three seconds against a missing file. A stage that cannot run should
        # be skipped, not retried.
        (Stage("fuse-attn-pyr47",
               ["scripts/fuse_head.py", "--tokens", "pyr33",
                "--pyr-stem", "pyr47", "--epochs", "12"],
               release=REL, est=60 * 60, retries=2,
               needs=config.STREET_CACHE / "pyr47_meta.npz"), False),
    ]

    for st, needs_tiles in plan:
        if O.now() > deadline:
            log("out of time before {}".format(st.name))
            break
        # Ask "is it already done?" before "does it still fit?", or finished
        # work is reported as skipped for time and the log stops being a record
        # of what actually happened.
        if st.satisfied():
            log("skip   {}  (already done)".format(st.name))
            continue
        if needs_tiles:
            if O.now() > tile_off:
                log("skip   {}  (past the {} tile cutoff)"
                    .format(st.name, O.hhmm(tile_off)))
                continue
            if not tiles_up():
                log("skip   {}  (tile server down)".format(st.name))
                continue
            # Do not start something that would still be running at the cutoff
            # and lose its tiles halfway: it would fail late and burn the slot.
            if O.now() + st.est > tile_off:
                log("skip   {}  (est {:.0f} min would cross the cutoff)"
                    .format(st.name, st.est / 60))
                continue
        run_stage(st, state, deadline)

    log("-" * 72)
    log("done; logs in {}".format(O.LOGS))
    if O.SAMPLER:
        log("system, whole run: " + O.SAMPLER.summary())


if __name__ == "__main__":
    main()
