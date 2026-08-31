"""Everything still open, against two deadlines rather than one.

The tile server goes away at 09:30, and **every beam rollout needs it** -- not
just evaluation. Training needs it too, because `--select hit` decodes 2,000
val images each epoch to choose which one to keep. So the schedule has a hard
inner cutoff: anything with a rollout in it has to start early enough to finish
before the tiles do. What is left after that is pure encoder and numpy work,
which never touches the network, so it fills the tail.

Stages carry `tiles=True/False` and the tile-dependent ones are skipped rather
than started once the cutoff passes -- a stage killed halfway by a dead tile
server writes no marker and wastes its whole slot.

Order inside the tile window is by value per minute:

  1. the headline arm is still climbing. Continuing s10_n400k_bank25 for two
     epochs took it 10.1 -> 8.6 km; whether it is done is one 30-minute run.
  2. the two encoders are combined 81/19 by accident, because only the joined
     vector is normalised and DINOv2's activations are 4x larger. Equal-norm
     wins the retrieval sweep on all three metrics. Rebuild, re-bank, retrain.
  3. cell8 is the number quoted externally and its arms were mistuned.
  4. the same rebalance applied to the 1.15M bank, which is where it should
     matter most -- last, because it is the only stage that cannot be salvaged
     if the night runs short.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import config
import overnight as O
from overnight import Stage, load_state, log, run_stage

REL = "s10"
EXT = "bank_ext"
W = "4.03"                          # equal-norm blend, from the retrieval sweep

BASE = "dual_c3.f16.npy"
BAL = "dual_bal.f16.npy"
BAL_BANK = "dual_bal_bank25.f16.npy"
STACKED = "dual_c3_bank25.f16.npy"

KNN_BASE_SEQ = config.knn_name(BASE, "sequence")
KNN_BAL_SEQ = config.knn_name(BAL, "sequence")
KNN_BANK_SEQ = config.knn_name(STACKED, "sequence", ext=EXT)
KNN_BALBANK_SEQ = config.knn_name(BAL_BANK, "sequence", ext=EXT)
KNN_C8_BANK = config.knn_name(STACKED, "cell8", ext=EXT)

ARCH = ["--pool", "attn", "--pool-q", "4", "--pos", "both",
        "--map-layers", "1", "--neg", "4",
        "--retr", "--retr-k", "16", "--retr-mode", "dual", "--d-key", "128"]


def train(tag, street, knn, mode, epochs, lr=None, init=None, limit=None):
    a = ["src/train.py", "--tag", tag, "--epochs", str(epochs), "--batch", "64",
         "--select", "hit", "--sel-n", "2000", "--val-n", "5000",
         "--split-mode", mode, "--street-file", street, "--knn-file", knn]
    if limit:
        a += ["--limit", str(limit)]
    if lr is not None:
        a += ["--lr", str(lr)]
    if init is not None:
        a += ["--init", init]
    return a + ARCH


def boot(tags, out):
    return ["scripts/bootstrap.py", "--tags", tags, "--split", "test",
            "--n", "5000", "--beam", "2", "--score-steps", "3",
            "--out", str(O.RUNS / out)]


def tiles_up():
    """Is the map server actually answering?

    The cutoff is a promise about when the server goes away, not a guarantee it
    stays up until then. Without this a dead server turns each remaining
    rollout stage into its full retry budget of failures, and burns the window
    that the tile-free stages could have used.
    """
    try:
        import tiles as T
        T.TileClient(config.TILE_SERVER, timeout=8.0, retries=1).health()
        return True
    except Exception as e:
        log("tile server not answering: {}".format(e))
        return False


def best_cell8_lr():
    """Whichever learning rate actually won, on the best evidence available.

    Prefer the cached per-image test errors the paired bootstrap leaves behind:
    5,000 images, the same protocol every other number here uses. Fall back to
    the checkpoint's own val_hit only if that eval never ran.

    The fallback is genuinely worse and the difference is not academic. These
    two arms landed at val_hit 0.0690 and 0.0665 -- 0.25 pp on 2,000 images,
    far inside noise -- so selecting on it is close to a coin flip, and it would
    have picked the arm whose hit rate went flat after epoch 1 over the one
    still climbing at epoch 4. A continuation compounds that choice.
    """
    import numpy as np
    cands = ("s10_cell8_bank25_lr1e4", "s10_cell8_bank25_lr3e4")
    scored = []
    for tag in cands:
        if not (config.CHECKPOINTS / (tag + ".pt")).exists():
            continue
        e = ROOT / "runs" / "errs" / (tag + "_test_5000r_k2_d3.npy")
        if e.exists():
            v = np.load(e)
            hit, src, km = float((v < 25).mean()), "test", float(np.median(v))
        else:
            import torch
            ck = torch.load(config.CHECKPOINTS / (tag + ".pt"),
                            map_location="cpu", weights_only=False)
            hit, src, km = ck.get("val_hit", 0.0), "val", ck.get("val_km", 0.0)
        log("  cell8 {}  <25km {:.4f} ({}, n={})  median {:.1f} km".format(
            tag, hit, src, 5000 if src == "test" else 2000, km))
        scored.append((hit, km, tag))
    if not scored:
        return None
    scored.sort(key=lambda r: -r[0])
    # Hit rate decides only when it can. These two landed 0.06 pp apart on the
    # hit rate -- indistinguishable -- while their medians differ by 45 km and
    # separate cleanly, so the tie-break is the metric that actually resolved.
    if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 0.005:
        scored.sort(key=lambda r: r[1])
        log("  hit rates within 0.5 pp -- indistinguishable at this n; "
            "breaking the tie on median km instead")
    return scored[0][-1]


def cutoff_at(hh, mm):
    """The next occurrence of hh:mm, as an epoch second."""
    now = datetime.now()
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += timedelta(days=1)
    return t.timestamp()


def main():
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    for d in (O.RUNS, O.LOGS, O.MARKS):
        d.mkdir(parents=True, exist_ok=True)
    state = load_state()
    deadline = O.now() + hours * 3600
    tiles_until = cutoff_at(9, 30)

    O.SAMPLER = O.Sampler()
    O.SAMPLER.start()
    log("=" * 72)
    log("marathon: continuation, encoder rebalance, cell8, tiling probe")
    log("deadline     {}  ({:.1f} h)".format(O.hhmm(deadline), hours))
    log("tile server  until {}  -- rollouts must finish by then".format(
        O.hhmm(tiles_until)))

    log("cell8 learning-rate sweep:")
    C8_WIN = best_cell8_lr() or "s10_cell8_bank25_lr1e4"
    log("  continuing {}".format(C8_WIN))

    # (stage, needs_tiles)
    plan = [
        # -- 1. is the headline arm done climbing? ------------------------
        (Stage("mar_cont4",
               train("s10_n400k_bank25_c4", STACKED, KNN_BANK_SEQ, "sequence",
                     2, init="s10_n400k_bank25_cont"),
               release=REL, est=35 * 60, retries=1), True),
        (Stage("mar_cont4_eval",
               boot("s10_n400k_bank25_cont,s10_n400k_bank25_c4",
                    "BOOTSTRAP_cont4.md"),
               release=REL, est=10 * 60, retries=1), True),

        # -- 2. equal-norm encoders, at the 400k bank ---------------------
        (Stage("mar_bal_concat",
               ["scripts/concat_street.py", "--a", "embeddings_c3",
                "--b", "siglip_c3", "--scale-b", W, "--out", "dual_bal"],
               release=REL, est=15 * 60, retries=1), False),
        (Stage("mar_bal_knn",
               ["scripts/build_knn.py", "--street-file", BAL,
                "--split-mode", "sequence", "--chunk", "512",
                "--bank-block", "200000"],
               release=REL, est=45 * 60, retries=1), False),
        (Stage("mar_bal_train",
               train("s10_bal", BAL, KNN_BAL_SEQ, "sequence", 2,
                     limit=400000),
               release=REL, est=35 * 60, retries=1), True),
        (Stage("mar_bal_eval",
               boot("s10_key_dual,s10_bal", "BOOTSTRAP_blend.md"),
               release=REL, est=10 * 60, retries=1), True),

        # -- 3. cell8: continue whichever LR won ---------------------------
        (Stage("mar_c8_cont",
               train(C8_WIN + "_c", STACKED, KNN_C8_BANK, "cell8", 2,
                     init=C8_WIN),
               release=REL, est=35 * 60, retries=1), True),
        (Stage("mar_c8_eval",
               boot("{0},{0}_c,s10_cell8_bank25_cont".format(C8_WIN),
                    "BOOTSTRAP_cell8_c.md"),
               release=REL, est=15 * 60, retries=1), True),

        # -- 4. the rebalance where it should matter most ------------------
        (Stage("mar_balext_concat",
               ["scripts/concat_street.py", "--a", "bank_ext_dino",
                "--b", "bank_ext_siglip", "--scale-b", W,
                "--out", "bank_ext_bal"],
               release=REL, est=20 * 60, retries=1), False),
        (Stage("mar_balbank_stack",
               ["scripts/stack_bank.py", "--base", "dual_bal",
                "--ext", "bank_ext_bal", "--out", "dual_bal_bank25"],
               release=REL, est=20 * 60, retries=1), False),
        (Stage("mar_balbank_knn",
               # --bank-ext names the extension's METADATA (image_id, x16,
               # y16, sequence), which describes the images and is the same
               # whichever embedding variant was built from them. The variant
               # is chosen by --street-file. Passing "bank_ext_bal" here sent
               # it looking for a bank_ext_bal_meta.npz that should not exist.
               ["scripts/build_knn.py", "--street-file", BAL_BANK,
                "--split-mode", "sequence", "--chunk", "512",
                "--bank-ext", EXT, "--bank-block", "150000"],
               release=REL, est=70 * 60, retries=1), False),
        (Stage("mar_balbank_train",
               train("s10_bal_bank25", BAL_BANK, KNN_BALBANK_SEQ, "sequence",
                     2, limit=400000),
               release=REL, est=40 * 60, retries=1), True),
        (Stage("mar_balbank_eval",
               boot("s10_n400k_bank25_cont,s10_bal_bank25",
                    "BOOTSTRAP_balbank.md"),
               release=REL, est=10 * 60, retries=1), True),

        # -- 5. tile-free, so it fills the tail after the tiles go ---------
        (Stage("mar_tile_probe",
               ["scripts/tile_probe.py", "--n", "30000", "--queries", "1500",
                "--dims", "256,512,1024,2048"],
               release=REL, est=90 * 60, retries=1), False),
    ]

    alive = True
    for st, needs in plan:
        if needs:
            if O.now() + st.est > tiles_until:
                log("skip   {}  -- needs tiles, would run past {}".format(
                    st.name, O.hhmm(tiles_until)))
                continue
            if not alive:
                log("skip   {}  -- tile server already gone".format(st.name))
                continue
            if not tiles_up():
                alive = False
                log("skip   {}  -- tile server down; dropping every remaining "
                    "rollout stage and going straight to the offline work"
                    .format(st.name))
                continue
        run_stage(st, state, deadline)

    # the digest is not a stage: it must run even if everything above failed,
    # and it reads only cached per-image errors, so it cannot fail on its own
    try:
        import subprocess
        subprocess.call([O.PY, "scripts/digest.py"], cwd=str(ROOT))
    except Exception as e:
        log("digest failed: {}".format(e))

    O.finish(state, deadline)


if __name__ == "__main__":
    main()
