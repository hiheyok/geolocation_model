"""Unattended pipeline: build the s10 release, then run the data x epochs grid.

Three things this has to survive, all of which have already happened once on
this machine: a stage crashing, the whole box rebooting mid-run, and the work
simply not fitting in the time available.

  * Every stage has a done-marker and is idempotent, so restarting the runner
    resumes rather than repeats.  The marker is written only after the stage
    exits zero.
  * Every stage has a retry budget, and a stage that keeps failing is recorded
    and stepped over instead of taking the run down with it.
  * There is a hard deadline.  Stages are ordered by value, the training grid
    is *sized* from the time left after prep rather than guessed in advance,
    and anything that no longer fits is dropped from the tail.

The experiment the grid runs: hold optimizer steps fixed and vary how many
images those steps see.  If 400k images at 4 epochs beats 25k at 64, data is
the binding constraint and the learning curve is still sloping; if they tie,
compute is, and the extra shards bought nothing.  One further arm is
deliberately *not* compute-matched -- 50k images for 80 epochs -- to ask
whether many passes over less data substitute for more data at all.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PY = sys.executable
RUNS = ROOT / "runs"
LOGS = RUNS / "logs"
MARKS = RUNS / "marks"
STATE = RUNS / "state.json"
UTIL = RUNS / "util.csv"

SHARDS = "00,01,02,03,04,05,06,07,08,09"
DINO = "vit_base_patch14_dinov2.lvd142m"
SIGLIP = "vit_base_patch16_siglip_224.v2_webli"

# The configuration behind the 224.7 km headline, so every arm differs from the
# shipping model only in how much data it saw and for how long.
SHIP = ["--street-file", "dual_c3.f16.npy", "--pool", "attn", "--pool-q", "4",
        "--pos", "both", "--map-layers", "1", "--neg", "4",
        "--retr", "--retr-k", "16", "--retr-mode", "dual", "--d-key", "128"]

BATCH = 64


# --------------------------------------------------------------- utilities --

def now():
    return time.time()


def hhmm(t):
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def log(msg):
    line = "[{}] {}".format(hhmm(now()), msg)
    print(line, flush=True)
    with open(RUNS / "runner.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": {}, "failed": {}, "notes": []}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


# ------------------------------------------------------ system utilisation --

class Sampler(threading.Thread):
    """GPU, CPU and memory every few seconds, tagged with the running stage.

    Utilisation and power alone were not enough to diagnose the last
    performance defect here -- a process spilling out of VRAM reports 100%
    utilisation while every access crosses PCIe -- so dedicated *and* shared
    adapter memory are both sampled.  Shared usage comes from a Windows
    performance counter and is polled rarely because it costs a shell.
    """

    def __init__(self, every=10.0, shared_every=6):
        super().__init__(daemon=True)
        self.every = every
        self.shared_every = shared_every
        self.stage = "idle"
        self.stop_flag = threading.Event()
        self.rows = []
        if not UTIL.exists():
            UTIL.write_text("t,stage,gpu_util,gpu_mem_mb,gpu_shared_mb,gpu_w,"
                            "gpu_c,cpu_pct,ram_used_gb\n", encoding="utf-8")

    def _nvidia(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
                 "power.draw,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            a = [p.strip() for p in out.stdout.strip().split("\n")[0].split(",")]
            return float(a[0]), float(a[1]), float(a[2]), float(a[3])
        except Exception:
            return -1.0, -1.0, -1.0, -1.0

    def _shared(self):
        """Windows: bytes of GPU memory that spilled into system DRAM."""
        ps = ("(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage'"
              " -ErrorAction SilentlyContinue).CounterSamples"
              " | Measure-Object -Property CookedValue -Sum"
              " | Select-Object -ExpandProperty Sum")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=30)
            return float(out.stdout.strip()) / 1e6
        except Exception:
            return -1.0

    def run(self):
        import psutil
        psutil.cpu_percent(None)
        i = 0
        shared = -1.0
        while not self.stop_flag.wait(self.every):
            u, m, w, c = self._nvidia()
            if i % self.shared_every == 0:
                shared = self._shared()
            cpu = psutil.cpu_percent(None)
            ram = psutil.virtual_memory().used / 1e9
            row = (now(), self.stage, u, m, shared, w, c, cpu, ram)
            self.rows.append(row)
            with open(UTIL, "a", encoding="utf-8") as fh:
                fh.write("{:.0f},{},{:.0f},{:.0f},{:.0f},{:.1f},{:.0f},"
                         "{:.1f},{:.2f}\n".format(*row))
            i += 1

    def summary(self, stage=None):
        r = [x for x in self.rows if stage is None or x[1] == stage]
        if not r:
            return "no samples"
        col = lambda j: np.array([x[j] for x in r if x[j] >= 0])
        f = lambda j, u: ("{:.0f}{}".format(col(j).mean(), u)
                          if len(col(j)) else "n/a")
        return ("gpu {} (peak {}), vram {}, shared {}, {}, cpu {}, ram {}"
                .format(f(2, "%"),
                        "{:.0f}%".format(col(2).max()) if len(col(2)) else "n/a",
                        f(3, " MB"), f(4, " MB"), f(5, " W"),
                        f(7, "%"), f(8, " GB").replace(" GB", " GB")))


SAMPLER = None


# ------------------------------------------------------------------ stages --

class Stage:
    def __init__(self, name, argv, release=None, est=600, retries=2,
                 check=None, critical=False):
        self.name = name
        self.argv = argv
        self.release = release
        self.est = est                  # seconds, for deadline planning
        self.retries = retries
        self.check = check              # () -> bool, "already satisfied"
        self.critical = critical        # if it fails, dependents are dropped

    def marker(self):
        return MARKS / (self.name + ".done")

    def satisfied(self):
        if self.marker().exists():
            return True
        if self.check is not None and self.check():
            self.marker().write_text("satisfied by check\n", encoding="utf-8")
            return True
        return False


def run_stage(st, state, deadline):
    if st.satisfied():
        log("skip   {}  (already done)".format(st.name))
        return True
    left = deadline - now()
    if left < st.est:
        log("drop   {}  (needs ~{:.0f} min, {:.0f} min left)"
            .format(st.name, st.est / 60, left / 60))
        state["failed"][st.name] = "dropped: out of time"
        save_state(state)
        return False

    env = dict(os.environ)
    if st.release:
        env["OSV_RELEASE"] = st.release
    env["PYTHONUNBUFFERED"] = "1"
    logf = LOGS / (st.name + ".log")

    for attempt in range(1, st.retries + 2):
        if SAMPLER:
            SAMPLER.stage = st.name
        t0 = now()
        log("start  {}{}  (est {:.0f} min, deadline in {:.0f} min)".format(
            st.name, "" if attempt == 1 else "  retry {}".format(attempt - 1),
            st.est / 60, (deadline - now()) / 60))
        with open(logf, "a", encoding="utf-8") as fh:
            fh.write("\n==== attempt {} at {} ====\n".format(attempt, hhmm(t0)))
            fh.flush()
            rc = subprocess.call([PY] + st.argv, cwd=str(ROOT), env=env,
                                 stdout=fh, stderr=subprocess.STDOUT)
        el = now() - t0
        if SAMPLER:
            SAMPLER.stage = "idle"
        if rc == 0:
            st.marker().write_text("{:.0f}s\n".format(el), encoding="utf-8")
            state["done"][st.name] = {"secs": el, "at": hhmm(now())}
            save_state(state)
            log("ok     {}  {:.1f} min   {}".format(
                st.name, el / 60, SAMPLER.summary(st.name) if SAMPLER else ""))
            return True
        tail = ""
        try:
            tail = "\n         ".join(
                logf.read_text(encoding="utf-8", errors="replace")
                .strip().split("\n")[-4:])
        except Exception:
            pass
        log("FAIL   {}  rc={} after {:.1f} min\n         {}"
            .format(st.name, rc, el / 60, tail))
        if now() > deadline:
            break
    state["failed"][st.name] = "rc={}".format(rc)
    save_state(state)
    return False


# ------------------------------------------------------- stage definitions --

def parquet_rows(release, name="dataset.parquet"):
    p = ROOT / "data" / "processed" / release / name
    if not p.exists():
        return 0
    import pyarrow.parquet as pq
    try:
        return pq.ParquetFile(p).metadata.num_rows
    except Exception:
        return 0


def emb_ok(release, fname, dim):
    """Shape is not evidence: open_memmap allocates the whole file before the
    first image is embedded, so a pass killed after 30 seconds leaves a
    correctly-shaped array of zeros.  Rows are written grouped by shard in
    sorted order, so a spread sample catches any truncated run.
    """
    p = ROOT / "cache" / "street" / release / fname
    n = parquet_rows(release)
    if not p.exists() or not n:
        return False
    try:
        a = np.load(p, mmap_mode="r")
        if a.shape != (n, dim):
            return False
        probe = np.linspace(0, n - 1, 96).astype(np.int64)
        rows = np.asarray(a[probe], dtype=np.float32)
        return bool((np.abs(rows).sum(1) > 0).all())
    except Exception:
        return False


def prep_stages():
    return [
        Stage("dataset", ["scripts/build_dataset.py", "--shard", SHARDS],
              release="s10", est=25 * 60, critical=True,
              check=lambda: parquet_rows("s10") >= 400000),
        Stage("embed_dino",
              ["scripts/embed_street.py", "--model", DINO, "--crops", "3",
               "--out", "embeddings_c3"],
              release="s10", est=90 * 60, critical=True,
              check=lambda: emb_ok("s10", "embeddings_c3.f16.npy", 2304)),
        Stage("embed_siglip",
              ["scripts/embed_street.py", "--model", SIGLIP, "--crops", "3",
               "--out", "siglip_c3"],
              release="s10", est=90 * 60, critical=True,
              check=lambda: emb_ok("s10", "siglip_c3.f16.npy", 2304)),
        Stage("concat", ["scripts/concat_street.py", "--a", "embeddings_c3",
                         "--b", "siglip_c3", "--out", "dual_c3"],
              release="s10", est=3 * 60, critical=True,
              check=lambda: emb_ok("s10", "dual_c3.f16.npy", 4608)),
        Stage("tiles", ["scripts/fetch_tiles.py", "--exhaustive-z8",
                        "--seed-from", str(ROOT / "cache" / "map" / "s01")],
              release="s10", est=45 * 60, critical=True),
        Stage("knn", ["scripts/build_knn.py", "--street-file", "dual_c3.f16.npy",
                      "--split-mode", "sequence", "--chunk", "256"],
              release="s10", est=25 * 60, critical=True),
    ]


def train_argv(tag, release, limit, epochs, sel_n=1000, extra=()):
    a = ["src/train.py", "--tag", tag, "--epochs", str(epochs),
         "--batch", str(BATCH), "--select", "km", "--sel-n", str(sel_n)] + SHIP
    if limit:
        a += ["--limit", str(limit)]
    return a + list(extra)


def eval_argv(tag, split, n=5000, ks="2", score_steps=3):
    return ["src/evaluate.py", "--tag", tag, "--split", split, "--n", str(n),
            "--ks", ks, "--score-steps", str(score_steps)]


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0,
                    help="wall clock from the ORIGINAL start, not from a restart")
    ap.add_argument("--start", type=float, default=0.0,
                    help="epoch seconds of the original start; 0 = now")
    ap.add_argument("--steps", type=int, default=0,
                    help="optimizer steps per compute-matched arm; 0 = size it "
                         "from the time left after prep")
    a = ap.parse_args()

    for d in (RUNS, LOGS, MARKS):
        d.mkdir(parents=True, exist_ok=True)

    state = load_state()
    start = a.start or state.get("start") or now()
    state["start"] = start
    deadline = start + a.hours * 3600
    save_state(state)

    global SAMPLER
    SAMPLER = Sampler()
    SAMPLER.start()

    log("=" * 72)
    log("overnight runner   started {}   deadline {}  ({:.1f} h left)"
        .format(hhmm(start), hhmm(deadline), (deadline - now()) / 3600))

    # ---- 1. prep the s10 release -----------------------------------------
    ok = True
    for st in prep_stages():
        if not run_stage(st, state, deadline):
            ok = False
            break

    n_img = parquet_rows("s10")
    n_train = 0
    if ok:
        try:
            import pyarrow.parquet as pq
            import splits as sp
            os.environ["OSV_RELEASE"] = "s10"
            tbl = pq.read_table(ROOT / "data/processed/s10/dataset.parquet")
            lab, sh = sp.read(tbl, "sequence")
            n_train = int((lab == "train").sum())
            log("release s10   {:,} images   {:,} train   split hash {}"
                .format(n_img, n_train, sh))
        except Exception as e:
            log("could not read the s10 split: {}".format(e))
            ok = False

    # ---- 2. the s01 control ----------------------------------------------
    # Two things changed at once -- the selection criterion and the amount of
    # data -- so retrain the shipping config on the ORIGINAL 50k release under
    # the new criterion.  Without it a win on s10 cannot be attributed.
    queue = [
        Stage("s01_km", train_argv("s01_km", "s01", 0, 20),
              release="s01", est=45 * 60, retries=1),
        Stage("s01_km_eval", eval_argv("s01_km", "test", ks="1,2,4"),
              release="s01", est=15 * 60, retries=1),
    ]
    for st in queue:
        run_stage(st, state, deadline)

    if not ok:
        log("prep incomplete -- stopping before the grid")
        return finish(state, deadline)

    # ---- 3. calibrate, then size the grid --------------------------------
    # One short arm measures seconds per optimizer step on this release, which
    # is what the rest of the plan is denominated in.  Guessing it is how a run
    # like this overruns.
    cal = Stage("calib", train_argv("calib_s10", "s10", 25000, 1, sel_n=256),
                release="s10", est=12 * 60, retries=1)
    run_stage(cal, state, deadline)
    cal_secs = state["done"].get("calib", {}).get("secs", 0)
    per_step = (cal_secs / max(1, 25000 / BATCH)) if cal_secs else 0.13
    log("calibration  {:.0f}s for {:.0f} steps  ->  {:.3f} s/step"
        .format(cal_secs, 25000 / BATCH, per_step))

    left = deadline - now()
    reserve = 60 * 60                       # final evaluation + report
    grid_budget = max(0.0, left - reserve)

    # The long arm is the user-requested question and gets its own slice; the
    # compute-matched sweep splits the rest five ways.
    long_n, long_ep = 50000, 80
    long_steps = long_ep * long_n / BATCH
    long_cost = long_steps * per_step

    sizes = [25000, 50000, 100000, 200000, min(400000, n_train)]
    if a.steps:
        S = a.steps
    else:
        share = max(0.0, grid_budget - long_cost) / max(1, len(sizes))
        S = int(max(3000, share / max(per_step, 1e-6)))
    log("budget  {:.1f} h left, {:.1f} h reserved for evaluation"
        .format(left / 3600, reserve / 3600))
    log("grid    {:,} optimizer steps per compute-matched arm "
        "(~{:.0f} min each); long arm {:,} steps (~{:.0f} min)"
        .format(S, S * per_step / 60, int(long_steps), long_cost / 60))

    arms = []
    for n in sizes:
        ep = max(1, int(round(S * BATCH / n)))
        tag = "s10_n{}k_e{}".format(n // 1000, ep)
        arms.append((tag, n, ep, ep * n / BATCH * per_step))
    arms.append(("s10_n50k_e80", long_n, long_ep, long_cost))

    log("arms:")
    for tag, n, ep, cost in arms:
        log("   {:<16} {:>7,} images x {:>3} epochs = {:>7,} steps  ~{:.0f} min"
            .format(tag, n, ep, int(ep * n / BATCH), cost / 60))

    for tag, n, ep, cost in arms:
        lim = 0 if n >= n_train else n
        st = Stage(tag, train_argv(tag, "s10", lim, ep), release="s10",
                   est=cost * 1.25 + 300, retries=1)
        run_stage(st, state, deadline)

    # ---- 4. evaluate everything that trained ------------------------------
    for tag, _, _, _ in arms:
        if not (MARKS / (tag + ".done")).exists():
            continue
        for split, ks in (("test", "2"), ("val", "1,2,4")):
            run_stage(Stage("{}_eval_{}".format(tag, split),
                            eval_argv(tag, split, ks=ks), release="s10",
                            est=(4 if ks == "2" else 10) * 60, retries=1),
                      state, deadline)

    return finish(state, deadline)


def finish(state, deadline):
    if SAMPLER:
        SAMPLER.stop_flag.set()
    log("-" * 72)
    log("done {}/{} stages, {} failed or dropped".format(
        len(state["done"]), len(state["done"]) + len(state["failed"]),
        len(state["failed"])))
    for k, v in state["failed"].items():
        log("   {:<20} {}".format(k, v))
    if SAMPLER:
        log("system, whole run: " + SAMPLER.summary())
    save_state(state)
    try:
        subprocess.call([PY, "scripts/report.py"], cwd=str(ROOT),
                        stdout=open(LOGS / "report.log", "w", encoding="utf-8"),
                        stderr=subprocess.STDOUT)
        log("report written to {}".format(RUNS / "REPORT.md"))
    except Exception as e:
        log("report failed: {}".format(e))
    log("logs in {}".format(LOGS))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
