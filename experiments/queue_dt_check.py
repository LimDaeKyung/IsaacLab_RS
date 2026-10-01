"""Physics dt diagnostic (EXPERIMENTS.md §9, pre-registered 2026-09-30). Runs after experiments/overnight4.py exits.

Same E4 checkpoints, same action period (1/60 s); only the physics step changes:
  A (current): sim.dt = 1/120, decimation 2  -> existing results in experiments/eval/e4_relheight_s{seed}
  B (finer)  : sim.dt = 1/240, decimation 4  -> experiments/eval/dt240_e4_s{seed}
Evaluation only, no training.

usage (conda env lerobot-arena, from ~/IsaacLab_RS):
  nohup setsid python experiments/queue_dt_check.py > experiments/queue_dt_check.out 2>&1 < /dev/null &
"""

import glob
import json
import os
import re
import statistics as st
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402

R = o.R
PLAY = "Isaac-Ant-Rough-E4-Play-v0"
SEEDS = (42, 43, 44, 45, 46)
DT240 = ["env.sim.dt=0.004166666666666667", "env.decimation=4", "env.sim.render_interval=4"]
START_DEADLINE = datetime(2026, 10, 1, 8, 30)  # ~30 min of evaluation must finish before the 09:00 team deadline


def queue_busy():
    return subprocess.run(["pgrep", "-f", "experiments/overnight4.py"], stdout=subprocess.DEVNULL).returncode == 0


def ckpt_of(prefix):
    c = sorted(glob.glob(f"logs/rsl_rl/ant/*_{prefix}/model_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    return c[-1] if c else None


def step_sizes(log_path):
    """(physics dt, env step dt) printed by ManagerBasedEnv, to confirm the hydra override took effect."""
    text = open(log_path).read()
    p = re.search(r"Physics step-size\s*:\s*([0-9.e-]+)", text)
    e = re.search(r"Environment step-size\s*:\s*([0-9.e-]+)", text)
    return (float(p.group(1)) if p else None, float(e.group(1)) if e else None)


def ho_fall(eval_dir):
    return st.mean(json.load(open(f"{eval_dir}/{c}.json"))["fall_rate"] for c in o.HELD_OUT)


def sweep(ckpt, out, extra):
    subprocess.run(["bash", f"{R}/eval_sweep.sh", ckpt, out, PLAY, *extra],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    while queue_busy():
        time.sleep(60)
    if datetime.now() > START_DEADLINE:
        o.log("dt check skipped: overnight4 finished after 08:30, not enough time before the 09:00 deadline")
        return
    o.log("## dt check start (E4 s42-46, sim.dt 1/240 + decimation 4 vs existing 1/120 + decimation 2)")

    # noise floor: re-run A for seed 42 to see how much the same setting moves between runs
    ck42 = ckpt_of("e4_relheight_s42")
    sweep(ck42, "experiments/eval/dt120_rerun_e4_s42", [])
    a0, a1 = o.metrics("experiments/eval/e4_relheight_s42"), o.metrics("experiments/eval/dt120_rerun_e4_s42")
    o.log(f"dt check noise floor (A re-run, s42): held-out {a0['held_out']:.1f} -> {a1['held_out']:.1f}, "
          f"flat {a0['flat']:.1f} -> {a1['flat']:.1f}")

    rows = []
    for seed in SEEDS:
        ck = ckpt_of(f"e4_relheight_s{seed}")
        out = f"experiments/eval/dt240_e4_s{seed}"
        sweep(ck, out, DT240)
        pdt, edt = step_sizes(f"{out}/flat_f1.0.log")
        if pdt is None or abs(pdt - 1 / 240) > 1e-6 or abs(edt - 1 / 60) > 1e-6:
            o.log(f"dt check s{seed}: override NOT applied (physics {pdt}, env step {edt}), stopping — see {out}/flat_f1.0.log")
            return
        a, b = o.metrics(f"experiments/eval/e4_relheight_s{seed}"), o.metrics(out)
        a["ho_fall"] = ho_fall(f"experiments/eval/e4_relheight_s{seed}")
        b["ho_fall"] = ho_fall(out)
        rows.append((seed, a, b))
        o.log(f"dt check s{seed}: held-out {a['held_out']:.1f} -> {b['held_out']:.1f} "
              f"(fall {100 * a['ho_fall']:.0f}% -> {100 * b['ho_fall']:.0f}%), seen {a['seen']:.1f} -> {b['seen']:.1f}, "
              f"flat {a['flat']:.1f} -> {b['flat']:.1f}")
        o.record({"dt_check": seed, "A_1_120": a, "B_1_240": b, "physics_dt": pdt, "env_dt": edt})

    diffs = [b["held_out"] - a["held_out"] for _, a, b in rows]
    md, sd = st.mean(diffs), st.stdev(diffs)
    half = 2.776 * sd / len(diffs) ** 0.5  # t(0.975, df=4)
    o.log(f"dt check summary (B - A, held-out, 5 paired seeds): mean {md:+.1f} [95% CI {md - half:+.1f}, {md + half:+.1f}], "
          f"B better in {sum(d > 0 for d in diffs)}/5 | pre-registered: |mean| < 3 and CI contains 0 -> 1/120 is enough")
    o.record({"dt_check_summary": {"diffs": diffs, "mean": md, "ci": [md - half, md + half]}})
    o.log("## dt check done")


if __name__ == "__main__":
    main()
