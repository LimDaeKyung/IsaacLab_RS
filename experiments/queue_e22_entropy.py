"""E22: entropy_coef dose-response 0.01 / 0.002 on E4 (EXPERIMENTS.md §11, pre-registered 2026-09-30).
Runs after overnight4.py, queue_dt_check.py and queue_e21_energy.py exit.

Training: Isaac-Ant-Rough-E4-v0 + agent.algorithm.entropy_coef=<value> (everything else as E4, 1000 it).
Scoring:  Isaac-Ant-Rough-E4-Play-v0 (entropy only affects training).
Existing points: 0 = E4 (e4_relheight_s{seed}), 0.005 = E6 (e6_s{seed}).

usage (conda env lerobot-arena, from ~/IsaacLab_RS):
  nohup setsid python experiments/queue_e22_entropy.py > experiments/queue_e22_entropy.out 2>&1 < /dev/null &
"""

import glob
import json
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402

R = o.R
TASK, PLAY = "Isaac-Ant-Rough-E4-v0", "Isaac-Ant-Rough-E4-Play-v0"
VALUES = (("0.01", "e22_ent01"), ("0.002", "e22_ent002"))  # 0.01 first: rsl_rl default, the untested side above 0.005
SEEDS = (42, 43)
DEADLINE = datetime(2026, 10, 1, 9, 0)  # new-experiment cutoff (extended by the user)
EST_RUN = timedelta(minutes=17)
WAIT_FOR = ("experiments/overnight4.py", "experiments/queue_dt_check.py", "experiments/queue_e21_energy.py")


def busy():
    return any(subprocess.run(["pgrep", "-f", p], stdout=subprocess.DEVNULL).returncode == 0 for p in WAIT_FOR)


def ckpt_of(prefix):
    c = sorted(glob.glob(f"logs/rsl_rl/ant/*_{prefix}/model_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    return c[-1] if c else None


def full(eval_dir):
    m = o.metrics(eval_dir)
    m["ho_fall"] = st.mean(json.load(open(f"{eval_dir}/{c}.json"))["fall_rate"] for c in o.HELD_OUT)
    return m


def main():
    while busy():
        time.sleep(60)
    o.log("## E22 (entropy_coef 0.01 / 0.002 on E4) start")
    res = {}
    for value, prefix in VALUES:
        for seed in SEEDS:
            if datetime.now() + EST_RUN > DEADLINE:
                o.log(f"E22 entropy {value} seed {seed} skipped: would end after 09:00")
                continue
            run = f"{prefix}_s{seed}"
            over = [f"agent.algorithm.entropy_coef={value}"]
            o.log(f"**E22 entropy {value} seed {seed} start** | task {TASK} | overrides {over} | 1000 it")
            code = o.run(["./isaaclab.sh", "-p", f"{R}/train.py", "--task", TASK, "--headless", "--seed", str(seed),
                          "--run_name", run, *over], f"experiments/train_logs/{run}.log")
            ck = ckpt_of(run)
            if code != 0 or not ck or not ck.endswith("model_999.pt"):
                o.log(f"E22 entropy {value} seed {seed} FAILED during training (see experiments/train_logs/{run}.log)")
                o.record({"exp": "E22", "entropy": value, "seed": seed, "status": "train_failed"})
                continue
            subprocess.run(["bash", f"{R}/eval_sweep.sh", ck, f"experiments/eval/{run}", PLAY, *over],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            m, ts = full(f"experiments/eval/{run}"), o.train_stats(os.path.dirname(ck))
            res[(value, seed)] = m
            o.record({"exp": "E22", "entropy": value, "seed": seed, "run": run, "ckpt": ck, "metrics": m, "train": ts})
            o.log(f"E22 entropy {value} seed {seed} result: {o.fmt(m)}, held-out fall {100 * m['ho_fall']:.0f}% "
                  f"| std {ts['std']:.3f}, lr {ts['lr']:.2e} | `{os.path.dirname(ck)}`")

    # dose-response table over both seeds: 0 (E4), 0.002, 0.005 (E6), 0.01
    points = [("0", "e4_relheight"), ("0.002", "e22_ent002"), ("0.005", "e6"), ("0.01", "e22_ent01")]
    rows = {}
    for value, prefix in points:
        ms = [full(f"experiments/eval/{prefix}_s{s}") for s in SEEDS
              if os.path.exists(f"experiments/eval/{prefix}_s{s}/boxes_mid_f0.2.json")]
        stds = [o.train_stats(os.path.dirname(ckpt_of(f"{prefix}_s{s}")))["std"] for s in SEEDS if ckpt_of(f"{prefix}_s{s}")]
        if len(ms) == len(SEEDS):
            rows[value] = {k: st.mean(m[k] for m in ms) for k in ("held_out", "ho_fall", "flat")}
            rows[value]["std"] = st.mean(stds)
            o.log(f"E22 table entropy {value}: held-out {rows[value]['held_out']:.1f}, held-out fall "
                  f"{100 * rows[value]['ho_fall']:.0f}%, flat {rows[value]['flat']:.1f}, final std {rows[value]['std']:.3f} (s42·43)")
    # pre-registered: a new value is "better than 0.005" with the §8.3 screening rule against E6 s42·43
    e6 = rows.get("0.005")
    for value in ("0.002", "0.01"):
        if value not in rows or not e6:
            o.log(f"E22 entropy {value}: incomplete → no verdict")
            continue
        c = rows[value]
        cond_a = c["held_out"] >= e6["held_out"] + 3 or (c["held_out"] >= e6["held_out"] - 1 and c["ho_fall"] <= e6["ho_fall"] - 0.10)
        cond_b = c["flat"] >= e6["flat"] - 5
        o.log(f"E22 entropy {value} vs 0.005 (E6): {'PASS' if cond_a and cond_b else 'fail'}")
        o.record({"screen": f"E22_{value}", "ref": "E6", "c": c, "r": e6, "pass": cond_a and cond_b})
    o.log("## E22 done")


if __name__ == "__main__":
    main()
