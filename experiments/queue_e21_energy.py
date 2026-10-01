"""E21: training-only energy penalty reduced -0.05 -> -0.02 (EXPERIMENTS.md §10, pre-registered 2026-09-30).
Runs after experiments/overnight4.py and experiments/queue_dt_check.py exit.

Training: Isaac-Ant-Rough-E4-v0 + env.rewards.energy.weight=-0.02 (everything else as E4, entropy 0, 1000 it).
Scoring:  Isaac-Ant-Rough-E4-Play-v0 with NO override, i.e. the original reward (energy -0.05).
Reference: E4 seeds 42, 43 (experiments/eval/e4_relheight_s{seed}).

usage (conda env lerobot-arena, from ~/IsaacLab_RS):
  nohup setsid python experiments/queue_e21_energy.py > experiments/queue_e21_energy.out 2>&1 < /dev/null &
"""

import glob
import json
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402

R = o.R
TASK, PLAY = "Isaac-Ant-Rough-E4-v0", "Isaac-Ant-Rough-E4-Play-v0"
ENERGY = ["env.rewards.energy.weight=-0.02"]
SEEDS = (42, 43)
START_DEADLINE = datetime(2026, 10, 1, 8, 25)  # 2 runs x ~16.5 min must end before the 09:00 new-experiment deadline (extended by the user)


def busy():
    return any(subprocess.run(["pgrep", "-f", p], stdout=subprocess.DEVNULL).returncode == 0
               for p in ("experiments/overnight4.py", "experiments/queue_dt_check.py"))


def ckpt_of(prefix):
    c = sorted(glob.glob(f"logs/rsl_rl/ant/*_{prefix}/model_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    return c[-1] if c else None


def ho_fall(eval_dir):
    return st.mean(json.load(open(f"{eval_dir}/{c}.json"))["fall_rate"] for c in o.HELD_OUT)


def flat_speed(eval_dir):
    return json.load(open(f"{eval_dir}/flat_f1.0.json"))["speed_x_mean"]


def raw_power(run_dir, weight):
    """Unweighted power term (per second of episode) at the end of training, to check the policy really spends more."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return st.mean(e.value for e in ea.Scalars("Episode_Reward/energy")[-50:]) / weight


def full(eval_dir):
    m = o.metrics(eval_dir)
    m["ho_fall"], m["flat_speed"] = ho_fall(eval_dir), flat_speed(eval_dir)
    return m


def main():
    while busy():
        time.sleep(60)
    if datetime.now() > START_DEADLINE:
        o.log("E21 skipped: earlier queues finished after 08:25, not enough time before the 09:00 deadline")
        return
    o.log("## E21 (training-only energy weight -0.05 -> -0.02) start")
    res = {}
    for seed in SEEDS:
        run = f"e21_energy02_s{seed}"
        o.log(f"**E21 seed {seed} start** | task {TASK} | overrides {ENERGY} | 1000 it")
        code = o.run(["./isaaclab.sh", "-p", f"{R}/train.py", "--task", TASK, "--headless", "--seed", str(seed),
                      "--run_name", run, *ENERGY], f"experiments/train_logs/{run}.log")
        ck = ckpt_of(run)
        if code != 0 or not ck or not ck.endswith("model_999.pt"):
            o.log(f"E21 seed {seed} FAILED during training (see experiments/train_logs/{run}.log)")
            o.record({"exp": "E21", "seed": seed, "status": "train_failed"})
            continue
        # scoring with the original reward: no override passed to the Play task
        subprocess.run(["bash", f"{R}/eval_sweep.sh", ck, f"experiments/eval/{run}", PLAY],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        c, r = full(f"experiments/eval/{run}"), full(f"experiments/eval/e4_relheight_s{seed}")
        pc = raw_power(os.path.dirname(ck), -0.02)
        pr = raw_power(os.path.dirname(ckpt_of(f"e4_relheight_s{seed}")), -0.05)
        res[seed] = (c, r)
        o.record({"exp": "E21", "seed": seed, "run": run, "ckpt": ck, "metrics": c, "ref_E4": r,
                  "raw_power": pc, "raw_power_E4": pr, "train": o.train_stats(os.path.dirname(ck))})
        o.log(f"E21 seed {seed} vs E4: held-out {c['held_out']:.1f} vs {r['held_out']:.1f}, held-out fall "
              f"{100 * c['ho_fall']:.0f}% vs {100 * r['ho_fall']:.0f}%, flat {c['flat']:.1f} vs {r['flat']:.1f}, "
              f"flat speed {c['flat_speed']:.2f} vs {r['flat_speed']:.2f}, raw power (train, per s) {pc:.1f} vs {pr:.1f} "
              f"| `{os.path.dirname(ck)}`")

    if len(res) < 2:
        o.log("E21: incomplete runs → not passed")
    else:
        avg = lambda k, i: st.mean(res[s][i][k] for s in SEEDS)  # noqa: E731
        c_ho, r_ho, c_fall, r_fall = avg("held_out", 0), avg("held_out", 1), avg("ho_fall", 0), avg("ho_fall", 1)
        c_flat, r_flat = avg("flat", 0), avg("flat", 1)
        cond_a = c_ho >= r_ho + 3 or (c_ho >= r_ho - 1 and c_fall <= r_fall - 0.10)
        cond_b = c_flat >= r_flat - 5
        o.log(f"E21 vs E4 (2-seed): held-out {c_ho:.1f} vs {r_ho:.1f}, held-out fall {100 * c_fall:.0f}% vs "
              f"{100 * r_fall:.0f}%, flat {c_flat:.1f} vs {r_flat:.1f} → {'PASS' if cond_a and cond_b else 'fail'}")
        o.record({"screen": "E21", "ref": "E4", "c": [c_ho, c_fall, c_flat], "r": [r_ho, r_fall, r_flat],
                  "pass": cond_a and cond_b})
    o.log("## E21 done")


if __name__ == "__main__":
    main()
