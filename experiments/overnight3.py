"""Round 3 (2026-09-30): C1 + 2×2 flat-speed study with the rules pre-registered in EXPERIMENTS.md §7.3.

usage (conda env lerobot-arena, from ~/IsaacLab_RS):
  nohup setsid python experiments/overnight3.py > experiments/overnight3.out 2>&1 < /dev/null &
"""

import glob
import json
import os
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402

R = o.R
ENT = ["agent.algorithm.entropy_coef=0.005"]
CELLS = ["C1", "C1Eyes", "C1FlatLanes", "C1EyesFlatLanes"]
E4_FLAT_3SEED = 69.3
E4_LOCKBOX = 46.9


def tasks3(cell):
    return f"Isaac-Ant-R3-{cell}-v0", f"Isaac-Ant-R3-{cell}-Play-v0"


def speed(eval_dir, cond="flat_f1.0"):
    with open(f"{eval_dir}/{cond}.json") as f:
        return json.load(f)["speed_x_mean"]


def run_cell(cell, seed):
    run_name = f"r3_{cell.lower()}_s{seed}"
    task, play = tasks3(cell)
    o.log(f"**{cell} seed {seed} start** | task {task} | overrides {ENT}")
    code = o.run(["./isaaclab.sh", "-p", f"{R}/train.py", "--task", task, "--headless", "--seed", str(seed),
                  "--run_name", run_name, *ENT], f"experiments/train_logs/{run_name}.log")
    dirs = sorted(glob.glob(f"logs/rsl_rl/ant/*_{run_name}"))
    ckpt = f"{dirs[-1]}/model_999.pt" if dirs else None
    if code != 0 or not ckpt or not os.path.exists(ckpt):
        o.log(f"{cell} seed {seed} FAILED during training (see experiments/train_logs/{run_name}.log)")
        o.record({"exp": f"R3-{cell}", "seed": seed, "status": "train_failed"})
        return None
    out = f"experiments/eval/{run_name}"
    subprocess.run(["bash", f"{R}/eval_sweep.sh", ckpt, out, play, *ENT], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    m = o.metrics(out)
    m["flat_speed"] = speed(out)
    ts = o.train_stats(dirs[-1])
    entry = {"exp": f"R3-{cell}", "seed": seed, "run_dir": dirs[-1], "ckpt": ckpt, "metrics": m, "train": ts}
    o.record(entry)
    o.log(f"{cell} seed {seed} result: {o.fmt(m)}, flat speed {m['flat_speed']:.2f} m/s | final std {ts['std']:.3f}, "
          f"lr {ts['lr']:.2e} | run `{dirs[-1]}`")
    return entry


def main():
    o.log("## round 3 start (C1 + 2×2 flat-speed study, rules: EXPERIMENTS.md §7.3)")
    results = {cell: run_cell(cell, 42) for cell in CELLS}
    ok = {c: e for c, e in results.items() if e}
    if "C1" not in ok:
        o.log("C1 failed → no selection possible, keep E4")
        return finish(None)
    c1_ho = ok["C1"]["metrics"]["held_out"]
    eligible = {c: e for c, e in ok.items() if e["metrics"]["held_out"] >= c1_ho - 2.0}
    cand = max(eligible, key=lambda c: eligible[c]["metrics"]["flat"])
    table = "; ".join(f"{c}: held-out {e['metrics']['held_out']:.1f}, flat {e['metrics']['flat']:.1f}, "
                      f"speed {e['metrics']['flat_speed']:.2f}" for c, e in ok.items())
    o.log(f"2×2 screening: {table}")
    o.log(f"selection rule (held-out ≥ C1−2 = {c1_ho - 2:.1f}, then highest flat) → candidate {cand}")
    o.record({"round3_selection": cand, "eligible": list(eligible)})

    seeds = {42: ok[cand]}
    for seed in (43, 44):
        e = run_cell(cand, seed)
        if e:
            seeds[seed] = e
    ho = [seeds[s]["metrics"]["held_out"] for s in sorted(seeds)]
    flat = [seeds[s]["metrics"]["flat"] for s in sorted(seeds)]
    spd = [seeds[s]["metrics"]["flat_speed"] for s in sorted(seeds)]
    cond1 = len(ho) == 3 and st.mean(ho) >= 47.0 and min(ho) >= 44.0
    cond2 = len(flat) == 3 and st.mean(flat) >= E4_FLAT_3SEED + 10.0
    o.log(f"{cand} 3-seed: held-out {[round(v, 1) for v in ho]} (mean {st.mean(ho):.1f}) → cond1 {cond1}; "
          f"flat {[round(v, 1) for v in flat]} (mean {st.mean(flat):.1f}) → cond2 {cond2}; "
          f"flat speed {[round(v, 2) for v in spd]} (mean {st.mean(spd):.2f} m/s)")
    replace = False
    if cond1 and cond2:
        _, play = tasks3(cand)
        name = f"R3-{cand}"
        subprocess.run(["bash", "experiments/lockbox_eval.sh", name, seeds[42]["ckpt"], play, *ENT],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        res = {}
        for t in ("lock_rails", "lock_gaps", "lock_pits", "lock_stones"):
            path = f"experiments/eval_lockbox/{name}/{t}.json"
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                res[t] = (round(r["return_mean"], 1), round(100 * r["fall_rate"]), round(r["distance_x_mean"], 1))
        lb = st.mean(v[0] for v in res.values()) if len(res) == 4 else float("nan")
        cond3 = lb >= E4_LOCKBOX - 3.0
        o.log(f"LOCKBOX {name} (seed 42): {res} → mean {lb:.1f} → cond3 {cond3}")
        o.record({"lockbox": name, "results": res, "mean": lb})
        replace = cond3
    o.log(f"replacement decision: {'REPLACE E4 with ' + cand if replace else 'keep E4'}")
    o.record({"round3_decision": "replace" if replace else "keep_E4", "candidate": cand})
    finish(seeds[42] if replace else None, cand if replace else None)


def finish(final, cell=None):
    if final is None:
        o.log("submission candidate stays E4 (smoke test and videos from round 1 / earlier apply)")
        o.log("## round 3 done")
        return
    _, play = tasks3(cell)
    out = f"experiments/eval/smoke_r3_{cell.lower()}"
    os.makedirs(out, exist_ok=True)
    for terrain, fric in (("flat", 1.0), ("boxes_mid", 1.0), ("flat", 0.2), ("boxes_mid", 0.2)):
        for mode in ("average", "multiply"):
            name = f"{terrain}_f{fric}_{mode}"
            code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain, "--friction",
                          str(fric), "--combine_mode", mode, "--checkpoint", final["ckpt"], "--output",
                          f"{out}/{name}.json", *ENT], f"{out}/{name}.log")
            msg = "FAILED"
            if code == 0 and os.path.exists(f"{out}/{name}.json"):
                with open(f"{out}/{name}.json") as f:
                    r = json.load(f)
                msg = f"OK, return {r['return_mean']:.1f}, fall {100 * r['fall_rate']:.0f}%"
            o.log(f"smoke R3-{cell} {name}: {msg}")
    code = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", play, "--headless", "--seed", "24",
                  "--num_envs", "100", "--checkpoint", final["ckpt"], *ENT], f"{out}/official_play_one_episode.log")
    with open(f"{out}/official_play_one_episode.log") as f:
        res = [line.strip() for line in f if "[RESULT]" in line]
    o.log(f"smoke R3-{cell} official play_one_episode.py: exit {code}, {res}")
    for terrain in ("flat", "boxes_mid"):
        code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain, "--num_envs", "16",
                      "--video", "--checkpoint", final["ckpt"], "--output", f"experiments/videos/R3_{cell}_{terrain}.json",
                      *ENT], f"experiments/videos/R3_{cell}_{terrain}.log")
        o.log(f"video R3_{cell}_{terrain}: {'OK' if code == 0 else 'FAILED'}")
    o.log("## round 3 done")


if __name__ == "__main__":
    main()
