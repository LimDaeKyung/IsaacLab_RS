"""Overnight sequential search on top of E4 (2026-09-29 → 09-30 09:00), with pre-registered adoption rules.

Order (one change per step, stacked on the current winner):
  E6  entropy_coef 0.0 -> 0.005 (retry 0.002 if the action std blows up above 1.0)
  E7  flat tile share 0.10 -> 0.25
  E8  3-step history of dynamic observation terms
  E9  actor/critic observation normalization
  E10 weak pushes + torso mass scaling (training only)
  then: seeds 43/44 of the final winner (only if it is not E4), and a Play smoke test.

Adoption (screening, seed 42 vs current winner):
  held-out mean >= winner + 3, or (held-out mean >= winner - 2 and flat return >= winner + 10).
Final replacement of E4: 3-seed held-out mean >= E4 3-seed mean + 3 and every seed above the E4 mean.

Every decision is appended to experiments/overnight_log.md and experiments/overnight_decisions.jsonl.

usage (conda env lerobot-arena, from ~/IsaacLab_RS):
  nohup setsid python experiments/overnight.py > experiments/overnight.out 2>&1 < /dev/null &
"""

import glob
import json
import os
import statistics as st
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
R = "scripts/reinforcement_learning/rsl_rl"
LOG_MD = "experiments/overnight_log.md"
LOG_JSONL = "experiments/overnight_decisions.jsonl"

SEEN = ["flat_f0.5", "boxes_low_f1.0", "boxes_mid_f1.0", "boxes_high_f1.0", "boxes_mid_f0.5",
        "rough_low_f1.0", "rough_high_f1.0", "slope_f1.0", "stairs_f1.0"]
HELD_OUT = ["ho_boxes_fine_f1.0", "ho_obstacles_f1.0", "ho_wave_f1.0", "flat_f0.1", "boxes_mid_f0.2"]
FLAG_ORDER = ("Flat", "Hist", "Push")


def log(text):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_MD, "a") as f:
        f.write(f"- [{stamp}] {text}\n")
    print(f"[OVERNIGHT] {stamp} {text}", flush=True)


def record(entry):
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def tasks(flags):
    suffix = "".join(f for f in FLAG_ORDER if f in flags)
    base = f"Isaac-Ant-Rough-E4-{suffix}" if suffix else "Isaac-Ant-Rough-E4"
    return f"{base}-v0", f"{base}-Play-v0"


def run(cmd, log_path):
    with open(log_path, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def train(flags, overrides, run_name, seed):
    task, _ = tasks(flags)
    code = run(["./isaaclab.sh", "-p", f"{R}/train.py", "--task", task, "--headless", "--seed", str(seed),
                "--run_name", run_name, *overrides], f"experiments/train_logs/{run_name}.log")
    dirs = sorted(glob.glob(f"logs/rsl_rl/ant/*_{run_name}"))
    ckpts = sorted(glob.glob(f"{dirs[-1]}/model_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3])) if dirs else []
    if code != 0 or not ckpts or not ckpts[-1].endswith("model_999.pt"):
        return None, None
    return dirs[-1], ckpts[-1]


def evaluate(flags, overrides, ckpt, run_name):
    _, play = tasks(flags)
    out = f"experiments/eval/{run_name}"
    subprocess.run(["bash", f"{R}/eval_sweep.sh", ckpt, out, play, *overrides],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def metrics(eval_dir):
    def load(c):
        with open(f"{eval_dir}/{c}.json") as f:
            return json.load(f)
    flat = load("flat_f1.0")
    return {
        "held_out": st.mean(load(c)["return_mean"] for c in HELD_OUT),
        "seen": st.mean(load(c)["return_mean"] for c in SEEN),
        "flat": flat["return_mean"],
        "flat_fall": flat["fall_rate"],
        "flat_len": flat["episode_length_mean"],
        "boxes_mid": load("boxes_mid_f1.0")["return_mean"],
    }


def train_stats(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    last = lambda t: ea.Scalars(t)[-1].value if t in tags else float("nan")  # noqa: E731
    return {"std": last("Policy/mean_noise_std"), "lr": last("Loss/learning_rate"),
            "train_reward": last("Train/mean_reward"), "train_len": last("Train/mean_episode_length")}


def fmt(m):
    return (f"held-out {m['held_out']:.1f}, seen {m['seen']:.1f}, flat {m['flat']:.1f} "
            f"(fall {100 * m['flat_fall']:.0f}%, len {m['flat_len']:.0f}), boxes±10 {m['boxes_mid']:.1f}")


def step(exp_id, hypothesis, flags, overrides, winner):
    run_name = f"{exp_id.lower()}_s42"
    log(f"**{exp_id} start** — {hypothesis} | task {tasks(flags)[0]} | overrides {overrides or '-'}")
    run_dir, ckpt = train(flags, overrides, run_name, 42)
    if ckpt is None:
        log(f"{exp_id} FAILED during training (see experiments/train_logs/{run_name}.log) → not adopted")
        record({"exp": exp_id, "status": "train_failed"})
        return None, None
    ts = train_stats(run_dir)
    m = metrics(evaluate(flags, overrides, ckpt, run_name))
    entry = {"exp": exp_id, "run_dir": run_dir, "ckpt": ckpt, "flags": flags, "overrides": overrides,
             "metrics": m, "train": ts}
    record(entry)
    log(f"{exp_id} result: {fmt(m)} | final std {ts['std']:.3f}, lr {ts['lr']:.2e} | run `{run_dir}`")
    return entry, ts


def adopt(entry, winner):
    m, w = entry["metrics"], winner["metrics"]
    by_ho = m["held_out"] >= w["held_out"] + 3.0
    by_flat = m["held_out"] >= w["held_out"] - 2.0 and m["flat"] >= w["flat"] + 10.0
    return by_ho or by_flat, (f"held-out {m['held_out']:.1f} vs winner {w['held_out']:.1f} (need +3), "
                              f"flat {m['flat']:.1f} vs {w['flat']:.1f} (need +10 with held-out ≥ −2)")


def main():
    log("## overnight run start (base: E4 seed 42)")
    winner = {"exp": "E4", "flags": (), "overrides": [], "metrics": metrics("experiments/eval/e4_relheight_s42"),
              "ckpt": sorted(glob.glob("logs/rsl_rl/ant/*_e4_relheight_s42/model_999.pt"))[0]}
    log(f"E4 reference: {fmt(winner['metrics'])}")

    plan = [
        ("E6", "entropy_coef 0.0→0.005: keep exploration alive (std collapsed to 0.02–0.06, adaptive LR hit its floor)",
         (), ["agent.algorithm.entropy_coef=0.005"]),
        ("E7", "flat tile share 0.10→0.25: flat regression is forgetting / too little flat experience", ("Flat",), []),
        ("E8", "3-step history of dynamic terms: short memory reduces falls", ("Hist",), []),
        ("E9", "actor/critic observation normalization",
         (), ["agent.policy.actor_obs_normalization=True", "agent.policy.critic_obs_normalization=True"]),
        ("E10", "weak pushes (±0.5 m/s every 10–15 s) + torso mass ×0.8–1.2 (training only)", ("Push",), []),
    ]
    for exp_id, hyp, add_flags, add_over in plan:
        flags = tuple(sorted(set(winner["flags"]) | set(add_flags), key=FLAG_ORDER.index))
        overrides = winner["overrides"] + add_over
        entry, ts = step(exp_id, hyp, flags, overrides, winner)
        if exp_id == "E6" and entry is not None and ts["std"] > 1.0:
            log(f"E6 action std {ts['std']:.2f} > 1.0 → retry with entropy_coef 0.002 (E6b)")
            overrides = winner["overrides"] + ["agent.algorithm.entropy_coef=0.002"]
            entry, ts = step("E6b", "entropy_coef 0.002 (E6 std blew up)", flags, overrides, winner)
        if entry is None:
            continue
        ok, why = adopt(entry, winner)
        log(f"{entry['exp']} {'ADOPTED' if ok else 'not adopted'} — {why}")
        record({"exp": entry["exp"], "decision": "adopted" if ok else "rejected", "why": why})
        if ok:
            winner = entry

    log(f"screening winner: {winner['exp']} ({fmt(winner['metrics'])})")
    final = {"exp": "E4", "flags": (), "overrides": [], "ckpt": sorted(glob.glob("logs/rsl_rl/ant/*_e4_relheight_s42/model_999.pt"))[0]}
    if winner["exp"] != "E4":
        e4_ho = [metrics(f"experiments/eval/e4_relheight_s{s}")["held_out"] for s in (42, 43, 44)]
        new_ho = [winner["metrics"]["held_out"]]
        for seed in (43, 44):
            run_name = f"{winner['exp'].lower()}_s{seed}"
            log(f"{winner['exp']} seed {seed} start")
            run_dir, ckpt = train(winner["flags"], winner["overrides"], run_name, seed)
            if ckpt is None:
                log(f"{winner['exp']} seed {seed} FAILED")
                continue
            new_ho.append(metrics(evaluate(winner["flags"], winner["overrides"], ckpt, run_name))["held_out"])
        e4_mean = st.mean(e4_ho)
        replace = len(new_ho) == 3 and st.mean(new_ho) >= e4_mean + 3.0 and min(new_ho) > e4_mean
        log(f"3-seed held-out: {winner['exp']} {[round(v, 1) for v in new_ho]} (mean {st.mean(new_ho):.1f}) vs "
            f"E4 {[round(v, 1) for v in e4_ho]} (mean {e4_mean:.1f}) → "
            f"{'REPLACE E4 as submission candidate' if replace else 'keep E4 as submission candidate'}")
        record({"exp": winner["exp"], "decision": "final_replace" if replace else "final_keep_E4",
                "new_held_out": new_ho, "e4_held_out": e4_ho})
        if replace:
            final = winner

    # Play smoke test of the submission candidate: terrain swapped like the TA does, two friction combine modes
    _, play = tasks(final["flags"])
    log(f"smoke test start: {final['exp']} on {play}")
    out = f"experiments/eval/smoke_{final['exp'].lower()}"
    os.makedirs(out, exist_ok=True)
    for terrain, fric in (("flat", 1.0), ("boxes_mid", 1.0), ("flat", 0.2), ("boxes_mid", 0.2)):
        for mode in ("average", "multiply"):
            name = f"{terrain}_f{fric}_{mode}"
            code = run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain,
                        "--friction", str(fric), "--combine_mode", mode, "--checkpoint", final["ckpt"],
                        "--output", f"{out}/{name}.json", *final["overrides"]], f"{out}/{name}.log")
            if code == 0 and os.path.exists(f"{out}/{name}.json"):
                with open(f"{out}/{name}.json") as f:
                    r = json.load(f)
                log(f"smoke {name}: OK, return {r['return_mean']:.1f} ± {r['return_std']:.1f}, fall {100 * r['fall_rate']:.0f}%")
            else:
                log(f"smoke {name}: FAILED (exit {code}, see {out}/{name}.log)")
    code = run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", play, "--headless", "--seed", "24",
                "--num_envs", "100", "--checkpoint", final["ckpt"], *final["overrides"]], f"{out}/official_play_one_episode.log")
    with open(f"{out}/official_play_one_episode.log") as f:
        res = [line.strip() for line in f if "[RESULT]" in line]
    log(f"smoke official play_one_episode.py (default Play terrain): exit {code}, {res}")
    log("## overnight run done")


if __name__ == "__main__":
    main()
