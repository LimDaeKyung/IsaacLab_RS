"""Batch 7 (share-fb): one change on top of the new submission F3a, at the same 1000 + 600 it budget.

H1 = F1 lineage (LowFric: ground multiply + robot friction 0.05-1.2) 1000 it -> warm-started 600 it (LowFric)
H2 = G1 lineage (training-only energy weight -0.02) 1000 it -> warm-started 600 it (energy -0.02)
Scoring: original reward, Play task of each lineage, entropy override only.
Pre-registered 2026-10-01 03:00 in experiments/EXPERIMENTS_overnight_fb.md (batch 7), before any batch-7 result.
"""

import json
import os
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight4 as r4  # noqa: E402
import queue_fb_b5 as b5  # noqa: E402

R = o.R
ENT = ["agent.algorithm.entropy_coef=0.005"]
ENERGY = ["env.rewards.energy.weight=-0.02"]
LF_TASK, LF_PLAY = "Isaac-Ant-R2-LowFricOracle-v0", "Isaac-Ant-R2-LowFricOracle-Play-v0"
OR_TASK, OR_PLAY = "Isaac-Ant-R2-Oracle-v0", "Isaac-Ant-R2-Oracle-Play-v0"

LINEAGES = {
    # name: (stage-1 run prefix, task, play, training overrides)
    "H1": ("f1", LF_TASK, LF_PLAY, ENT),
    "H2": ("g1", OR_TASK, OR_PLAY, ENT + ENERGY),
}


def log(text):
    o.log(f"[fb] {text}")


def wait_for_run(run_name):
    while subprocess.run(["pgrep", "-f", "--", f"--run_name {run_name}( |$)"], stdout=subprocess.DEVNULL).returncode == 0:
        time.sleep(30)


def gpu_training_busy():
    return subprocess.run(["pgrep", "-f", r"^\S*/python3? scripts/reinforcement_learning/rsl_rl/(train|train_finetune)\.py"], stdout=subprocess.DEVNULL).returncode == 0


def train(run_name, task, seed, train_over, iters, init=None):
    """Train unless the final checkpoint already exists; returns the checkpoint or None."""
    wait_for_run(run_name)
    while gpu_training_busy():  # never share the 8 GB GPU with another training (OOM at 02:57, see log)
        time.sleep(30)
    ck = r4.ckpt_of(run_name)
    if ck and ck.endswith(f"model_{iters - 1}.pt"):
        return ck
    minutes = r4.EST_FT if init else r4.EST_RUN * iters / 1000
    if not b5.can_start(minutes):
        log(f"{run_name} skipped: would start after 08:00 or end too late")
        return None
    script = "train_finetune.py" if init else "train.py"
    log(f"**{run_name} start** | {script} | task {task} | train overrides {train_over} | {iters} it"
        + (f" | warm start `{init}`" if init else ""))
    cmd = ["./isaaclab.sh", "-p", f"{R}/{script}", "--task", task, "--headless", "--seed", str(seed),
           "--run_name", run_name, "--max_iterations", str(iters), *train_over]
    if init:
        cmd += ["--init_checkpoint", init]
    code = o.run(cmd, f"experiments/train_logs/{run_name}.log")
    ck = r4.ckpt_of(run_name)
    if code != 0 or not ck or not ck.endswith(f"model_{iters - 1}.pt"):
        log(f"{run_name} FAILED (see experiments/train_logs/{run_name}.log)")
        return None
    return ck


def lineage(name, seed):
    prefix, task, play, over = LINEAGES[name]
    ck1 = train(f"{prefix}_s{seed}", task, seed, over, 1000)
    if ck1 is None:
        return None
    run2 = f"{name.lower()}_s{seed}"
    ck2 = train(run2, task, seed, over, 600, init=ck1)
    if ck2 is None:
        return None
    if not os.path.exists(f"experiments/eval/{run2}/boxes_mid_f0.2.json"):
        subprocess.run(["bash", f"{R}/eval_sweep.sh", ck2, f"experiments/eval/{run2}", play, *ENT],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    m = r4.full_metrics(run2)
    m["mult"], m["mult_fall"] = b5.eval_multiply(run2, play, ck2)
    m["ckpt"] = ck2
    ts = o.train_stats(os.path.dirname(ck2))
    o.record({"batch": 7, "exp": name, "seed": seed, "run": run2, "ckpt": ck2, "metrics": m, "train": ts})
    log(f"{name} s{seed}: held-out {m['held_out']:.1f}, flat {m['flat']:.1f} (speed {m['flat_speed']:.2f}), "
        f"ho-fall {100 * m['ho_fall']:.0f}%, boxes μ0.2 multiply {m['mult']:.1f} (fall {100 * m['mult_fall']:.0f}%) "
        f"| std {ts['std']:.3f}, lr {ts['lr']:.2e}")
    return m


def f3a(seed):
    m = r4.full_metrics(f"f3a_s{seed}")
    m["mult"], m["mult_fall"] = b5.eval_multiply(f"f3a_s{seed}", OR_PLAY, r4.ckpt_of(f"f3a_s{seed}"))
    return m


def main():
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    log(f"## batch 7 start (one change on top of submission {state['submission']}, same 1000+600 it budget)")
    ref = [f3a(s) for s in (42, 43)]
    ho0, fl0, mu0 = b5.mean(ref, "held_out"), b5.mean(ref, "flat"), b5.mean(ref, "mult")
    log(f"F3a s42/43 reference: held-out {ho0:.1f}, flat {fl0:.1f}, boxes μ0.2 multiply {mu0:.1f}")

    # H2 stage 1 for seed 42 is already running (started by the cancelled batch 6); H1 stage 1 exists for 42/43
    screen = {"H1": [lineage("H1", s) for s in (42, 43)], "H2": [lineage("H2", s) for s in (42, 43)]}
    passed = []
    for exp, ms in screen.items():
        if any(m is None for m in ms):
            log(f"screen {exp}: incomplete")
            continue
        ho, fl, mu = b5.mean(ms, "held_out"), b5.mean(ms, "flat"), b5.mean(ms, "mult")
        if exp == "H1":
            ok = ((ho >= ho0 - 1 and mu >= mu0 + 5) or ho >= ho0 + 3) and fl >= fl0 - 5
        else:
            ok = (ho >= ho0 + 3 or (ho >= ho0 - 1 and fl >= fl0 + 10)) and fl >= fl0 - 5
        log(f"screen {exp}: held-out {ho:.1f} vs {ho0:.1f}, flat {fl:.1f} vs {fl0:.1f}, μ0.2 multiply {mu:.1f} vs "
            f"{mu0:.1f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 7, "exp": exp, "held_out": ho, "flat": fl, "mult": mu, "decision": "PASS" if ok else "fail"})
        if ok:
            passed.append((ho, exp))
    passed.sort(reverse=True)
    log(f"batch 7 screening passed: {[e for _, e in passed] or 'none'}")

    for _, exp in passed:
        base = {s: f3a(s) for s in (47, 48, 49)}
        cand = {s: lineage(exp, s) for s in (47, 48, 49)}
        seeds = [s for s in (47, 48, 49) if cand.get(s)]
        if len(seeds) < 3:
            log(f"confirm {exp}: only {len(seeds)} seeds → no replacement decision")
            continue
        diffs = [cand[s]["held_out"] - base[s]["held_out"] for s in seeds]
        d, wins = st.mean(diffs), sum(x > 0 for x in diffs)
        half = r4.T_975_DF2 * st.stdev(diffs) / 3 ** 0.5
        fl_c, fl_b = b5.mean([cand[s] for s in seeds], "flat"), b5.mean([base[s] for s in seeds], "flat")
        mu_c, mu_b = b5.mean([cand[s] for s in seeds], "mult"), b5.mean([base[s] for s in seeds], "mult")
        fl_ok = fl_c >= fl_b - 5
        log(f"confirm {exp} vs F3a (s47–49): held-out diff {d:+.1f} [95% CI {d - half:+.1f}, {d + half:+.1f}], wins "
            f"{wins}/3, flat {fl_c:.1f} vs {fl_b:.1f}, μ0.2 multiply {mu_c:.1f} vs {mu_b:.1f}")
        # H1 is judged on its target condition as pre-registered; H2 on held-out
        if exp == "H1":
            mu_diffs = [cand[s]["mult"] - base[s]["mult"] for s in seeds]
            cond = (d >= -1 and st.mean(mu_diffs) >= 5 and sum(x > 0 for x in mu_diffs) >= 2) or (d >= 3 and wins >= 2)
        else:
            cond = d >= 3 and wins >= 2
        o.record({"batch": 7, "stage": "confirm", "exp": exp, "diffs": diffs, "wins": wins, "flat_ok": fl_ok,
                  "cond": cond})
        if not (cond and fl_ok):
            log(f"confirm {exp}: replacement conditions not met → keep F3a")
            continue
        _, _, play, _ = LINEAGES[exp]
        ck = cand[47]["ckpt"]
        subprocess.run(["bash", "experiments/lockbox_eval.sh", f"fb_{exp}_s47", ck, play, *ENT],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vals = []
        for t in ("lock_rails", "lock_gaps", "lock_pits", "lock_stones"):
            try:
                with open(f"experiments/eval_lockbox/fb_{exp}_s47/{t}.json") as f:
                    vals.append(json.load(f)["return_mean"])
            except (OSError, KeyError, json.JSONDecodeError):
                vals.append(float("nan"))
        lb = st.mean(vals)
        need = state["lockbox"] - 3
        ok = lb >= need
        log(f"LOCKBOX {exp} s47 (lockbox reused, 5th model): {[round(v, 1) for v in vals]} → mean {lb:.1f} "
            f"(need ≥ {need:.1f}) → {'REPLACE F3a' if ok else 'keep F3a'}")
        if not ok:
            continue
        os.makedirs(f"experiments/eval/smoke_fb_{exp}", exist_ok=True)
        for terrain in ("flat", "boxes_mid"):
            for fric in ("1.0", "0.2"):
                for mode in ("average", "multiply"):
                    name = f"{terrain}_f{fric}_{mode}"
                    code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain,
                                  "--friction", fric, "--combine_mode", mode, "--checkpoint", ck, "--output",
                                  f"experiments/eval/smoke_fb_{exp}/{name}.json", *ENT],
                                 f"experiments/train_logs/smoke_fb_{exp}_{name}.log")
                    log(f"smoke {exp} {name}: {'OK' if code == 0 else 'FAILED'}")
        code = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", play, "--headless",
                      "--num_envs", "100", "--checkpoint", ck, *ENT], f"experiments/train_logs/smoke_fb_{exp}_official.log")
        log(f"smoke {exp} official play_one_episode: exit {code}")
        new_state = {"submission": exp, "task": LINEAGES[exp][1], "play": play, "ckpt": ck,
                     "score_overrides": ENT, "train_overrides": LINEAGES[exp][3], "lockbox": lb,
                     "runs": {str(s): f"{exp.lower()}_s{s}" for s in (42, 43, 47, 48, 49)},
                     "previous": state["submission"]}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(new_state, f, indent=1)
        log(f"NEW SUBMISSION: {exp} `{ck}` (play {play})")
        break
    log("## batch 7 done")


if __name__ == "__main__":
    main()
