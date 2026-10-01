"""Batch 5 (share-fb): E15 one-change screening F1/F3a/F3b/F2, then confirmation on fresh seeds 47-49.

Pre-registration: EXPERIMENTS.md chapter 13 (2026-09-30 22:01), copied into experiments/EXPERIMENTS_overnight_fb.md.
Waits for the share-27 queues (overnight4, dt check, E21, E22, T1) to exit, then runs on the single GPU.
usage (conda env lerobot-arena, from ~/IsaacLab_RS):
  nohup setsid python experiments/queue_fb_b5.py > experiments/queue_fb_b5.out 2>&1 < /dev/null &
"""

import json
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight4 as r4  # noqa: E402

R = o.R
ENT = ["agent.algorithm.entropy_coef=0.005"]
NO_NEW_TRAIN_AFTER = datetime(2026, 10, 1, 8, 0)
PEER_QUEUES = ("overnight4.py", "queue_dt_check.py", "queue_e21_energy.py", "queue_e22_entropy.py",
               "queue_t1_teacher.py")

E15_TASK, E15_PLAY = "Isaac-Ant-R2-Oracle-v0", "Isaac-Ant-R2-Oracle-Play-v0"
SUBMISSION = "logs/rsl_rl/ant/2026-09-30_18-52-10_e15_s44/model_999.pt"
LOCKBOX_E15 = 55.2


def log(text):
    o.log(f"[fb] {text}")


def peer_busy():
    # anchor on the interpreter so shells that merely contain the file name do not match
    for name in PEER_QUEUES:
        pattern = rf"^(\S*/)?python3? experiments/{name}"
        if subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL).returncode == 0:
            return True
    return False


def can_start(minutes):
    return datetime.now().timestamp() + minutes * 60 <= NO_NEW_TRAIN_AFTER.timestamp() + 20 * 60 \
        and datetime.now() < NO_NEW_TRAIN_AFTER


def eval_multiply(run_name, play, ckpt):
    out = f"experiments/eval/{run_name}/boxes_mid_f0.2_multiply.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", "boxes_mid", "--friction", "0.2",
               "--combine_mode", "multiply", "--num_envs", "100", "--checkpoint", ckpt, "--output", out, *ENT],
              out.replace(".json", ".log"))
    try:
        with open(out) as f:
            d = json.load(f)
        return d["return_mean"], d["fall_rate"]
    except (OSError, KeyError, json.JSONDecodeError):
        return float("nan"), float("nan")


def run(label, task, play, run_name, seed, iters=1000, init=None):
    if not can_start(r4.EST_FT if init else r4.EST_RUN * iters / 1000):
        log(f"{label} seed {seed} skipped: would start after 08:00 or end too late")
        return None
    m = r4.train_eval(label, task, play, run_name, seed, ENT, iters=iters, init=init)
    if m is None:
        return None
    m["mult"], m["mult_fall"] = eval_multiply(run_name, play, r4.ckpt_of(run_name))
    log(f"{label} s{seed}: held-out {m['held_out']:.1f}, flat {m['flat']:.1f} (speed {m['flat_speed']:.2f}), "
        f"ho-fall {100 * m['ho_fall']:.0f}%, boxes μ0.2 multiply {m['mult']:.1f} (fall {100 * m['mult_fall']:.0f}%)")
    return m


def e15_metrics(seed):
    run_name = f"e15_s{seed}"
    m = r4.full_metrics(run_name)
    m["mult"], m["mult_fall"] = eval_multiply(run_name, E15_PLAY, r4.ckpt_of(run_name))
    return m


def mean(ms, key):
    return st.mean(m[key] for m in ms)


def main():
    while peer_busy():
        time.sleep(60)
    log("## batch 5 start (E15 one-change screening; rules EXPERIMENTS.md §13 / EXPERIMENTS_overnight_fb.md)")
    ref = [e15_metrics(s) for s in (42, 43)]
    log(f"E15 s42/43 reference: held-out {mean(ref, 'held_out'):.1f}, flat {mean(ref, 'flat'):.1f}, "
        f"boxes μ0.2 multiply {mean(ref, 'mult'):.1f}")

    screen = {}
    for s in (42, 43):
        screen.setdefault("F1", []).append(run("F1", "Isaac-Ant-R2-LowFricOracle-v0", "Isaac-Ant-R2-LowFricOracle-Play-v0",
                                               f"f1_s{s}", s))
    for s in (42, 43):
        screen.setdefault("F3a", []).append(run("F3a", E15_TASK, E15_PLAY, f"f3a_s{s}", s, iters=600,
                                                init=r4.ckpt_of(f"e15_s{s}")))
    for s in (42, 43):
        screen.setdefault("F3b", []).append(run("F3b", E15_TASK, E15_PLAY, f"f3b_s{s}", s, iters=1600))
    for s in (42, 43):
        screen.setdefault("F2", []).append(run("F2", "Isaac-Ant-R4-OracleVar-v0", "Isaac-Ant-R4-OracleVar-Play-v0",
                                               f"f2_s{s}", s))

    ho0, fl0, mu0 = mean(ref, "held_out"), mean(ref, "flat"), mean(ref, "mult")
    passed = []
    for exp, ms in screen.items():
        if any(m is None for m in ms):
            log(f"screen {exp}: incomplete → not judged")
            o.record({"batch": 5, "exp": exp, "decision": "incomplete"})
            continue
        ho, fl, mu = mean(ms, "held_out"), mean(ms, "flat"), mean(ms, "mult")
        if exp == "F1":
            ok = ((ho >= ho0 - 1 and mu >= mu0 + 5) or ho >= ho0 + 3) and fl >= fl0 - 5
        elif exp == "F3a":
            ok = ho >= ho0 + 3
        elif exp == "F2":
            ok = ho >= ho0 + 3 and fl >= fl0 - 5
        else:
            ok = None  # F3b: interpretation only
        verdict = {True: "PASS", False: "fail", None: "interpretation only"}[ok]
        log(f"screen {exp}: held-out {ho:.1f} vs {ho0:.1f}, flat {fl:.1f} vs {fl0:.1f}, "
            f"μ0.2 multiply {mu:.1f} vs {mu0:.1f} → {verdict}")
        o.record({"batch": 5, "exp": exp, "held_out": ho, "flat": fl, "mult": mu, "decision": verdict})
        if ok:
            passed.append((ho, exp))
    passed.sort(reverse=True)
    log(f"batch 5 screening passed: {[e for _, e in passed] or 'none'}")

    specs = {
        "F1": ("Isaac-Ant-R2-LowFricOracle-v0", "Isaac-Ant-R2-LowFricOracle-Play-v0", 1000, False),
        "F3a": (E15_TASK, E15_PLAY, 600, True),
        "F2": ("Isaac-Ant-R4-OracleVar-v0", "Isaac-Ant-R4-OracleVar-Play-v0", 1000, False),
    }
    base = {}
    for _, exp in passed:
        task, play, iters, warm = specs[exp]
        # E15 on the fresh seeds (shared across candidates)
        for s in (47, 48, 49):
            if s not in base:
                base[s] = run("E15", E15_TASK, E15_PLAY, f"e15_s{s}", s)
        cand = {}
        for s in (47, 48, 49):
            init = r4.ckpt_of(f"e15_s{s}") if warm else None
            if warm and init is None:
                continue
            cand[s] = run(exp, task, play, f"{exp.lower()}_s{s}", s, iters=iters, init=init)
        seeds = [s for s in (47, 48, 49) if base.get(s) and cand.get(s)]
        if len(seeds) < 3:
            log(f"confirm {exp}: only {len(seeds)} paired seeds → no replacement decision")
            continue
        diffs = [cand[s]["held_out"] - base[s]["held_out"] for s in seeds]
        d = st.mean(diffs)
        half = r4.T_975_DF2 * st.stdev(diffs) / (3 ** 0.5)
        wins = sum(x > 0 for x in diffs)
        fl_ok = mean([cand[s] for s in seeds], "flat") >= mean([base[s] for s in seeds], "flat") - 5
        c1, c2 = d >= 3, wins >= 2
        log(f"confirm {exp} vs E15 (s47–49): held-out diff {d:+.1f} [95% CI {d - half:+.1f}, {d + half:+.1f}], "
            f"wins {wins}/3, flat ok {fl_ok} → cond1 {c1}, cond2 {c2}")
        o.record({"batch": 5, "stage": "confirm", "exp": exp, "diffs": diffs, "wins": wins, "flat_ok": fl_ok})
        if not (c1 and c2 and fl_ok):
            continue
        ck = r4.ckpt_of(f"{exp.lower()}_s47")
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
        ok = lb >= LOCKBOX_E15 - 3
        log(f"LOCKBOX {exp} s47 (reused lockbox, 3rd model): {[round(v, 1) for v in vals]} → mean {lb:.1f} "
            f"(need ≥ {LOCKBOX_E15 - 3:.1f}) → {'REPLACE E15' if ok else 'keep E15'}")
        o.record({"batch": 5, "stage": "lockbox", "exp": exp, "lockbox": vals, "replace": ok})
        if ok:
            os.makedirs(f"experiments/eval/smoke_fb_{exp}", exist_ok=True)
            log(f"NEW SUBMISSION CANDIDATE: {exp} `{ck}` (play task {play}); smoke tests follow")
            with open("experiments/fb_state.json", "w") as f:
                json.dump({"submission": exp, "task": task, "play": play, "ckpt": ck, "iters": iters,
                           "warm_from_e15": warm, "runs": {str(s): f"{exp.lower()}_s{s}" for s in (47, 48, 49)},
                           "lockbox": lb}, f, indent=1)
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
                          "--num_envs", "100", "--checkpoint", ck, *ENT],
                         f"experiments/train_logs/smoke_fb_{exp}_official.log")
            log(f"smoke {exp} official play_one_episode: exit {code}")
            break
    log("## batch 5 done")


if __name__ == "__main__":
    main()
