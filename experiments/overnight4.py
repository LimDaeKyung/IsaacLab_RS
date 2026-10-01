"""Round 4 phases B(smoke)–C–D–E, started after experiments/queue_r4A.py. Rules: EXPERIMENTS.md §8 (pre-registered).

usage (conda env lerobot-arena, from ~/IsaacLab_RS):
  nohup setsid python experiments/overnight4.py > experiments/overnight4.out 2>&1 < /dev/null &
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
ENT = ["agent.algorithm.entropy_coef=0.005"]
NEW_EXP_DEADLINE = datetime(2026, 10, 1, 7, 0)
BASELINE = "logs/rsl_rl/ant/2026-09-17_13-20-27_ant_baseline/model_999.pt"
T_975_DF2 = 4.303
EST_RUN = 16.5  # minutes per 1000-it run incl. evaluation (measured)
EST_FT = 12.0  # minutes per 600-it warm-started run incl. evaluation


def ckpt_of(run_name):
    c = sorted(glob.glob(f"logs/rsl_rl/ant/*_{run_name}/model_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
    return c[-1] if c else None


def ho_fall(eval_dir):
    vals = []
    for c in o.HELD_OUT:
        with open(f"{eval_dir}/{c}.json") as f:
            vals.append(json.load(f)["fall_rate"])
    return st.mean(vals)


def full_metrics(run_name):
    d = f"experiments/eval/{run_name}"
    m = o.metrics(d)
    m["ho_fall"] = ho_fall(d)
    with open(f"{d}/flat_f1.0.json") as f:
        m["flat_speed"] = json.load(f)["speed_x_mean"]
    return m


def train_eval(label, task, play, run_name, seed, overrides, iters=1000, init=None):
    if os.path.exists(f"experiments/eval/{run_name}/boxes_mid_f0.2.json"):
        o.log(f"{label} seed {seed}: already evaluated (`{run_name}`), reusing")
        return full_metrics(run_name)
    script = "train_finetune.py" if init else "train.py"
    o.log(f"**{label} seed {seed} start** | task {task} | overrides {overrides or '-'} | {iters} it"
          + (f" | warm start `{init}`" if init else ""))
    cmd = ["./isaaclab.sh", "-p", f"{R}/{script}", "--task", task, "--headless", "--seed", str(seed),
           "--run_name", run_name, "--max_iterations", str(iters)] + (["--init_checkpoint", init] if init else [])
    code = o.run(cmd + overrides, f"experiments/train_logs/{run_name}.log")
    ckpt = ckpt_of(run_name)
    if code != 0 or not ckpt or not ckpt.endswith(f"model_{iters - 1}.pt"):
        o.log(f"{label} seed {seed} FAILED during training (see experiments/train_logs/{run_name}.log)")
        o.record({"exp": label, "seed": seed, "status": "train_failed"})
        return None
    subprocess.run(["bash", f"{R}/eval_sweep.sh", ckpt, f"experiments/eval/{run_name}", play, *overrides],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    m = full_metrics(run_name)
    ts = o.train_stats(os.path.dirname(ckpt))
    o.record({"exp": label, "seed": seed, "run": run_name, "ckpt": ckpt, "metrics": m, "train": ts})
    o.log(f"{label} seed {seed} result: {o.fmt(m)}, held-out fall {100 * m['ho_fall']:.0f}%, flat speed "
          f"{m['flat_speed']:.2f} | std {ts['std']:.3f}, lr {ts['lr']:.2e} | `{os.path.dirname(ckpt)}`")
    return m


def smoke(name, task, overrides, init=None, num_envs=64, iters=2):
    script = "train_finetune.py" if init else "train.py"
    cmd = ["./isaaclab.sh", "-p", f"{R}/{script}", "--task", task, "--headless", "--num_envs", str(num_envs),
           "--max_iterations", str(iters), "--run_name", f"smoke_{name}"] + (["--init_checkpoint", init] if init else [])
    code = o.run(cmd + overrides, f"experiments/train_logs/smoke_{name}.log")
    dirs = glob.glob(f"logs/rsl_rl/ant/*_smoke_{name}")
    ok = code == 0 and bool(dirs) and os.path.exists(f"{dirs[0]}/model_{iters - 1}.pt")
    o.log(f"smoke {name} ({task}): {'OK' if ok else 'FAILED'}")
    return ok, (dirs[0] if dirs else None)


def fall_penalty_diagnostic(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    fp = ea.Scalars("Episode_Reward/fall_penalty")
    term = ea.Scalars("Episode_Termination/torso_height")
    rows = []
    for a, b in zip(fp[5:], term[5:]):  # skip the first iterations while the weight is being set
        expected = b.value * (-30.0 / 60.0) / 16.0  # falls per reset x weight x dt, normalised by episode_length_s
        rows.append((a.value, expected))
    ratio = st.mean(r[0] for r in rows) / st.mean(r[1] for r in rows) if rows and st.mean(r[1] for r in rows) else float("nan")
    o.log(f"E17 diagnostic (FallConst, weight −30 from the start): logged Episode_Reward/fall_penalty mean "
          f"{st.mean(r[0] for r in rows):.5f} vs expected (fall fraction × −30 × dt / 16 s) {st.mean(r[1] for r in rows):.5f} "
          f"→ ratio {ratio:.3f} (1.0 = per-episode sum equals falls × weight × dt)")
    o.record({"diagnostic": "fall_penalty_sum", "ratio": ratio, "rows": rows[:20]})
    return ratio


def minutes_to(deadline):
    return (deadline - datetime.now()).total_seconds() / 60.0


def main():
    while "## round 4 phase A done" not in open(o.LOG_MD).read():
        time.sleep(30)
    o.log("## round 4 phase B (smoke) start")
    e6 = {s: ckpt_of(f"e6_s{s}") for s in (42, 43, 44, 45, 46)}
    ok = {}
    ok["Air"], _ = smoke("r4_air", "Isaac-Ant-R4-Air-v0", ENT)
    ok["FallRamp"], _ = smoke("r4_fallramp", "Isaac-Ant-R4-FallRamp-v0", ENT, init=e6[42])
    okc, diag_dir = smoke("r4_fallconst_diag", "Isaac-Ant-R4-FallConst-v0", ENT, init=e6[42], num_envs=512, iters=40)
    if okc:
        fall_penalty_diagnostic(diag_dir)
    ok["Asym"], asym_dir = smoke("r4_asym", "Isaac-Ant-R4-Asym-v0", ENT)
    if ok["Asym"]:
        # the critic group must compute on a plane and Play must use the actor only
        ck = f"{asym_dir}/model_1.pt"
        c1 = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", "Isaac-Ant-R4-Asym-Play-v0", "--terrain", "flat",
                    "--num_envs", "16", "--checkpoint", ck, "--output", "logs/eval/check/smoke_r4_asym_flat.json"],
                   "logs/eval/check/smoke_r4_asym_flat.log")
        c2 = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", "Isaac-Ant-R4-Asym-Play-v0", "--headless",
                    "--num_envs", "16", "--checkpoint", ck], "logs/eval/check/smoke_r4_asym_play_one_episode.log")
        ok["Asym"] = c1 == 0 and c2 == 0
        o.log(f"smoke Asym on plane (evaluate.py exit {c1}) and play_one_episode.py (exit {c2}) → "
              f"{'OK' if ok['Asym'] else 'FAILED'}")
    ok["Push"], _ = smoke("r4_push", "Isaac-Ant-R4-Push-v0", ENT)
    ok["EyesNoise"], _ = smoke("r4_eyesnoise", "Isaac-Ant-R4-EyesNoise-v0", ENT)
    for d in glob.glob("logs/rsl_rl/ant/*_smoke_r4_*"):
        os.rename(d, f"logs/rsl_rl/_smoke/{os.path.basename(d)}")

    # ---- C: screening, seeds 42 and 43 ----
    o.log("## round 4 phase C (screening) start")
    ideas = [
        ("E15", "Isaac-Ant-R2-Oracle-v0", "Isaac-Ant-R2-Oracle-Play-v0", ENT, 1000, None, "v3", True),
        ("E16", "Isaac-Ant-R4-Air-v0", "Isaac-Ant-R4-Air-Play-v0", ENT, 1000, None, "e6", ok["Air"]),
        ("E17c", "Isaac-Ant-Rough-E4-v0", "Isaac-Ant-Rough-E4-Play-v0", ENT, 600, "e6", None, ok["FallRamp"]),
        ("E17", "Isaac-Ant-R4-FallRamp-v0", "Isaac-Ant-R4-FallRamp-Play-v0", ENT, 600, "e6", "e17c", ok["FallRamp"]),
        ("E18", "Isaac-Ant-R4-Asym-v0", "Isaac-Ant-R4-Asym-Play-v0", ENT, 1000, None, "e6", ok["Asym"]),
        ("E19", "Isaac-Ant-R4-Push-v0", "Isaac-Ant-R4-Push-Play-v0", ENT, 1000, None, "e6", ok["Push"]),
        ("E20", "Isaac-Ant-R4-EyesNoise-v0", "Isaac-Ant-R4-EyesNoise-Play-v0", ENT, 1000, None, "r3_c1eyesflatlanes",
         ok["EyesNoise"]),
    ]
    # time guard: estimated tail after C = D (3 candidates x 3 seeds) + lockbox + smoke + videos
    tail = 9 * EST_RUN + 20
    skip_order = ["E20", "E19", "E18"]
    screen = {}
    for exp, task, play, over, iters, init_kind, ref, enabled in ideas:
        if not enabled:
            o.log(f"{exp} skipped: its smoke test failed")
            continue
        remaining = [i[0] for i in ideas if i[0] not in screen and i[0] != exp]
        need = 2 * (EST_FT if iters == 600 else EST_RUN)
        if exp in skip_order and minutes_to(NEW_EXP_DEADLINE) < need + tail:
            o.log(f"{exp} skipped: time guard ({minutes_to(NEW_EXP_DEADLINE):.0f} min to 07:00)")
            continue
        screen[exp] = {}
        for seed in (42, 43):
            init = e6[seed] if init_kind == "e6" else None
            screen[exp][seed] = train_eval(exp, task, play, f"{exp.lower()}_s{seed}", seed, over, iters, init)
        screen[exp]["meta"] = (task, play, over, iters, init_kind, ref)

    def ref_metrics(ref, seed):
        name = {"v3": f"v3_s{seed}", "e6": f"e6_s{seed}", "e17c": f"e17c_s{seed}",
                "r3_c1eyesflatlanes": f"r3_c1eyesflatlanes_s{seed}"}[ref]
        return full_metrics(name)

    passed = []
    for exp, runs in screen.items():
        ref = runs["meta"][5]
        if ref is None:
            continue
        if not all(runs.get(s) for s in (42, 43)):
            o.log(f"{exp}: incomplete screening runs → not passed")
            continue
        c_ho = st.mean(runs[s]["held_out"] for s in (42, 43))
        c_fall = st.mean(runs[s]["ho_fall"] for s in (42, 43))
        c_flat = st.mean(runs[s]["flat"] for s in (42, 43))
        r = [ref_metrics(ref, s) for s in (42, 43)]
        r_ho, r_fall, r_flat = st.mean(x["held_out"] for x in r), st.mean(x["ho_fall"] for x in r), st.mean(x["flat"] for x in r)
        cond_a = c_ho >= r_ho + 3 or (c_ho >= r_ho - 1 and c_fall <= r_fall - 0.10)
        cond_b = c_flat >= r_flat - 5
        verdict = cond_a and cond_b
        o.log(f"C {exp} vs {ref} (2-seed): held-out {c_ho:.1f} vs {r_ho:.1f}, held-out fall {100 * c_fall:.0f}% vs "
              f"{100 * r_fall:.0f}%, flat {c_flat:.1f} vs {r_flat:.1f} → {'PASS' if verdict else 'fail'}")
        o.record({"screen": exp, "ref": ref, "c": [c_ho, c_fall, c_flat], "r": [r_ho, r_fall, r_flat], "pass": verdict})
        if verdict:
            passed.append((c_ho, exp))
    passed = [e for _, e in sorted(passed, reverse=True)[:3]]
    o.log(f"C passed (top 3 by held-out): {passed or 'none'}")

    # ---- D: seeds 44-46 for passed candidates ----
    for exp in passed:
        task, play, over, iters, init_kind, _ = screen[exp]["meta"]
        for seed in (44, 45, 46):
            init = e6[seed] if init_kind == "e6" else None
            screen[exp][seed] = train_eval(exp, task, play, f"{exp.lower()}_s{seed}", seed, over, iters, init)

    # ---- E: final selection on paired seeds 44-46 ----
    pool = {"E4": "e4_relheight", "E6": "e6", "V3": "v3"}
    pool.update({exp: exp.lower() for exp in passed})
    vals = {}
    for name, prefix in pool.items():
        vals[name] = {}
        for seed in (42, 43, 44, 45, 46):
            if os.path.exists(f"experiments/eval/{prefix}_s{seed}/boxes_mid_f0.2.json"):
                vals[name][seed] = full_metrics(f"{prefix}_s{seed}")
    e4 = vals["E4"]
    lines, eligible = [], []
    for name, v in vals.items():
        seeds_all = " ".join(f"s{s}:{v[s]['held_out']:.1f}/{v[s]['flat']:.1f}" for s in sorted(v))
        if not all(s in v for s in (44, 45, 46)):
            lines.append(f"{name}: incomplete seeds 44-46 ({seeds_all})")
            continue
        diffs = [v[s]["held_out"] - e4[s]["held_out"] for s in (44, 45, 46)]
        md = st.mean(diffs)
        sd = st.stdev(diffs)
        ci = (md - T_975_DF2 * sd / 3 ** 0.5, md + T_975_DF2 * sd / 3 ** 0.5)
        ho = st.mean(v[s]["held_out"] for s in (44, 45, 46))
        e4ho = st.mean(e4[s]["held_out"] for s in (44, 45, 46))
        flat = st.mean(v[s]["flat"] for s in (44, 45, 46))
        e4flat = st.mean(e4[s]["flat"] for s in (44, 45, 46))
        wins = sum(d > 0 for d in diffs)
        c1, c2, c3 = ho >= e4ho + 3, wins >= 2, flat >= e4flat - 5
        lines.append(f"{name}: held-out(44-46) {ho:.1f} vs E4 {e4ho:.1f}, paired diff mean {md:+.1f} "
                     f"[95% CI {ci[0]:+.1f}, {ci[1]:+.1f}], wins {wins}/3, flat {flat:.1f} vs {e4flat:.1f} → "
                     f"c1 {c1} c2 {c2} c3 {c3} | all seeds held-out/flat: {seeds_all}")
        o.record({"final_compare": name, "diffs": diffs, "ci": ci, "ho": ho, "flat": flat, "conds": [c1, c2, c3]})
        if name != "E4" and c1 and c2 and c3:
            eligible.append((ho, name))
    for line in lines:
        o.log("E " + line)
    final = "E4"
    if eligible:
        best = max(eligible)[1]
        prefix = pool[best]
        play = {"E6": "Isaac-Ant-Rough-E4-Play-v0", "V3": "Isaac-Ant-R2-Oracle-Play-v0"}.get(best) or screen[best]["meta"][1]
        ck = ckpt_of(f"{prefix}_s44")
        subprocess.run(["bash", "experiments/lockbox_eval.sh", best, ck, play, *ENT], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        res = {}
        for t in ("lock_rails", "lock_gaps", "lock_pits", "lock_stones"):
            p = f"experiments/eval_lockbox/{best}/{t}.json"
            if os.path.exists(p):
                with open(p) as f:
                    r = json.load(f)
                res[t] = (round(r["return_mean"], 1), round(100 * r["fall_rate"]), round(r["distance_x_mean"], 1))
        lb = st.mean(x[0] for x in res.values()) if len(res) == 4 else float("nan")
        o.log(f"LOCKBOX {best} (seed 44 checkpoint): {res} → mean {lb:.1f} (need ≥ 43.9)")
        o.record({"lockbox": best, "results": res, "mean": lb})
        final = best if lb >= 43.9 else "E4"
        final_ckpt, final_play = ck, play
    o.log(f"round 4 final decision: submission = {final}")
    o.record({"round4_final": final})

    if final != "E4":
        out = f"experiments/eval/smoke_r4_{final.lower()}"
        os.makedirs(out, exist_ok=True)
        for terrain, fric in (("flat", 1.0), ("boxes_mid", 1.0), ("flat", 0.2), ("boxes_mid", 0.2)):
            for mode in ("average", "multiply"):
                name = f"{terrain}_f{fric}_{mode}"
                code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", final_play, "--terrain", terrain,
                              "--friction", str(fric), "--combine_mode", mode, "--checkpoint", final_ckpt, "--output",
                              f"{out}/{name}.json", *ENT], f"{out}/{name}.log")
                o.log(f"smoke {final} {name}: {'OK' if code == 0 else 'FAILED'}")
        code = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", final_play, "--headless", "--seed", "24",
                      "--num_envs", "100", "--checkpoint", final_ckpt, *ENT], f"{out}/official_play_one_episode.log")
        with open(f"{out}/official_play_one_episode.log") as f:
            res = [line.strip() for line in f if "[RESULT]" in line]
        o.log(f"smoke {final} official play_one_episode.py: exit {code}, {res}")
        for terrain in ("flat", "boxes_mid"):
            code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", final_play, "--terrain", terrain,
                          "--num_envs", "16", "--video", "--checkpoint", final_ckpt, "--output",
                          f"experiments/videos/R4_{final}_{terrain}.json", *ENT], f"experiments/videos/R4_{final}_{terrain}.log")
            o.log(f"video R4_{final}_{terrain}: {'OK' if code == 0 else 'FAILED'}")
    else:
        o.log("final stays E4: smoke test (round 1) and E0/E4 flat·boxes videos already exist")
    o.log("## round 4 done")


if __name__ == "__main__":
    main()
