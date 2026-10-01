"""Round 2 (2026-09-30, must finish before 09:00), started automatically after experiments/overnight.py.

Improvement experiments (adoption rule as in round 1, vs the current winner, seed 42):
  E11 ④ fall-suppression fine-tune: warm start from the winner checkpoint, std reset to 0.4, 600 it,
         training-only rewards (fall penalty, ground clearance, roll/pitch rate)           [warm-started run]
  E12 ⑤ flat first, rough later: warm start from the flat baseline model_999, winner config, 1000 it [warm-started]
  E13 ⑦ low friction: ground combined by multiply + robot friction 0.05-1.2 (effective friction down to 0.05)
  E14 ① feet air-time reward (contact sensors on the feet, training only)
Verification experiments (not adopted, recorded as is):
  V1 ② curriculum re-test: entropy-on winner config without curriculum
  V2 ③ long training: entropy-on winner config, 3000 it (compare std / lr curves with E5)
  V3 ⑥ oracle: E4 settings trained on boxes ±10 cm only, compared on boxes ±10 cm
Then: seeds 43/44 of a round-2 winner (3-seed replacement rule vs E4), Play smoke test, lockbox (once), videos.
A deadline guard skips optional steps (V2, then V3, then extra videos) when they would end after 08:30.
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
import overnight as o  # noqa: E402  (shared helpers: log, record, run, metrics, train_stats, fmt, adopt)

R = o.R
R2_FLAGS = ("Flat", "Hist", "Push", "Safe", "LowFric", "Air", "NoCurr", "Oracle")
DEADLINE = datetime(2026, 9, 30, 8, 30)
BASELINE = "logs/rsl_rl/ant/2026-09-17_13-20-27_ant_baseline/model_999.pt"
# measured on this laptop (minutes): 1000 it training incl. start-up, 15-condition eval sweep
EST = {"train1000": 13.0, "eval": 4.5, "train600": 9.0, "train3000": 38.0, "smoke": 3.0, "lockbox": 3.0, "video": 1.2}


def minutes_left():
    return (DEADLINE - datetime.now()).total_seconds() / 60.0


def fits(step_min, reserve_min):
    """True if a step of ``step_min`` still leaves ``reserve_min`` for the mandatory tail before the deadline."""
    return minutes_left() >= step_min + reserve_min


def tasks2(flags):
    suffix = "".join(f for f in R2_FLAGS if f in flags)
    base = f"Isaac-Ant-R2-{suffix}" if suffix else "Isaac-Ant-Rough-E4"
    return f"{base}-v0", f"{base}-Play-v0"


def train2(flags, overrides, run_name, seed, iters=1000, init=None, reset_std=None):
    task, _ = tasks2(flags)
    script = "train_finetune.py" if init else "train.py"
    cmd = ["./isaaclab.sh", "-p", f"{R}/{script}", "--task", task, "--headless", "--seed", str(seed),
           "--run_name", run_name, "--max_iterations", str(iters)]
    if init:
        cmd += ["--init_checkpoint", init]
    if reset_std is not None:
        cmd += ["--reset_std", str(reset_std)]
    code = o.run(cmd + overrides, f"experiments/train_logs/{run_name}.log")
    dirs = sorted(glob.glob(f"logs/rsl_rl/ant/*_{run_name}"))
    last = f"model_{iters - 1}.pt"
    ckpt = f"{dirs[-1]}/{last}" if dirs else None
    if code != 0 or not ckpt or not os.path.exists(ckpt):
        return None, None
    return dirs[-1], ckpt


def eval2(flags, overrides, ckpt, run_name):
    _, play = tasks2(flags)
    out = f"experiments/eval/{run_name}"
    subprocess.run(["bash", f"{R}/eval_sweep.sh", ckpt, out, play, *overrides],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


def experiment(exp_id, kind, hypothesis, flags, overrides, iters=1000, init=None, reset_std=None, seed=42):
    run_name = f"{exp_id.lower()}_s{seed}"
    note = f" | warm start from `{init}` (std reset {reset_std})" if init else ""
    o.log(f"**{exp_id} start** ({kind}) — {hypothesis} | task {tasks2(flags)[0]} | overrides {overrides or '-'} | "
          f"{iters} it{note}")
    run_dir, ckpt = train2(flags, overrides, run_name, seed, iters, init, reset_std)
    if ckpt is None:
        o.log(f"{exp_id} FAILED during training (see experiments/train_logs/{run_name}.log)")
        o.record({"exp": exp_id, "status": "train_failed"})
        return None
    ts = o.train_stats(run_dir)
    m = o.metrics(eval2(flags, overrides, ckpt, run_name))
    entry = {"exp": exp_id, "kind": kind, "run_dir": run_dir, "ckpt": ckpt, "flags": list(flags),
             "overrides": overrides, "metrics": m, "train": ts, "warm_start": init, "iters": iters, "seed": seed}
    o.record(entry)
    o.log(f"{exp_id} result: {o.fmt(m)} | final std {ts['std']:.3f}, lr {ts['lr']:.2e} | run `{run_dir}`")
    return entry


def smoke_train(name, flags, overrides, init=None):
    task, _ = tasks2(flags)
    script = "train_finetune.py" if init else "train.py"
    cmd = ["./isaaclab.sh", "-p", f"{R}/{script}", "--task", task, "--headless", "--num_envs", "64",
           "--max_iterations", "2", "--run_name", f"smoke_{name}"]
    if init:
        cmd += ["--init_checkpoint", init, "--reset_std", "0.4"]
    code = o.run(cmd + overrides, f"experiments/train_logs/smoke_{name}.log")
    ok = code == 0 and bool(glob.glob(f"logs/rsl_rl/ant/*_smoke_{name}/model_1.pt"))
    for d in glob.glob(f"logs/rsl_rl/ant/*_smoke_{name}"):
        os.makedirs("logs/rsl_rl/_smoke", exist_ok=True)
        os.rename(d, f"logs/rsl_rl/_smoke/{os.path.basename(d)}")
    o.log(f"smoke {name} ({task}{', warm start' if init else ''}): {'OK' if ok else 'FAILED'}")
    return ok


def with_entropy(overrides):
    return overrides if any("entropy_coef" in x for x in overrides) else overrides + ["agent.algorithm.entropy_coef=0.005"]


def lockbox(name, ckpt, flags, overrides, play=None):
    play = play or tasks2(flags)[1]
    subprocess.run(["bash", "experiments/lockbox_eval.sh", name, ckpt, play, *overrides],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = {}
    for t in ("lock_rails", "lock_gaps", "lock_pits", "lock_stones"):
        path = f"experiments/eval_lockbox/{name}/{t}.json"
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            res[t] = (round(r["return_mean"], 1), round(100 * r["fall_rate"]), round(r["distance_x_mean"], 1))
    mean = st.mean(v[0] for v in res.values()) if res else float("nan")
    o.log(f"LOCKBOX {name}: {res} → mean return {mean:.1f}")
    o.record({"lockbox": name, "results": res, "mean": mean})
    return res


def video(name, ckpt, flags, overrides, terrain):
    _, play = tasks2(flags)
    out = f"experiments/videos/{name}_{terrain}.json"
    code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain, "--num_envs", "16",
                  "--video", "--checkpoint", ckpt, "--output", out, *overrides],
                 f"experiments/videos/{name}_{terrain}.log")
    o.log(f"video {name}_{terrain}: {'OK' if code == 0 else 'FAILED'}")


def load_round1():
    entries, adopted, final_decision = {}, [], None
    with open(o.LOG_JSONL) as f:
        for line in f:
            e = json.loads(line)
            if "metrics" in e and "exp" in e:
                entries[e["exp"]] = e
            if e.get("decision") == "adopted":
                adopted.append(e["exp"])
            if e.get("decision") in ("final_replace", "final_keep_E4"):
                final_decision = e
    return entries, adopted, final_decision


def main():
    while "## overnight run done" not in open(o.LOG_MD).read():
        time.sleep(30)
    if not os.environ.get("R2_RESTART"):
        o.log("## round 2 start")
    e4 = {"exp": "E4", "flags": [], "overrides": [], "metrics": o.metrics("experiments/eval/e4_relheight_s42"),
          "ckpt": sorted(glob.glob("logs/rsl_rl/ant/*_e4_relheight_s42/model_999.pt"))[0]}
    entries, adopted, final1 = load_round1()
    winner = entries[adopted[-1]] if adopted else e4
    winner = {**winner, "flags": list(winner["flags"])}
    final = winner if (final1 and final1["decision"] == "final_replace") else e4
    o.log(f"round-1 screening winner: {winner['exp']} (flags {winner['flags']}, overrides {winner['overrides']}); "
          f"round-1 submission candidate: {final['exp']}")

    if os.environ.get("R2_RESTART"):
        # restart after fixing train_finetune.py: the other smoke tests passed and E0's single lockbox run is done
        while subprocess.run(["pgrep", "-f", "lockbox_eval.sh E0"], stdout=subprocess.DEVNULL).returncode == 0:
            time.sleep(10)
        o.log("round 2 restarted after fixing train_finetune.py (load_state_dict return value); "
              "LowFric+Air and Oracle smoke tests had passed, E0 lockbox finished in the first attempt")
        ok_safe = smoke_train("r2_safe_warm_retry", winner["flags"] + ["Safe"], winner["overrides"], init=winner["ckpt"])
        ok_lowfric_air, ok_oracle = True, True
        res = {}
        for t in ("lock_rails", "lock_gaps", "lock_pits", "lock_stones"):
            path = f"experiments/eval_lockbox/E0/{t}.json"
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                res[t] = (round(r["return_mean"], 1), round(100 * r["fall_rate"]), round(r["distance_x_mean"], 1))
        o.log(f"LOCKBOX E0: {res}")
        o.record({"lockbox": "E0", "results": res})
    else:
        # smoke tests of the new code paths (GPU is free now)
        ok_safe = smoke_train("r2_safe_warm", winner["flags"] + ["Safe"], winner["overrides"], init=winner["ckpt"])
        ok_lowfric_air = smoke_train("r2_lowfric_air", winner["flags"] + ["LowFric", "Air"], winner["overrides"])
        ok_oracle = smoke_train("r2_oracle_nocurr", ["Oracle"], [])
        # E0 on the lockbox: E0 is not a selection target; this also is E0's single lockbox evaluation
        lockbox("E0", BASELINE, [], [], play="Isaac-Ant-E0-Play-v0")

    tail = EST["smoke"] * 3 + EST["lockbox"] * 2 + EST["video"] * 6 + 2 * (EST["train1000"] + EST["eval"])
    improvements = []
    if ok_safe:
        improvements.append(("E11", "④ fall suppression: fall penalty −300, clearance < 0.45 m ×−5, roll/pitch rate ×−0.05 "
                             "(training only), warm start + std reset 0.4", ["Safe"], [], 600, "winner", 0.4))
    improvements.append(("E12", "⑤ flat first, rough later: warm start from the flat baseline (std reset 0.5)",
                         [], [], 1000, "baseline", 0.5))
    if ok_lowfric_air:
        improvements.append(("E13", "⑦ low friction: ground multiply + robot friction 0.05–1.2 → effective 0.05–1.2",
                             ["LowFric"], [], 1000, None, None))
        improvements.append(("E14", "① feet air-time reward ×2.0 (threshold 0.1 s, training only)",
                             ["Air"], [], 1000, None, None))
    round2_adopted = False
    for exp_id, hyp, add_flags, add_over, iters, init_kind, reset_std in improvements:
        est = (EST["train600"] if iters == 600 else EST["train1000"]) + EST["eval"]
        if not fits(est, tail):
            o.log(f"{exp_id} skipped: would end after the deadline guard ({minutes_left():.0f} min left)")
            continue
        flags = [f for f in R2_FLAGS if f in set(winner["flags"]) | set(add_flags)]
        overrides = winner["overrides"] + add_over
        init = {"winner": winner["ckpt"], "baseline": BASELINE, None: None}[init_kind]
        if init_kind == "baseline":
            # the flat baseline has 60-dim observations and no normalizer: drop incompatible winner parts
            dropped = [f for f in flags if f == "Hist"] + [x for x in overrides if "obs_normalization" in x]
            flags = [f for f in flags if f != "Hist"]
            overrides = [x for x in overrides if "obs_normalization" not in x]
            if dropped:
                o.log(f"E12: dropped {dropped} from the winner config to stay weight-compatible with the baseline")
        entry = experiment(exp_id, "improvement", hyp, flags, overrides, iters, init, reset_std)
        if entry is None:
            continue
        entry["base_exp"] = winner["exp"]
        ok, why = o.adopt(entry, winner)
        o.log(f"{exp_id} {'ADOPTED' if ok else 'not adopted'} — {why}")
        o.record({"exp": exp_id, "decision": "adopted" if ok else "rejected", "why": why})
        if ok:
            winner, round2_adopted = entry, True

    ent_over = with_entropy(winner["overrides"])
    verifications = [
        ("V1", "② curriculum re-test with exploration on: entropy-on winner config without curriculum",
         [f for f in R2_FLAGS if f in set(winner["flags"]) | {"NoCurr"}], ent_over, 1000, EST["train1000"]),
        ("V2", "③ long training with exploration on: entropy-on winner config, 3000 it (vs E5 std/lr collapse)",
         list(winner["flags"]), ent_over, 3000, EST["train3000"]),
        ("V3", "⑥ oracle: E4 settings trained on boxes ±10 cm only", ["Oracle"], [], 1000, EST["train1000"]),
    ]
    if not ok_oracle:
        verifications = [v for v in verifications if v[0] != "V3"]
    seeds_needed = 2 * (EST["train1000"] + EST["eval"]) if round2_adopted else 0
    tail2 = EST["smoke"] * 3 + EST["lockbox"] * 2 + EST["video"] * 6 + seeds_needed
    # guard order: drop V2 first, then V3
    for vid, hyp, flags, overrides, iters, est_train in sorted(verifications, key=lambda v: {"V1": 0, "V3": 1, "V2": 2}[v[0]]):
        if not fits(est_train + EST["eval"], tail2):
            o.log(f"{vid} skipped: would end after the deadline guard ({minutes_left():.0f} min left)")
            continue
        entry = experiment(vid, "verification", hyp, flags, overrides, iters)
        if entry and vid == "V3":
            o.log(f"V3 oracle boxes±10 {entry['metrics']['boxes_mid']:.1f} vs winner {winner['metrics']['boxes_mid']:.1f} "
                  f"→ winner reaches {100 * winner['metrics']['boxes_mid'] / max(entry['metrics']['boxes_mid'], 1e-6):.0f}% "
                  f"of the oracle")

    if round2_adopted:
        e4_ho = [o.metrics(f"experiments/eval/e4_relheight_s{s}")["held_out"] for s in (42, 43, 44)]
        new_ho = [winner["metrics"]["held_out"]]
        for seed in (43, 44):
            init = None
            if winner["exp"] == "E12":
                init = BASELINE  # only one flat baseline exists
            elif winner["exp"] == "E11":
                # warm start from the same-seed checkpoint of the model E11 was built on, if it exists
                base = winner.get("base_exp", "E4")
                pattern = "e4_relheight" if base == "E4" else base.lower()
                same_seed = sorted(glob.glob(f"logs/rsl_rl/ant/*_{pattern}_s{seed}/model_*.pt"),
                                   key=lambda p: int(p.rsplit("_", 1)[1][:-3]))
                init = same_seed[-1] if same_seed else winner.get("warm_start")
                o.log(f"E11 seed {seed}: warm start from {init}")
            e = experiment(winner["exp"], "seed repeat", "reproducibility", winner["flags"], winner["overrides"],
                           winner.get("iters", 1000), init, {"E11": 0.4, "E12": 0.5}.get(winner["exp"]), seed=seed)
            if e:
                new_ho.append(e["metrics"]["held_out"])
        e4_mean = st.mean(e4_ho)
        replace = len(new_ho) == 3 and st.mean(new_ho) >= e4_mean + 3.0 and min(new_ho) > e4_mean
        o.log(f"3-seed held-out: {winner['exp']} {[round(v, 1) for v in new_ho]} (mean {st.mean(new_ho):.1f}) vs E4 "
              f"{[round(v, 1) for v in e4_ho]} (mean {e4_mean:.1f}) → "
              f"{'REPLACE E4 as submission candidate' if replace else 'keep previous submission candidate'}")
        o.record({"exp": winner["exp"], "decision": "final_replace" if replace else "final_keep",
                  "new_held_out": new_ho, "e4_held_out": e4_ho})
        if replace:
            final = winner

    o.log(f"submission candidate: {final['exp']}")
    if final["exp"] != "E4" and (final1 is None or final["exp"] != final1.get("exp")):
        _, play = tasks2(final["flags"])
        out = f"experiments/eval/smoke_{final['exp'].lower()}"
        os.makedirs(out, exist_ok=True)
        for terrain, fric in (("flat", 1.0), ("boxes_mid", 1.0), ("flat", 0.2), ("boxes_mid", 0.2)):
            for mode in ("average", "multiply"):
                name = f"{terrain}_f{fric}_{mode}"
                code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain,
                              "--friction", str(fric), "--combine_mode", mode, "--checkpoint", final["ckpt"],
                              "--output", f"{out}/{name}.json", *final["overrides"]], f"{out}/{name}.log")
                o.log(f"smoke {final['exp']} {name}: {'OK' if code == 0 else 'FAILED'}")
    else:
        o.log("smoke test of the submission candidate already done in round 1 (same model)")

    lockbox("E4", e4["ckpt"], [], [])
    if final["exp"] != "E4":
        lockbox(f"final_{final['exp']}", final["ckpt"], final["flags"], final["overrides"])

    for terrain in ("flat", "boxes_mid", "stairs"):
        if final["exp"] != "E4" and fits(2 * EST["video"], 0):
            video(f"final_{final['exp']}", final["ckpt"], final["flags"], final["overrides"], terrain)
    o.log("## round 2 done")


if __name__ == "__main__":
    main()
