"""Batch 10 (share-fb, prepared 2026-10-01 11:10, NOT started): train on every evaluation terrain type ("AllMix").

Pre-registration: experiments/EXPERIMENTS_overnight_fb.md, section "배치 10".
Test set = lockbox 1 (4 types) + lockbox 2 (4 types, fixed 2026-10-01 11:00) = "unseen8"; the 5 held-out types become
training terrains here and are reported for information only.
usage (conda env lerobot-arena, from ~/IsaacLab_RS; nothing else using the GPU):
  python experiments/queue_fb_b10.py > experiments/queue_fb_b10.out 2>&1
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
import queue_fb_b5 as b5  # noqa: E402
import queue_fb_b7 as b7  # noqa: E402

R = o.R
ENT = ["agent.algorithm.entropy_coef=0.005"]
TASK, PLAY = "Isaac-Ant-R5-AllMix-v0", "Isaac-Ant-R5-AllMix-Play-v0"
REF_PLAY = "Isaac-Ant-R2-Oracle-Play-v0"  # F3a
LOCK1 = ("lock_rails", "lock_gaps", "lock_pits", "lock_stones")
LOCK2 = ("lock2_cylinders", "lock2_cones", "lock2_tilted_blocks", "lock2_platform")
BASELINE = "logs/rsl_rl/ant/2026-09-17_13-20-27_ant_baseline/model_999.pt"
NO_START_AFTER_HOURS = 6.0  # this batch must not start a run more than 6 h after it began


def log(text):
    o.log(f"[fb] [b10] {text}")


def isaac_busy():
    """Any other Isaac Lab training / playing / evaluation process (e.g. the user watching play.py)."""
    # anchored to the python process itself (14:15 a launch shell containing the script text blocked the queue)
    pattern = r"^\S*/python3? scripts/reinforcement_learning/rsl_rl/(train|train_finetune|play|play_one_episode|evaluate)\.py"
    return subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL).returncode == 0


def wait_gpu():
    while isaac_busy():
        time.sleep(30)


def evaluate(ckpt, play, terrain, out, over):
    for _ in range(2):  # one retry: Isaac Sim occasionally aborts (12:41 "free(): corrupted unsorted chunks")
        if os.path.exists(out):
            break
        wait_gpu()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain, "--num_envs", "100",
               "--checkpoint", ckpt, "--output", out, *over], out.replace(".json", ".log"))
    with open(out) as f:
        return json.load(f)


def unseen8(run, ckpt, play, over):
    vals = {}
    for t in LOCK1 + LOCK2:
        vals[t] = evaluate(ckpt, play, t, f"experiments/eval_unseen8/{run}/{t}.json", over)["return_mean"]
    return vals


def train(run, seed, iters):
    wait_gpu()
    return b7.train(run, TASK, seed, ENT, iters)


def metrics(run, ckpt, play, over):
    for _ in range(2):  # one retry of the sweep if any condition crashed (see evaluate)
        if os.path.exists(f"experiments/eval/{run}/boxes_mid_f0.2.json") and all(
                os.path.exists(f"experiments/eval/{run}/{c}.json") for c in o.HELD_OUT):
            break
        wait_gpu()
        subprocess.run(["bash", f"{R}/eval_sweep.sh", ckpt, f"experiments/eval/{run}", play, *over],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    m = r4.full_metrics(run)
    u = unseen8(run, ckpt, play, over)
    m["unseen8"] = st.mean(u.values())
    m["lock1"] = st.mean(u[t] for t in LOCK1)
    m["lock2"] = st.mean(u[t] for t in LOCK2)
    m["unseen8_detail"] = u
    return m


def fmt(name, m):
    return (f"{name}: unseen8 {m['unseen8']:.1f} (lock1 {m['lock1']:.1f}, lock2 {m['lock2']:.1f}), "
            f"held-out(now trained) {m['held_out']:.1f}, flat {m['flat']:.1f} (speed {m['flat_speed']:.2f})")


def main():
    t0 = time.time()
    # the shared train() helper still carried the overnight guard "no new training after 2026-10-01 08:00";
    # this batch has its own 6 h limit (NO_START_AFTER_HOURS), so move the shared guard to the LMS deadline
    b5.NO_NEW_TRAIN_AFTER = datetime(2026, 10, 6, 20, 0)
    log("## batch 10 start (AllMix: train on every EVAL_TERRAINS type; test = lockbox 1 + 2)")

    # 0) smoke: config + terrain build + 2 training iterations; E0 sanity on lockbox 2 (E0 is not a candidate)
    wait_gpu()
    ok, _ = r4.smoke("b10_allmix", TASK, ENT)
    if not ok:
        log("smoke AllMix FAILED → batch stopped (see experiments/train_logs/smoke_b10_allmix.log)")
        return
    e0 = {t: evaluate(BASELINE, "Isaac-Ant-E0-Play-v0", t, f"experiments/eval_unseen8/E0/{t}.json", [])["return_mean"]
          for t in LOCK2}
    log("E0 on lockbox 2 (sanity, not used for selection): " + ", ".join(f"{k} {v:.1f}" for k, v in e0.items()))

    # reference: F3a s42/43
    ref = [metrics(f"f3a_s{s}", r4.ckpt_of(f"f3a_s{s}"), REF_PLAY, ENT) for s in (42, 43)]
    for s, m in zip((42, 43), ref):
        log(fmt(f"F3a s{s}", m))
    u0, fl0 = b5.mean(ref, "unseen8"), b5.mean(ref, "flat")

    # 1) screening
    variants = {"M16": 1600, "M30": 3000}
    passed = []
    for name, iters in variants.items():
        ms = []
        for s in (42, 43):
            if time.time() - t0 > NO_START_AFTER_HOURS * 3600:
                log(f"{name} s{s} skipped: time limit")
                ms.append(None)
                continue
            run = f"{name.lower()}_s{s}"
            ck = train(run, s, iters)
            ms.append(metrics(run, ck, PLAY, ENT) if ck else None)
            if ms[-1]:
                log(fmt(f"{name} s{s}", ms[-1]))
        if any(m is None for m in ms):
            log(f"screen {name}: incomplete")
            continue
        u, fl = b5.mean(ms, "unseen8"), b5.mean(ms, "flat")
        ok = u >= u0 + 3 and fl >= fl0 - 5
        log(f"screen {name}: unseen8 {u:.1f} vs F3a {u0:.1f}, flat {fl:.1f} vs {fl0:.1f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 10, "exp": name, "unseen8": u, "flat": fl, "decision": "PASS" if ok else "fail"})
        if ok:
            passed.append((u, name, iters))
    passed.sort(reverse=True)
    if not passed:
        log("## batch 10 done: no candidate passed screening → keep F3a")
        return

    # 2) confirmation of the best passer on seeds 47-49 against F3a s47-49 (paired)
    _, name, iters = passed[0]
    base = {s: metrics(f"f3a_s{s}", r4.ckpt_of(f"f3a_s{s}"), REF_PLAY, ENT) for s in (47, 48, 49)}
    cand = {}
    for s in (47, 48, 49):
        if time.time() - t0 > NO_START_AFTER_HOURS * 3600:
            log(f"{name} s{s} skipped: time limit")
            continue
        run = f"{name.lower()}_s{s}"
        ck = train(run, s, iters)
        if ck:
            cand[s] = metrics(run, ck, PLAY, ENT)
            log(fmt(f"{name} s{s}", cand[s]))
    if len(cand) < 3:
        log(f"confirm {name}: only {len(cand)} seeds → keep F3a")
        return
    diffs = [cand[s]["unseen8"] - base[s]["unseen8"] for s in (47, 48, 49)]
    d, wins = st.mean(diffs), sum(x > 0 for x in diffs)
    half = r4.T_975_DF2 * st.stdev(diffs) / 3 ** 0.5
    fl_ok = b5.mean(list(cand.values()), "flat") >= b5.mean(list(base.values()), "flat") - 5
    replace = d >= 3 and wins >= 2 and fl_ok
    log(f"confirm {name} vs F3a (s47–49, unseen8): diff {d:+.1f} [95% CI {d - half:+.1f}, {d + half:+.1f}], "
        f"wins {wins}/3, flat ok {fl_ok} → {'REPLACE F3a' if replace else 'keep F3a'}")
    o.record({"batch": 10, "stage": "confirm", "exp": name, "diffs": diffs, "wins": wins, "flat_ok": fl_ok,
              "replace": replace})
    if replace:
        ck = r4.ckpt_of(f"{name.lower()}_s47")
        os.makedirs(f"experiments/eval/smoke_fb_{name}", exist_ok=True)
        for terrain in ("flat", "boxes_mid"):
            for fric in ("1.0", "0.2"):
                for mode in ("average", "multiply"):
                    n = f"{terrain}_f{fric}_{mode}"
                    code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", PLAY, "--terrain", terrain,
                                  "--friction", fric, "--combine_mode", mode, "--checkpoint", ck, "--output",
                                  f"experiments/eval/smoke_fb_{name}/{n}.json", *ENT],
                                 f"experiments/train_logs/smoke_fb_{name}_{n}.log")
                    log(f"smoke {name} {n}: {'OK' if code == 0 else 'FAILED'}")
        code = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", PLAY, "--headless", "--num_envs", "100",
                      "--checkpoint", ck, *ENT], f"experiments/train_logs/smoke_fb_{name}_official.log")
        log(f"smoke {name} official play_one_episode: exit {code}")
        with open("experiments/fb_state.json") as f:
            state = json.load(f)
        state = {**state, "previous": state["submission"], "submission": name, "task": TASK, "play": PLAY, "ckpt": ck,
                 "description": f"AllMix (every eval terrain type) {iters} it", "runs": {str(s): f"{name.lower()}_s{s}"
                                                                                          for s in (42, 43, 47, 48, 49)}}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(state, f, indent=1)
        log(f"NEW SUBMISSION: {name} `{ck}`")
    log("## batch 10 done")


if __name__ == "__main__":
    main()
