"""Batch 11 (share-fb, pre-registered 2026-10-01 15:25): raise the unseen-terrain evaluation score.

Primary metric "unseen14" = mean return over 14 conditions never used for training the F3a lineage:
held-out 5 (eval sweep) + lockbox 1 (4) + lockbox 2 (4) + boxes ±10 μ0.2 with multiply friction combine (1).
Final untouched check: lockbox 3 (4 harder variants, fixed 15:20), evaluated once at the very end.

Order: 0) symmetry check, LSTM/symmetry smoke  1) B1 energy (H2, 5 existing seeds)  2) A1 bigger net
3) C1 low friction (H1, 5 seeds)  4) A3 symmetry  5) A4 LSTM  6) final pick + lockbox 3 + replacement.
Runs after experiments/queue_fb_b10.py exits.
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
import overnight4 as r4  # noqa: E402
import queue_fb_b5 as b5  # noqa: E402
import queue_fb_b7 as b7  # noqa: E402
import queue_fb_b10 as b10  # noqa: E402

R = o.R
ENT = ["agent.algorithm.entropy_coef=0.005"]
NET = ["agent.policy.actor_hidden_dims=[512,256,128]", "agent.policy.critic_hidden_dims=[512,256,128]"]
ORACLE, ORACLE_PLAY = "Isaac-Ant-R2-Oracle-v0", "Isaac-Ant-R2-Oracle-Play-v0"
LF, LF_PLAY = "Isaac-Ant-R2-LowFricOracle-v0", "Isaac-Ant-R2-LowFricOracle-Play-v0"
SYM, SYM_PLAY = "Isaac-Ant-R6-OracleSym-v0", "Isaac-Ant-R6-OracleSym-Play-v0"
LSTM, LSTM_PLAY = "Isaac-Ant-R6-OracleLSTM-v0", "Isaac-Ant-R6-OracleLSTM-Play-v0"
LOCK3 = ("lock3_obstacles_high", "lock3_wave_big", "lock3_rails_high", "lock3_pits_deep")
MAX_HOURS = 14.0
b5.NO_NEW_TRAIN_AFTER = datetime(2026, 10, 6, 20, 0)  # shared guard: LMS deadline; this batch uses MAX_HOURS
T0 = time.time()


def log(text):
    o.log(f"[fb] [b11] {text}")


def busy(name):
    return subprocess.run(["pgrep", "-f", rf"^(\S*/)?python3? experiments/{name}"],
                          stdout=subprocess.DEVNULL).returncode == 0


def in_time():
    return time.time() - T0 < MAX_HOURS * 3600


def unseen14(run, play, over):
    ck = r4.ckpt_of(run)
    m = b10.metrics(run, ck, play, over)
    m["mult"], _ = b5.eval_multiply(run, play, ck) if over == ENT else eval_mult(run, play, ck, over)
    vals = [json.load(open(f"experiments/eval/{run}/{c}.json"))["return_mean"] for c in o.HELD_OUT]
    vals += list(m["unseen8_detail"].values()) + [m["mult"]]
    m["unseen14"] = st.mean(vals)
    return m


def eval_mult(run, play, ck, over):
    out = f"experiments/eval/{run}/boxes_mid_f0.2_multiply.json"
    if not os.path.exists(out):
        b10.wait_gpu()
        o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", "boxes_mid", "--friction", "0.2",
               "--combine_mode", "multiply", "--num_envs", "100", "--checkpoint", ck, "--output", out, *over],
              out.replace(".json", ".log"))
    d = json.load(open(out))
    return d["return_mean"], d["fall_rate"]


def train(run, task, seed, over, iters, init=None):
    if not in_time():
        log(f"{run} skipped: batch time limit")
        return None
    b10.wait_gpu()
    return b7.train(run, task, seed, over, iters, init=init)


def fmt(name, m):
    return (f"{name}: unseen14 {m['unseen14']:.1f} (held-out {m['held_out']:.1f}, lock1 {m['lock1']:.1f}, "
            f"lock2 {m['lock2']:.1f}, μ0.2 mult {m['mult']:.1f}), flat {m['flat']:.1f} (speed {m['flat_speed']:.2f})")


def f3a(seed):
    return unseen14(f"f3a_s{seed}", ORACLE_PLAY, ENT)


def paired(name, cand, base, need_wins):
    seeds = [s for s in cand if cand[s] and base.get(s)]
    diffs = [cand[s]["unseen14"] - base[s]["unseen14"] for s in seeds]
    if len(diffs) < 3:
        log(f"{name}: only {len(diffs)} paired seeds → no decision")
        return None
    d = st.mean(diffs)
    t = {3: 4.303, 4: 3.182, 5: 2.776}.get(len(diffs), 2.0)
    half = t * st.stdev(diffs) / len(diffs) ** 0.5
    wins = sum(x > 0 for x in diffs)
    fl = st.mean(cand[s]["flat"] for s in seeds) - st.mean(base[s]["flat"] for s in seeds)
    ok = d >= 3 and wins >= need_wins and fl >= -5
    log(f"{name} vs F3a (seeds {seeds}): unseen14 diff {d:+.1f} [95% CI {d - half:+.1f}, {d + half:+.1f}], wins "
        f"{wins}/{len(diffs)}, flat diff {fl:+.1f} → {'PASSES replacement conditions' if ok else 'no'}")
    o.record({"batch": 11, "stage": "decision", "exp": name, "seeds": seeds, "diffs": diffs, "wins": wins,
              "flat_diff": fl, "pass": ok})
    return (d, name) if ok else None


def screen_and_confirm(name, task, play, over, iters=1600):
    """Screen on seeds 42/43 vs F3a, confirm on 47-49 if it passes (pre-registered thresholds)."""
    ms = {}
    for s in (42, 43):
        run = f"{name.lower()}_s{s}"
        if train(run, task, s, over, iters) is None:
            return None
        ms[s] = unseen14(run, play, over)
        log(fmt(f"{name} s{s}", ms[s]))
    u = st.mean(m["unseen14"] for m in ms.values()) - st.mean(f3a(s)["unseen14"] for s in (42, 43))
    fl = st.mean(m["flat"] for m in ms.values()) - st.mean(f3a(s)["flat"] for s in (42, 43))
    ok = u >= 3 and fl >= -5
    log(f"screen {name}: unseen14 diff {u:+.1f}, flat diff {fl:+.1f} vs F3a s42/43 → {'PASS' if ok else 'fail'}")
    o.record({"batch": 11, "stage": "screen", "exp": name, "unseen14_diff": u, "flat_diff": fl, "pass": ok})
    if not ok:
        return None
    for s in (47, 48, 49):
        run = f"{name.lower()}_s{s}"
        if train(run, task, s, over, iters) is not None:
            ms[s] = unseen14(run, play, over)
            log(fmt(f"{name} s{s}", ms[s]))
    return paired(name, {s: ms.get(s) for s in (47, 48, 49)}, {s: f3a(s) for s in (47, 48, 49)}, need_wins=2)


def main():
    while busy("queue_fb_b10.py"):
        time.sleep(60)
    log("## batch 11 start (primary metric unseen14; final untouched check lockbox 3)")
    specs = {}  # name -> (play task, overrides, s47 run) for the final step

    # 0) checks
    b10.wait_gpu()
    code = o.run(["./isaaclab.sh", "-p", "experiments/check_symmetry.py"], "experiments/train_logs/check_symmetry.log")
    sym_ok = os.path.exists("source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_symmetry_map.json")
    log(f"symmetry check exit {code}: {'map written (PASS)' if sym_ok else 'FAIL → A3 skipped'}")
    if sym_ok:
        b10.wait_gpu()
        sym_ok, _ = r4.smoke("b11_sym", SYM, [])
        log(f"smoke symmetry training: {'OK' if sym_ok else 'FAILED → A3 skipped'}")
    b10.wait_gpu()
    lstm_ok, d = r4.smoke("b11_lstm", LSTM, [])
    if lstm_ok:
        ck = sorted(glob.glob(f"{d}/model_*.pt"))[-1] if d else None
        code = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", LSTM_PLAY, "--headless",
                      "--num_envs", "4", "--checkpoint", ck], "experiments/train_logs/smoke_b11_lstm_official.log")
        lstm_ok = code == 0
    log(f"smoke LSTM training + official play_one_episode: {'OK' if lstm_ok else 'FAILED → A4 skipped'}")

    passed = []
    # 1) B1: energy -0.02 lineage (H2) already has seeds 42, 43, 47, 48, 49 → evaluation only
    h2 = {s: unseen14(f"h2_s{s}", ORACLE_PLAY, ENT) for s in (42, 43, 47, 48, 49)}
    for s, m in h2.items():
        log(fmt(f"H2 s{s}", m))
    r = paired("B1 (H2 energy)", h2, {s: f3a(s) for s in h2}, need_wins=3)
    if r:
        passed.append(r)
        specs["B1 (H2 energy)"] = (ORACLE_PLAY, ENT, "h2_s47")

    # 2) A1: bigger network [512, 256, 128], 1600 it
    r = screen_and_confirm("N1", ORACLE, ORACLE_PLAY, ENT + NET)
    if r:
        passed.append(r)
        specs["N1"] = (ORACLE_PLAY, ENT + NET, "n1_s47")

    # 3) C1: low-friction lineage (H1 = F1 1000 it + 600 it), complete seeds 47-49
    for s in (47, 48, 49):
        ck1 = train(f"f1_s{s}", LF, s, ENT, 1000)
        if ck1:
            train(f"h1_s{s}", LF, s, ENT, 600, init=ck1)
    h1 = {s: unseen14(f"h1_s{s}", LF_PLAY, ENT) for s in (42, 43, 47, 48, 49) if r4.ckpt_of(f"h1_s{s}")}
    for s, m in h1.items():
        log(fmt(f"H1 s{s}", m))
    r = paired("C1 (H1 low friction)", h1, {s: f3a(s) for s in h1}, need_wins=3)
    if r:
        passed.append(r)
        specs["C1 (H1 low friction)"] = (LF_PLAY, ENT, "h1_s47")

    # 4) A3: left-right symmetry augmentation
    if sym_ok:
        r = screen_and_confirm("SY", SYM, SYM_PLAY, [])
        if r:
            passed.append(r)
            specs["SY"] = (SYM_PLAY, [], "sy_s47")

    # 5) A4: LSTM policy
    if lstm_ok:
        r = screen_and_confirm("LS", LSTM, LSTM_PLAY, [])
        if r:
            passed.append(r)
            specs["LS"] = (LSTM_PLAY, [], "ls_s47")

    # 6) final (amended 2026-10-01 19:45, before batch 11 started): compare against the CURRENT submission.
    #    Batch 10 may have replaced F3a by M30 (AllMix 3000 it). M30 trained on the held-out 5, so the common
    #    metric between M30 and F3a-lineage candidates is unseen8 (lockbox 1 + 2), and lockbox 3 is the final check.
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    log(f"candidates meeting replacement conditions vs F3a: {[n for _, n in sorted(passed, reverse=True)] or 'none'}; "
        f"current submission before the final step: {state['submission']}")

    def lock3(run, play, over):
        ck = r4.ckpt_of(run)
        v = {t: b10.evaluate(ck, play, t, f"experiments/eval_lockbox3/{run}/{t}.json", over)["return_mean"] for t in LOCK3}
        log(f"lockbox 3 {run}: " + ", ".join(f"{k[6:]} {x:.1f}" for k, x in v.items()) + f" → {st.mean(v.values()):.1f}")
        return st.mean(v.values())

    l3_f3a = lock3("f3a_s47", ORACLE_PLAY, ENT)
    cur, cur_l3, cur_u8 = "F3a", l3_f3a, None
    if state["submission"] == "M30":
        l3_m30 = lock3("m30_s47", "Isaac-Ant-R5-AllMix-Play-v0", ENT)
        if l3_m30 >= l3_f3a - 3:
            cur, cur_l3 = "M30", l3_m30
            cur_u8 = {s: b10.metrics(f"m30_s{s}", r4.ckpt_of(f"m30_s{s}"), "Isaac-Ant-R5-AllMix-Play-v0", ENT)["unseen8"]
                      for s in (47, 48, 49)}
            log(f"M30 keeps the submission: lockbox 3 {l3_m30:.1f} ≥ F3a {l3_f3a:.1f} − 3")
        else:
            log(f"M30 REVERTED to F3a: lockbox 3 {l3_m30:.1f} < F3a {l3_f3a:.1f} − 3")
            state = {**state, "submission": "F3a", "play": ORACLE_PLAY, "ckpt": r4.ckpt_of("f3a_s47"),
                     "note": "M30 reverted by the lockbox-3 check (batch 11)"}
            with open("experiments/fb_state.json", "w") as f:
                json.dump(state, f, indent=1)

    for _, name in sorted(passed, reverse=True):
        play, over, run = specs[name]
        if cur == "M30":
            # candidate must also beat M30 on the common unseen8 metric, paired on seeds 47-49
            prefix = run.rsplit("_s", 1)[0]
            cu = {s: b10.metrics(f"{prefix}_s{s}", r4.ckpt_of(f"{prefix}_s{s}"), play, over)["unseen8"]
                  for s in (47, 48, 49) if r4.ckpt_of(f"{prefix}_s{s}")}
            diffs = [cu[s] - cur_u8[s] for s in cu]
            ok_u8 = len(diffs) == 3 and st.mean(diffs) >= 3 and sum(x > 0 for x in diffs) >= 2
            log(f"{name} vs M30 (s47–49, unseen8): diffs {[round(x, 1) for x in diffs]} → {'PASS' if ok_u8 else 'keep M30'}")
            if not ok_u8:
                continue
        l3m = lock3(run, play, over)
        ok = l3m >= cur_l3 - 3
        log(f"lockbox 3 {name}: {l3m:.1f} (need ≥ {cur_l3 - 3:.1f}) → {'REPLACE ' + cur if ok else 'keep ' + cur}")
        if not ok:
            continue
        ck = r4.ckpt_of(run)
        os.makedirs(f"experiments/eval/smoke_b11_{run}", exist_ok=True)
        for terrain in ("flat", "boxes_mid"):
            for fric in ("1.0", "0.2"):
                for mode in ("average", "multiply"):
                    n = f"{terrain}_f{fric}_{mode}"
                    code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain,
                                  "--friction", fric, "--combine_mode", mode, "--checkpoint", ck, "--output",
                                  f"experiments/eval/smoke_b11_{run}/{n}.json", *over],
                                 f"experiments/train_logs/smoke_b11_{run}_{n}.log")
                    log(f"smoke {name} {n}: {'OK' if code == 0 else 'FAILED'}")
        code = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", play, "--headless", "--num_envs",
                      "100", "--checkpoint", ck, *over], f"experiments/train_logs/smoke_b11_{run}_official.log")
        log(f"smoke {name} official play_one_episode: exit {code}")
        state = {**state, "previous": cur, "submission": name, "play": play, "ckpt": ck, "score_overrides": over,
                 "lockbox3": l3m}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(state, f, indent=1)
        log(f"NEW SUBMISSION: {name} `{ck}`")
        break
    log("## batch 11 done")


if __name__ == "__main__":
    main()
