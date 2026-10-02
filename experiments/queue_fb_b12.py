"""Batch 12 (share-fb, pre-registered 2026-10-01 21:05): develop the M30 line against its weak points.

Weak points of M30 (batch 10): more falls on almost every unseen terrain, weaker on low friction (boxes μ0.2:
66.6 vs F3a 74.4), stepping stones stuck around 40, symmetry not used (mapping fixed after batch 11).
Candidates (each one change on the M30 recipe = AllMix, entropy 0.005):
  M40  = M30 + 1000 more iterations (warm start)          -> does the length trend continue?
  LF30 = AllMix + low-friction (multiply, robot 0.05-1.2), 3000 it  -> friction weakness
  SY30 = AllMix + left-right symmetry augmentation, 3000 it (only if check_symmetry passes) -> data efficiency/falls
Metric "unseen9" = lockbox 1 (4) + lockbox 2 (4) + boxes ±10 μ0.2 multiply. Final untouched check: lockbox 4.
Runs after experiments/queue_fb_b11.py exits; no new training after Fri 2026-10-02 23:59.
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
import queue_fb_b10 as b10  # noqa: E402

R = o.R
ENT = ["agent.algorithm.entropy_coef=0.005"]
b5.NO_NEW_TRAIN_AFTER = datetime(2026, 10, 2, 23, 59)
ALLMIX, ALLMIX_PLAY = "Isaac-Ant-R5-AllMix-v0", "Isaac-Ant-R5-AllMix-Play-v0"
LOWF = "Isaac-Ant-R7-AllMixLowFric-v0"
SYM, SYM_PLAY = "Isaac-Ant-R7-AllMixSym-v0", "Isaac-Ant-R7-AllMixSym-Play-v0"
LOCK4 = {"lock4_stones_hard": [], "lock4_cylinders_tall": [], "lock4_boxes_fine12": [],
         "lock4_wave_lowfric": ["--friction", "0.3", "--combine_mode", "multiply"]}
# how to evaluate any possible current submission (name -> run prefix, play task, score overrides)
MODELS = {"F3a": ("f3a", "Isaac-Ant-R2-Oracle-Play-v0", ENT), "M30": ("m30", ALLMIX_PLAY, ENT),
          "B1 (H2 energy)": ("h2", "Isaac-Ant-R2-Oracle-Play-v0", ENT),
          "C1 (H1 low friction)": ("h1", "Isaac-Ant-R2-LowFricOracle-Play-v0", ENT),
          "N1": ("n1", "Isaac-Ant-R2-Oracle-Play-v0", ENT + ["agent.policy.actor_hidden_dims=[512,256,128]",
                                                          "agent.policy.critic_hidden_dims=[512,256,128]"]),
          "SY": ("sy", "Isaac-Ant-R6-OracleSym-Play-v0", []), "LS": ("ls", "Isaac-Ant-R6-OracleLSTM-Play-v0", []),
          "M40": ("m40", ALLMIX_PLAY, ENT), "LF30": ("lf30", ALLMIX_PLAY, ENT), "SY30": ("sy30", SYM_PLAY, [])}


def log(text):
    o.log(f"[fb] [b12] {text}")


def busy(pattern):
    return subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL).returncode == 0


def u9(run, play, over):
    ck = r4.ckpt_of(run)
    m = b10.metrics(run, ck, play, over)
    out = f"experiments/eval/{run}/boxes_mid_f0.2_multiply.json"
    for _ in range(2):  # one retry (see queue_fb_b10.evaluate)
        if os.path.exists(out):
            break
        b10.wait_gpu()
        o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", "boxes_mid", "--friction", "0.2",
               "--combine_mode", "multiply", "--num_envs", "100", "--checkpoint", ck, "--output", out, *over],
              out.replace(".json", ".log"))
    m["mult"] = json.load(open(out))["return_mean"]
    m["unseen9"] = st.mean(list(m["unseen8_detail"].values()) + [m["mult"]])
    return m


def fmt(name, m):
    return (f"{name}: unseen9 {m['unseen9']:.1f} (lock1 {m['lock1']:.1f}, lock2 {m['lock2']:.1f}, μ0.2 mult "
            f"{m['mult']:.1f}), flat {m['flat']:.1f} (speed {m['flat_speed']:.2f}, fall {100 * m['flat_fall']:.0f}%)")


def lock4(run, play, over):
    ck = r4.ckpt_of(run)
    v = {t: b10.evaluate(ck, play, t, f"experiments/eval_lockbox4/{run}/{t}.json", over + extra)["return_mean"]
         for t, extra in LOCK4.items()}
    log(f"lockbox 4 {run}: " + ", ".join(f"{k[6:]} {x:.1f}" for k, x in v.items()) + f" → {st.mean(v.values()):.1f}")
    return st.mean(v.values())


def run_candidate(name, seed):
    run = f"{name.lower()}_s{seed}"
    b10.wait_gpu()
    if name == "M40":
        ck = b7.train(run, ALLMIX, seed, ENT, 1000, init=r4.ckpt_of(f"m30_s{seed}"))
    elif name == "LF30":
        ck = b7.train(run, LOWF, seed, ENT, 3000)
    else:
        ck = b7.train(run, SYM, seed, [], 3000)
    if ck is None:
        return None
    _, play, over = MODELS[name]
    m = u9(run, play, over)
    log(fmt(f"{name} s{seed}", m))
    return m


def main():
    while busy(r"^(\S*/)?python3? experiments/queue_fb_b11.py") or busy("run_fb_b10_b11.sh"):
        time.sleep(60)
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    log(f"## batch 12 start (current submission: {cur}; metric unseen9; final check lockbox 4)")

    # 0) symmetry check with the geometric y-mirror pairing
    b10.wait_gpu()
    code = o.run(["./isaaclab.sh", "-p", "experiments/check_symmetry.py"], "experiments/train_logs/check_symmetry_b12.log")
    sym_ok = os.path.exists("source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/ant_symmetry_map.json")
    log(f"symmetry check (y-mirror by geometry) exit {code}: {'PASS' if sym_ok else 'FAIL → SY30 skipped'}")
    if sym_ok:
        b10.wait_gpu()
        sym_ok, _ = r4.smoke("b12_sym", SYM, [])
        log(f"smoke AllMix+symmetry training: {'OK' if sym_ok else 'FAILED → SY30 skipped'}")

    ref = {s: u9(f"m30_s{s}", ALLMIX_PLAY, ENT) for s in (42, 43)}
    for s, m in ref.items():
        log(fmt(f"M30 s{s}", m))
    u0, f0, mu0 = (st.mean(m[k] for m in ref.values()) for k in ("unseen9", "flat", "mult"))

    # 1) screening (seeds 42/43) against M30
    passed = []
    for name in ["M40", "LF30"] + (["SY30"] if sym_ok else []):
        ms = [run_candidate(name, s) for s in (42, 43)]
        if any(m is None for m in ms):
            log(f"screen {name}: incomplete")
            continue
        u, fl, mu = (st.mean(m[k] for m in ms) for k in ("unseen9", "flat", "mult"))
        target = name == "LF30" and u >= u0 - 1 and mu >= mu0 + 5
        ok = (u >= u0 + 3 or target) and fl >= f0 - 5
        log(f"screen {name}: unseen9 {u:.1f} vs M30 {u0:.1f}, μ0.2 mult {mu:.1f} vs {mu0:.1f}, flat {fl:.1f} vs "
            f"{f0:.1f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 12, "stage": "screen", "exp": name, "unseen9": u, "mult": mu, "flat": fl, "pass": ok})
        if ok:
            passed.append((u - u0, name))
    passed.sort(reverse=True)

    # 2) confirmation on seeds 47-49 against the CURRENT submission (paired), best two passers
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    prefix, cplay, cover = MODELS[cur]
    base = {s: u9(f"{prefix}_s{s}", cplay, cover) for s in (47, 48, 49) if r4.ckpt_of(f"{prefix}_s{s}")}
    confirmed = []
    for _, name in passed[:2]:
        cand = {s: run_candidate(name, s) for s in (47, 48, 49)}
        seeds = [s for s in (47, 48, 49) if cand.get(s) and base.get(s)]
        if len(seeds) < 3:
            log(f"confirm {name}: only {len(seeds)} seeds")
            continue
        d = [cand[s]["unseen9"] - base[s]["unseen9"] for s in seeds]
        dm = [cand[s]["mult"] - base[s]["mult"] for s in seeds]
        fl = st.mean(cand[s]["flat"] for s in seeds) - st.mean(base[s]["flat"] for s in seeds)
        half = r4.T_975_DF2 * st.stdev(d) / 3 ** 0.5
        ok = (st.mean(d) >= 3 and sum(x > 0 for x in d) >= 2) or (
            name == "LF30" and st.mean(d) >= -1 and st.mean(dm) >= 5 and sum(x > 0 for x in dm) >= 2)
        ok = ok and fl >= -5
        log(f"confirm {name} vs {cur} (s47–49): unseen9 diff {st.mean(d):+.1f} [95% CI {st.mean(d) - half:+.1f}, "
            f"{st.mean(d) + half:+.1f}], wins {sum(x > 0 for x in d)}/3, μ0.2 mult diff {st.mean(dm):+.1f}, flat diff "
            f"{fl:+.1f} → {'PASSES' if ok else 'no'}")
        o.record({"batch": 12, "stage": "confirm", "exp": name, "vs": cur, "diffs": d, "mult_diffs": dm,
                  "flat_diff": fl, "pass": ok})
        if ok:
            confirmed.append((st.mean(d), name))

    # 3) final: lockbox 4 (untouched) for the best confirmed candidate vs the current submission
    if confirmed:
        confirmed.sort(reverse=True)
        _, name = confirmed[0]
        l_cur = lock4(f"{prefix}_s47", cplay, cover)
        _, play, over = MODELS[name]
        l_new = lock4(f"{name.lower()}_s47", play, over)
        if l_new >= l_cur - 3:
            ck = r4.ckpt_of(f"{name.lower()}_s47")
            os.makedirs(f"experiments/eval/smoke_b12_{name}", exist_ok=True)
            for terrain in ("flat", "boxes_mid"):
                for fric in ("1.0", "0.2"):
                    for mode in ("average", "multiply"):
                        n = f"{terrain}_f{fric}_{mode}"
                        c = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain,
                                   "--friction", fric, "--combine_mode", mode, "--checkpoint", ck, "--output",
                                   f"experiments/eval/smoke_b12_{name}/{n}.json", *over],
                                  f"experiments/train_logs/smoke_b12_{name}_{n}.log")
                        log(f"smoke {name} {n}: {'OK' if c == 0 else 'FAILED'}")
            c = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", play, "--headless", "--num_envs",
                       "100", "--checkpoint", ck, *over], f"experiments/train_logs/smoke_b12_{name}_official.log")
            log(f"smoke {name} official play_one_episode: exit {c}")
            state = {**state, "previous": cur, "submission": name, "play": play, "ckpt": ck, "score_overrides": over,
                     "lockbox4": l_new}
            with open("experiments/fb_state.json", "w") as f:
                json.dump(state, f, indent=1)
            log(f"NEW SUBMISSION: {name} `{ck}` (lockbox 4 {l_new:.1f} vs {cur} {l_cur:.1f})")
        else:
            log(f"lockbox 4: {name} {l_new:.1f} < {cur} {l_cur:.1f} − 3 → keep {cur}")
    else:
        log(f"no candidate confirmed → keep {cur}")
    log("## batch 12 done")


if __name__ == "__main__":
    main()
