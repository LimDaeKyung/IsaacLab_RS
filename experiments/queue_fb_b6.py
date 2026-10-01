"""Batch 6 (share-fb): E15 + training-only energy penalty (G1), E15 + best E22 entropy (G2, conditional).

Pre-registered 2026-09-30 22:35 in experiments/EXPERIMENTS_overnight_fb.md (batch 6), before any batch-5 / E22 / T1 result.
Runs after experiments/queue_fb_b5.py exits. Scoring always uses the original reward (no energy override at eval).
usage: python experiments/queue_fb_b6.py > experiments/queue_fb_b6.out 2>&1
"""

import glob
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
ENERGY = ["env.rewards.energy.weight=-0.02"]  # training only
E15_TASK, E15_PLAY = b5.E15_TASK, b5.E15_PLAY
E6_S42_43_HELD_OUT = 46.7  # E6 (entropy 0.005 on E4) seeds 42/43, reference for the E22 dose-response


def log(text):
    o.log(f"[fb] {text}")


def busy(name):
    return subprocess.run(["pgrep", "-f", rf"^(\S*/)?python3? experiments/{name}"],
                          stdout=subprocess.DEVNULL).returncode == 0


def train_eval(label, task, play, run_name, seed, train_over, eval_over, iters=1000):
    """Like overnight4.train_eval, but training and scoring overrides are separate."""
    if not os.path.exists(f"experiments/eval/{run_name}/boxes_mid_f0.2.json"):
        if not b5.can_start(r4.EST_RUN * iters / 1000):
            log(f"{label} seed {seed} skipped: would start after 08:00 or end too late")
            return None
        log(f"**{label} seed {seed} start** | task {task} | train overrides {train_over} | score overrides {eval_over}")
        code = o.run(["./isaaclab.sh", "-p", f"{R}/train.py", "--task", task, "--headless", "--seed", str(seed),
                      "--run_name", run_name, "--max_iterations", str(iters), *train_over],
                     f"experiments/train_logs/{run_name}.log")
        ckpt = r4.ckpt_of(run_name)
        if code != 0 or not ckpt or not ckpt.endswith(f"model_{iters - 1}.pt"):
            log(f"{label} seed {seed} FAILED during training (see experiments/train_logs/{run_name}.log)")
            return None
        subprocess.run(["bash", f"{R}/eval_sweep.sh", ckpt, f"experiments/eval/{run_name}", play, *eval_over],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ckpt = r4.ckpt_of(run_name)
    m = r4.full_metrics(run_name)
    m["mult"], m["mult_fall"] = b5.eval_multiply(run_name, play, ckpt)
    ts = o.train_stats(os.path.dirname(ckpt))
    o.record({"batch": 6, "exp": label, "seed": seed, "run": run_name, "ckpt": ckpt, "metrics": m, "train": ts})
    log(f"{label} s{seed}: held-out {m['held_out']:.1f}, flat {m['flat']:.1f} (speed {m['flat_speed']:.2f}), "
        f"ho-fall {100 * m['ho_fall']:.0f}%, boxes μ0.2 multiply {m['mult']:.1f} | std {ts['std']:.3f}, lr {ts['lr']:.2e}")
    return m


def e22_best():
    """Best E22 entropy value if its 2-seed held-out beats E6 (0.005) s42/43 by >= 3, else None."""
    best = None
    for value in ("0.01", "0.002"):
        dirs = [d for d in glob.glob("experiments/eval/*") if "e22" in os.path.basename(d) and
                value.replace(".", "") in os.path.basename(d).replace(".", "")]
        vals = []
        for d in dirs:
            try:
                vals.append(r4.full_metrics(os.path.basename(d))["held_out"])
            except (OSError, KeyError, ValueError):
                pass
        if len(vals) >= 2:
            ho = st.mean(vals)
            log(f"E22 entropy {value}: held-out {ho:.1f} over {len(vals)} seeds (E6 0.005 s42/43 = {E6_S42_43_HELD_OUT})")
            if ho >= E6_S42_43_HELD_OUT + 3 and (best is None or ho > best[1]):
                best = (value, ho)
    return best[0] if best else None


def main():
    while busy("queue_fb_b5.py") or busy("queue_t1_teacher.py") or busy("queue_e22_entropy.py"):
        time.sleep(60)
    log("## batch 6 start (G1 energy −0.02 training-only on E15, G2 conditional E22 entropy)")
    state = {}
    if os.path.exists("experiments/fb_state.json"):
        with open("experiments/fb_state.json") as f:
            state = json.load(f)
    ref = [b5.e15_metrics(s) for s in (42, 43)]
    ho0, fl0 = b5.mean(ref, "held_out"), b5.mean(ref, "flat")

    ideas = [("G1", ENT + ENERGY, ENT)]
    ent = e22_best()
    if ent and ent != "0.005":
        ideas.append(("G2", [f"agent.algorithm.entropy_coef={ent}"], [f"agent.algorithm.entropy_coef={ent}"]))
    else:
        log("G2 skipped: no E22 entropy value beat 0.005 by ≥ 3 (pre-registered condition)")

    passed = []
    for exp, tr, ev in ideas:
        ms = [train_eval(exp, E15_TASK, E15_PLAY, f"{exp.lower()}_s{s}", s, tr, ev) for s in (42, 43)]
        if any(m is None for m in ms):
            log(f"screen {exp}: incomplete")
            continue
        ho, fl = b5.mean(ms, "held_out"), b5.mean(ms, "flat")
        ok = (ho >= ho0 + 3 or (ho >= ho0 - 1 and fl >= fl0 + 10)) and fl >= fl0 - 5
        log(f"screen {exp}: held-out {ho:.1f} vs E15 {ho0:.1f}, flat {fl:.1f} vs {fl0:.1f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 6, "exp": exp, "held_out": ho, "flat": fl, "decision": "PASS" if ok else "fail"})
        if ok:
            passed.append((ho, exp, tr, ev))
    passed.sort(key=lambda x: -x[0])

    # confirmation against the CURRENT submission (E15, or the batch-5 replacement)
    sub = state.get("submission", "E15")
    sub_lock = state.get("lockbox", b5.LOCKBOX_E15)
    for _, exp, tr, ev in passed:
        base = {}
        for s in (47, 48, 49):
            if sub == "E15":
                base[s] = b5.run("E15", E15_TASK, E15_PLAY, f"e15_s{s}", s)
            else:
                run = state["runs"][str(s)]
                base[s] = r4.full_metrics(run) if os.path.exists(f"experiments/eval/{run}/boxes_mid_f0.2.json") else None
        cand = {s: train_eval(exp, E15_TASK, E15_PLAY, f"{exp.lower()}_s{s}", s, tr, ev) for s in (47, 48, 49)}
        seeds = [s for s in (47, 48, 49) if base.get(s) and cand.get(s)]
        if len(seeds) < 3:
            log(f"confirm {exp}: only {len(seeds)} paired seeds → no replacement decision")
            continue
        diffs = [cand[s]["held_out"] - base[s]["held_out"] for s in seeds]
        d, wins = st.mean(diffs), sum(x > 0 for x in diffs)
        half = r4.T_975_DF2 * st.stdev(diffs) / 3 ** 0.5
        fl_ok = b5.mean([cand[s] for s in seeds], "flat") >= b5.mean([base[s] for s in seeds], "flat") - 5
        log(f"confirm {exp} vs {sub} (s47–49): held-out diff {d:+.1f} [95% CI {d - half:+.1f}, {d + half:+.1f}], "
            f"wins {wins}/3, flat ok {fl_ok}")
        o.record({"batch": 6, "stage": "confirm", "exp": exp, "vs": sub, "diffs": diffs, "wins": wins, "flat_ok": fl_ok})
        if not (d >= 3 and wins >= 2 and fl_ok):
            continue
        ck = r4.ckpt_of(f"{exp.lower()}_s47")
        subprocess.run(["bash", "experiments/lockbox_eval.sh", f"fb_{exp}_s47", ck, E15_PLAY, *ev],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        vals = []
        for t in ("lock_rails", "lock_gaps", "lock_pits", "lock_stones"):
            try:
                with open(f"experiments/eval_lockbox/fb_{exp}_s47/{t}.json") as f:
                    vals.append(json.load(f)["return_mean"])
            except (OSError, KeyError, json.JSONDecodeError):
                vals.append(float("nan"))
        lb = st.mean(vals)
        ok = lb >= sub_lock - 3
        log(f"LOCKBOX {exp} s47 (reused lockbox): {[round(v, 1) for v in vals]} → mean {lb:.1f} (need ≥ {sub_lock - 3:.1f})"
            f" → {'REPLACE ' + sub if ok else 'keep ' + sub}")
        if ok:
            with open("experiments/fb_state.json", "w") as f:
                json.dump({"submission": exp, "task": E15_TASK, "play": E15_PLAY, "ckpt": ck, "iters": 1000,
                           "train_overrides": tr, "score_overrides": ev,
                           "runs": {str(s): f"{exp.lower()}_s{s}" for s in (47, 48, 49)}, "lockbox": lb}, f, indent=1)
            log(f"NEW SUBMISSION CANDIDATE: {exp} `{ck}` (play {E15_PLAY}, score overrides {ev})")
            break
    log("## batch 6 done")


if __name__ == "__main__":
    main()
