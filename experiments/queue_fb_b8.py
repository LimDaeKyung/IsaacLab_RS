"""Batch 8 (share-fb): K1 = F3a + another warm-started 600 it (total 2200 it), then final-model videos.

Pre-registered 2026-10-01 05:02 in experiments/EXPERIMENTS_overnight_fb.md (batch 8), before any batch-8 result.
"""

import json
import os
import statistics as st
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight4 as r4  # noqa: E402
import queue_fb_b5 as b5  # noqa: E402
import queue_fb_b7 as b7  # noqa: E402

R = o.R
ENT = b7.ENT
OR_TASK, OR_PLAY = b7.OR_TASK, b7.OR_PLAY


def log(text):
    o.log(f"[fb] {text}")


def k1(seed):
    run = f"k1_s{seed}"
    ck = b7.train(run, OR_TASK, seed, ENT, 600, init=r4.ckpt_of(f"f3a_s{seed}"))
    if ck is None:
        return None
    if not os.path.exists(f"experiments/eval/{run}/boxes_mid_f0.2.json"):
        subprocess.run(["bash", f"{R}/eval_sweep.sh", ck, f"experiments/eval/{run}", OR_PLAY, *ENT],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    m = r4.full_metrics(run)
    m["mult"], m["mult_fall"] = b5.eval_multiply(run, OR_PLAY, ck)
    m["ckpt"] = ck
    ts = o.train_stats(os.path.dirname(ck))
    o.record({"batch": 8, "exp": "K1", "seed": seed, "run": run, "ckpt": ck, "metrics": m, "train": ts})
    log(f"K1 s{seed}: held-out {m['held_out']:.1f}, flat {m['flat']:.1f} (speed {m['flat_speed']:.2f}), "
        f"ho-fall {100 * m['ho_fall']:.0f}%, boxes μ0.2 multiply {m['mult']:.1f} | std {ts['std']:.3f}, lr {ts['lr']:.2e}")
    return m


def videos(name, ckpt, play):
    os.makedirs("experiments/videos", exist_ok=True)
    for terrain in ("flat", "boxes_mid", "ho_obstacles", "stairs"):
        out = f"experiments/videos/FINAL_{name}_{terrain}.json"
        code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain, "--num_envs", "16",
                      "--video", "--checkpoint", ckpt, "--output", out, *ENT], out.replace(".json", ".log"))
        log(f"video {name} {terrain}: {'OK' if code == 0 else 'FAILED'}")


def main():
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    log("## batch 8 start (K1 = F3a + 600 it, total 2200 it)")
    ref = [b7.f3a(s) for s in (42, 43)]
    ho0, fl0 = b5.mean(ref, "held_out"), b5.mean(ref, "flat")
    ms = [k1(s) for s in (42, 43)]
    replaced = False
    if all(ms):
        ho, fl = b5.mean(ms, "held_out"), b5.mean(ms, "flat")
        ok = ho >= ho0 + 3 and fl >= fl0 - 5
        log(f"screen K1: held-out {ho:.1f} vs F3a {ho0:.1f}, flat {fl:.1f} vs {fl0:.1f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 8, "exp": "K1", "held_out": ho, "flat": fl, "decision": "PASS" if ok else "fail"})
        if ok:
            base = {s: b7.f3a(s) for s in (47, 48, 49)}
            cand = {s: k1(s) for s in (47, 48, 49)}
            seeds = [s for s in (47, 48, 49) if cand.get(s)]
            if len(seeds) == 3:
                diffs = [cand[s]["held_out"] - base[s]["held_out"] for s in seeds]
                d, wins = st.mean(diffs), sum(x > 0 for x in diffs)
                half = r4.T_975_DF2 * st.stdev(diffs) / 3 ** 0.5
                fl_ok = b5.mean(list(cand.values()), "flat") >= b5.mean(list(base.values()), "flat") - 5
                log(f"confirm K1 vs F3a (s47–49): held-out diff {d:+.1f} [95% CI {d - half:+.1f}, {d + half:+.1f}], "
                    f"wins {wins}/3, flat ok {fl_ok}")
                if d >= 3 and wins >= 2 and fl_ok:
                    ck = cand[47]["ckpt"]
                    subprocess.run(["bash", "experiments/lockbox_eval.sh", "fb_K1_s47", ck, OR_PLAY, *ENT],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    vals = []
                    for t in ("lock_rails", "lock_gaps", "lock_pits", "lock_stones"):
                        with open(f"experiments/eval_lockbox/fb_K1_s47/{t}.json") as f:
                            vals.append(json.load(f)["return_mean"])
                    lb = st.mean(vals)
                    need = state["lockbox"] - 3
                    log(f"LOCKBOX K1 s47 (lockbox reused, 5th model): {[round(v, 1) for v in vals]} → mean {lb:.1f} "
                        f"(need ≥ {need:.1f})")
                    if lb >= need:
                        os.makedirs("experiments/eval/smoke_fb_K1", exist_ok=True)
                        for terrain in ("flat", "boxes_mid"):
                            for fric in ("1.0", "0.2"):
                                for mode in ("average", "multiply"):
                                    name = f"{terrain}_f{fric}_{mode}"
                                    code = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", OR_PLAY,
                                                  "--terrain", terrain, "--friction", fric, "--combine_mode", mode,
                                                  "--checkpoint", ck, "--output",
                                                  f"experiments/eval/smoke_fb_K1/{name}.json", *ENT],
                                                 f"experiments/train_logs/smoke_fb_K1_{name}.log")
                                    log(f"smoke K1 {name}: {'OK' if code == 0 else 'FAILED'}")
                        code = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", OR_PLAY, "--headless",
                                      "--num_envs", "100", "--checkpoint", ck, *ENT],
                                     "experiments/train_logs/smoke_fb_K1_official.log")
                        log(f"smoke K1 official play_one_episode: exit {code}")
                        state = {**state, "submission": "K1", "ckpt": ck, "lockbox": lb, "previous": "F3a",
                                 "description": "F3a + warm-started 600 it (total 2200 it)",
                                 "runs": {str(s): f"k1_s{s}" for s in (42, 43, 47, 48, 49)}}
                        with open("experiments/fb_state.json", "w") as f:
                            json.dump(state, f, indent=1)
                        log(f"NEW SUBMISSION: K1 `{ck}`")
                        replaced = True
    log(f"final submission after batch 8: {state['submission']} `{state['ckpt']}`" + (" (replaced)" if replaced else ""))
    videos(state["submission"], state["ckpt"], state["play"])
    log("## batch 8 done")


if __name__ == "__main__":
    main()
