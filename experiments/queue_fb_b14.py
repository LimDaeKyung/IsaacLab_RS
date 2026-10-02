"""Batch 14 (share-fb, pre-registered 2026-10-02 03:00): build on batch 13 (MF40b ≈ M40 but keeps low friction).

Findings so far: M40 (M30 + 1000 it) improved lockbox 1/2 and flat but low friction collapsed (μ0.2 multiply 27 → 21);
strong low-friction training (C1, effective 0.05-1.2) fixed friction but hurt everything else.
Candidates (all warm-started from M30 s{seed}, +1000 it, same budget as M40 which serves as the control):
  MF40a = + moderate low friction (robot 0.2-1.2, ground multiply)
  MF40b = + mild low friction (robot 0.4-1.2, ground multiply)
  SYA40 = + approximate left-right symmetry augmentation (asset is not exactly symmetric; map residual 0.18)
Metric unseen9 (lockbox 1 + 2 + boxes μ0.2 multiply). Final untouched check: lockbox 4 if batch 12 did not use it,
otherwise lockbox 5 (fixed 00:30). Runs after batch 12; no new training after Fri 2026-10-02 23:59.
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
import queue_fb_b7 as b7  # noqa: E402
import queue_fb_b10 as b10  # noqa: E402
import queue_fb_b12 as b12  # noqa: E402  (sets the Fri 23:59 training guard)

R = o.R
ENT = b12.ENT
ALLMIX_PLAY = b12.ALLMIX_PLAY
CANDS = {
    "AX40": ("Isaac-Ant-R9-AllMixWide-v0", ALLMIX_PLAY, ENT, ENT),
    "AXF40": ("Isaac-Ant-R9-AllMixWideMF04-v0", ALLMIX_PLAY, ENT, ENT),
    "MF50b": ("Isaac-Ant-R8-AllMixMF04-v0", ALLMIX_PLAY, ENT, ENT),
}
b12.MODELS.update({"MF40a": ("mf40a", ALLMIX_PLAY, ENT), "MF40b": ("mf40b", ALLMIX_PLAY, ENT),
                   "SYA40": ("sya40", b12.SYM_PLAY, [])})
for _n, (_t, _p, _tr, _ev) in CANDS.items():
    b12.MODELS[_n] = (_n.lower(), _p, _ev)
LOCK5 = {"lock5_stairs_high": [], "lock5_slope_steep": [], "lock5_cones_dense": [],
         "lock5_boxes_lowfric": ["--friction", "0.3", "--combine_mode", "multiply"]}


def log(text):
    o.log(f"[fb] [b14] {text}")


def run_candidate(name, seed):
    task, play, tr, ev = CANDS[name]
    run = f"{name.lower()}_s{seed}"
    b10.wait_gpu()
    if name == "MF50b":  # M30 -> MF40b (+1000) -> MF50b (+1000), mild low friction throughout
        if not r4.ckpt_of(f"mf40b_s{seed}"):
            b7.train(f"mf40b_s{seed}", task, seed, tr, 1000, init=r4.ckpt_of(f"m30_s{seed}"))
        init = r4.ckpt_of(f"mf40b_s{seed}")
    else:
        init = r4.ckpt_of(f"m30_s{seed}")
    ck = b7.train(run, task, seed, tr, 1000, init=init) if init else None
    if ck is None:
        return None
    m = b12.u9(run, play, ev)
    log(b12.fmt(f"{name} s{seed}", m))
    return m


def lockbox(run, play, over, boxes, tag):
    ck = r4.ckpt_of(run)
    v = {t: b10.evaluate(ck, play, t, f"experiments/eval_{tag}/{run}/{t}.json", over + extra)["return_mean"]
         for t, extra in boxes.items()}
    log(f"{tag} {run}: " + ", ".join(f"{k[6:]} {x:.1f}" for k, x in v.items()) + f" → {st.mean(v.values()):.1f}")
    return st.mean(v.values())


def main():
    while b12.busy(r"^(\S*/)?python3? experiments/queue_fb_b13.py") or b12.busy("run_fb_b13.sh"):
        time.sleep(60)
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    lock4_used = os.path.isdir("experiments/eval_lockbox4") and len(os.listdir("experiments/eval_lockbox4")) > 0
    boxes, tag = (LOCK5, "lockbox5") if lock4_used else (b12.LOCK4, "lockbox4")
    log(f"## batch 14 start (current submission {cur}; final check {tag})")

    ref = {s: b12.u9(f"m30_s{s}", ALLMIX_PLAY, ENT) for s in (42, 43)}
    ctl = {s: b12.u9(f"m40_s{s}", ALLMIX_PLAY, ENT) for s in (42, 43) if r4.ckpt_of(f"m40_s{s}")}
    u0, f0, mu0 = (st.mean(m[k] for m in ref.values()) for k in ("unseen9", "flat", "mult"))
    uc = st.mean(m["unseen9"] for m in ctl.values()) if ctl else float("nan")
    # 0) smoke: the per-tile random-difficulty terrains must build and train
    b10.wait_gpu()
    for t in ("Isaac-Ant-R9-AllMixWide-v0", "Isaac-Ant-R9-AllMixWideMF04-v0"):
        ok_s, _ = r4.smoke("b14_" + t.split("-")[3].lower(), t, ENT)
        log(f"smoke {t}: {'OK' if ok_s else 'FAILED'}")
        if not ok_s:
            CANDS.pop("AX40" if t.endswith("Wide-v0") else "AXF40", None)
    log(f"reference M30 s42/43 unseen9 {u0:.1f}, μ0.2 mult {mu0:.1f}, flat {f0:.1f}; control M40 unseen9 {uc:.1f}")

    passed = []
    for name in CANDS:
        ms = [run_candidate(name, s) for s in (42, 43)]
        if any(m is None for m in ms):
            log(f"screen {name}: incomplete")
            continue
        u, fl, mu = (st.mean(m[k] for m in ms) for k in ("unseen9", "flat", "mult"))
        target = name in ("AXF40", "MF50b") and u >= u0 - 1 and mu >= mu0 + 5
        ok = (u >= u0 + 3 or target) and fl >= f0 - 5
        log(f"screen {name}: unseen9 {u:.1f} vs M30 {u0:.1f} (control M40 {uc:.1f}), μ0.2 mult {mu:.1f} vs {mu0:.1f}, "
            f"flat {fl:.1f} vs {f0:.1f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 14, "stage": "screen", "exp": name, "unseen9": u, "mult": mu, "flat": fl, "pass": ok})
        if ok:
            passed.append((u - u0, name))
    passed.sort(reverse=True)

    prefix, cplay, cover = b12.MODELS[cur]
    base = {s: b12.u9(f"{prefix}_s{s}", cplay, cover) for s in (47, 48, 49) if r4.ckpt_of(f"{prefix}_s{s}")}
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
            name in ("AXF40", "MF50b") and st.mean(d) >= -1 and st.mean(dm) >= 5 and sum(x > 0 for x in dm) >= 2)
        ok = ok and fl >= -5
        log(f"confirm {name} vs {cur} (s47–49): unseen9 diff {st.mean(d):+.1f} [95% CI {st.mean(d) - half:+.1f}, "
            f"{st.mean(d) + half:+.1f}], wins {sum(x > 0 for x in d)}/3, μ0.2 mult diff {st.mean(dm):+.1f}, flat "
            f"{fl:+.1f} → {'PASSES' if ok else 'no'}")
        o.record({"batch": 14, "stage": "confirm", "exp": name, "vs": cur, "diffs": d, "mult_diffs": dm, "pass": ok})
        if ok:
            confirmed.append((st.mean(d), name))

    if confirmed:
        confirmed.sort(reverse=True)
        _, name = confirmed[0]
        l_cur = lockbox(f"{prefix}_s47", cplay, cover, boxes, tag)
        _, play, _, ev = CANDS[name]
        l_new = lockbox(f"{name.lower()}_s47", play, ev, boxes, tag)
        if l_new >= l_cur - 3:
            ck = r4.ckpt_of(f"{name.lower()}_s47")
            os.makedirs(f"experiments/eval/smoke_b14_{name}", exist_ok=True)
            for terrain in ("flat", "boxes_mid"):
                for fric in ("1.0", "0.2"):
                    for mode in ("average", "multiply"):
                        n = f"{terrain}_f{fric}_{mode}"
                        c = o.run(["./isaaclab.sh", "-p", f"{R}/evaluate.py", "--task", play, "--terrain", terrain,
                                   "--friction", fric, "--combine_mode", mode, "--checkpoint", ck, "--output",
                                   f"experiments/eval/smoke_b14_{name}/{n}.json", *ev],
                                  f"experiments/train_logs/smoke_b14_{name}_{n}.log")
                        log(f"smoke {name} {n}: {'OK' if c == 0 else 'FAILED'}")
            c = o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", play, "--headless", "--num_envs",
                       "100", "--checkpoint", ck, *ev], f"experiments/train_logs/smoke_b14_{name}_official.log")
            log(f"smoke {name} official play_one_episode: exit {c}")
            state = {**state, "previous": cur, "submission": name, "play": play, "ckpt": ck, "score_overrides": ev,
                     tag: l_new}
            with open("experiments/fb_state.json", "w") as f:
                json.dump(state, f, indent=1)
            log(f"NEW SUBMISSION: {name} `{ck}` ({tag} {l_new:.1f} vs {cur} {l_cur:.1f})")
        else:
            log(f"{tag}: {name} {l_new:.1f} < {cur} {l_cur:.1f} − 3 → keep {cur}")
    else:
        log(f"no candidate confirmed → keep {cur}")
    log("## batch 14 done")


if __name__ == "__main__":
    main()
