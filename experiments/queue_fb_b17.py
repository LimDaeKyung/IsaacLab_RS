"""Batch 17 (share-fb, pre-registered 2026-10-02 10:27, user request 10:20: "넘어지지 않게끔 보완하는 쪽으로 학습").

SP50 scores highest but falls in 50-60 % of episodes. The scored reward has no fall penalty beyond the lost future
reward, so the fast gait trades falls for speed. Here a training-only fall penalty (removed in Play, so the scored
reward is unchanged) is added while continuing SP50 s{seed} for +1000 it on the same terrain/friction:
  C60  = no penalty (control, same budget; run already started by batch 16)
  FP20 = -20 per fall (weight -1200 x 1/60 s)
  FP50 = -50 per fall (weight -3000 x 1/60 s)
Fall metric: mean fall rate over the 8 lockbox-1/2 terrains ("unseen fall") and on flat ground.
Screen (s42/43) and confirm (s47-49), paired with the parent SP50 of the same seed:
  pass = unseen fall ≤ parent − 15 points AND unseen9 ≥ parent − 10 (confirm: fall lower in ≥ 2 of 3 seeds).
  C60 is judged with the batch-16 rule (score) and only reported here as the control.
Final (lockbox 6, untouched): SP50 s47 vs the best confirmed FP candidate s47 (return and fall rate) + official boxes.
  Replace the submission if lockbox6 ≥ SP50 − 5, official ≥ SP50 − 10 and lockbox-6 fall ≤ SP50 − 15 points;
  otherwise keep SP50 and report the candidate as the low-fall alternative.
Then the postponed batch-16 candidates (CF30, CH) run as batch 18 if time remains. No new training after 23:59.
"""

import glob
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight4 as r4  # noqa: E402
import queue_fb_b7 as b7  # noqa: E402
import queue_fb_b10 as b10  # noqa: E402
import queue_fb_b12 as b12  # noqa: E402  (sets the Fri 23:59 training guard)
import queue_fb_b14 as b14  # noqa: E402
import queue_fb_b15 as b15  # noqa: E402
import queue_fb_b16 as b16  # noqa: E402

ENT = b12.ENT
ALLMIX_PLAY = b12.ALLMIX_PLAY
T = "Isaac-Ant-R9-AllMix{}-v0"
CANDS = {"C60": T.format("WideMF04"), "FP20": T.format("WideMF04FP20"), "FP50": T.format("WideMF04FP50")}
for _n in CANDS:
    b12.MODELS[_n] = (_n.lower(), ALLMIX_PLAY, ENT)


def log(text):
    o.log(f"[fb] [b17] {text}")


def parent(s):
    return f"sp50_s{s}"


def unseen_fall(run):
    vals = [json.load(open(f))["fall_rate"] for f in glob.glob(f"experiments/eval_unseen8/{run}/*.json")]
    return st.mean(vals) if vals else float("nan")


def lock_fall(run, tag):
    vals = [json.load(open(f))["fall_rate"] for f in glob.glob(f"experiments/eval_{tag}/{run}/*.json")]
    return st.mean(vals) if vals else float("nan")


def run(name, seed):
    init = r4.ckpt_of(parent(seed))
    b10.wait_gpu()
    ck = b7.train(f"{name.lower()}_s{seed}", CANDS[name], seed, ENT, 1000, init=init) if init else None
    if ck is None:
        return None
    r = f"{name.lower()}_s{seed}"
    m = b12.u9(r, ALLMIX_PLAY, ENT)
    m["ufall"] = unseen_fall(r)
    log(b12.fmt(f"{name} s{seed}", m) + f", unseen fall {100 * m['ufall']:.0f}%")
    return m


def judge(ms, ref):
    seeds = sorted(ms)
    du = [ms[s]["unseen9"] - ref[s]["unseen9"] for s in seeds]
    dfall = [100 * (ms[s]["ufall"] - ref[s]["ufall"]) for s in seeds]
    dflat = [100 * (ms[s]["flat_fall"] - ref[s]["flat_fall"]) for s in seeds]
    ok = st.mean(dfall) <= -15 and st.mean(du) >= -10
    if len(seeds) == 3:
        ok = ok and sum(x < 0 for x in dfall) >= 2
    return ok, du, dfall, dflat


def main():
    while b12.busy(r"^(\S*/)?python3? experiments/queue_fb_b16.py") or b12.busy("run_fb_b16[.]sh"):
        time.sleep(60)
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    log(f"## batch 17 start (current submission {cur}; fall reduction on the SP50 line; final check lockbox6)")
    b14.log = log
    ref = {}
    for s in (42, 43, 47, 48, 49):
        ref[s] = b12.u9(parent(s), ALLMIX_PLAY, ENT)
        ref[s]["ufall"] = unseen_fall(parent(s))
        log(f"parent {parent(s)}: unseen9 {ref[s]['unseen9']:.1f}, unseen fall {100 * ref[s]['ufall']:.0f}%, flat fall "
            f"{100 * ref[s]['flat_fall']:.0f}%")

    passed = []
    for name in CANDS:
        ms = {s: run(name, s) for s in (42, 43)}
        if any(m is None for m in ms.values()):
            log(f"screen {name}: incomplete")
            continue
        ok, du, dfall, dflat = judge(ms, ref)
        if name == "C60":
            sok, d, dm, df, _ = b15.judge(name, ms, ref)
            log(f"control C60: unseen9 diff {st.mean(d):+.1f}, mult {st.mean(dm):+.1f}, unseen fall diff "
                f"{st.mean(dfall):+.0f} points, flat fall diff {st.mean(dflat):+.0f} → score rule {'PASS' if sok else 'fail'}")
            o.record({"batch": 17, "stage": "control", "exp": name, "diffs": d, "fall_diffs": dfall, "score_pass": sok})
            continue
        log(f"screen {name}: unseen9 diff {st.mean(du):+.1f} ({', '.join(f'{x:+.1f}' for x in du)}), unseen fall diff "
            f"{st.mean(dfall):+.0f} points, flat fall diff {st.mean(dflat):+.0f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 17, "stage": "screen", "exp": name, "diffs": du, "fall_diffs": dfall, "pass": ok})
        if ok:
            passed.append((st.mean(du), name))
    passed.sort(reverse=True)

    confirmed = []
    for _, name in passed:
        ms = {s: run(name, s) for s in (47, 48, 49)}
        ms = {s: m for s, m in ms.items() if m}
        if len(ms) < 3:
            log(f"confirm {name}: only {len(ms)} seeds")
            continue
        ok, du, dfall, dflat = judge(ms, ref)
        half = r4.T_975_DF2 * st.stdev(du) / 3 ** 0.5
        log(f"confirm {name} (s47–49): unseen9 diff {st.mean(du):+.1f} [95% CI {st.mean(du) - half:+.1f}, "
            f"{st.mean(du) + half:+.1f}], unseen fall diff {st.mean(dfall):+.0f} points ({', '.join(f'{x:+.0f}' for x in dfall)}), "
            f"flat fall diff {st.mean(dflat):+.0f} → {'PASSES' if ok else 'no'}")
        o.record({"batch": 17, "stage": "confirm", "exp": name, "diffs": du, "fall_diffs": dfall, "pass": ok})
        if ok:
            confirmed.append((st.mean(du), name))

    if confirmed:
        confirmed.sort(reverse=True)
        name = confirmed[0][1]
        cur_run = os.path.basename(os.path.dirname(state["ckpt"])).split("_", 2)[2]
        new_run = f"{name.lower()}_s47"
        l0, f0 = b14.lockbox(cur_run, ALLMIX_PLAY, ENT, b16.LOCK6, "lockbox6"), b15.official(cur_run)
        l1, f1 = b14.lockbox(new_run, ALLMIX_PLAY, ENT, b16.LOCK6, "lockbox6"), b15.official(new_run)
        k0, k1 = lock_fall(cur_run, "lockbox6"), lock_fall(new_run, "lockbox6")
        log(f"final {cur} ({cur_run}): lockbox6 {l0:.1f} (fall {100 * k0:.0f}%), official {f0:.1f}")
        log(f"final {name} ({new_run}): lockbox6 {l1:.1f} (fall {100 * k1:.0f}%), official {f1:.1f}")
        if l1 >= l0 - 5 and f1 >= f0 - 10 and 100 * (k1 - k0) <= -15:
            ck = r4.ckpt_of(new_run)
            state = {**state, "previous": cur, "submission": name, "play": ALLMIX_PLAY, "ckpt": ck,
                     "score_overrides": ENT, "lockbox6": l1, "lockbox6_fall": k1,
                     "official_play_one_episode_boxes": f1, "note": f"batch 17 final ({new_run})"}
            with open("experiments/fb_state.json", "w") as f:
                json.dump(state, f, indent=1)
            log(f"NEW SUBMISSION: {name} `{ck}` (fewer falls)")
        else:
            log(f"keep {cur}; {name} ({new_run}) is the low-fall alternative")
            state = {**state, "low_fall_alternative": {"name": name, "ckpt": r4.ckpt_of(new_run), "lockbox6": l1,
                                                       "lockbox6_fall": k1, "official": f1}}
            with open("experiments/fb_state.json", "w") as f:
                json.dump(state, f, indent=1)
    else:
        log("no fall-reduction candidate confirmed")
    log("## batch 17 done")


if __name__ == "__main__":
    main()
