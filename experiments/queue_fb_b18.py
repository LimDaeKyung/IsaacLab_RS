"""Batch 18 (share-fb, pre-registered 2026-10-02 12:10, before the FP20 confirm and lockbox-6 results).

Batch 17 screen: FP20 (-20 per fall) cut the unseen fall rate by 20 points without losing score (+2.9); FP50 broke the
gait (-49). The control C60 (+1000 it, no penalty) gained +17 score but barely fewer falls (-7 points). So more
training raises the score and the -20 penalty lowers falls; combine them by continuing the FP20 line:
  FPL = fp20_s{seed} + 1000 it, penalty -20 per fall   (longer training with the same penalty)
  FPU = fp20_s{seed} + 1000 it, penalty -30 per fall   (a gentle step up after -20; -50 at once failed)
Paired with the parent fp20 of the same seed. Pass if
  (unseen9 ≥ parent + 3 and unseen fall ≤ parent + 5 points) or (unseen fall ≤ parent − 10 points and unseen9 ≥ parent − 5)
(confirm s47-49: the same on the mean, and the deciding quantity improves in ≥ 2 of 3 seeds).
Final (lockbox 7, fixed 12:08): current submission vs the best confirmed candidate s47 (score, fall rate, official boxes).
Replace if official ≥ current − 10 and [(lockbox7 ≥ current + 3 and lockbox7 fall ≤ current + 5 points) or
(lockbox7 fall ≤ current − 10 points and lockbox7 ≥ current − 5)]. No new training after Fri 2026-10-02 23:59.
"""

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
import queue_fb_b17 as b17  # noqa: E402

ENT = b12.ENT
ALLMIX_PLAY = b12.ALLMIX_PLAY
T = "Isaac-Ant-R9-AllMix{}-v0"
CANDS = {"FPL": T.format("WideMF04FP20"), "FPU": T.format("WideMF04FP30")}
LOCK7 = {"lock7_gaps_wide": [], "lock7_pyramids_high": [], "lock7_wave_short": [],
         "lock7_rough_lowfric": ["--friction", "0.4", "--combine_mode", "multiply"]}
for _n in CANDS:
    b12.MODELS[_n] = (_n.lower(), ALLMIX_PLAY, ENT)


def log(text):
    o.log(f"[fb] [b18] {text}")


def parent(s):
    return f"fp20_s{s}"


def metrics(run):
    m = b12.u9(run, ALLMIX_PLAY, ENT)
    m["ufall"] = b17.unseen_fall(run)
    return m


def run(name, seed):
    init = r4.ckpt_of(parent(seed))
    b10.wait_gpu()
    ck = b7.train(f"{name.lower()}_s{seed}", CANDS[name], seed, ENT, 1000, init=init) if init else None
    if ck is None:
        return None
    m = metrics(f"{name.lower()}_s{seed}")
    log(b12.fmt(f"{name} s{seed}", m) + f", unseen fall {100 * m['ufall']:.0f}%")
    return m


def judge(ms, ref):
    seeds = sorted(ms)
    du = [ms[s]["unseen9"] - ref[s]["unseen9"] for s in seeds]
    dfall = [100 * (ms[s]["ufall"] - ref[s]["ufall"]) for s in seeds]
    score_way = st.mean(du) >= 3 and st.mean(dfall) <= 5
    fall_way = st.mean(dfall) <= -10 and st.mean(du) >= -5
    if len(seeds) == 3:
        score_way = score_way and sum(x > 0 for x in du) >= 2
        fall_way = fall_way and sum(x < 0 for x in dfall) >= 2
    return score_way or fall_way, du, dfall


def main():
    while b12.busy(r"^(\S*/)?python3? experiments/queue_fb_b17.py") or b12.busy("run_fb_b17[.]sh"):
        time.sleep(60)
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    log(f"## batch 18 start (current submission {cur}; continue the FP20 line; final check lockbox7)")
    b14.log = log
    ref = {}
    for s in (42, 43, 47, 48, 49):
        if r4.ckpt_of(parent(s)):
            ref[s] = metrics(parent(s))
            log(f"parent {parent(s)}: unseen9 {ref[s]['unseen9']:.1f}, unseen fall {100 * ref[s]['ufall']:.0f}%")

    passed = []
    for name in CANDS:
        ms = {s: run(name, s) for s in (42, 43)}
        if any(m is None for m in ms.values()) or any(s not in ref for s in ms):
            log(f"screen {name}: incomplete")
            continue
        ok, du, dfall = judge(ms, ref)
        log(f"screen {name}: unseen9 diff {st.mean(du):+.1f} ({', '.join(f'{x:+.1f}' for x in du)}), unseen fall diff "
            f"{st.mean(dfall):+.0f} points → {'PASS' if ok else 'fail'}")
        o.record({"batch": 18, "stage": "screen", "exp": name, "diffs": du, "fall_diffs": dfall, "pass": ok})
        if ok:
            passed.append((st.mean(du) - st.mean(dfall) / 2, name))
    passed.sort(reverse=True)

    confirmed = []
    for key, name in passed:
        ms = {s: run(name, s) for s in (47, 48, 49)}
        ms = {s: m for s, m in ms.items() if m and s in ref}
        if len(ms) < 3:
            log(f"confirm {name}: only {len(ms)} seeds")
            continue
        ok, du, dfall = judge(ms, ref)
        half = r4.T_975_DF2 * st.stdev(du) / 3 ** 0.5
        log(f"confirm {name} (s47–49): unseen9 diff {st.mean(du):+.1f} [95% CI {st.mean(du) - half:+.1f}, "
            f"{st.mean(du) + half:+.1f}], unseen fall diff {st.mean(dfall):+.0f} points "
            f"({', '.join(f'{x:+.0f}' for x in dfall)}) → {'PASSES' if ok else 'no'}")
        o.record({"batch": 18, "stage": "confirm", "exp": name, "diffs": du, "fall_diffs": dfall, "pass": ok})
        if ok:
            confirmed.append((key, name))

    if not confirmed:
        log(f"no candidate confirmed → keep {cur}")
        log("## batch 18 done")
        return
    name = max(confirmed)[1]
    cur_run = os.path.basename(os.path.dirname(state["ckpt"])).split("_", 2)[2]
    new_run = f"{name.lower()}_s47"
    res = {}
    for n, r in ((cur, cur_run), (name, new_run)):
        res[n] = (b14.lockbox(r, ALLMIX_PLAY, ENT, LOCK7, "lockbox7"), b17.lock_fall(r, "lockbox7"), b15.official(r))
        log(f"final {n} ({r}): lockbox7 {res[n][0]:.1f} (fall {100 * res[n][1]:.0f}%), official {res[n][2]:.1f}")
    (l0, k0, f0), (l1, k1, f1) = res[cur], res[name]
    dk = 100 * (k1 - k0)
    if f1 >= f0 - 10 and ((l1 >= l0 + 3 and dk <= 5) or (dk <= -10 and l1 >= l0 - 5)):
        ck = r4.ckpt_of(new_run)
        state = {**state, "previous": cur, "submission": name, "play": ALLMIX_PLAY, "ckpt": ck, "score_overrides": ENT,
                 "lockbox7": l1, "lockbox7_fall": k1, "official_play_one_episode_boxes": f1,
                 "note": f"batch 18 final ({new_run})"}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(state, f, indent=1)
        log(f"NEW SUBMISSION: {name} `{ck}` (lockbox7 {l1:.1f} vs {l0:.1f}, fall {dk:+.0f} points, official {f1:.1f})")
    else:
        log(f"keep {cur}: {name} lockbox7 {l1:.1f} vs {l0:.1f}, fall {dk:+.0f} points, official {f1:.1f} vs {f0:.1f}")
    log("## batch 18 done")


if __name__ == "__main__":
    main()
