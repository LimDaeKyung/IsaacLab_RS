"""Batch 16 (share-fb, pre-registered 2026-10-02 08:13, before the batch-15 confirm and lockbox-5 results).

Continues whatever line batch 15 leaves as the submission (SP50, SPF20, axf40_s49 or AXF40), +1000 it per candidate,
each seed warm-started from the same seed of that line, so every comparison is paired with its own parent:
  C60  = same training environment as the parent line (control: is more training still useful?)
  CF30 = friction 0.3-1.2 (between the mild 0.4 and the moderate 0.2 settings; low friction is the main weakness)
  CH   = bump heights raised by one third (boxes ≤ 20 cm, boxes_fine/obstacles ≤ 16 cm, wave ≤ 25 cm; tall boxes and
         stones are weak points), friction as in the parent line
Rules as in batch 15. Final check: lockbox 6 (fixed 08:12) on the current submission and the confirmed candidates
(s47), plus the official boxes score; replace only with lockbox 6 ≥ current + 3 and official ≥ current.
No new training after Fri 2026-10-02 23:59.
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

ENT = b12.ENT
ALLMIX_PLAY = b12.ALLMIX_PLAY
T = "Isaac-Ant-R9-AllMix{}-v0"
LOCK6 = {"lock6_stairs_down": [], "lock6_slope_down": [], "lock6_rough_heavy": [],
         "lock6_boxobst_midfric": ["--friction", "0.5", "--combine_mode", "average"]}
# submission -> (parent run for seed s, friction suffix of the parent's training environment)
LINES = {
    "SP50": (lambda s: f"sp50_s{s}", "MF04"),
    "SPF20": (lambda s: f"spf20_s{s}", "MF02"),
    "AXF40_s49": (lambda s: "axf40_s49", "MF04"),
    "AXF40": (lambda s: f"axf40_s{s}", "MF04"),
}


def log(text):
    o.log(f"[fb] [b16] {text}")


def main():
    while b12.busy(r"^(\S*/)?python3? experiments/queue_fb_b15.py") or b12.busy("run_fb_b15.sh"):
        time.sleep(60)
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    if cur not in LINES:
        log(f"## batch 16 not started: unknown submission {cur}")
        return
    parent, fr = LINES[cur]
    cands = {"C60": T.format(f"Wide{fr}"), "CF30": T.format("WideMF03"), "CH": T.format(f"Hard{fr}")}
    log(f"## batch 16 start (line {cur}, parent {parent('{s}')}; candidates {cands}; final check lockbox6)")
    b14.log = log
    for n in cands:
        b12.MODELS[n] = (n.lower(), ALLMIX_PLAY, ENT)

    ref = {s: b12.u9(parent(s), ALLMIX_PLAY, ENT) for s in (42, 43, 47, 48, 49) if r4.ckpt_of(parent(s))}
    for s, m in ref.items():
        log(b12.fmt(f"parent {parent(s)}", m))

    def run(name, seed):
        init = r4.ckpt_of(parent(seed))
        b10.wait_gpu()
        ck = b7.train(f"{name.lower()}_s{seed}", cands[name], seed, ENT, 1000, init=init) if init else None
        if ck is None:
            return None
        m = b12.u9(f"{name.lower()}_s{seed}", ALLMIX_PLAY, ENT)
        log(b12.fmt(f"{name} s{seed}", m))
        return m

    passed = []
    for name in cands:
        ms = {s: run(name, s) for s in (42, 43)}
        if any(m is None for m in ms.values()) or any(s not in ref for s in ms):
            log(f"screen {name}: incomplete")
            continue
        ok, d, dm, df, falls = b15.judge(name, ms, ref)
        log(f"screen {name}: unseen9 diff {st.mean(d):+.1f} ({', '.join(f'{x:+.1f}' for x in d)}), μ0.2 mult diff "
            f"{st.mean(dm):+.1f}, flat {df:+.1f}, flat fall {100 * falls:.0f}% → {'PASS' if ok else 'fail'}")
        o.record({"batch": 16, "stage": "screen", "exp": name, "diffs": d, "mult_diffs": dm, "flat_diff": df,
                  "pass": ok})
        if ok:
            passed.append((st.mean(d), name))
    passed.sort(reverse=True)

    confirmed = []
    for _, name in passed[:2]:
        ms = {s: run(name, s) for s in (47, 48, 49)}
        ms = {s: m for s, m in ms.items() if m and s in ref}
        if len(ms) < 3:
            log(f"confirm {name}: only {len(ms)} seeds")
            continue
        ok, d, dm, df, falls = b15.judge(name, ms, ref)
        half = r4.T_975_DF2 * st.stdev(d) / 3 ** 0.5
        log(f"confirm {name} (s47–49): unseen9 diff {st.mean(d):+.1f} [95% CI {st.mean(d) - half:+.1f}, "
            f"{st.mean(d) + half:+.1f}], μ0.2 mult diff {st.mean(dm):+.1f}, flat {df:+.1f}, flat fall "
            f"{100 * falls:.0f}% → {'PASSES' if ok else 'no'}")
        o.record({"batch": 16, "stage": "confirm", "exp": name, "diffs": d, "mult_diffs": dm, "pass": ok})
        if ok:
            confirmed.append(name)

    if not confirmed:
        log(f"no candidate confirmed → keep {cur}")
        log("## batch 16 done")
        return
    cur_run = os.path.basename(os.path.dirname(state["ckpt"])).split("_", 2)[2]
    pool = {cur: cur_run, **{n: f"{n.lower()}_s47" for n in confirmed}}
    res = {n: (b14.lockbox(r, ALLMIX_PLAY, ENT, LOCK6, "lockbox6"), b15.official(r)) for n, r in pool.items()}
    for n, (lb, of) in res.items():
        log(f"final {n} ({pool[n]}): lockbox6 {lb:.1f}, official boxes {of:.1f}")
    l0, f0 = res[cur]
    best = max((n for n in res if n != cur), key=lambda n: res[n][0])
    lb, of = res[best]
    if lb >= l0 + 3 and of >= f0:
        ck = r4.ckpt_of(pool[best])
        state = {**state, "previous": cur, "submission": f"{cur}+{best}", "play": ALLMIX_PLAY, "ckpt": ck,
                 "score_overrides": ENT, "lockbox6": lb, "official_play_one_episode_boxes": of,
                 "note": f"batch 16 final ({pool[best]})"}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(state, f, indent=1)
        log(f"NEW SUBMISSION: {cur}+{best} `{ck}` (lockbox6 {lb:.1f} vs {l0:.1f}; official {of:.1f} vs {f0:.1f})")
    else:
        log(f"keep {cur}: best {best} lockbox6 {lb:.1f} vs {l0:.1f} (+3 needed), official {of:.1f} vs {f0:.1f}")
    log("## batch 16 done")


if __name__ == "__main__":
    main()
