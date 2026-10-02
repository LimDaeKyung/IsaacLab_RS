"""Batch 19 (share-fb, pre-registered 2026-10-02 14:14, before the FPU results and before any lockbox-7/8 result).

Remaining weak points of the fall-penalised line (FP20): low friction (μ0.2 multiply 18-27, FPL even -3) and tall
bumps / stones. Batch 16 meant to test exactly these two (CF30, CH) but was stopped before any result for the
fall-reduction request; here they run on the current line, keeping its fall penalty:
  FPF = parent + 1000 it, friction 0.3-1.2 (instead of 0.4-1.2)
  FPH = parent + 1000 it, bump heights raised by one third (boxes ≤ 20 cm, boxes_fine/obstacles ≤ 16 cm, wave ≤ 25 cm)
Parent line = current submission after batch 18 (FP20 -> fp20_s{seed} with -20 per fall, FPU -> fpu_s{seed} with -30).
Rules as in batch 18 (paired with the parent; score way or fall way). Final: lockbox 7 if batch 18 did not use it,
else lockbox 8 (fixed 14:12). No new training after Fri 2026-10-02 23:59.
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
import queue_fb_b18 as b18  # noqa: E402

ENT = b12.ENT
ALLMIX_PLAY = b12.ALLMIX_PLAY
T = "Isaac-Ant-R9-AllMix{}-v0"
LINES = {"FP20": ("fp20", "20"), "FPU": ("fpu", "30")}
LOCK8 = {"lock8_boxes_tall_sparse": [], "lock8_stairs_down_hf": [], "lock8_stones_wide": [],
         "lock8_slope_lowfric": ["--friction", "0.5", "--combine_mode", "multiply"]}


def log(text):
    o.log(f"[fb] [b19] {text}")


def main():
    while b12.busy(r"^(\S*/)?python3? experiments/queue_fb_b18.py"):
        time.sleep(60)
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    if cur not in LINES:
        log(f"## batch 19 not started: unknown submission {cur}")
        return
    prefix, pen = LINES[cur]
    cands = {"FPF": T.format(f"WideMF03FP{pen}"), "FPH": T.format(f"HardMF04FP{pen}")}
    used7 = os.path.isdir("experiments/eval_lockbox7") and len(os.listdir("experiments/eval_lockbox7")) > 0
    boxes, tag = (LOCK8, "lockbox8") if used7 else (b18.LOCK7, "lockbox7")
    log(f"## batch 19 start (line {cur} = {prefix}_s{{s}}, penalty -{pen}/fall; candidates {cands}; final {tag})")
    b14.log = log
    for n in cands:
        b12.MODELS[n] = (n.lower(), ALLMIX_PLAY, ENT)

    def parent(s):
        return f"{prefix}_s{s}"

    ref = {s: b18.metrics(parent(s)) for s in (42, 43, 47, 48, 49) if r4.ckpt_of(parent(s))}
    for s, m in ref.items():
        log(f"parent {parent(s)}: unseen9 {m['unseen9']:.1f}, mult {m['mult']:.1f}, unseen fall {100 * m['ufall']:.0f}%")

    def run(name, seed):
        init = r4.ckpt_of(parent(seed))
        b10.wait_gpu()
        ck = b7.train(f"{name.lower()}_s{seed}", cands[name], seed, ENT, 1000, init=init) if init else None
        if ck is None:
            return None
        m = b18.metrics(f"{name.lower()}_s{seed}")
        log(b12.fmt(f"{name} s{seed}", m) + f", unseen fall {100 * m['ufall']:.0f}%")
        return m

    passed = []
    for name in cands:
        ms = {s: run(name, s) for s in (42, 43)}
        if any(m is None for m in ms.values()) or any(s not in ref for s in ms):
            log(f"screen {name}: incomplete")
            continue
        ok, du, dfall = b18.judge(ms, ref)
        dm = st.mean(ms[s]["mult"] - ref[s]["mult"] for s in ms)
        log(f"screen {name}: unseen9 diff {st.mean(du):+.1f} ({', '.join(f'{x:+.1f}' for x in du)}), unseen fall diff "
            f"{st.mean(dfall):+.0f} points, μ0.2 mult diff {dm:+.1f} → {'PASS' if ok else 'fail'}")
        o.record({"batch": 19, "stage": "screen", "exp": name, "diffs": du, "fall_diffs": dfall, "pass": ok})
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
        ok, du, dfall = b18.judge(ms, ref)
        half = r4.T_975_DF2 * st.stdev(du) / 3 ** 0.5
        log(f"confirm {name} (s47–49): unseen9 diff {st.mean(du):+.1f} [95% CI {st.mean(du) - half:+.1f}, "
            f"{st.mean(du) + half:+.1f}], unseen fall diff {st.mean(dfall):+.0f} points → {'PASSES' if ok else 'no'}")
        o.record({"batch": 19, "stage": "confirm", "exp": name, "diffs": du, "fall_diffs": dfall, "pass": ok})
        if ok:
            confirmed.append((key, name))

    if not confirmed:
        log(f"no candidate confirmed → keep {cur}")
        log("## batch 19 done")
        return
    name = max(confirmed)[1]
    cur_run = os.path.basename(os.path.dirname(state["ckpt"])).split("_", 2)[2]
    new_run = f"{name.lower()}_s47"
    res = {}
    for n, r in ((cur, cur_run), (name, new_run)):
        res[n] = (b14.lockbox(r, ALLMIX_PLAY, ENT, boxes, tag), b17.lock_fall(r, tag), b15.official(r))
        log(f"final {n} ({r}): {tag} {res[n][0]:.1f} (fall {100 * res[n][1]:.0f}%), official {res[n][2]:.1f}")
    (l0, k0, f0), (l1, k1, f1) = res[cur], res[name]
    dk = 100 * (k1 - k0)
    if f1 >= f0 - 10 and ((l1 >= l0 + 3 and dk <= 5) or (dk <= -10 and l1 >= l0 - 5)):
        ck = r4.ckpt_of(new_run)
        state = {**state, "previous": cur, "submission": name, "play": ALLMIX_PLAY, "ckpt": ck, "score_overrides": ENT,
                 tag: l1, f"{tag}_fall": k1, "official_play_one_episode_boxes": f1,
                 "note": f"batch 19 final ({new_run})"}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(state, f, indent=1)
        log(f"NEW SUBMISSION: {name} `{ck}` ({tag} {l1:.1f} vs {l0:.1f}, fall {dk:+.0f} points, official {f1:.1f})")
    else:
        log(f"keep {cur}: {name} {tag} {l1:.1f} vs {l0:.1f}, fall {dk:+.0f} points, official {f1:.1f} vs {f0:.1f}")
    log("## batch 19 done")


if __name__ == "__main__":
    main()
