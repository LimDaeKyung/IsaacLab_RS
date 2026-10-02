"""Batch 15 (share-fb, pre-registered 2026-10-02 06:22): the fast gait of axf40_s49 and the low-friction weakness.

Finding after batch 14: axf40_s49 learned a fast gait (flat 10.4 m/s) that falls often (58-82 %) but scores much
higher with our reward than the careful gait of axf40_s47 (unseen9 76.9 vs 69.5, lockbox 4 63.1 vs 45.6, official
boxes play_one_episode 86.9 vs 57.1). Its only clear loss is low friction (μ0.2 multiply 15.4, wave_lowfric 18.4).
Candidates (+1000 it each, AllMix-Wide terrain):
  SP50  = axf40_s49 + 1000 it, friction 0.4-1.2   (does the fast gait survive and improve when trained further?)
  SPF20 = axf40_s49 + 1000 it, friction 0.2-1.2   (fast gait + its low-friction weakness)
  AXF20 = axf40_s{seed} + 1000 it, friction 0.2-1.2 (careful line + low friction; compared with AXF40 per seed)
SP lines share one parent, so their seeds differ only in training noise; they are compared with the parent's fixed
score. Final check: lockbox 5 (untouched) on the current submission, axf40_s49 and every confirmed candidate (s47).
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

R = o.R
ENT = b12.ENT
ALLMIX_PLAY = b12.ALLMIX_PLAY
MF04, MF02 = "Isaac-Ant-R9-AllMixWideMF04-v0", "Isaac-Ant-R9-AllMixWideMF02-v0"
CANDS = {  # name: (task, parent run for seed s)
    "SP50": (MF04, lambda s: "axf40_s49"),
    "SPF20": (MF02, lambda s: "axf40_s49"),
    "AXF20": (MF02, lambda s: f"axf40_s{s}"),
}
for _n in CANDS:
    b12.MODELS[_n] = (_n.lower(), ALLMIX_PLAY, ENT)
b12.MODELS["AXF40"] = ("axf40", ALLMIX_PLAY, ENT)
LOCK5 = b14.LOCK5


def log(text):
    o.log(f"[fb] [b15] {text}")


def run_candidate(name, seed):
    task, parent = CANDS[name]
    run = f"{name.lower()}_s{seed}"
    init = r4.ckpt_of(parent(seed))
    b10.wait_gpu()
    ck = b7.train(run, task, seed, ENT, 1000, init=init) if init else None
    if ck is None:
        return None
    m = b12.u9(run, ALLMIX_PLAY, ENT)
    log(b12.fmt(f"{name} s{seed}", m))
    return m


def judge(name, ms, ref):
    """Same rule for screen (2 seeds) and confirm (3 seeds); ref holds per-seed reference metrics."""
    seeds = sorted(ms)
    d = [ms[s]["unseen9"] - ref[s]["unseen9"] for s in seeds]
    dm = [ms[s]["mult"] - ref[s]["mult"] for s in seeds]
    df = st.mean(ms[s]["flat"] for s in seeds) - st.mean(ref[s]["flat"] for s in seeds)
    need = 1 if len(seeds) == 2 else 2  # screen: mean rule (as in batches 13-14); confirm: 2 of 3 seeds
    ok = ((st.mean(d) >= 3 and sum(x > 0 for x in d) >= need)
          or (st.mean(d) >= -1 and st.mean(dm) >= 5 and sum(x > 0 for x in dm) >= need)) and df >= -5
    falls = st.mean(ms[s]["flat_fall"] for s in seeds)
    return ok, d, dm, df, falls


def lockbox(run):
    return b14.lockbox(run, ALLMIX_PLAY, ENT, LOCK5, "lockbox5")


def official(run):
    out = f"experiments/train_logs/official_{run}.log"
    if not os.path.exists(out):
        b10.wait_gpu()
        o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", ALLMIX_PLAY, "--headless", "--num_envs",
               "100", "--checkpoint", r4.ckpt_of(run), *ENT], out)
    for line in open(out):
        if "Episode reward total" in line:
            return float(line.split("mean=")[1].split(",")[0])
    return float("nan")


def main():
    while b12.busy(r"^(\S*/)?python3? experiments/queue_fb_b14.py") or b12.busy("run_fb_b14.sh"):
        time.sleep(60)
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]
    log(f"## batch 15 start (current submission {cur}; final check lockbox5)")
    b14.log = log  # lockbox() logs under this batch

    parent = b12.u9("axf40_s49", ALLMIX_PLAY, ENT)
    axf = {s: b12.u9(f"axf40_s{s}", ALLMIX_PLAY, ENT) for s in (42, 43, 47, 48, 49)}
    log(f"reference axf40_s49 (SP parent) unseen9 {parent['unseen9']:.1f}, mult {parent['mult']:.1f}, flat "
        f"{parent['flat']:.1f} (fall {100 * parent['flat_fall']:.0f}%)")

    def ref_for(name, seeds):
        return {s: parent for s in seeds} if name.startswith("SP") else {s: axf[s] for s in seeds}

    passed = []
    for name in CANDS:
        ms = {s: run_candidate(name, s) for s in (42, 43)}
        if any(m is None for m in ms.values()):
            log(f"screen {name}: incomplete")
            continue
        ok, d, dm, df, falls = judge(name, ms, ref_for(name, ms))
        log(f"screen {name}: unseen9 diff {st.mean(d):+.1f} ({', '.join(f'{x:+.1f}' for x in d)}), μ0.2 mult diff "
            f"{st.mean(dm):+.1f}, flat {df:+.1f}, flat fall {100 * falls:.0f}% → {'PASS' if ok else 'fail'}")
        o.record({"batch": 15, "stage": "screen", "exp": name, "diffs": d, "mult_diffs": dm, "flat_diff": df,
                  "pass": ok})
        if ok:
            passed.append((st.mean(d), name))
    passed.sort(reverse=True)

    confirmed = []
    for _, name in passed[:2]:
        ms = {s: run_candidate(name, s) for s in (47, 48, 49)}
        ms = {s: m for s, m in ms.items() if m}
        if len(ms) < 3:
            log(f"confirm {name}: only {len(ms)} seeds")
            continue
        ok, d, dm, df, falls = judge(name, ms, ref_for(name, ms))
        half = r4.T_975_DF2 * st.stdev(d) / 3 ** 0.5
        log(f"confirm {name} (s47–49): unseen9 diff {st.mean(d):+.1f} [95% CI {st.mean(d) - half:+.1f}, "
            f"{st.mean(d) + half:+.1f}], μ0.2 mult diff {st.mean(dm):+.1f}, flat {df:+.1f}, flat fall "
            f"{100 * falls:.0f}% → {'PASSES' if ok else 'no'}")
        o.record({"batch": 15, "stage": "confirm", "exp": name, "diffs": d, "mult_diffs": dm, "pass": ok})
        if ok:
            confirmed.append(name)

    # final: lockbox 5 + official boxes score; replace only with a clear lockbox-5 margin (+3) and official ≥ current
    cur_run = os.path.basename(os.path.dirname(state["ckpt"])).split("_", 2)[2]
    pool = {cur: cur_run, "AXF40_s49": "axf40_s49", **{n: f"{n.lower()}_s47" for n in confirmed}}
    res = {n: (lockbox(r), official(r)) for n, r in pool.items()}
    for n, (lb, of) in res.items():
        log(f"final {n} ({pool[n]}): lockbox5 {lb:.1f}, official boxes {of:.1f}")
    l0, f0 = res[cur]
    best = max((n for n in res if n != cur), key=lambda n: res[n][0])
    lb, of = res[best]
    if lb >= l0 + 3 and of >= f0:
        run = pool[best]
        ck = r4.ckpt_of(run)
        state = {**state, "previous": cur, "submission": best, "play": ALLMIX_PLAY, "ckpt": ck, "score_overrides": ENT,
                 "lockbox5": lb, "official_play_one_episode_boxes": of, "note": f"batch 15 final ({run})"}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(state, f, indent=1)
        log(f"NEW SUBMISSION: {best} `{ck}` (lockbox5 {lb:.1f} vs {cur} {l0:.1f}; official {of:.1f} vs {f0:.1f})")
    else:
        log(f"keep {cur}: best other {best} lockbox5 {lb:.1f} vs {l0:.1f} (+3 needed), official {of:.1f} vs {f0:.1f}")
    log("## batch 15 done")


if __name__ == "__main__":
    main()
