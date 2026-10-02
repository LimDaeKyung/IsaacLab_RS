"""Batch 21 (share-fb, pre-registered 2026-10-02 18:20): evaluation only, no training.

(a) decision: FP20 seed chosen by the validation metric (unseen9) = fp20_s49; lockbox 8 (untouched) vs current fp20_s47.
(b) descriptive: the whole lineage on lockbox 7 (untouched), never used for any decision.
(c) check: the official command without the entropy override gives the same FP20 score.
"""

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight4 as r4  # noqa: E402
import queue_fb_b10 as b10  # noqa: E402
import queue_fb_b12 as b12  # noqa: E402
import queue_fb_b14 as b14  # noqa: E402
import queue_fb_b15 as b15  # noqa: E402
import queue_fb_b17 as b17  # noqa: E402
import queue_fb_b18 as b18  # noqa: E402
import queue_fb_b19 as b19  # noqa: E402

R = o.R
ENT = b12.ENT
AM = b12.ALLMIX_PLAY
ORACLE = "Isaac-Ant-R2-Oracle-Play-v0"
M = "docs/assignment1/models"
# name: (run used as the output folder, checkpoint, play task)
LINEAGE = {
    "E0": ("e0_flat_baseline", f"{M}/e0_flat_baseline.pt", "Isaac-Ant-E0-Play-v0"),
    "E4": ("e4_mixed_relheight_s42", f"{M}/e4_mixed_relheight_s42.pt", "Isaac-Ant-Rough-E4-Play-v0"),
    "E15": ("e15_s44", f"{M}/e15_boxes_entropy_s44.pt", ORACLE),
    "F3a": ("f3a_s47", f"{M}/f3a_final_s47.pt", ORACLE),
    "M30": ("m30_s47", None, AM),
    "AXF40": ("axf40_s47", None, AM),
    "SP50": ("sp50_s47", None, AM),
    "FP20": ("fp20_s47", None, AM),
    "FP20_s49": ("fp20_s49", None, AM),
}


def log(text):
    o.log(f"[fb] [b21] {text}")


def lockbox(run, ckpt, play, boxes, tag):
    v, falls = {}, []
    for t, extra in boxes.items():
        r = b10.evaluate(ckpt, play, t, f"experiments/eval_{tag}/{run}/{t}.json", ENT + extra)
        v[t] = r["return_mean"]
        falls.append(r["fall_rate"])
    return v, st.mean(v.values()), st.mean(falls)


def main():
    b14.log = log
    log("## batch 21 start (evaluation only)")
    with open("experiments/fb_state.json") as f:
        state = json.load(f)
    cur = state["submission"]

    # (a) decision on lockbox 8
    res = {}
    for name, run in (("FP20", "fp20_s47"), ("FP20_s49", "fp20_s49")):
        v, mean, fall = lockbox(run, r4.ckpt_of(run), AM, b19.LOCK8, "lockbox8")
        res[name] = (mean, fall, b15.official(run))
        log(f"lockbox8 {run}: " + ", ".join(f"{k[6:]} {x:.1f}" for k, x in v.items())
            + f" → {mean:.1f} (fall {100 * fall:.0f}%), official {res[name][2]:.1f}")
    (l0, k0, f0), (l1, k1, f1) = res["FP20"], res["FP20_s49"]
    dk = 100 * (k1 - k0)
    if cur == "FP20" and f1 >= f0 - 10 and ((l1 >= l0 + 3 and dk <= 5) or (dk <= -10 and l1 >= l0 - 5)):
        ck = r4.ckpt_of("fp20_s49")
        state = {**state, "previous": cur, "submission": "FP20_s49", "play": AM, "ckpt": ck, "score_overrides": ENT,
                 "lockbox8": l1, "lockbox8_fall": k1, "official_play_one_episode_boxes": f1,
                 "note": "batch 21: FP20 seed chosen by unseen9, confirmed on lockbox 8"}
        with open("experiments/fb_state.json", "w") as f:
            json.dump(state, f, indent=1)
        log(f"NEW SUBMISSION: FP20_s49 `{ck}` (lockbox8 {l1:.1f} vs {l0:.1f}, fall {dk:+.0f} points, official {f1:.1f})")
    else:
        log(f"keep {cur}: fp20_s49 lockbox8 {l1:.1f} vs {l0:.1f}, fall {dk:+.0f} points, official {f1:.1f} vs {f0:.1f}")

    # (b) descriptive lineage table on lockbox 7
    rows = {}
    for name, (run, ck, play) in LINEAGE.items():
        ck = ck or r4.ckpt_of(run)
        v, mean, fall = lockbox(run, ck, play, b18.LOCK7, "lockbox7")
        rows[name] = {"lockbox7": mean, "fall": fall, **{k[6:]: x for k, x in v.items()}}
        log(f"lockbox7 {name}: " + ", ".join(f"{k[6:]} {x:.1f}" for k, x in v.items())
            + f" → {mean:.1f} (fall {100 * fall:.0f}%)")
    with open("experiments/eval_lockbox7/lineage.json", "w") as f:
        json.dump(rows, f, indent=1)

    # (c) official command exactly as the TA runs it (no entropy override)
    out = "experiments/train_logs/official_fp20_s47_no_override.log"
    b10.wait_gpu()
    o.run(["./isaaclab.sh", "-p", f"{R}/play_one_episode.py", "--task", AM, "--headless", "--seed", "24", "--num_envs",
           "100", "--checkpoint", r4.ckpt_of("fp20_s47")], out)
    line = [x for x in open(out) if "Episode reward total" in x]
    log(f"official fp20_s47 without override (seed 24): {line[0].strip() if line else 'NO RESULT'}")
    log("## batch 21 done")


if __name__ == "__main__":
    main()
