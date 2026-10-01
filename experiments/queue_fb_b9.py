"""Batch 9 (share-fb): reporting only, no model selection.

1) F3a on seeds 44-46 (E15 s44-46 + warm-started 600 it) so that F3a has seeds 42-49 and pairs with E4/E15.
2) Post-hoc lockbox for every seed of E4 (42-46), E15 (42-49) and F3a (42-49) -> mean ± sd across seeds.
The submission (F3a s47) was fixed before this batch; nothing here changes it.
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
LOCK = ("lock_rails", "lock_gaps", "lock_pits", "lock_stones")


def log(text):
    o.log(f"[fb] {text}")


def lockbox(run, play, over):
    out = f"experiments/eval_lockbox/posthoc_{run}"
    if not all(os.path.exists(f"{out}/{t}.json") for t in LOCK):
        subprocess.run(["bash", "experiments/lockbox_eval.sh", f"posthoc_{run}", r4.ckpt_of(run), play, *over],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    vals = {}
    for t in LOCK:
        with open(f"{out}/{t}.json") as f:
            vals[t] = json.load(f)["return_mean"]
    return vals


def main():
    log("## batch 9 start (reporting only: F3a s44-46, post-hoc lockbox for all seeds)")
    for s in (44, 45, 46):
        run = f"f3a_s{s}"
        ck = b7.train(run, b7.OR_TASK, s, ENT, 600, init=r4.ckpt_of(f"e15_s{s}"))
        if ck and not os.path.exists(f"experiments/eval/{run}/boxes_mid_f0.2.json"):
            subprocess.run(["bash", f"{R}/eval_sweep.sh", ck, f"experiments/eval/{run}", b7.OR_PLAY, *ENT],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ck:
            m = b7.f3a(s)
            log(f"F3a s{s}: held-out {m['held_out']:.1f}, flat {m['flat']:.1f} (speed {m['flat_speed']:.2f}), "
                f"ho-fall {100 * m['ho_fall']:.0f}%, boxes μ0.2 multiply {m['mult']:.1f}")

    groups = {
        "E4": ([f"e4_relheight_s{s}" for s in range(42, 47)], "Isaac-Ant-Rough-E4-Play-v0", []),
        "E15": ([f"e15_s{s}" for s in range(42, 50)], b7.OR_PLAY, ENT),
        "F3a": ([f"f3a_s{s}" for s in range(42, 50)], b7.OR_PLAY, ENT),
    }
    summary = {}
    for name, (runs, play, over) in groups.items():
        per_seed = {}
        for run in runs:
            if r4.ckpt_of(run) is None:
                continue
            v = lockbox(run, play, over)
            per_seed[run] = v
            log(f"posthoc lockbox {run}: " + ", ".join(f"{t[5:]} {x:.1f}" for t, x in v.items())
                + f" → mean {st.mean(v.values()):.1f}")
        means = [st.mean(v.values()) for v in per_seed.values()]
        summary[name] = {"per_seed": per_seed, "mean": st.mean(means), "sd": st.stdev(means) if len(means) > 1 else 0.0,
                         "n": len(means)}
        log(f"posthoc lockbox {name}: {summary[name]['mean']:.1f} ± {summary[name]['sd']:.1f} (n={len(means)})")
    with open("experiments/posthoc_lockbox_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    log("## batch 9 done")


if __name__ == "__main__":
    main()
