"""Batch 23 (share-fb, pre-registered 2026-10-02 19:58): descriptive, no submission change.

C60 (SP50 + 1000 it, no fall penalty) had the highest screen score (unseen9 121.8 vs FP20 107.7, s42/43) but falls
more (44% vs 31%) and was never confirmed. Train C60 on s47-49 and report it next to FP20 on the same seeds
(unseen9, unseen fall, official boxes), so the user can choose between the score-first and the fewer-falls model.
All lockboxes are used, so no lockbox decides anything here; the submission stays FP20 s49.
"""

import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight4 as r4  # noqa: E402
import queue_fb_b7 as b7  # noqa: E402
import queue_fb_b10 as b10  # noqa: E402
import queue_fb_b12 as b12  # noqa: E402  (sets the Fri 23:59 training guard)
import queue_fb_b15 as b15  # noqa: E402
import queue_fb_b18 as b18  # noqa: E402

ENT = b12.ENT


def log(text):
    o.log(f"[fb] [b23] {text}")


def main():
    log("## batch 23 start (descriptive: C60 on s47-49 vs FP20)")
    for s in (47, 48, 49):
        b10.wait_gpu()
        b7.train(f"c60_s{s}", "Isaac-Ant-R9-AllMixWideMF04-v0", s, ENT, 1000, init=r4.ckpt_of(f"sp50_s{s}"))
    rows = {}
    for name in ("c60", "fp20"):
        ms = {s: b18.metrics(f"{name}_s{s}") for s in (47, 48, 49) if r4.ckpt_of(f"{name}_s{s}")}
        for s, m in ms.items():
            log(f"{name}_s{s}: unseen9 {m['unseen9']:.1f}, unseen fall {100 * m['ufall']:.0f}%, flat fall "
                f"{100 * m['flat_fall']:.0f}%, μ0.2 mult {m['mult']:.1f}, official {b15.official(f'{name}_s{s}'):.1f}")
        rows[name] = ms
    if len(rows["c60"]) == 3:
        d = [rows["c60"][s]["unseen9"] - rows["fp20"][s]["unseen9"] for s in (47, 48, 49)]
        df = [100 * (rows["c60"][s]["ufall"] - rows["fp20"][s]["ufall"]) for s in (47, 48, 49)]
        log(f"C60 − FP20 (s47–49): unseen9 {st.mean(d):+.1f} ({', '.join(f'{x:+.1f}' for x in d)}), unseen fall "
            f"{st.mean(df):+.0f} points")
    log("## batch 23 done")


if __name__ == "__main__":
    main()
