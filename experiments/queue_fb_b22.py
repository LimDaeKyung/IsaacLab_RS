"""Batch 22 (share-fb, pre-registered 2026-10-02 18:37): descriptive fall-penalty sweep, no submission change.

Batch 17 trained SP50 s42/43 for +1000 it with penalties 0 (C60), -20 (FP20) and -50 (FP50) per fall. This adds
-10 (FP10) and -35 (FP35) on the same parents, seeds and budget, giving a 5-point dose-response curve of
penalty size -> unseen9 score and unseen fall rate for the analysis. Nothing here can change the submission.
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
import queue_fb_b18 as b18  # noqa: E402

ENT = b12.ENT
T = "Isaac-Ant-R9-AllMix{}-v0"
NEW = {"FP10": T.format("WideMF04FP10"), "FP35": T.format("WideMF04FP35")}
CURVE = [("C60", 0), ("FP10", -10), ("FP20", -20), ("FP35", -35), ("FP50", -50)]


def log(text):
    o.log(f"[fb] [b22] {text}")


def main():
    log("## batch 22 start (descriptive penalty sweep from SP50 s42/43)")
    for name, task in NEW.items():
        for s in (42, 43):
            b10.wait_gpu()
            b7.train(f"{name.lower()}_s{s}", task, s, ENT, 1000, init=r4.ckpt_of(f"sp50_s{s}"))
    for name, pen in CURVE:
        ms = [b18.metrics(f"{name.lower()}_s{s}") for s in (42, 43) if r4.ckpt_of(f"{name.lower()}_s{s}")]
        if len(ms) < 2:
            log(f"curve {name} ({pen}/fall): incomplete")
            continue
        log(f"curve {name} ({pen}/fall): unseen9 {st.mean(m['unseen9'] for m in ms):.1f}, unseen fall "
            f"{100 * st.mean(m['ufall'] for m in ms):.0f}%, flat fall {100 * st.mean(m['flat_fall'] for m in ms):.0f}%, "
            f"flat speed {st.mean(m['flat_speed'] for m in ms):.2f}, μ0.2 mult {st.mean(m['mult'] for m in ms):.1f}")
    log("## batch 22 done")


if __name__ == "__main__":
    main()
