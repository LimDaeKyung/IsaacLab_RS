"""Round 4 phase A: extra seeds for E4 (45, 46), E6 (45, 46), V3 (43-46). Pre-registered in EXPERIMENTS.md §8.2."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overnight as o  # noqa: E402
import overnight2 as o2  # noqa: E402

ENT = ["agent.algorithm.entropy_coef=0.005"]
o.log("## round 4 phase A start (E4 s45-46, E6 s45-46, V3 s43-46)")
for seed in (45, 46):
    o2.experiment("E4_relheight", "A seed", "E4 re-check", [], [], 1000, seed=seed)
for seed in (45, 46):
    o2.experiment("E6", "A seed", "E6 re-check", [], ENT, 1000, seed=seed)
for seed in (43, 44, 45, 46):
    o2.experiment("V3", "A seed", "V3 re-check", ["Oracle"], [], 1000, seed=seed)
o.log("## round 4 phase A done")
