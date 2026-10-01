"""Collect evaluation JSONs into markdown tables.

usage: python experiments/summarize.py [eval_dir ...]   (default: every folder in experiments/eval)
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (file stem, label, group)
CONDITIONS = [
    ("flat_f1.0", "flat", "seen"),
    ("flat_f0.5", "flat, friction 0.5", "seen"),
    ("boxes_low_f1.0", "boxes ±5cm", "seen"),
    ("boxes_mid_f1.0", "boxes ±10cm", "seen"),
    ("boxes_high_f1.0", "boxes ±15cm", "seen"),
    ("boxes_mid_f0.5", "boxes ±10cm, friction 0.5", "seen"),
    ("rough_low_f1.0", "rough 0-5cm", "seen"),
    ("rough_high_f1.0", "rough 0-10cm", "seen"),
    ("slope_f1.0", "pyramid slope 0.2", "seen"),
    ("stairs_f1.0", "pyramid stairs 10cm", "seen"),
    ("ho_boxes_fine_f1.0", "fine boxes 0.3m ±10cm", "held-out"),
    ("ho_obstacles_f1.0", "discrete obstacles 10cm", "held-out"),
    ("ho_wave_f1.0", "waves 15cm", "held-out"),
    ("flat_f0.1", "flat, friction 0.1", "held-out"),
    ("boxes_mid_f0.2", "boxes ±10cm, friction 0.2", "held-out"),
]


def load(eval_dir):
    rows = {}
    for stem, _, _ in CONDITIONS:
        path = os.path.join(eval_dir, f"{stem}.json")
        if os.path.exists(path):
            with open(path) as f:
                rows[stem] = json.load(f)
    return rows


def main():
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "eval", "*")))
    results = {os.path.basename(d.rstrip("/")): load(d) for d in dirs}

    for name, rows in results.items():
        print(f"\n### {name}\n")
        print("| group | condition | return (mean ± std) | fall % | upside-down % | distance x (m) |")
        print("|---|---|---:|---:|---:|---:|")
        for stem, label, group in CONDITIONS:
            r = rows.get(stem)
            if r is None:
                continue
            print(
                f"| {group} | {label} | {r['return_mean']:.1f} ± {r['return_std']:.1f} | {100 * r['fall_rate']:.0f} |"
                f" {100 * r.get('upside_down_at_end_rate', float('nan')):.0f} | {r['distance_x_mean']:.1f} |"
            )

    # compact comparison of mean returns across experiments
    names = list(results)
    print("\n### mean return by experiment\n")
    print("| group | condition | " + " | ".join(names) + " |")
    print("|---|---|" + "---:|" * len(names))
    for stem, label, group in CONDITIONS:
        cells = [f"{results[n][stem]['return_mean']:.1f}" if stem in results[n] else "-" for n in names]
        print(f"| {group} | {label} | " + " | ".join(cells) + " |")
    for group in ("seen", "held-out"):
        stems = [s for s, _, g in CONDITIONS if g == group and s != "flat_f1.0"]
        cells = []
        for n in names:
            vals = [results[n][s]["return_mean"] for s in stems if s in results[n]]
            cells.append(f"**{sum(vals) / len(vals):.1f}**" if vals else "-")
        label = "average excluding flat" if group == "seen" else "average"
        print(f"| {group} | {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
