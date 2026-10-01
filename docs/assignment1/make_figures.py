"""Figures for the assignment README (static PNGs, regenerated from experiments/eval* JSON files).

usage (conda env lerobot-arena, from ~/IsaacLab_RS):  python docs/assignment1/make_figures.py
"""

import glob
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = "docs/assignment1/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CONTEXT, FINAL = "#8f8d87", "#2a78d6"  # de-emphasised context vs the final model (validated pair, labels on every bar)

for f in font_manager.findSystemFonts():
    if "NotoSansCJK-Regular" in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

HELD_OUT = ["ho_boxes_fine_f1.0", "ho_obstacles_f1.0", "ho_wave_f1.0", "flat_f0.1", "boxes_mid_f0.2"]
LOCK = ["lock_rails", "lock_gaps", "lock_pits", "lock_stones"]
MODELS = [  # label, eval dirs, lockbox dirs
    ("E0\n평지 원본", ["E0_baseline_reltermination"], ["E0"]),
    ("E4\n혼합 지형", [f"e4_relheight_s{s}" for s in range(42, 47)], [f"posthoc_e4_relheight_s{s}" for s in range(42, 47)]),
    ("E15\n박스+탐색", [f"e15_s{s}" for s in range(42, 50)], [f"posthoc_e15_s{s}" for s in range(42, 50)]),
    ("F3a (최종)\n+600 it", [f"f3a_s{s}" for s in range(42, 50)], [f"posthoc_f3a_s{s}" for s in range(42, 50)]),
]


def load(d):
    return {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{d}/*.json")}


def per_seed(dirs, base, conds):
    vals = []
    for d in dirs:
        r = load(f"{base}/{d}")
        vals.append(st.mean(r[c]["return_mean"] for c in conds))
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def fig_progression():
    panels = [("처음 보는 지형 5종", "experiments/eval", HELD_OUT, 1),
              ("봉인 시험장 4종", "experiments/eval_lockbox", LOCK, 2),
              ("평지 (원본 환경)", "experiments/eval", ["flat_f1.0"], 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, (title, base, conds, idx) in zip(axes, panels):
        style(ax)
        for i, m in enumerate(MODELS):
            mean, sd, n = per_seed(m[idx], base, conds)
            color = FINAL if i == len(MODELS) - 1 else CONTEXT
            ax.bar(i, mean, width=0.6, color=color, zorder=2)
            if n > 1:
                ax.errorbar(i, mean, yerr=sd, color=INK2, capsize=3, linewidth=1, zorder=3)
            ax.text(i, mean + sd + 2.5, f"{mean:.1f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if color == FINAL else "normal")
        ax.set_xticks(range(len(MODELS)), [m[0] for m in MODELS], fontsize=9)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_ylim(0, 140)
    axes[0].set_ylabel("에피소드 리턴 (100 envs 평균)")
    fig.suptitle("모델별 점수: 막대는 seed 평균, 선은 seed 간 표준편차 (E0 1개, E4 5개, E15·F3a 8개)",
                 fontsize=11, x=0.01, ha="left", color=INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/score_progression.png", dpi=150)


def fig_terrains():
    names = [("ho_boxes_fine_f1.0", "좁은 박스"), ("ho_obstacles_f1.0", "장애물"), ("ho_wave_f1.0", "물결"),
             ("flat_f0.1", "평지 μ0.1"), ("boxes_mid_f0.2", "박스 μ0.2"), ("lock_rails", "레일*"),
             ("lock_gaps", "틈*"), ("lock_pits", "구덩이*"), ("lock_stones", "징검다리*")]
    e0 = {**load("experiments/eval/E0_baseline_reltermination"), **load("experiments/eval_lockbox/E0")}
    fa = [{**load(f"experiments/eval/f3a_s{s}"), **load(f"experiments/eval_lockbox/posthoc_f3a_s{s}")} for s in range(42, 50)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    style(ax)
    w = 0.38
    for i, (k, label) in enumerate(names):
        a = e0[k]["return_mean"]
        b = st.mean(r[k]["return_mean"] for r in fa)
        ax.bar(i - w / 2 - 0.01, a, width=w, color=CONTEXT, zorder=2, label="E0 평지 원본" if i == 0 else None)
        ax.bar(i + w / 2 + 0.01, b, width=w, color=FINAL, zorder=2, label="F3a 최종 (8 seed 평균)" if i == 0 else None)
        ax.text(i - w / 2 - 0.01, a + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(i + w / 2 + 0.01, b + 1.5, f"{b:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(names)), [n for _, n in names], fontsize=10)
    ax.axvline(4.5, color=GRID, linewidth=1)
    ax.set_ylim(0, 110)
    ax.set_ylabel("에피소드 리턴")
    ax.set_title("학습에 쓰지 않은 지형별 점수 (* 봉인 시험장: 모델 선택에도 쓰지 않음)", fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/unseen_terrains.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_progression()
    fig_terrains()
    print("saved", os.listdir(OUT))
