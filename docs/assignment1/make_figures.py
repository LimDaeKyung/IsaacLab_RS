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


def fig_lineage():
    """Every stage on lockbox 7 (fixed 10-02 12:08, used by no decision; experiments/queue_fb_b21.py)."""
    rows = json.load(open("experiments/eval_lockbox7/lineage.json"))
    labels = {"E0": "E0\n평지 원본", "E4": "E4\n발밑 기준", "E15": "E15\n박스+탐색", "F3a": "F3a\n+600 it",
              "M30": "M30\n전 지형", "AXF40": "AXF40\n높이 연속", "SP50": "SP50\n빠른 걸음",
              "FP20": "FP20 s47\n넘어짐 벌점", "FP20_s49": "FP20 s49\n(최종)"}
    names = list(labels)
    panels = [("봉인 시험장 7 점수 (4종 평균)", "lockbox7", 1, "{:.1f}", 140),
              ("봉인 시험장 7 넘어짐 비율", "fall", 100, "{:.0f}%", 100)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    for ax, (title, key, scale, fmt, top) in zip(axes, panels):
        style(ax)
        for i, n in enumerate(names):
            v = rows[n][key] * scale
            color = FINAL if n == "FP20_s49" else CONTEXT
            ax.bar(i, v, width=0.62, color=color, zorder=2)
            ax.text(i, v + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if color == FINAL else "normal")
        ax.set_xticks(range(len(names)), [labels[n] for n in names], fontsize=8)
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_ylim(0, top)
    axes[0].set_ylabel("에피소드 리턴 (100 envs 평균)")
    fig.suptitle("단계별 모델을 같은 처음 보는 지형에서 비교 (넓은 틈 35 cm, 피라미드 15 cm, 짧은 물결, 저마찰 요철; 각 모델의 대표 seed 1개)",
                 fontsize=11, x=0.01, ha="left", color=INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/lineage_lockbox7.png", dpi=150)


def fig_penalty():
    """Training-only fall penalty sweep from SP50 s42/43, +1000 it each (batches 17 and 22)."""
    curve = [("c60", 0), ("fp10", -10), ("fp20", -20), ("fp35", -35), ("fp50", -50)]
    score, fall = [], []
    for prefix, _ in curve:
        s_vals, f_vals = [], []
        for s in (42, 43):
            u = load(f"experiments/eval_unseen8/{prefix}_s{s}")
            mult = json.load(open(f"experiments/eval/{prefix}_s{s}/boxes_mid_f0.2_multiply.json"))["return_mean"]
            s_vals.append(st.mean([r["return_mean"] for r in u.values()] + [mult]))
            f_vals.append(100 * st.mean(r["fall_rate"] for r in u.values()))
        score.append(st.mean(s_vals))
        fall.append(st.mean(f_vals))
    xs = [p for _, p in curve]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    for ax, ys, title, fmt, top in ((axes[0], score, "처음 보는 지형 점수 (unseen9)", "{:.0f}", 140),
                                    (axes[1], fall, "처음 보는 지형 넘어짐 비율", "{:.0f}%", 60)):
        style(ax)
        ax.plot(xs, ys, color=CONTEXT, linewidth=2, zorder=2)
        for x, y in zip(xs, ys):
            final = x == -20
            ax.plot(x, y, "o", markersize=9 if final else 7, color=FINAL if final else CONTEXT, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=2)
            ax.text(x, y + top * 0.04, fmt.format(y), ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if final else "normal")
        ax.set_xticks(xs, [f"{x}" for x in xs])
        ax.set_xlabel("학습 전용 넘어짐 벌점 (넘어질 때마다, 점)")
        ax.set_title(title, fontsize=12, loc="left")
        ax.set_ylim(0, top)
        ax.invert_xaxis()
    fig.suptitle("넘어짐 벌점 크기에 따른 변화: SP50에서 +1,000 it, seed 42·43 평균 (−20 = 최종 FP20)",
                 fontsize=11, x=0.01, ha="left", color=INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/penalty_sweep.png", dpi=150)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_progression()
    fig_terrains()
    fig_lineage()
    fig_penalty()
    print("saved", os.listdir(OUT))
