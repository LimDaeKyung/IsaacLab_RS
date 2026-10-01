#!/usr/bin/env bash
# Headless evaluation of one checkpoint over all terrain / friction conditions.
# usage: eval_sweep.sh <checkpoint.pt> <output_dir> [play_task] [extra evaluate.py / hydra args...]
set -u
CKPT=$1
OUT=$2
TASK=${3:-Isaac-Ant-Eval-v0}
shift 3 2>/dev/null || shift $#
EXTRA=("$@")
mkdir -p "$OUT"
# seen: terrain types used in training (with fixed, unseen parameters and terrain seed)
# held-out: terrain types / friction never used in training
CONDITIONS=(
  "flat 1.0" "flat 0.5"
  "boxes_low 1.0" "boxes_mid 1.0" "boxes_high 1.0" "boxes_mid 0.5"
  "rough_low 1.0" "rough_high 1.0"
  "slope 1.0" "stairs 1.0"
  "ho_boxes_fine 1.0" "ho_obstacles 1.0" "ho_wave 1.0" "flat 0.1" "boxes_mid 0.2"
)
for cond in "${CONDITIONS[@]}"; do
  read -r terrain friction <<< "$cond"
  name="${terrain}_f${friction}"
  echo "[SWEEP] $name"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate.py \
    --task "$TASK" --terrain "$terrain" --friction "$friction" \
    --checkpoint "$CKPT" --output "$OUT/$name.json" "${EXTRA[@]}" > "$OUT/$name.log" 2>&1 \
    || echo "[SWEEP] $name FAILED (see $OUT/$name.log)"
done
