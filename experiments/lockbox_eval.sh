#!/usr/bin/env bash
# Lockbox evaluation (terrains fixed 2026-09-30 00:04). Run ONCE at the very end for E0, E4 and the final model only.
# usage: lockbox_eval.sh <name> <checkpoint.pt> <play_task> [extra evaluate.py / hydra args...]
set -u
NAME=$1; CKPT=$2; TASK=$3; shift 3
cd "$(dirname "$0")/.."
OUT=experiments/eval_lockbox/$NAME
mkdir -p "$OUT"
for t in lock_rails lock_gaps lock_pits lock_stones; do
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate.py --task "$TASK" --terrain $t \
    --checkpoint "$CKPT" --output "$OUT/$t.json" "$@" > "$OUT/$t.log" 2>&1 || echo "[LOCKBOX] $NAME $t FAILED"
done
