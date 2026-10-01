#!/usr/bin/env bash
# Train one experiment and evaluate its last checkpoint on the full terrain sweep (headless).
# usage: run_experiment.sh <EXP_ID e.g. E2> <run_name e.g. e2_curriculum_s42> [seed] [extra train args...]
set -u
EXP=$1
RUN=$2
SEED=${3:-42}
shift 3 || shift $#
cd "$(dirname "$0")/../../.."
# skip experiments whose full evaluation already exists (lets an interrupted queue be restarted)
if [ "$(ls "experiments/eval/${RUN}"/*.json 2>/dev/null | wc -l)" -ge 15 ]; then
  echo "[EXP] $EXP $RUN already evaluated, skipping"
  exit 0
fi
echo "[EXP] $EXP $RUN seed=$SEED start $(date '+%F %T')"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task "Isaac-Ant-Rough-${EXP}-v0" --headless --seed "$SEED" --run_name "$RUN" "$@" \
  > "experiments/train_logs/${RUN}.log" 2>&1
RUN_DIR=$(ls -d logs/rsl_rl/ant/*_"${RUN}" | tail -1)
CKPT=$(ls "$RUN_DIR"/model_*.pt | sort -t_ -k2 -n | tail -1)
echo "[EXP] $EXP trained -> $CKPT $(date '+%F %T')"
scripts/reinforcement_learning/rsl_rl/eval_sweep.sh "$CKPT" "experiments/eval/${RUN}" "Isaac-Ant-Rough-${EXP}-Play-v0"
echo "[EXP] $EXP done $(date '+%F %T')"
