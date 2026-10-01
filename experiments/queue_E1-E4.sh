#!/usr/bin/env bash
# Overnight queue: E1-E4 (same budget, seed 42) and baseline re-scored with the common termination rule.
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
R=scripts/reinforcement_learning/rsl_rl
$R/run_experiment.sh E1 e1_terrain_s42 42
$R/run_experiment.sh E2 e2_curriculum_s42 42
$R/run_experiment.sh E3 e3_friction_s42 42
$R/run_experiment.sh E4 e4_relheight_s42 42
echo "[EXP] E0 rescore start $(date '+%F %T')"
$R/eval_sweep.sh logs/rsl_rl/ant/2026-09-17_13-20-27_ant_baseline/model_999.pt experiments/eval/E0_baseline_reltermination Isaac-Ant-E0-Play-v0
echo "[QUEUE] all done $(date '+%F %T')"
