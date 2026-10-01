#!/usr/bin/env bash
# Third queue: seeds 43/44 for E1-E3 so every ablation step can be reported as mean ± std over 3 seeds.
cd "$(dirname "$0")/.."
until grep -q "\[QUEUE\] all done" experiments/queue_2.log; do sleep 20; done
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
R=scripts/reinforcement_learning/rsl_rl
for seed in 43 44; do
  $R/run_experiment.sh E1 e1_terrain_s$seed $seed
  $R/run_experiment.sh E2 e2_curriculum_s$seed $seed
  $R/run_experiment.sh E3 e3_friction_s$seed $seed
done
echo "[QUEUE] all done $(date '+%F %T')"
