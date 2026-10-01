#!/usr/bin/env bash
# Second queue: reproducibility (E4 seeds 43/44), curriculum re-test on the best stack (E4b), longer budget (E5).
cd "$(dirname "$0")/.."
until grep -q "\[QUEUE\] all done" experiments/queue_E1-E4.log; do sleep 20; done
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
R=scripts/reinforcement_learning/rsl_rl
$R/run_experiment.sh E4b e4b_nocurriculum_s42 42
$R/run_experiment.sh E4 e4_relheight_s43 43
$R/run_experiment.sh E4 e4_relheight_s44 44
$R/run_experiment.sh E4 e5_e4_3000it_s42 42 --max_iterations 3000
echo "[QUEUE] all done $(date '+%F %T')"
