#!/usr/bin/env bash
# Resume the remaining work after an interruption. Finished experiments are skipped automatically.
# usage (from ~/IsaacLab_RS):  nohup setsid experiments/resume.sh > experiments/resume.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
R=scripts/reinforcement_learning/rsl_rl
# 1) seeds 43/44 for E1-E3 (reproducibility of every ablation step)
for seed in 43 44; do
  $R/run_experiment.sh E1 e1_terrain_s$seed $seed
  $R/run_experiment.sh E2 e2_curriculum_s$seed $seed
  $R/run_experiment.sh E3 e3_friction_s$seed $seed
done
echo "[QUEUE] seeds all done $(date '+%F %T')"
# 2) headless behaviour videos, E0 vs E4 (needs free GPU memory, so after training)
E0=logs/rsl_rl/ant/2026-09-17_13-20-27_ant_baseline/model_999.pt
E4=$(ls logs/rsl_rl/ant/*_e4_relheight_s42/model_999.pt)
for t in flat boxes_mid ho_obstacles stairs; do
  [ -f experiments/videos/E0_$t.json ] || ./isaaclab.sh -p $R/evaluate.py --task Isaac-Ant-E0-Play-v0 --terrain $t --num_envs 16 --video \
    --checkpoint $E0 --output experiments/videos/E0_$t.json > experiments/videos/E0_$t.log 2>&1 || echo "[VIDEO] E0 $t FAILED"
  [ -f experiments/videos/E4_$t.json ] || ./isaaclab.sh -p $R/evaluate.py --task Isaac-Ant-Rough-E4-Play-v0 --terrain $t --num_envs 16 --video \
    --checkpoint $E4 --output experiments/videos/E4_$t.json > experiments/videos/E4_$t.log 2>&1 || echo "[VIDEO] E4 $t FAILED"
  echo "[VIDEO] $t done $(date '+%T')"
done
echo "[QUEUE] all done $(date '+%F %T')"
