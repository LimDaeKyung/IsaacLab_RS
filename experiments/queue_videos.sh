#!/usr/bin/env bash
# Headless behaviour videos (first episode of env 0) for E0 vs E4 after all training finished (needs free GPU memory).
cd "$(dirname "$0")/.."
until grep -q "\[QUEUE\] all done" experiments/queue_3.log; do sleep 30; done
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
E0=logs/rsl_rl/ant/2026-09-17_13-20-27_ant_baseline/model_999.pt
E4=$(ls logs/rsl_rl/ant/*_e4_relheight_s42/model_999.pt)
for t in flat boxes_mid ho_obstacles stairs; do
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate.py --task Isaac-Ant-E0-Play-v0 --terrain $t --num_envs 16 --video \
    --checkpoint $E0 --output experiments/videos/E0_$t.json > experiments/videos/E0_$t.log 2>&1 || echo "[VIDEO] E0 $t FAILED"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate.py --task Isaac-Ant-Rough-E4-Play-v0 --terrain $t --num_envs 16 --video \
    --checkpoint $E4 --output experiments/videos/E4_$t.json > experiments/videos/E4_$t.log 2>&1 || echo "[VIDEO] E4 $t FAILED"
  echo "[VIDEO] $t done $(date '+%T')"
done
echo "[QUEUE] videos all done $(date '+%F %T')"
