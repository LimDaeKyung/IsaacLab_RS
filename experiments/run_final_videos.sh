#!/usr/bin/env bash
# final model (FP20 s49) videos + the official command exactly as in EVAL_COMMAND.txt (no training running)
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
CK=docs/assignment1/models/fp20_final_s49.pt
for T in boxes_mid flat stairs lock5_stairs_high lock5_slope_steep lock_stones ho_wave; do
  timeout 600 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate.py --task Isaac-Ant-R5-AllMix-Play-v0 \
    --terrain $T --num_envs 4 --checkpoint $CK --output experiments/videos_final/$T.json --video --headless \
    > experiments/videos_final/$T.log 2>&1
  echo "[final] video $T exit $? $(date '+%T')"
done
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py --task Isaac-Ant-R5-AllMix-Play-v0 \
  --seed 24 --num_envs 100 --checkpoint $CK > experiments/train_logs/official_final_docs_ckpt.log 2>&1
echo "[final] official $(grep 'Episode reward total' experiments/train_logs/official_final_docs_ckpt.log)"
echo "[final] done $(date '+%T')"
