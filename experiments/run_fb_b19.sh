#!/usr/bin/env bash
# waits for batch 18, records the remaining FP20 videos while no training runs (CPU physics), then runs batch 19
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner19] start $(date '+%F %T')"
while pgrep -f "^python experiments/queue_fb_b1[78][.]py" > /dev/null; do sleep 60; done
CK=logs/rsl_rl/ant/2026-10-02_12-07-57_fp20_s47/model_999.pt
for T in flat stairs lock5_stairs_high lock5_slope_steep lock_stones ho_wave; do
  [ -f "experiments/videos_fp20/$T.json" ] && continue
  timeout 300 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/evaluate.py --task Isaac-Ant-R5-AllMix-Play-v0 \
    --terrain $T --num_envs 4 --device cpu --checkpoint $CK --output experiments/videos_fp20/$T.json --video --headless \
    agent.algorithm.entropy_coef=0.005 > experiments/videos_fp20/$T.log 2>&1
  echo "[videos] fp20 $T exit $? $(date '+%T')"
done
pkill -9 -f "^/home/share/miniforge3/envs/lerobot-arena/bin/python scripts/reinforcement_learning/rsl_rl/evaluate.py.*videos_fp20" || true
python experiments/queue_fb_b19.py > experiments/queue_fb_b19.out 2>&1
echo "[runner19] batch 19 exit $? $(date '+%F %T')"
