#!/usr/bin/env bash
# Detached runner for batch 10 then batch 11 (survives the agent's background-job time limit).
# usage: nohup setsid bash experiments/run_fb_b10_b11.sh > experiments/run_fb_b10_b11.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner] batch 10 start $(date '+%F %T')"
python experiments/queue_fb_b10.py > experiments/queue_fb_b10.out 2>&1
echo "[runner] batch 10 exit $? $(date '+%F %T')"
python experiments/queue_fb_b11.py > experiments/queue_fb_b11.out 2>&1
echo "[runner] batch 11 exit $? $(date '+%F %T')"
