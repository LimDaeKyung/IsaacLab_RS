#!/usr/bin/env bash
# 12:45 restart after the 12:41 Isaac Sim abort: batch 17 resumes (finished runs and evals are reused), then batch 18
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner17] restart $(date '+%F %T')"
python experiments/queue_fb_b17.py > experiments/queue_fb_b17.out 2>&1
echo "[runner17] batch 17 exit $? $(date '+%F %T')"
python experiments/queue_fb_b18.py > experiments/queue_fb_b18.out 2>&1
echo "[runner18] batch 18 exit $? $(date '+%F %T')"
