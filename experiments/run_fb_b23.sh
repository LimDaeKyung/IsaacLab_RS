#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner23] start $(date '+%F %T')"
python experiments/queue_fb_b23.py > experiments/queue_fb_b23.out 2>&1
echo "[runner23] batch 23 exit $? $(date '+%F %T')"
