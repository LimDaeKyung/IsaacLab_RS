#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner16] start $(date '+%F %T')"
python experiments/queue_fb_b16.py > experiments/queue_fb_b16.out 2>&1
echo "[runner16] batch 16 exit $? $(date '+%F %T')"
