#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner14] start $(date '+%F %T')"
python experiments/queue_fb_b14.py > experiments/queue_fb_b14.out 2>&1
echo "[runner14] batch 14 exit $? $(date '+%F %T')"
