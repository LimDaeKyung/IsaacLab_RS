#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner17] start $(date '+%F %T')"
python experiments/queue_fb_b17.py > experiments/queue_fb_b17.out 2>&1
echo "[runner17] batch 17 exit $? $(date '+%F %T')"
