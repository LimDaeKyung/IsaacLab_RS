#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner22] start $(date '+%F %T')"
python experiments/queue_fb_b22.py > experiments/queue_fb_b22.out 2>&1
echo "[runner22] batch 22 exit $? $(date '+%F %T')"
