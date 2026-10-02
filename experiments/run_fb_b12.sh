#!/usr/bin/env bash
# Detached runner for batch 12 (waits for batch 11 inside the script).
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner12] start $(date '+%F %T')"
python experiments/queue_fb_b12.py > experiments/queue_fb_b12.out 2>&1
echo "[runner12] batch 12 exit $? $(date '+%F %T')"
