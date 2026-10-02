#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner13] start $(date '+%F %T')"
python experiments/queue_fb_b13.py > experiments/queue_fb_b13.out 2>&1
echo "[runner13] batch 13 exit $? $(date '+%F %T')"
