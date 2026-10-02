#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner18] start $(date '+%F %T')"
python experiments/queue_fb_b18.py > experiments/queue_fb_b18.out 2>&1
echo "[runner18] batch 18 exit $? $(date '+%F %T')"
