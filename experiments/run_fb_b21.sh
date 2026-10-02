#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner21] start $(date '+%F %T')"
python experiments/queue_fb_b21.py > experiments/queue_fb_b21.out 2>&1
echo "[runner21] batch 21 exit $? $(date '+%F %T')"
