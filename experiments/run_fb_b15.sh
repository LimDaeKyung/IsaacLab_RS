#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner15] start $(date '+%F %T')"
python experiments/queue_fb_b15.py > experiments/queue_fb_b15.out 2>&1
echo "[runner15] batch 15 exit $? $(date '+%F %T')"
