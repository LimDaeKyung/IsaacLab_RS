#!/usr/bin/env bash
# batch 20 (last autonomous batch); queue_fb_b20.py itself waits for batch 19 to finish
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-arena
echo "[runner20] start $(date '+%F %T')"
python experiments/queue_fb_b20.py > experiments/queue_fb_b20.out 2>&1
echo "[runner20] batch 20 exit $? $(date '+%F %T')"
