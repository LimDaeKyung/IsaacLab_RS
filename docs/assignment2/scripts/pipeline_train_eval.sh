#!/bin/bash
# Wait for Mimic generation -> BC-RNN train (seed 101, TA config) -> play 100 rollouts (seed 42, horizon 800)
set -u
S=./experiments/assignment2_logs; mkdir -p $S
cd /home/share/IsaacLab_RS
source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-arena
TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
DATA=./datasets/toy2box_auto_30_mimic_500.hdf5
NAME=auto30_mimic500_s101

echo "[pipeline] waiting for generation to finish"
until grep -q "^EXIT" $S/generate.log; do sleep 20; done
grep "^EXIT" $S/generate.log
[ -f "$DATA" ] || { echo "[pipeline] ERROR: $DATA missing"; exit 1; }

echo "[pipeline] train start $(date +%H:%M)"
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task $TASK --algo bc --dataset $DATA --name $NAME > $S/train.log 2>&1
echo "[pipeline] train exit $? at $(date +%H:%M)"

CKPT=$(ls -t logs/robomimic/$TASK/$NAME/*/models/model_epoch_2000.pth 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "[pipeline] ERROR: epoch 2000 checkpoint not found"; exit 1; }
echo "[pipeline] checkpoint $CKPT"

echo "[pipeline] eval start $(date +%H:%M)"
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task $TASK --checkpoint $CKPT --num_rollouts 100 --horizon 800 --seed 42 --headless > $S/play.log 2>&1
echo "[pipeline] eval exit $? at $(date +%H:%M)"
grep -E "Successful trials|Success rate" $S/play.log
echo "[pipeline] DONE"
