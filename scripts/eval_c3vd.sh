#!/usr/bin/env bash
# C3VD MEDE eval (split defaults: unc_scale=0.1, num_aug=6)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python evaluate_depth_tta.py \
  --eval_mono --eval_split c3vd \
  --data_path "${DATA_PATH:-/mnt/data2_hdd/beilei/Dataset/C3VD_undistort}" \
  --load_weights_folder "${WEIGHTS:-../MEDTTA/logs/endodac_col}" \
  --pretrained_path "${PRETRAINED:-../MEDTTA/pretrained_model}" \
  --backbone_size base --lora_type dvlora \
  --height 256 --width 320 \
  --tta_method mede \
  "$@"
