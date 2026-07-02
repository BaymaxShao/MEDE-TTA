#!/usr/bin/env bash
# Hamlyn MEDE eval (split defaults: unc_scale=0, num_aug=2)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python evaluate_depth_tta.py \
  --eval_mono --eval_split hamlyn \
  --data_path "${DATA_PATH:-/mnt/data2_hdd/beilei/Dataset/Hamlyn}" \
  --load_weights_folder "${WEIGHTS:-../MEDTTA/logs/endodac_base}" \
  --pretrained_path "${PRETRAINED:-../MEDTTA/pretrained_model}" \
  --backbone_size base --lora_type dvlora \
  --height 256 --width 320 \
  --tta_method mede \
  "$@"
