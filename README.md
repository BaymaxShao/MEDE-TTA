# MEDE-TTA: Monocular Endoscopic Depth Estimation via Test-Time Adaptation

Minimal release of **MEDE-TTA**: Aug-Teacher + Reset + scale-invariant consistency on EndoDAC (DV-LoRA).

Supported benchmarks: **Hamlyn**, **C3VD** only.

## Setup

```bash
pip install -r requirements.txt
```

Checkpoints:
- `{load_weights_folder}/depth_model.pth`
- `pretrained_model/depth_anything_vitb14.pth` (if not fully contained in checkpoint)

## Evaluation

**Hamlyn** (weights: `endodac_base`):
```bash
bash scripts/eval_hamlyn.sh
# equivalent minimal command:
python evaluate_depth_tta.py \
  --eval_mono --eval_split hamlyn \
  --data_path /path/to/Hamlyn \
  --load_weights_folder /path/to/endodac_base \
  --pretrained_path /path/to/pretrained_model \
  --backbone_size base --lora_type dvlora \
  --height 256 --width 320 \
  --tta_method mede
```

**C3VD** (weights: `endodac_col`, 8 default sequences):
```bash
bash scripts/eval_c3vd.sh
# equivalent minimal command:
python evaluate_depth_tta.py \
  --eval_mono --eval_split c3vd \
  --data_path /path/to/C3VD_undistort \
  --load_weights_folder /path/to/endodac_col \
  --pretrained_path /path/to/pretrained_model \
  --backbone_size base --lora_type dvlora \
  --height 256 --width 320 \
  --tta_method mede
```

**Source-only baseline:** `--tta_method source`

