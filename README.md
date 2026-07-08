# MEDE-TTA: Monocular Endoscopic Depth Estimation via Test-Time Adaptation

Released code of **MEDE-TTA**: **Training-free Zero-shot** Monocular Depth Estimation in **New Endoscopic Environment**.

## Setup

**Environment**:
```bash
pip install -r requirements.txt
```

**Checkpoints**:
- Download pretrained weight of EndoDAC trained on SCARED from [their github repo](https://github.com/BeileiCui/EndoDAC), extracting the content to `logs/endodac_base`
- Download pretrained weight of EndoDAC trained on SimCol from [Google Driver](https://drive.google.com/file/d/1sVAg_SW4im9dC1ZAkhy3nAZAMw77qWio/view?usp=sharing) to `logs/endodac_col`
- Download pretrained weight of Depth Anything from [Google Driver](https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll/view) to `pretrained_model`

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

