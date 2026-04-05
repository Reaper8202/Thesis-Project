# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

PyTorch U-Net for counting quantum dots (QDs) in fluorescence microscopy images via density map regression. The model predicts a Gaussian-smoothed density map; `density_map.sum()` is the count. This sidesteps direct detection, which fails at high dot density or low SNR.

**Hardware target:** GTX 1660 Super (6 GB VRAM)  
**Dataset:** ~100 training images + ImageJ CSV annotations  
**Primary metric:** RMSE on predicted dot count

## Commands

```bash
# EDA — verify data loaded correctly before training
python run.py --mode eda

# Train (standard)
python run.py --mode train --epochs 200 --batch_size 8

# Train with torch.compile (~15–30% faster on PyTorch 2.0+)
python run.py --mode train --compile

# Evaluate best_model.pth on test set
python run.py --mode test

# Single image inference
python run.py --mode predict --image path/to/image.tif

# Check model size
python -c "from src.model import UNet, count_parameters; m = UNet(); print(f'{count_parameters(m):,}')"
```

**Key CLI flags:** `--seed 42` (reproducibility), `--lr 1e-4` (learning rate), `--count_weight 0.01` (loss balance), `--patience 30`, `--sigma 3.0` (density map Gaussian), `--compile`. Override data paths with `--images_dir` / `--annotations_dir` (defaults: `data/images/`, `data/annotations/`).

## Architecture

The pipeline is: `preprocess.py` → `dataset.py` → `model.py` ← `train.py` → `evaluate.py`.

**`src/preprocess.py`** — loads images at native bit depth (uint8/uint16/float32 TIF) keeping float32 throughout to avoid quantisation. Uses percentile normalisation (p1/p99) instead of min/max to handle hot/dead pixels. Annotations are ImageJ CSV with X/Y columns (case-insensitive). Annotation coordinates are rescaled when images are resized.

**`src/dataset.py`** — `QuantumDotDataset` generates density maps on the fly from point annotations via `generate_density_map()` (Gaussian kernel, σ=3.0, integral = point count). Non-augmented datasets (val/test) cache their tensors after first access — density maps are deterministic when `augment=False` so recomputing each epoch is wasteful. Zoom-crop augmentation removes out-of-bounds points and updates `count_tensor` accordingly.

**`src/model.py`** — U-Net with `base_filters=32` (~7.76 M params). Key non-defaults:
- `AvgPool2d` instead of MaxPool (preserves density energy in feature maps)
- `Softplus(β=10)` output (smooth non-negative activation; ReLU kills gradients below zero)
- `LeakyReLU(0.1)` in ConvBlocks (prevents dead neurons on small dataset)
- `Dropout2d` on encoders (0.1) and bottleneck (0.2) (drops whole channels, stronger than standard Dropout for conv layers)

**`src/train.py`** — `CountAwareLoss` = MSE(density map) + λ·MAE(total count). Uses AMP (`torch.amp.autocast`), `CosineAnnealingWarmRestarts(T_0=50, T_mult=2)`, gradient clipping (max_norm=1.0), and early stopping on val RMSE. Optional `torch.compile` wraps the model after `.to(device)`. Training curves are saved with EMA smoothing.

**`src/evaluate.py`** — reports MAE/RMSE with 95% bootstrap CIs (2000 resamples) and a per-quartile breakdown. Saves a scatter plot + residual plot. Test loader uses `num_workers=4`.

**`src/predict.py`** — single-image inference. `predict_single_image()` loads, resizes (PIL mode `'F'` + LANCZOS to match training), normalises, and runs the model. `load_trained_model()` loads `best_model.pth`. `predict_and_visualize()` saves a 3-panel figure (input / density map / overlay with count).

**`run.py`** — orchestrates everything. Uses stratified train/val/test split (70/15/15) binned by count quartile so all count ranges appear in each partition — important for ~100-image datasets. Falls back to random split if stratification fails. Outputs are written to `outputs/` (model checkpoint, training curves, evaluation plots, `prediction.png`).

## Data Constraints

- Place images in `data/images/` and annotations in `data/annotations/`; filenames must match exactly (e.g., `img001.tif` ↔ `img001.csv`)
- CSV must have `X` and `Y` columns (case-insensitive)
- Augmentation (flips, rotations, zoom-crop, brightness, contrast, gamma, noise, blur) is training-only
- Count in the loss always reflects post-augmentation points (zoom-crop can remove edge points)
- Validation/inference runs with `model.eval()` — Dropout2d is disabled, results are deterministic
