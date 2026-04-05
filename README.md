# Quantum Dot Counter

A PyTorch U-Net model for counting quantum dots (QDs) in fluorescence microscopy images using density map regression. Instead of detecting individual dots, the model learns to predict a Gaussian-smoothed density map whose integral equals the dot count — a formulation that is robust to overlap, low signal-to-noise, and varying dot appearance.

**Hardware target:** NVIDIA GTX 1660 Super (6 GB VRAM)  
**Dataset:** ~100 annotated fluorescence microscopy images  
**Primary metric:** RMSE on predicted dot count

---

## Table of Contents

1. [Setup](#setup)
2. [Data Format](#data-format)
3. [Quick Start](#quick-start)
4. [All CLI Options](#all-cli-options)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Data Preprocessing](#data-preprocessing)
8. [Data Augmentation](#data-augmentation)
9. [Evaluation Outputs](#evaluation-outputs)
10. [Project Structure](#project-structure)

---

## Setup

```bash
pip install torch torchvision numpy pandas scipy scikit-learn matplotlib pillow
```

Verify your GPU is visible:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Data Format

```
data/
  images/          # PNG, JPG, TIF — any bit depth, any size, grayscale or RGB
  annotations/     # One CSV per image, filename must match exactly
outputs/           # Created automatically — checkpoints, logs, figures
```

**CSV format** — ImageJ Multi-point output (or any CSV with `X` and `Y` columns):

```
X,Y
142.3,88.1
205.7,301.4
...
```

Column names are case-insensitive. The coordinates are pixel positions in the *original* image before resizing.

---

## Quick Start

### 1. Explore your data before training

```bash
python run.py --mode eda
```

Saves visualizations of the first 5 samples with annotation overlays to `outputs/eda/`. Use this to verify that annotations loaded correctly and that counts look reasonable.

### 2. Train

```bash
python run.py --mode train --epochs 200 --batch_size 8
```

The model trains for up to 200 epochs with early stopping (patience 30). The best checkpoint by validation RMSE is saved to `outputs/best_model.pth`.

### 3. Evaluate on the test set

```bash
python run.py --mode test
```

Prints MAE, RMSE, and MAPE with 95% bootstrap confidence intervals, plus a per-quartile breakdown. Saves a scatter plot and residual plot to `outputs/pred_vs_gt.png`.

### 4. Predict on a single image

```bash
python run.py --mode predict --image path/to/image.tif
```

Saves a three-panel figure (raw image / density map / overlay with count) to `outputs/prediction.png`.

### 5. Train with `torch.compile` (PyTorch 2.0+, ~15–30% faster)

```bash
python run.py --mode train --compile
```

---

## All CLI Options

| Argument | Default | Description |
|---|---|---|
| `--mode` | `train` | `train`, `test`, `predict`, `eda` |
| `--images_dir` | `data/images` | Path to image folder |
| `--annotations_dir` | `data/annotations` | Path to CSV annotation folder |
| `--output_dir` | `outputs` | Where to save checkpoints and figures |
| `--image` | — | Image path for `--mode predict` |
| `--target_size` | `256` | Resize images to this square resolution |
| `--sigma` | `3.0` | Gaussian sigma for density map generation (pixels) |
| `--batch_size` | `8` | Training batch size |
| `--epochs` | `200` | Maximum training epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--patience` | `30` | Early stopping patience (epochs) |
| `--count_weight` | `0.01` | Weight on count MAE term in loss |
| `--seed` | `42` | Random seed for reproducibility |
| `--compile` | off | Enable `torch.compile()` for faster training |

**Example — tune loss weight and use a larger batch:**

```bash
python run.py --mode train --count_weight 0.05 --batch_size 16 --epochs 300 --patience 50
```

---

## Model Architecture

The model is a standard encoder-decoder U-Net with skip connections, sized for 256×256 grayscale input on a 6 GB GPU.

```
Input  (B, 1, 256, 256)
  │
  ├── Encoder
  │     enc1: ConvBlock(1   → 32)   + Dropout2d(0.10)    256×256
  │     enc2: ConvBlock(32  → 64)   + Dropout2d(0.10)    128×128
  │     enc3: ConvBlock(64  → 128)  + Dropout2d(0.10)     64×64
  │     enc4: ConvBlock(128 → 256)  + Dropout2d(0.10)     32×32
  │
  ├── Bottleneck
  │     ConvBlock(256 → 512) + Dropout2d(0.20)            16×16
  │
  └── Decoder (with skip connections from encoder)
        up4 + cat(enc4): ConvBlock(512 → 256)             32×32
        up3 + cat(enc3): ConvBlock(256 → 128)             64×64
        up2 + cat(enc2): ConvBlock(128 → 64)             128×128
        up1 + cat(enc1): ConvBlock(64  → 32)             256×256
        Conv1×1(32 → 1) → Softplus(β=10)
  │
Output (B, 1, 256, 256)  — predicted density map, non-negative
```

Each `ConvBlock` is: Conv3×3 → BatchNorm → LeakyReLU(0.1) × 2.

**Total parameters: ~7.76 M**

### Design Decisions

#### Density Map Regression vs. Detection

Direct dot detection (e.g., template matching, peak finding) fails when dots overlap at high density or have varying appearance. Density map regression sidesteps this by learning a smooth signal whose integral is the count. This is the same approach used in crowd counting (CSRNet, MCNN) and transfers well to fluorescence QD images.

The predicted count is simply `density_map.sum()`.

#### AvgPool2d instead of MaxPool2d

MaxPool discards 75% of activations per 2×2 region, keeping only the strongest response. For density estimation this is destructive — spatial density information is lost in the pooled feature maps, hurting count accuracy. AvgPool preserves the total density energy across each pooled region.

#### Softplus(β=10) output activation instead of ReLU

The output density map must be non-negative, so some activation is needed. ReLU has zero gradient for negative pre-activations, which prevents the model from correcting over-suppressed regions during backpropagation. Softplus — `log(1 + exp(β·x)) / β` — is smooth and strictly positive everywhere, giving non-zero gradients throughout. With β=10 it closely approximates ReLU above zero while keeping gradients alive below it.

#### LeakyReLU(0.1) in ConvBlocks instead of ReLU

With a small dataset (~100 images), dead neuron syndrome (ReLU units that produce zero and never recover) is a real risk. LeakyReLU passes a small negative slope (0.1) for negative activations, keeping all units trainable throughout.

#### Dropout2d for regularisation on a small dataset

Standard Dropout drops individual activations. Dropout2d drops entire feature map channels, which is more effective for convolutional layers because spatially correlated activations within a channel are either all dropped or all kept. This prevents the network from relying on any specific channel's response and provides stronger regularisation per dropped unit. Rates are 0.1 on encoder blocks (light regularisation to preserve spatial features) and 0.2 on the bottleneck (heavier, since the bottleneck is the most overfit-prone part with the smallest spatial resolution).

#### Base filter count = 32

At 256×256 input with 6 GB VRAM, 32 base filters give ~7.76 M parameters and use roughly 2–3 GB of VRAM during training with AMP (mixed precision). This leaves room for a batch size of 8 without running out of memory, while still providing enough capacity for the task.

---

## Training Pipeline

### Loss Function: `CountAwareLoss`

```
L = MSE(density_pred, density_gt) + λ · MAE(sum(pred), sum(gt))
```

The MSE term optimises the spatial structure of the density map (pixel-level accuracy). The MAE term on the total count directly penalises count errors, providing a strong gradient signal early in training when the density map is still diffuse. Default λ = 0.01 (tune with `--count_weight`).

### Optimizer and Scheduler

- **Adam** with `lr=1e-4`, `weight_decay=1e-4`
- **CosineAnnealingWarmRestarts** (`T_0=50`, `T_mult=2`)

Cosine annealing with warm restarts avoids the need for manual learning rate tuning. The LR decays smoothly from `lr` to near-zero over `T_0` epochs, then restarts. Each restart cycle is twice as long as the previous (`T_mult=2`), giving the model time to explore coarse structure early and refine fine structure later.

Weight decay (L2 regularisation) on Adam provides additional regularisation complementary to Dropout2d.

### Automatic Mixed Precision (AMP)

Training uses `torch.amp.autocast` and `GradScaler` to run the forward pass in float16 where safe, falling back to float32 for numerically sensitive operations. On the GTX 1660 Super this roughly halves VRAM usage and increases throughput by ~40–50% due to the GPU's Tensor Core utilisation.

Gradient scaling compensates for the reduced dynamic range of float16 gradients, preventing underflow during backpropagation.

### Gradient Clipping

Gradients are clipped to `max_norm=1.0` before each optimizer step. This stabilises training when the loss landscape is steep — particularly important early in training with a small dataset where the model can make large erratic updates.

### Early Stopping

Training stops if validation RMSE does not improve for `patience` (default 30) consecutive epochs. The best checkpoint is saved independently of early stopping, so the final model is always the best-seen rather than the last.

### Data Split

The dataset is split 70% train / 15% val / 15% test. Splitting uses **stratified sampling by count quartile** — images are binned into four equal-frequency count ranges before splitting, ensuring each partition represents the full range of dot densities. With only ~100 images, a naive random split has a meaningful probability of placing all high-density images in one partition.

---

## Data Preprocessing

All preprocessing is handled by `src/preprocess.py` and applied once at load time.

1. **Grayscale conversion** — RGB images are converted using luminance weights (0.299R + 0.587G + 0.114B). 16-bit TIFs (PIL modes `I`, `I;16`) and float TIFs (mode `F`) are read directly as float32 without quantisation.

2. **Resize to 256×256** — Uses PIL mode `F` (float32) with LANCZOS resampling throughout, avoiding the uint8 quantisation that would occur if the image were converted to 8-bit before resizing. Annotation coordinates are rescaled by the same scale factors.

3. **Percentile normalisation** — Normalises to [0, 1] using the 1st and 99th percentiles instead of min/max. This is robust to hot pixels and dead pixels, which are common in fluorescence microscopy and would otherwise compress the dynamic range of the entire image to near-zero.

---

## Data Augmentation

Augmentation is applied only to the training set, online (per epoch), in `src/dataset.py`. Validation and test sets see the original images with no augmentation.

All augmentations are chosen specifically for fluorescence microscopy:

| Augmentation | Probability | Rationale |
|---|---|---|
| Horizontal flip | 50% | QDs have no orientation |
| Vertical flip | 50% | QDs have no orientation |
| 90° rotation (k = 0–3) | uniform | QD images have no preferred axis |
| Zoom crop (0.75–0.95×) | 50% | Scale invariance; points outside crop are removed from the count |
| Brightness ×(0.7–1.3) | 50% | Simulates laser power and exposure variation |
| Contrast stretch (0.75–1.25×) | 50% | Simulates detector response differences |
| Gamma correction (0.7–1.5) | 50% | Simulates detector non-linearity |
| Gaussian noise σ=(0.01–0.05) | 50% | Simulates shot noise and read noise |
| Gaussian blur σ=(0.3–1.5) | 50% | Simulates defocus and diffraction-limited PSF variation |

Geometric transforms are applied first (they must update both image and point coordinates). Photometric transforms are applied after (image only). Blur is applied last so it interacts with the final intensity values.

**Count consistency:** The zoom-crop augmentation may crop out points near the edges. The ground-truth count tensor passed to the loss is always updated to reflect only the points that remain inside the cropped region.

---

## Evaluation Outputs

Running `--mode test` produces:

| Output | Location | Description |
|---|---|---|
| Console metrics | stdout | MAE, RMSE, MAPE, correlation with 95% bootstrap CIs |
| Per-quartile table | stdout | RMSE broken down by GT count range |
| Per-image table | stdout | GT count, predicted count, error for every test image |
| Scatter + residual plot | `outputs/pred_vs_gt.png` | Left: predicted vs. GT with OLS fit and CI band; Right: residuals vs. GT |

**Bootstrap confidence intervals** are computed by resampling the test errors 2000 times with replacement, giving a distribution-free estimate of uncertainty on the metrics. These are suitable for reporting in thesis tables.

Running `--mode train` additionally produces:

| Output | Location | Description |
|---|---|---|
| Best checkpoint | `outputs/best_model.pth` | Weights at lowest validation RMSE |
| Final checkpoint | `outputs/final_model.pth` | Weights at end of training |
| Training curves | `outputs/training_curves.png` | Loss / MAE / RMSE per epoch with EMA smoothing and best-epoch marker |
| Test predictions | `outputs/test_predictions/` | Three-panel visualisations for up to 10 test images |

---

## Project Structure

```
├── run.py                  # CLI entry point
├── src/
│   ├── preprocess.py       # Image loading, resizing, normalisation, annotation parsing
│   ├── dataset.py          # PyTorch Dataset; density map generation; augmentation
│   ├── model.py            # U-Net architecture
│   ├── train.py            # Training loop, loss, scheduler, AMP
│   ├── evaluate.py         # Test set metrics, bootstrap CIs, plots
│   └── predict.py          # Single-image inference and visualisation
├── data/
│   ├── images/             # Input images (PNG, JPG, TIF)
│   └── annotations/        # ImageJ-format CSVs (one per image)
└── outputs/                # Created automatically at runtime
```
