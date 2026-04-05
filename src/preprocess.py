"""
Preprocessing pipeline:
- Load images of any size/format (uint8, uint16, float32 TIF, RGB, RGBA)
- Convert to grayscale float32
- Resize to standard size with float precision (no uint8 quantization)
- Rescale annotations to match
- Normalize intensity with percentile-based robust scaling
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


def load_image_as_grayscale(image_path):
    """
    Load any image and return a grayscale float32 numpy array.

    Handles: uint8 (L, RGB, RGBA), 16-bit int (I, I;16), float32 TIF (F).
    High-bit-depth modes are returned without quantization; normalize_image()
    handles the range later via percentile scaling.
    """
    img = Image.open(image_path)

    # High-bit-depth modes: return raw float32 without quantization.
    # PIL mode 'I'   — 32-bit signed int (what Pillow uses for 16-bit TIFFs).
    # PIL mode 'I;16'— raw 16-bit unsigned (older Pillow / direct raw access).
    # PIL mode 'F'   — 32-bit float (some scientific instruments).
    # normalize_image() handles any value range via percentile clipping.
    if img.mode in ('I', 'I;16', 'F'):
        return np.array(img, dtype=np.float32)

    # Flatten alpha before any colour conversion
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Grayscale conversion
    if img.mode == 'RGB':
        # Luminance weights (0.299R + 0.587G + 0.114B).
        # For fluorescence images that are pseudo-coloured single-channel
        # (R==G==B), this is a no-op. For genuinely coloured fluorescence
        # where QD signal lives in one channel, consider extracting that
        # channel explicitly before calling this function.
        img = img.convert('L')
    elif img.mode != 'L':
        img = img.convert('L')

    return np.array(img, dtype=np.float32)


def load_annotations(csv_path):
    """
    Load annotations from ImageJ-style CSV.
    Returns list of (x, y) tuples.
    """
    df = pd.read_csv(csv_path)

    x_col = None
    y_col = None

    for col in df.columns:
        if col.strip().upper() == 'X':
            x_col = col
        elif col.strip().upper() == 'Y':
            y_col = col

    if x_col is None or y_col is None:
        raise ValueError(f"Could not find X and Y columns in {csv_path}. "
                         f"Found columns: {list(df.columns)}")

    points = list(zip(df[x_col].values, df[y_col].values))
    return points


def resize_image_and_points(image, points, target_size=(256, 256)):
    """
    Resize image and rescale annotation points accordingly.

    Uses PIL mode 'F' (float32) throughout to avoid uint8 quantization
    artifacts. LANCZOS resampling reduces aliasing when downscaling.

    Args:
        image: numpy float32 array (H, W), any value range
        points: list of (x, y) tuples
        target_size: (height, width)

    Returns:
        resized_image (float32), rescaled_points
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # PIL mode 'F' accepts any float32 value range — no uint8 conversion needed.
    # LANCZOS provides the highest-quality anti-aliased downsampling.
    img_pil = Image.fromarray(image.astype(np.float32), mode='F')
    img_pil = img_pil.resize((target_w, target_h), Image.LANCZOS)
    resized_image = np.array(img_pil, dtype=np.float32)

    rescaled_points = [(x * scale_x, y * scale_y) for x, y in points]

    return resized_image, rescaled_points


def normalize_image(image):
    """
    Percentile-based normalization to [0, 1].
    Robust to outlier hot/dead pixels common in fluorescence microscopy.
    Works correctly for any input range (uint8, 16-bit, float TIF).
    """
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)

    if p99 - p1 < 1e-6:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - p1) / (p99 - p1)
    normalized = np.clip(normalized, 0, 1)
    return normalized.astype(np.float32)


def preprocess_dataset(images_dir, annotations_dir, target_size=(256, 256)):
    """
    Load and preprocess all image-annotation pairs.

    Returns:
        list of dicts: [{'name': str, 'image': np.array, 'points': list, 'count': int}, ...]
    """
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    dataset = []
    skipped = []

    for img_path in image_files:
        name = img_path.stem

        csv_path = annotations_dir / f"{name}.csv"
        if not csv_path.exists():
            skipped.append(name)
            continue

        try:
            image = load_image_as_grayscale(img_path)
            points = load_annotations(csv_path)
            image, points = resize_image_and_points(image, points, target_size)
            image = normalize_image(image)

            dataset.append({
                'name': name,
                'image': image,
                'points': points,
                'count': len(points)
            })

        except Exception as e:
            print(f"Error processing {name}: {e}")
            skipped.append(name)

    print(f"Successfully loaded: {len(dataset)} image-annotation pairs")
    if skipped:
        print(f"Skipped (no matching CSV or error): {skipped}")

    return dataset


def visualize_sample(sample, save_path=None):
    """Visualize an image with its annotation points overlaid."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(sample['image'], cmap='gray')
    axes[0].set_title(f"{sample['name']} (raw)")
    axes[0].axis('off')

    axes[1].imshow(sample['image'], cmap='gray')
    if sample['points']:
        xs, ys = zip(*sample['points'])
        axes[1].scatter(xs, ys, c='red', s=20, marker='+', linewidths=1)
    axes[1].set_title(f"{sample['name']} — {sample['count']} QDs")
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()
