"""
Inference: load a trained model, predict dot segmentation map, extract count.

Count method: sum(softmax[:, 1]) / (pi * mask_radius^2)
Localization: connected components on softmax[:, 1] > 0.001 (DeepTrack2 default)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure

from src.preprocess import load_image_as_grayscale, normalize_image
from src.model import UNet
from PIL import Image


def predict_single_image(model, image_path, device, target_size=(256, 256), mask_radius=5):
    """
    Predict dot count and segmentation for a single image.

    Returns:
        predicted_count: float  (probability-sum method)
        dot_prob_map:    (H, W) numpy array — P(dot) per pixel
        positions:       (N, 2) numpy array — detected dot centroids (row, col)
        processed_image: (H, W) numpy array
    """
    model.eval()

    image = load_image_as_grayscale(image_path)
    img_pil = Image.fromarray(image.astype(np.float32), mode='F')
    img_pil = img_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
    image = np.array(img_pil, dtype=np.float32)
    image = normalize_image(image)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)                          # (1, 2, H, W)
        probs  = F.softmax(logits, dim=1)                    # (1, 2, H, W)

    dot_prob_map = probs[0, 1].cpu().numpy()                 # (H, W)

    # Count via probability sum (handles overlapping dots)
    dot_area = math.pi * mask_radius ** 2
    predicted_count = float(dot_prob_map.sum() / dot_area)

    # Localise via connected components on softmax > 0.001 (DeepTrack2 threshold)
    binary = dot_prob_map > 0.001
    labeled = measure.label(binary)
    props   = measure.regionprops(labeled)
    positions = np.array([p.centroid for p in props]) if props else np.zeros((0, 2))

    return predicted_count, dot_prob_map, positions, image


def predict_and_visualize(model, image_path, device, save_path=None,
                          gt_count=None, mask_radius=5):
    """Predict and save a 3-panel figure."""
    count, dot_prob_map, positions, image = predict_single_image(
        model, image_path, device, mask_radius=mask_radius
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Input image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Panel 2: Dot probability map
    im = axes[1].imshow(dot_prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('P(dot) — segmentation output')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Panel 3: Detections overlaid
    axes[2].imshow(image, cmap='gray')
    if len(positions) > 0:
        axes[2].scatter(positions[:, 1], positions[:, 0],
                        s=80, facecolors='none', edgecolors='cyan', linewidths=1.2)
    title = f'Count (prob-sum): {count:.1f}  |  CC detections: {len(positions)}'
    if gt_count is not None:
        title += f'  (GT: {gt_count})'
    axes[2].set_title(title, fontsize=9)
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction to {save_path}")
    else:
        plt.show()
    plt.close()

    return count


def load_trained_model(model_path, device):
    """Load a trained U-Net model."""
    model = UNet(in_channels=1, out_channels=2, base_filters=32)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model
