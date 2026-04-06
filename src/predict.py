"""
Inference: load a trained model, predict density map, extract count.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocess import load_image_as_grayscale, normalize_image
from src.model import UNet
from PIL import Image


def predict_single_image(model, image_path, device, target_size=(256, 256)):
    """
    Predict quantum dot count for a single image.

    Returns:
        predicted_count: float
        density_map: numpy array (H, W)
        processed_image: numpy array (H, W)
    """
    model.eval()

    # Load and preprocess
    image = load_image_as_grayscale(image_path)

    # Resize
    img_pil = Image.fromarray(image.astype(np.float32), mode='F')
    img_pil = img_pil.resize((target_size[1], target_size[0]), Image.LANCZOS)
    image = np.array(img_pil, dtype=np.float32)

    # Normalize
    image = normalize_image(image)

    # To tensor
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    # Predict
    with torch.no_grad():
        density_map_tensor, count_tensor = model(image_tensor)

    density_map = density_map_tensor.squeeze().cpu().numpy()  # (H, W)
    predicted_count = count_tensor.item()                     # use count head

    return predicted_count, density_map, image


def predict_and_visualize(model, image_path, device, save_path=None, gt_count=None):
    """Predict and create a visualization."""
    predicted_count, density_map, image = predict_single_image(model, image_path, device)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Predicted density map
    im = axes[1].imshow(density_map, cmap='hot')
    axes[1].set_title('Predicted Density Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(density_map, cmap='hot', alpha=0.5)
    title = f'Predicted Count: {predicted_count:.1f}'
    if gt_count is not None:
        title += f' (GT: {gt_count})'
    axes[2].set_title(title)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction to {save_path}")
    else:
        plt.show()

    plt.close()

    return predicted_count


def load_trained_model(model_path, device):
    """Load a trained U-Net model."""
    model = UNet(in_channels=1, out_channels=1, base_filters=32)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model