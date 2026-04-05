"""
PyTorch Dataset that generates Gaussian density maps from point annotations.
Includes data augmentation tailored for fluorescence microscopy.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from PIL import Image


def generate_density_map(height, width, points, sigma=3.0):
    """
    Generate a Gaussian density map from point annotations.

    Each point becomes a Gaussian blob. The sum of the entire
    density map equals the number of points.

    Args:
        height: image height
        width: image width
        points: list of (x, y) tuples
        sigma: standard deviation of Gaussian kernel

    Returns:
        density_map: numpy array (H, W), where sum ≈ len(points)
    """
    density = np.zeros((height, width), dtype=np.float32)

    if len(points) == 0:
        return density

    for x, y in points:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            density[yi, xi] += 1.0

    density = gaussian_filter(density, sigma=sigma, mode='constant')

    return density


class QuantumDotDataset(Dataset):
    """
    PyTorch Dataset for quantum dot counting.

    Each sample returns:
        image: (1, H, W) tensor
        density_map: (1, H, W) tensor
        count: scalar tensor (post-augmentation point count)
    """

    def __init__(self, data_list, sigma=3.0, augment=False):
        """
        Args:
            data_list: list of dicts from preprocess_dataset()
            sigma: Gaussian sigma for density maps
            augment: whether to apply data augmentation
        """
        self.data = data_list
        self.sigma = sigma
        self.augment = augment
        # Cache density maps for non-augmented datasets (val/test).
        # Maps are deterministic when augment=False, so regenerating them
        # each epoch wastes CPU cycles in the DataLoader workers.
        self._cache = {} if not augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return cached tensors for non-augmented datasets (val/test)
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        sample = self.data[idx]

        image = sample['image'].copy()       # (H, W), float32 in [0, 1]
        points = list(sample['points'])       # list of (x, y)
        h, w = image.shape

        if self.augment:
            image, points = self._augment(image, points, h, w)

        # Regenerate h, w after augmentation (zoom crop may have changed them
        # back to original size, but rotation can transiently swap them)
        h, w = image.shape

        density_map = generate_density_map(h, w, points, sigma=self.sigma)

        image_tensor = torch.from_numpy(image).unsqueeze(0)        # (1, H, W)
        density_tensor = torch.from_numpy(density_map).unsqueeze(0) # (1, H, W)
        # Use post-augmentation point count (zoom crop may remove edge points)
        count_tensor = torch.tensor(float(len(points)), dtype=torch.float32)

        result = (image_tensor, density_tensor, count_tensor)

        if self._cache is not None:
            self._cache[idx] = result

        return result

    def _augment(self, image, points, h, w):
        """
        Apply random augmentations to image and point annotations.

        Geometric transforms first (they affect both image and points),
        then photometric transforms (image only), then blur last.

        Augmentations chosen for fluorescence microscopy:
        - Flips/rotations: QDs have no orientation, fully valid.
        - Brightness/contrast/gamma: simulate laser power variation,
          detector response curves, and exposure differences.
        - Gaussian noise: simulates shot noise and detector read noise.
        - Zoom crop: scale invariance; points outside crop are discarded.
        - Gaussian blur: simulates defocus, diffraction limits, and
          aberrations common in fluorescence microscopes.
        """

        # ---- Geometric transforms (image + points) ----

        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            points = [(w - 1 - x, y) for x, y in points]

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            points = [(x, h - 1 - y) for x, y in points]

        # Random 90-degree rotation (safe for square 256x256 images)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k).copy()
            for _ in range(k):
                points = [(h - 1 - y, x) for x, y in points]
                h, w = w, h

        # Random zoom via crop-and-resize
        # PIL mode 'F' (float32) preserves full precision; no uint8 quantization.
        if np.random.rand() > 0.5:
            zoom = np.random.uniform(0.75, 0.95)
            crop_h = int(h * zoom)
            crop_w = int(w * zoom)
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)

            image = image[top:top + crop_h, left:left + crop_w]

            img_pil = Image.fromarray(image.astype(np.float32), mode='F')
            img_pil = img_pil.resize((w, h), Image.BILINEAR)
            image = np.array(img_pil, dtype=np.float32)

            scale_x = w / crop_w
            scale_y = h / crop_h
            new_points = []
            for x, y in points:
                nx = (x - left) * scale_x
                ny = (y - top) * scale_y
                if 0 <= nx < w and 0 <= ny < h:
                    new_points.append((nx, ny))
            points = new_points

        # ---- Photometric transforms (image only) ----
        # Image is float32 in [0, 1] at this point.

        # Random brightness adjustment (simulates laser power / exposure)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image * factor, 0, 1).astype(np.float32)

        # Random contrast adjustment (mean-preserving stretch)
        if np.random.rand() > 0.5:
            contrast_factor = np.random.uniform(0.75, 1.25)
            mean_val = image.mean()
            image = np.clip((image - mean_val) * contrast_factor + mean_val, 0, 1).astype(np.float32)

        # Random gamma correction (simulates detector non-linearity).
        # For x in [0, 1]: x^gamma stays in [0, 1], no clipping needed.
        # gamma < 1 brightens mid-tones; gamma > 1 darkens them.
        if np.random.rand() > 0.5:
            gamma = np.random.uniform(0.7, 1.5)
            image = np.power(image, gamma).astype(np.float32)

        # Random Gaussian noise (simulates shot noise and read noise)
        if np.random.rand() > 0.5:
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1).astype(np.float32)

        # Random Gaussian blur (simulates defocus and diffraction limits).
        # Applied last so blur interacts with the final intensity values.
        if np.random.rand() > 0.5:
            blur_sigma = np.random.uniform(0.3, 1.5)
            image = gaussian_filter(image, sigma=blur_sigma).astype(np.float32)

        return image, points
