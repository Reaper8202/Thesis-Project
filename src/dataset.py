"""
PyTorch Dataset that generates binary dot masks from point annotations.
Includes data augmentation tailored for fluorescence microscopy.

Target format: (H, W) LongTensor with values 0 (background) or 1 (dot).
This feeds directly into CrossEntropyLoss, matching the DeepTrack2 approach
of training on a binary segmentation task rather than density map regression.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from PIL import Image


def generate_dot_mask(height, width, points, radius=5):
    """
    Generate a binary segmentation mask from point annotations.

    Each dot centre becomes a filled disk of `radius` pixels.
    Overlapping disks are merged (equivalent to DeepTrack2's
    SampleToMasks with merge_method="or").

    Args:
        height, width: image dimensions
        points: list of (x, y) tuples (column, row)
        radius: disk radius in pixels

    Returns:
        mask: (H, W) int64 array, values in {0, 1}
    """
    mask = np.zeros((height, width), dtype=np.int64)
    if len(points) == 0:
        return mask

    yy, xx = np.ogrid[:height, :width]
    r2 = radius ** 2
    for x, y in points:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            mask[(xx - xi) ** 2 + (yy - yi) ** 2 <= r2] = 1

    return mask


class QuantumDotDataset(Dataset):
    """
    PyTorch Dataset for quantum dot counting via binary segmentation.

    Each sample returns:
        image:      (1, H, W) float32 tensor
        mask:       (H, W)    int64 tensor  — 0=background, 1=dot
        count:      scalar float32 tensor   — number of (post-aug) dots
    """

    def __init__(self, data_list, mask_radius=5, augment=False):
        """
        Args:
            data_list:   list of dicts from preprocess_dataset()
            mask_radius: disk radius in pixels for dot mask generation
            augment:     whether to apply data augmentation
        """
        self.data = data_list
        self.mask_radius = mask_radius
        self.augment = augment
        self._cache = {} if not augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        sample = self.data[idx]
        image = sample['image'].copy()   # (H, W), float32 in [0, 1]
        points = list(sample['points'])  # list of (x, y)
        h, w = image.shape

        if self.augment:
            image, points = self._augment(image, points, h, w)

        h, w = image.shape

        mask = generate_dot_mask(h, w, points, radius=self.mask_radius)

        image_tensor = torch.from_numpy(image).unsqueeze(0)      # (1, H, W) float32
        mask_tensor  = torch.from_numpy(mask)                    # (H, W)    int64
        count_tensor = torch.tensor(float(len(points)), dtype=torch.float32)

        result = (image_tensor, mask_tensor, count_tensor)

        if self._cache is not None:
            self._cache[idx] = result

        return result

    def _augment(self, image, points, h, w):
        """
        Random augmentations for fluorescence microscopy.
        Geometric transforms first (affect image + points),
        then photometric transforms (image only).
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            points = [(w - 1 - x, y) for x, y in points]

        # Random vertical flip
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            points = [(x, h - 1 - y) for x, y in points]

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k).copy()
            for _ in range(k):
                points = [(h - 1 - y, x) for x, y in points]
                h, w = w, h

        # Random zoom via crop-and-resize
        if np.random.rand() > 0.5:
            zoom = np.random.uniform(0.75, 0.95)
            crop_h = int(h * zoom)
            crop_w = int(w * zoom)
            top  = np.random.randint(0, h - crop_h + 1)
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
                ny = (y - top)  * scale_y
                if 0 <= nx < w and 0 <= ny < h:
                    new_points.append((nx, ny))
            points = new_points

        # Brightness
        if np.random.rand() > 0.5:
            image = np.clip(image * np.random.uniform(0.7, 1.3), 0, 1).astype(np.float32)

        # Contrast (mean-preserving)
        if np.random.rand() > 0.5:
            m = image.mean()
            image = np.clip((image - m) * np.random.uniform(0.75, 1.25) + m, 0, 1).astype(np.float32)

        # Gamma
        if np.random.rand() > 0.5:
            image = np.power(image, np.random.uniform(0.7, 1.5)).astype(np.float32)

        # Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, np.random.uniform(0.01, 0.05), image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1).astype(np.float32)

        # Gaussian blur
        if np.random.rand() > 0.5:
            image = gaussian_filter(image, sigma=np.random.uniform(0.3, 1.5)).astype(np.float32)

        return image, points
