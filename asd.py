import deeptrack as dt
import numpy as np
import tifffile
import os
import csv
import matplotlib.pyplot as plt

IMAGE_SIZE = 512

# Shared mutable state — re-sampled once per image so all particles in one
# image draw from the same bounding box.
# Small region = tightly packed/dense; full image = sparse/spread out.
pos_state = {"x0": 0, "y0": 0, "x1": IMAGE_SIZE, "y1": IMAGE_SIZE}
count_state = {"n": 50}

def _resample_layout():
    """Pick a random bounding box and dot count for the next image."""
    region_size = int(np.random.uniform(40, IMAGE_SIZE))
    x0 = np.random.randint(0, IMAGE_SIZE - region_size + 1)
    y0 = np.random.randint(0, IMAGE_SIZE - region_size + 1)
    pos_state.update({"x0": x0, "y0": y0,
                      "x1": x0 + region_size, "y1": y0 + region_size})
    count_state["n"] = np.random.randint(1, 101)  # 1–100 dots

# 1. Optics — fixed to a realistic widefield fluorescence microscope.
# Narrow ranges so the model sees a consistent PSF across all images.
# NA 1.2–1.4 (oil/water immersion), resolution 80–120 nm/px (fine sampling
# so dots span ~10–15 px), wavelength 500–600 nm (GFP/QD emission band).
optics = dt.Fluorescence(
    NA=lambda: np.random.uniform(1.2, 1.4),
    resolution=lambda: np.random.uniform(0.08e-6, 0.12e-6),
    magnification=10,
    wavelength=lambda: np.random.uniform(500e-9, 600e-9),
    padding=(32, 32, 32, 32),
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
)

# 2. Particles — in-focus only (z≈0) so all dots have the same sharp PSF.
# Intensity narrow enough that dots are always visible but realistically
# variable. Position drawn from the per-image bounding box.
qd = dt.PointParticle(
    position=lambda: np.array([
        np.random.uniform(pos_state["x0"], pos_state["x1"]),
        np.random.uniform(pos_state["y0"], pos_state["y1"]),
    ]),
    z=lambda: np.random.uniform(-0.5, 0.5),   # near-focus only
    intensity=lambda: np.random.uniform(500, 3000),
)

qd_population = qd ^ (lambda: count_state["n"])

# 3. Noise — raised SNR floor (10–40) so dots are always visible above noise.
# Gaussian sigma kept small (1–5) so dot shape is not smeared.
background = dt.Add(value=lambda: np.random.uniform(50, 150))
poisson_noise = dt.Poisson(
    snr=lambda: np.random.uniform(10, 40),
    background=background.value,
)
gaussian_noise = dt.Gaussian(mu=0, sigma=lambda: np.random.uniform(1, 5))

# 4. Pipeline
pipeline = optics(qd_population) >> background >> poisson_noise >> gaussian_noise

# store_properties() lets us read back the exact dot positions from each
# resolved image for ground-truth annotation.
pipeline.store_properties()

# 5. Output directories — structured to match the training pipeline's
# expected layout (data/images/ + data/annotations/).
dataset_dir = "synthetic_qd_dataset_diverse"
images_dir = os.path.join(dataset_dir, "images")
annotations_dir = os.path.join(dataset_dir, "annotations")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

num_images = 3000
print(f"Generating {num_images} synthetic Quantum Dot images...")

for i in range(num_images):
    _resample_layout()
    pipeline.update()
    image = pipeline.resolve()

    image_2d = np.squeeze(image)

    name = f"qd_sim_{i:03d}"

    # Save image
    tifffile.imwrite(
        os.path.join(images_dir, f"{name}.tif"),
        image_2d.astype(np.float32),
    )

    # Extract ground-truth positions from stored properties.
    # DeepTrack stores each particle's property dict in image.properties;
    # position is (row, col) so we swap to (X=col, Y=row) for ImageJ CSV format.
    positions = []
    for prop in image.properties:
        if "position" in prop:
            pos = prop["position"]
            positions.append((float(pos[1]), float(pos[0])))  # X, Y

    # Save annotation CSV — matches the training pipeline's expected format
    with open(os.path.join(annotations_dir, f"{name}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["X", "Y"])
        writer.writerows(positions)

    region_px = pos_state["x1"] - pos_state["x0"]
    print(f"  [{i+1:03d}/{num_images}] dots={len(positions):3d}  "
          f"region={region_px}px  "
          f"origin=({pos_state['x0']},{pos_state['y0']})")

print(f"\nDataset saved to ./{dataset_dir}/")
print(f"  Images      -> {images_dir}/")
print(f"  Annotations -> {annotations_dir}/")

# 6. Save one example as a preview (no blocking plt.show)
pipeline.update()
example = np.squeeze(pipeline.resolve())
plt.figure(figsize=(6, 6))
plt.imshow(example, cmap="magma")
plt.title("Synthetic Quantum Dot Example")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(dataset_dir, "example_preview.png"), dpi=100, bbox_inches="tight")
plt.close()
print("Example preview saved.")
