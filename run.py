"""
Main entry point for the quantum dot counter.

Usage:
    python run.py                  # Train and evaluate
    python run.py --mode test      # Evaluate only (requires trained model)
    python run.py --mode predict --image path/to/image.png  # Single image prediction
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import preprocess_dataset, visualize_sample
from src.dataset import QuantumDotDataset
from src.model import UNet, count_parameters
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_and_visualize, load_trained_model


def main():
    parser = argparse.ArgumentParser(description='Quantum Dot Counter')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'predict', 'eda'],
                        help='Mode to run')
    parser.add_argument('--images_dir', type=str, default='data/images')
    parser.add_argument('--annotations_dir', type=str, default='data/annotations')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path for predict mode')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_radius', type=int, default=8,
                        help='Dot disk radius in pixels for binary mask (default: 8)')
    parser.add_argument('--pos_weight', type=float, default=15.0,
                        help='CrossEntropyLoss weight for dot class vs background (default: 15)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--compile', action='store_true',
                        help='Compile model with torch.compile() for faster training (PyTorch 2.0+)')
    parser.add_argument('--plateau_scheduler', action='store_true',
                        help='Use ReduceLROnPlateau instead of CosineAnnealingWarmRestarts')

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device — prefer CUDA, then Apple MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    target_size = (args.target_size, args.target_size)

    # ============================================================
    # MODE: EDA — Explore your data
    # ============================================================
    if args.mode == 'eda':
        print("\n=== EXPLORATORY DATA ANALYSIS ===\n")
        dataset = preprocess_dataset(args.images_dir, args.annotations_dir, target_size)

        if len(dataset) == 0:
            print("No data found! Check your file paths and naming.")
            return

        # Print stats
        counts = [s['count'] for s in dataset]
        print(f"\nDataset Statistics:")
        print(f"  Total images: {len(dataset)}")
        print(f"  QD counts: min={min(counts)}, max={max(counts)}, "
              f"mean={np.mean(counts):.1f}, median={np.median(counts):.1f}")

        # Visualize first 5 samples
        output_dir = Path(args.output_dir) / 'eda'
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(dataset[:5]):
            visualize_sample(sample, save_path=output_dir / f'sample_{i}.png')

        print(f"\nEDA visualizations saved to {output_dir}/")
        return

    # ============================================================
    # MODE: PREDICT — Single image inference
    # ============================================================
    if args.mode == 'predict':
        if args.image is None:
            print("Error: --image path required for predict mode")
            return

        model_path = Path(args.output_dir) / 'best_model.pth'
        if not model_path.exists():
            print(f"Error: No trained model found at {model_path}")
            print("Run training first: python run.py --mode train")
            return

        model = load_trained_model(model_path, device)
        save_path = Path(args.output_dir) / 'prediction.png'
        count = predict_and_visualize(model, args.image, device, save_path=save_path)
        print(f"\nPredicted count: {count:.1f}")
        return

    # ============================================================
    # LOAD AND SPLIT DATA (for train and test modes)
    # ============================================================
    print("\n=== Loading and preprocessing data ===\n")
    dataset = preprocess_dataset(args.images_dir, args.annotations_dir, target_size)

    if len(dataset) == 0:
        print("No data found! Check your file paths and naming.")
        print(f"  Images dir: {args.images_dir}")
        print(f"  Annotations dir: {args.annotations_dir}")
        print("  Image files should be .png, .jpg, .tif, etc.")
        print("  CSV files must have matching names (e.g., img1.png ↔ img1.csv)")
        return

    if len(dataset) < 5:
        print(f"Warning: Only {len(dataset)} samples found. Need at least 5 for train/val/test split.")
        print("Using all data for both training and validation (not recommended).")
        train_data = dataset
        val_data = dataset
        test_data = dataset
    else:
        # Stratified split by count quartile — ensures all count ranges are represented
        # in each partition, critical for small datasets (~100 images).
        split_done = False
        if _HAS_PANDAS:
            try:
                counts_arr = np.array([s['count'] for s in dataset])
                count_bins = pd.qcut(counts_arr, q=4, labels=False, duplicates='drop')
                train_data, temp_data = train_test_split(
                    dataset, test_size=0.3, random_state=args.seed, stratify=count_bins)
                temp_counts = np.array([s['count'] for s in temp_data])
                temp_bins = pd.qcut(temp_counts, q=2, labels=False, duplicates='drop')
                val_data, test_data = train_test_split(
                    temp_data, test_size=0.5, random_state=args.seed, stratify=temp_bins)
                print("Using stratified split (by count quartile).")
                split_done = True
            except ValueError:
                pass

        if not split_done:
            # Fall back to random split
            train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=args.seed)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed)
            print("Using random split.")

    print(f"Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

    # Create PyTorch datasets
    train_dataset = QuantumDotDataset(train_data, mask_radius=args.mask_radius, augment=True)
    val_dataset   = QuantumDotDataset(val_data,   mask_radius=args.mask_radius, augment=False)
    test_dataset  = QuantumDotDataset(test_data,  mask_radius=args.mask_radius, augment=False)

    # ============================================================
    # MODE: TRAIN
    # ============================================================
    if args.mode == 'train':
        print("\n=== Building model ===\n")
        model = UNet(in_channels=1, out_channels=2, base_filters=32)
        print(f"U-Net parameters: {count_parameters(model):,}")

        # Quick sanity check
        sample_img, sample_mask, sample_count = train_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample mask shape:  {sample_mask.shape}  dtype={sample_mask.dtype}")
        print(f"Dot pixels: {(sample_mask == 1).sum().item()}  GT count: {sample_count.item():.0f}")

        print("\n=== Starting training ===\n")
        model, history = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            device=device,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            pos_weight=args.pos_weight,
            mask_radius=args.mask_radius,
            patience=args.patience,
            compile_model=args.compile,
            use_plateau_scheduler=args.plateau_scheduler,
        )

        # Evaluate on test set
        print("\n=== Evaluating on test set ===\n")
        results = evaluate_model(model, test_dataset, device,
                                 output_dir=args.output_dir,
                                 mask_radius=args.mask_radius)

        # Visualize predictions on test set
        print("\n=== Saving test predictions ===\n")
        pred_dir = Path(args.output_dir) / 'test_predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(test_data[:10]):
            img_name   = sample['name']
            images_dir = Path(args.images_dir)
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                img_path = images_dir / f"{img_name}{ext}"
                if img_path.exists():
                    predict_and_visualize(
                        model, img_path, device,
                        save_path=pred_dir / f'{img_name}_pred.png',
                        gt_count=sample['count'],
                        mask_radius=args.mask_radius,
                    )
                    break

    # ============================================================
    # MODE: TEST — Evaluate existing model
    # ============================================================
    elif args.mode == 'test':
        model_path = Path(args.output_dir) / 'best_model.pth'
        if not model_path.exists():
            print(f"Error: No trained model found at {model_path}")
            return

        model = load_trained_model(model_path, device)
        results = evaluate_model(model, test_dataset, device,
                                 output_dir=args.output_dir,
                                 mask_radius=args.mask_radius)


if __name__ == '__main__':
    main()