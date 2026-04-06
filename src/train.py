"""
Training loop for binary dot segmentation U-Net.

Loss: CrossEntropyLoss(weight=[1, pos_weight]) — directly from the
      DeepTrack2 multi-particle tracking tutorial (weight=[1, 10]).

Count at inference: sum(softmax[:, 1]) / (pi * mask_radius^2).
This handles overlapping dots better than connected-component counting.
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path


def logits_to_counts(logits, mask_radius):
    """
    Derive dot count from segmentation logits.

    sum(P(dot)) / dot_area  handles overlapping dots gracefully —
    each dot contributes its area of probability mass regardless of
    whether it overlaps a neighbour.
    """
    probs = torch.softmax(logits, dim=1)     # (B, 2, H, W)
    dot_probs = probs[:, 1]                   # (B, H, W)
    dot_area = math.pi * mask_radius ** 2
    return dot_probs.sum(dim=(1, 2)) / dot_area   # (B,)


def train_model(
    train_dataset,
    val_dataset,
    model,
    device,
    output_dir='outputs',
    num_epochs=200,
    batch_size=8,
    learning_rate=1e-4,
    pos_weight=10.0,
    mask_radius=5,
    patience=30,
    compile_model=False,
    use_plateau_scheduler=False,
):
    """
    Train the binary segmentation U-Net.

    Returns:
        model:   trained model (best checkpoint loaded)
        history: dict with per-epoch metrics
    """
    torch.backends.cudnn.benchmark = True

    _log_path = Path(output_dir) / 'training_progress.txt'
    def _log(msg):
        print(msg, flush=True)
        with open(_log_path, 'a') as _f:
            _f.write(msg + '\n')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _workers = 0 if device.type == 'mps' else min(4, (os.cpu_count() or 4) // 2)
    _pin     = device.type == 'cuda'
    _persist = _workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=_workers, persistent_workers=_persist,
        pin_memory=_pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=_workers, persistent_workers=_persist,
        pin_memory=_pin,
    )

    # CrossEntropyLoss with class weights [background=1, dot=pos_weight]
    # Mirrors the DeepTrack2 tutorial: CrossEntropyLoss(weight=[1, 10])
    weight = torch.tensor([1.0, pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if use_plateau_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=10, min_lr=1e-6, verbose=True)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    _amp_enabled   = device.type == 'cuda'
    _scaler_device = 'cuda' if device.type == 'cuda' else 'cpu'
    try:
        scaler = torch.amp.GradScaler(_scaler_device, enabled=_amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled)

    model = model.to(device)

    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_rmse': [], 'val_rmse': [],
    }
    best_val_rmse  = float('inf')
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training on {device}")
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")
    print(f"Batch: {batch_size}  LR: {learning_rate}  pos_weight: {pos_weight}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        t0 = time.time()

        # ---- Training ----
        model.train()
        train_losses, train_errs, train_sq_errs = [], [], []

        for images, masks, counts in train_loader:
            images = images.to(device)
            masks  = masks.to(device)          # (B, H, W) int64

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=_amp_enabled):
                logits = model(images)         # (B, 2, H, W)
                loss   = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            with torch.no_grad():
                pred_counts = logits_to_counts(logits, mask_radius).cpu().numpy()
                gt_counts   = counts.numpy()
                train_errs.extend(np.abs(pred_counts - gt_counts))
                train_sq_errs.extend((pred_counts - gt_counts) ** 2)

        # ---- Validation ----
        model.eval()
        val_losses, val_errs, val_sq_errs = [], [], []

        with torch.no_grad():
            for images, masks, counts in val_loader:
                images = images.to(device)
                masks  = masks.to(device)

                logits = model(images)
                loss   = criterion(logits, masks)

                val_losses.append(loss.item())

                pred_counts = logits_to_counts(logits, mask_radius).cpu().numpy()
                gt_counts   = counts.numpy()
                val_errs.extend(np.abs(pred_counts - gt_counts))
                val_sq_errs.extend((pred_counts - gt_counts) ** 2)

        # Metrics
        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        train_mae  = np.mean(train_errs)
        val_mae    = np.mean(val_errs)
        train_rmse = np.sqrt(np.mean(train_sq_errs))
        val_rmse   = np.sqrt(np.mean(val_sq_errs))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)

        if use_plateau_scheduler:
            scheduler.step(val_rmse)
        else:
            scheduler.step()

        if val_rmse < best_val_rmse:
            best_val_rmse     = val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            _log(f"Epoch {epoch+1:>3d}/{num_epochs} | "
                 f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                 f"Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | "
                 f"Train RMSE: {train_rmse:.2f} | Val RMSE: {val_rmse:.2f} | "
                 f"Best RMSE: {best_val_rmse:.2f} | {elapsed:.1f}s")

        if epochs_no_improve >= patience:
            _log(f"\nEarly stopping at epoch {epoch+1} "
                 f"(no improvement for {patience} epochs)")
            break

    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    _plot_training_curves(history, output_dir)
    model.load_state_dict(torch.load(output_dir / 'best_model.pth', weights_only=True))
    _log(f"\nTraining complete! Best Val RMSE: {best_val_rmse:.2f}")
    return model, history


def _ema(values, alpha=0.85):
    smoothed, s = [], values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        smoothed.append(s)
    return smoothed


def _plot_training_curves(history, output_dir):
    epochs = np.arange(1, len(history['train_loss']) + 1)

    TRAIN_COLOR = '#2166ac'
    VAL_COLOR   = '#d6604d'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ('train_loss',  'val_loss',  'Loss (log scale)', 'CE Loss',    True),
        ('train_mae',   'val_mae',   'Count MAE',        'MAE (dots)', False),
        ('train_rmse',  'val_rmse',  'Count RMSE',       'RMSE (dots)',False),
    ]

    for ax, (tk, vk, title, ylabel, use_log) in zip(axes, panels):
        t_raw, v_raw = history[tk], history[vk]
        ax.plot(epochs, t_raw,      color=TRAIN_COLOR, alpha=0.2, linewidth=0.8)
        ax.plot(epochs, v_raw,      color=VAL_COLOR,   alpha=0.2, linewidth=0.8)
        ax.plot(epochs, _ema(t_raw), color=TRAIN_COLOR, linewidth=2, label='Train')
        ax.plot(epochs, _ema(v_raw), color=VAL_COLOR,   linewidth=2, label='Val')

        best_ep = int(np.argmin(v_raw)) + 1
        ax.axvline(best_ep, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.annotate(f'best\n(ep {best_ep})', xy=(best_ep, min(v_raw)),
                    xytext=(6, 6), textcoords='offset points', fontsize=7, color='gray')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        if use_log:
            ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")
