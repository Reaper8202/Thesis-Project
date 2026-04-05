import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path


class CountAwareLoss(nn.Module):
    """
    Combined loss: weighted MSE on density map + MAE on total count.

    Plain MSE on a mostly-zero density map gives near-zero gradient on
    background pixels (the vast majority) and almost nothing on the few
    dot pixels.  Boosting dot pixels with `dot_weight` keeps the model
    from collapsing to the trivial "always predict zero background" solution.

    L = mean(w * (pred - target)^2) + count_weight * MAE(sum(pred), sum(gt))
    where w = 1 + (dot_weight - 1) * (target > 0)
    """

    def __init__(self, count_weight=0.01, dot_weight=1.0):
        super().__init__()
        self.count_weight = count_weight
        self.dot_weight = dot_weight

    def forward(self, pred, target):
        # Weighted MSE: dot pixels can be boosted vs background via dot_weight.
        # dot_weight=1 → plain MSE (works well when dots are consistent/visible).
        weights = 1.0 + (self.dot_weight - 1.0) * (target > 0).float()
        weighted_mse = (weights * (pred - target) ** 2).mean()

        # Count loss: absolute error on total count (sum of density map)
        pred_count = pred.sum(dim=(1, 2, 3))     # (B,)
        target_count = target.sum(dim=(1, 2, 3))  # (B,)
        count_loss = torch.abs(pred_count - target_count).mean()

        return weighted_mse + self.count_weight * count_loss


def train_model(
    train_dataset,
    val_dataset,
    model,
    device,
    output_dir='outputs',
    num_epochs=200,
    batch_size=8,
    learning_rate=1e-4,
    count_loss_weight=0.01,
    dot_weight=100.0,
    patience=30,
    compile_model=False,
    use_plateau_scheduler=False,
):
    """
    Full training loop.

    Returns:
        model: trained model
        history: dict with training metrics
    """
    torch.backends.cudnn.benchmark = True

    # Direct file logger — bypasses all stdout buffering issues
    _log_path = Path(output_dir) / 'training_progress.txt'
    def _log(msg):
        print(msg, flush=True)
        with open(_log_path, 'a') as _f:
            _f.write(msg + '\n')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data loaders
    # num_workers=0 on macOS: worker processes re-import all modules on spawn
    # (Mac default) which is prohibitively slow with heavy packages like deeptrack.
    # pin_memory only helps with CUDA DMA transfers.
    _workers = 0 if device.type == 'mps' else 4
    _pin = device.type == 'cuda'
    _persist = _workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=_workers, persistent_workers=_persist,
        pin_memory=_pin, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=_workers, persistent_workers=_persist,
        pin_memory=_pin
    )

    # Loss, optimizer, scheduler
    criterion = CountAwareLoss(count_weight=count_loss_weight, dot_weight=dot_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if use_plateau_scheduler:
        # Halve LR whenever val RMSE stops improving for 10 epochs
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=10, min_lr=1e-6, verbose=True)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    # Use new torch.amp API (torch.cuda.amp.* deprecated in PyTorch 2.3)
    _amp_enabled = device.type == 'cuda'
    _scaler_device = 'cuda' if device.type == 'cuda' else 'cpu'
    try:
        scaler = torch.amp.GradScaler(_scaler_device, enabled=_amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled)  # PyTorch < 2.3 fallback

    model = model.to(device)

    # torch.compile: ~10-30% training speedup on PyTorch 2.0+ via kernel fusion
    if compile_model:
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
        else:
            print("Warning: torch.compile() not available (requires PyTorch 2.0+). Skipping.")

    # Tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mae': [], 'val_mae': [],
        'train_rmse': [], 'val_rmse': [],
    }
    best_val_rmse = float('inf')
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f"Training on {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}, LR: {learning_rate}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        t0 = time.time()

        # ---- Training ----
        model.train()
        train_losses = []
        train_count_errors = []
        train_count_sq_errors = []

        for images, density_maps, counts in train_loader:
            images = images.to(device)
            density_maps = density_maps.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=_amp_enabled):
                predictions = model(images)
                loss = criterion(predictions, density_maps)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            # Count error
            with torch.no_grad():
                pred_counts = predictions.sum(dim=(1, 2, 3)).cpu().numpy()
                gt_counts = counts.numpy()
                abs_errs = np.abs(pred_counts - gt_counts)
                sq_errs = (pred_counts - gt_counts) ** 2
                train_count_errors.extend(abs_errs)
                train_count_sq_errors.extend(sq_errs)

        # ---- Validation ----
        model.eval()
        val_losses = []
        val_count_errors = []
        val_count_sq_errors = []

        with torch.no_grad():
            for images, density_maps, counts in val_loader:
                images = images.to(device)
                density_maps = density_maps.to(device)

                predictions = model(images)
                loss = criterion(predictions, density_maps)

                val_losses.append(loss.item())

                pred_counts = predictions.sum(dim=(1, 2, 3)).cpu().numpy()
                gt_counts = counts.numpy()
                abs_errs = np.abs(pred_counts - gt_counts)
                sq_errs = (pred_counts - gt_counts) ** 2
                val_count_errors.extend(abs_errs)
                val_count_sq_errors.extend(sq_errs)

        # Metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_mae = np.mean(train_count_errors)
        val_mae = np.mean(val_count_errors)
        train_rmse = np.sqrt(np.mean(np.array(train_count_sq_errors)))
        val_rmse = np.sqrt(np.mean(np.array(val_count_sq_errors)))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)

        # Learning rate scheduling
        if use_plateau_scheduler:
            scheduler.step(val_rmse)
        else:
            scheduler.step()

        # Early stopping check
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
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
            _log(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')

    # Plot training curves
    _plot_training_curves(history, output_dir)

    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth', weights_only=True))

    _log(f"\nTraining complete! Best Val RMSE: {best_val_rmse:.2f}")
    return model, history


def _ema(values, alpha=0.85):
    """Exponential moving average for smoothing noisy training curves."""
    smoothed, s = [], values[0]
    for v in values:
        s = alpha * s + (1 - alpha) * v
        smoothed.append(s)
    return smoothed


def _plot_training_curves(history, output_dir):
    """Save publication-quality training curves with EMA smoothing."""
    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Color palette (colorblind-safe)
    TRAIN_COLOR = '#2166ac'   # blue
    VAL_COLOR   = '#d6604d'   # red-orange

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ('train_loss',  'val_loss',  'Loss (log scale)', 'Loss',      True),
        ('train_mae',   'val_mae',   'Count MAE',        'MAE (dots)', False),
        ('train_rmse',  'val_rmse',  'Count RMSE',       'RMSE (dots)',False),
    ]

    for ax, (train_key, val_key, title, ylabel, use_log) in zip(axes, panels):
        t_raw = history[train_key]
        v_raw = history[val_key]
        t_smooth = _ema(t_raw)
        v_smooth = _ema(v_raw)

        # Raw traces (faint)
        ax.plot(epochs, t_raw, color=TRAIN_COLOR, alpha=0.2, linewidth=0.8)
        ax.plot(epochs, v_raw, color=VAL_COLOR,   alpha=0.2, linewidth=0.8)
        # Smoothed traces (bold)
        ax.plot(epochs, t_smooth, color=TRAIN_COLOR, linewidth=2, label='Train')
        ax.plot(epochs, v_smooth, color=VAL_COLOR,   linewidth=2, label='Val')

        # Mark best validation epoch
        best_epoch = int(np.argmin(v_raw)) + 1
        ax.axvline(best_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.annotate(f'best\n(ep {best_epoch})', xy=(best_epoch, min(v_raw)),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=7, color='gray')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(frameon=False, fontsize=10)
        if use_log:
            ax.set_yscale('log')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_dir / 'training_curves.png'}")
