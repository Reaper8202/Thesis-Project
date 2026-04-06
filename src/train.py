import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path


class DualHeadLoss(nn.Module):
    """
    Loss for the dual-head UNet:

      1. Weighted MSE on the density map (spatial supervision).
         dot_weight upweights non-zero (dot) pixels to fight the class
         imbalance — background pixels dominate a density map.

      2. MAE on the count-head output (direct count supervision, strong weight).
         This is the primary count loss: the count head gets its own direct
         gradient path, avoiding the old problem of summing a density map
         that had converged to a near-zero constant.

      3. MAE on the density-map sum (soft count regulariser, weak weight).
         Keeps the density map geometrically consistent with the count.

    L = MSE_weighted(density) + head_weight * MAE(count_head, gt)
                              + 0.01        * MAE(sum(density), gt)
    """

    def __init__(self, head_weight=1.0, dot_weight=100.0):
        super().__init__()
        self.head_weight = head_weight
        self.dot_weight = dot_weight

    def forward(self, pred_density, pred_count, target_density, target_count):
        # 1. Weighted MSE on density map
        weights = 1.0 + (self.dot_weight - 1.0) * (target_density > 0).float()
        density_loss = (weights * (pred_density - target_density) ** 2).mean()

        # 2. Count head MAE (strong direct supervision)
        head_count_loss = torch.abs(pred_count - target_count).mean()

        # 3. Density-map sum MAE (soft spatial regulariser)
        density_sum = pred_density.sum(dim=(1, 2, 3))
        density_count_loss = torch.abs(density_sum - target_count).mean()

        return density_loss + self.head_weight * head_count_loss + 0.01 * density_count_loss


def train_model(
    train_dataset,
    val_dataset,
    model,
    device,
    output_dir='outputs',
    num_epochs=200,
    batch_size=8,
    learning_rate=1e-4,
    count_loss_weight=1.0,
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

    _log_path = Path(output_dir) / 'training_progress.txt'
    def _log(msg):
        print(msg, flush=True)
        with open(_log_path, 'a') as _f:
            _f.write(msg + '\n')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _workers = 0 if device.type == 'mps' else min(4, (os.cpu_count() or 4) // 2)
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

    criterion = DualHeadLoss(head_weight=count_loss_weight, dot_weight=dot_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if use_plateau_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=10, min_lr=1e-6, verbose=True)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    _amp_enabled = device.type == 'cuda'
    _scaler_device = 'cuda' if device.type == 'cuda' else 'cpu'
    try:
        scaler = torch.amp.GradScaler(_scaler_device, enabled=_amp_enabled)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled)

    model = model.to(device)

    if compile_model:
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
        else:
            print("Warning: torch.compile() not available (requires PyTorch 2.0+). Skipping.")

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
            counts_dev = counts.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=_amp_enabled):
                pred_density, pred_count = model(images)
                loss = criterion(pred_density, pred_count, density_maps, counts_dev)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

            with torch.no_grad():
                # Use count head as primary count prediction
                pred_counts_np = pred_count.cpu().numpy()
                gt_counts_np = counts.numpy()
                abs_errs = np.abs(pred_counts_np - gt_counts_np)
                sq_errs = (pred_counts_np - gt_counts_np) ** 2
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
                counts_dev = counts.to(device)

                pred_density, pred_count = model(images)
                loss = criterion(pred_density, pred_count, density_maps, counts_dev)

                val_losses.append(loss.item())

                pred_counts_np = pred_count.cpu().numpy()
                gt_counts_np = counts.numpy()
                abs_errs = np.abs(pred_counts_np - gt_counts_np)
                sq_errs = (pred_counts_np - gt_counts_np) ** 2
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

        if use_plateau_scheduler:
            scheduler.step(val_rmse)
        else:
            scheduler.step()

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

    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    _plot_training_curves(history, output_dir)
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

    TRAIN_COLOR = '#2166ac'
    VAL_COLOR   = '#d6604d'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ('train_loss',  'val_loss',  'Loss (log scale)', 'Loss',       True),
        ('train_mae',   'val_mae',   'Count MAE',        'MAE (dots)', False),
        ('train_rmse',  'val_rmse',  'Count RMSE',       'RMSE (dots)',False),
    ]

    for ax, (train_key, val_key, title, ylabel, use_log) in zip(axes, panels):
        t_raw = history[train_key]
        v_raw = history[val_key]
        t_smooth = _ema(t_raw)
        v_smooth = _ema(v_raw)

        ax.plot(epochs, t_raw, color=TRAIN_COLOR, alpha=0.2, linewidth=0.8)
        ax.plot(epochs, v_raw, color=VAL_COLOR,   alpha=0.2, linewidth=0.8)
        ax.plot(epochs, t_smooth, color=TRAIN_COLOR, linewidth=2, label='Train')
        ax.plot(epochs, v_smooth, color=VAL_COLOR,   linewidth=2, label='Val')

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
