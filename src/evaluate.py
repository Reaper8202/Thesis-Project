"""
Evaluation metrics and test set analysis.
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import measure
import matplotlib.pyplot as plt
from pathlib import Path


def _bootstrap_ci(values, stat_fn, n_boot=2000, ci=0.95, seed=42):
    rng  = np.random.default_rng(seed)
    boot = [stat_fn(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return lo, hi


def evaluate_model(model, test_dataset, device, output_dir='outputs',
                   batch_size=8, mask_radius=5):
    """
    Evaluate model on test set.

    Count method: sum(softmax[:, 1]) / (pi * mask_radius^2)

    Returns:
        results: dict with metrics and per-image predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    _workers = min(4, (os.cpu_count() or 4) // 2)
    _pin     = str(device).startswith('cuda')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=_workers, persistent_workers=_workers > 0, pin_memory=_pin,
    )

    all_pred_counts = []
    all_gt_counts   = []

    with torch.no_grad():
        for images, masks, counts in test_loader:
            images = images.to(device)
            logits = model(images)                                   # (B, 2, H, W)
            probs  = F.softmax(logits, dim=1)                       # (B, 2, H, W)
            binary = (probs[:, 1] > 0.5).cpu().numpy()              # (B, H, W)

            # Connected-component counting — same method as DeepTrack2 tutorial
            for b in range(binary.shape[0]):
                labeled = measure.label(binary[b])
                all_pred_counts.append(float(labeled.max()))
            all_gt_counts.extend(counts.numpy())

    all_pred_counts = np.array(all_pred_counts)
    all_gt_counts   = np.array(all_gt_counts)

    errors     = all_pred_counts - all_gt_counts
    abs_errors = np.abs(errors)

    mae  = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(abs_errors / np.maximum(all_gt_counts, 1)) * 100
    correlation = (np.corrcoef(all_gt_counts, all_pred_counts)[0, 1]
                   if len(all_gt_counts) > 1 else float('nan'))

    mae_lo,  mae_hi  = _bootstrap_ci(abs_errors, np.mean)
    rmse_lo, rmse_hi = _bootstrap_ci(errors ** 2, lambda x: np.sqrt(np.mean(x)))

    quartile_stats = []
    if len(all_gt_counts) >= 8:
        quartiles = np.percentile(all_gt_counts, [0, 25, 50, 75, 100])
        for q in range(4):
            mask = (all_gt_counts >= quartiles[q]) & (all_gt_counts < quartiles[q + 1])
            if mask.sum() > 0:
                quartile_stats.append({
                    'range': f'{quartiles[q]:.0f}–{quartiles[q+1]:.0f}',
                    'n':    int(mask.sum()),
                    'mae':  np.mean(abs_errors[mask]),
                    'rmse': np.sqrt(np.mean(errors[mask] ** 2)),
                })

    results = {
        'mae': mae, 'mae_ci': (mae_lo, mae_hi),
        'rmse': rmse, 'rmse_ci': (rmse_lo, rmse_hi),
        'mape': mape, 'correlation': correlation,
        'pred_counts': all_pred_counts,
        'gt_counts':   all_gt_counts,
        'quartile_stats': quartile_stats,
    }

    print(f"\n{'='*52}")
    print(f"TEST SET EVALUATION ({len(all_gt_counts)} images)")
    print(f"{'='*52}")
    print(f"MAE:         {mae:.2f}  (95% CI: {mae_lo:.2f}–{mae_hi:.2f})")
    print(f"RMSE:        {rmse:.2f}  (95% CI: {rmse_lo:.2f}–{rmse_hi:.2f})")
    print(f"MAPE:        {mape:.1f}%")
    print(f"Correlation: {correlation:.4f}")
    print(f"{'='*52}")

    if quartile_stats:
        print(f"\nPer-quartile breakdown (GT count range):")
        print(f"  {'Range':<12} {'N':>4}  {'MAE':>7}  {'RMSE':>7}")
        print(f"  {'-'*38}")
        for q in quartile_stats:
            print(f"  {q['range']:<12} {q['n']:>4}  {q['mae']:>7.2f}  {q['rmse']:>7.2f}")

    print(f"\n{'Image':<8} {'GT':>6} {'Pred':>8} {'Error':>8}")
    print('-' * 34)
    for i in range(len(all_gt_counts)):
        err = errors[i]
        print(f"{i+1:<8} {all_gt_counts[i]:>6.0f} {all_pred_counts[i]:>8.1f} {err:>+8.1f}")

    _plot_pred_vs_gt(all_gt_counts, all_pred_counts, results, output_dir)
    return results


def _plot_pred_vs_gt(gt_counts, pred_counts, results, output_dir):
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    mae      = results['mae']
    corr     = results['correlation']
    mae_lo, mae_hi = results['mae_ci']
    abs_err  = np.abs(pred_counts - gt_counts)

    # Predicted vs Ground Truth
    ax = axes[0]
    sc = ax.scatter(gt_counts, pred_counts, c=abs_err, cmap='YlOrRd',
                    s=60, alpha=0.85, edgecolors='k', linewidths=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, label='|Error| (dots)', shrink=0.85)

    lo  = min(gt_counts.min(), pred_counts.min())
    hi  = max(gt_counts.max(), pred_counts.max())
    pad = (hi - lo) * 0.05
    ax_range = [lo - pad, hi + pad]
    ax.plot(ax_range, ax_range, 'k--', linewidth=1.5, label='Identity', zorder=2)

    if len(gt_counts) > 2:
        slope, intercept, r_val, p_val, se = scipy_stats.linregress(gt_counts, pred_counts)
        x_fit = np.linspace(ax_range[0], ax_range[1], 200)
        y_fit = slope * x_fit + intercept
        n, x_bar = len(gt_counts), gt_counts.mean()
        Sxx    = np.sum((gt_counts - x_bar) ** 2)
        t_crit = scipy_stats.t.ppf(0.975, df=n - 2)
        ci_band = t_crit * se * np.sqrt(1/n + (x_fit - x_bar)**2 / np.maximum(Sxx, 1e-9))
        ax.plot(x_fit, y_fit, color='#2166ac', linewidth=1.5,
                label=f'OLS (slope={slope:.2f})')
        ax.fill_between(x_fit, y_fit - ci_band, y_fit + ci_band,
                        color='#2166ac', alpha=0.15, label='95% CI')

    ax.set_xlim(ax_range); ax.set_ylim(ax_range); ax.set_aspect('equal')
    ax.set_xlabel('Ground Truth Count', fontsize=12)
    ax.set_ylabel('Predicted Count', fontsize=12)
    ax.set_title(f'Predicted vs Ground Truth\n'
                 f'MAE = {mae:.2f} [{mae_lo:.2f}–{mae_hi:.2f}]   R = {corr:.3f}',
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25)

    # Residuals
    ax2 = axes[1]
    residuals = pred_counts - gt_counts
    rmse = results['rmse']
    ax2.scatter(gt_counts, residuals, c=abs_err, cmap='YlOrRd',
                s=60, alpha=0.85, edgecolors='k', linewidths=0.4)
    ax2.axhline(0,     color='k',    linestyle='--', linewidth=1.5)
    ax2.axhline( rmse, color='gray', linestyle=':',  linewidth=1, label=f'±RMSE ({rmse:.1f})')
    ax2.axhline(-rmse, color='gray', linestyle=':',  linewidth=1)
    ax2.fill_between(ax_range, -rmse, rmse, color='gray', alpha=0.08)
    ax2.set_xlim(ax_range)
    ax2.set_xlabel('Ground Truth Count', fontsize=12)
    ax2.set_ylabel('Residual (Pred − GT)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.legend(frameon=False, fontsize=9)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / 'pred_vs_gt.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plots to {output_dir / 'pred_vs_gt.png'}")
