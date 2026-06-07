"""Training-log plotters.

Renders summary figures from the CSVs that PPOTrainer / ImitationTrainer
write incrementally during training:

  - weights/ppo_log.csv         → ppo_log.png   (multi-panel: training stats,
                                                 reward components, eval metrics,
                                                 termination-reason fractions)
  - weights/imitation_log.csv   → imitation_log.png  (loss / accuracy / lr / grad)

Both functions are no-ops if the CSV is missing or malformed; they're meant
to be called at end-of-training and never block the run.

PPO traces are noisy; we draw the raw signal at low alpha and overlay a
rolling-mean smoothed line on top.  Smoothing window scales with run
length so 20-iter and 1000-iter runs both look reasonable.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use('Agg')  # headless — required on cluster nodes without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _smooth(series: pd.Series, window: int) -> pd.Series:
    """Centred rolling mean; falls back to identity when the window is 1."""
    if window <= 1 or len(series) < 2:
        return series
    return series.rolling(window, center=True, min_periods=1).mean()


def _plot_with_smooth(ax, x: pd.Series, y: pd.Series, window: int, *, label: str, color: Optional[str] = None, raw_alpha: float = 0.25) -> None:
    """Raw line at low alpha plus a rolling-mean overlay.  Single label."""
    if y.dropna().empty:
        return
    ax.plot(x, y, color=color, alpha=raw_alpha, linewidth=0.8)
    ax.plot(x, _smooth(y, window), color=color, label=label, linewidth=1.6)


def _has_any(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    """True if any of ``cols`` is present in ``df`` with at least one non-NaN value."""
    return any(c in df.columns and df[c].notna().any() for c in cols)


# ─────────────────────────────────────────────────────────────────────────────
# PPO log
# ─────────────────────────────────────────────────────────────────────────────

# Reward component columns expected in PPOTrainer's CSV.  Kept in sync with
# RewardCalculator.BREAKDOWN_KEYS by convention; if a column is missing
# (older log) we silently skip it.
_PPO_REWARD_COMPONENTS = (
    'r_coverage',
    'r_frontier',
    'r_near',
    'r_off_vessel',
    'r_revisit',
    'r_step_cost',
    'r_shaping',
    'r_terminal',
)
_PPO_VAL_COLS = (
    'val_coverage',
    'val_f1',
    'val_cldice',
)
_PPO_VAL_RECALL = (
    'val_recall_thin',
    'val_recall_med',
    'val_recall_thick',
)
_PPO_TERM_COLS = (
    'term_stop_frac',
    'term_off_track_frac',
    'term_max_steps_frac',
    'term_oob_frac',
)


def plot_ppo_log(csv_path: str, out_path: Optional[str] = None) -> Optional[str]:
    """Render the PPO training log to a PNG next to the CSV.

    Returns the output path on success, or None if the CSV is missing /
    empty / malformed.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if df.empty or 'iteration' not in df.columns:
        return None

    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + '.png'

    n = len(df)
    window = max(5, n // 20)  # ~5% of the run, min 5
    iters = df['iteration']

    # ── Layout: 5 panels stacked vertically ──────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(11, 16), sharex=True)
    (ax_reward, ax_loss, ax_components, ax_eval, ax_term) = axes

    # Panel 1: mean episode reward + episode length (twin y).
    _plot_with_smooth(ax_reward, iters, df.get('mean_reward'), window, label=f'mean_reward (smoothed w={window})', color='tab:blue')
    ax_reward.set_ylabel('mean episode reward', color='tab:blue')
    ax_reward.tick_params(axis='y', labelcolor='tab:blue')
    ax_reward.grid(True, alpha=0.3)
    ax_reward.set_title(f'PPO training log — {n} iterations  (window={window})')

    if 'mean_ep_length' in df.columns:
        ax_len = ax_reward.twinx()
        _plot_with_smooth(ax_len, iters, df['mean_ep_length'], window, label='mean_ep_length', color='tab:orange')
        ax_len.set_ylabel('mean episode length', color='tab:orange')
        ax_len.tick_params(axis='y', labelcolor='tab:orange')
        # Combine legends from both axes
        h1, l1 = ax_reward.get_legend_handles_labels()
        h2, l2 = ax_len.get_legend_handles_labels()
        ax_reward.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
    else:
        ax_reward.legend(loc='upper left', fontsize=8)

    # Panel 2: policy/value loss + entropy.
    if _has_any(df, ('policy_loss', 'value_loss')):
        if 'policy_loss' in df.columns:
            _plot_with_smooth(ax_loss, iters, df['policy_loss'], window, label='policy_loss', color='tab:red')
        if 'value_loss' in df.columns:
            _plot_with_smooth(ax_loss, iters, df['value_loss'], window, label='value_loss', color='tab:purple')
        ax_loss.set_ylabel('loss')
        ax_loss.grid(True, alpha=0.3)

        if 'entropy' in df.columns:
            ax_ent = ax_loss.twinx()
            _plot_with_smooth(ax_ent, iters, df['entropy'], window, label='entropy', color='tab:green')
            ax_ent.set_ylabel('entropy', color='tab:green')
            ax_ent.tick_params(axis='y', labelcolor='tab:green')
            h1, l1 = ax_loss.get_legend_handles_labels()
            h2, l2 = ax_ent.get_legend_handles_labels()
            ax_loss.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
        else:
            ax_loss.legend(loc='upper right', fontsize=8)

    # Panel 3: per-component reward means (one line per BREAKDOWN_KEY present).
    component_cols = [c for c in _PPO_REWARD_COMPONENTS if c in df.columns]
    if component_cols:
        cmap = plt.get_cmap('tab10')
        for i, c in enumerate(component_cols):
            _plot_with_smooth(ax_components, iters, df[c], window, label=c, color=cmap(i % 10))
        ax_components.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax_components.set_ylabel('per-step reward (mean)')
        ax_components.grid(True, alpha=0.3)
        ax_components.legend(loc='upper right', fontsize=7, ncol=2)

    # Panel 4: validation metrics (sparse — only present at eval iters).
    val_present = [c for c in (_PPO_VAL_COLS + _PPO_VAL_RECALL) if c in df.columns]
    val_drawn = False
    if val_present:
        cmap = plt.get_cmap('Set1')
        for i, c in enumerate(val_present):
            ys = df[c]
            ys_clean = ys.dropna()
            if ys_clean.empty:
                continue
            xs_clean = iters[ys_clean.index]
            ax_eval.plot(xs_clean, ys_clean, marker='o', markersize=3, linewidth=1.3, label=c, color=cmap(i % 9))
            val_drawn = True
        if val_drawn:
            ax_eval.set_ylabel('eval metric')
            ax_eval.set_ylim(0.0, 1.0)
            ax_eval.grid(True, alpha=0.3)
            ax_eval.legend(loc='lower right', fontsize=7, ncol=2)

    # Panel 5: termination-reason fractions as a stacked area chart.
    term_present = [c for c in _PPO_TERM_COLS if c in df.columns]
    if term_present:
        # Replace NaN with 0 only for stacking; smoothing tolerates NaN already.
        stack = np.stack([_smooth(df[c].fillna(0.0), window).values for c in term_present], axis=0)
        cmap = plt.get_cmap('tab20')
        ax_term.stackplot(iters, stack, labels=term_present, colors=[cmap(i % 20) for i in range(len(term_present))], alpha=0.8)
        ax_term.set_ylabel('termination fraction')
        ax_term.set_ylim(0.0, 1.0)
        ax_term.legend(loc='upper right', fontsize=7, ncol=4)
        ax_term.grid(True, alpha=0.3)

    axes[-1].set_xlabel('iteration')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Imitation log
# ─────────────────────────────────────────────────────────────────────────────


def plot_imitation_log(csv_path: str, out_path: Optional[str] = None) -> Optional[str]:
    """Render the imitation training log to a PNG next to the CSV."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if df.empty or 'epoch' not in df.columns:
        return None

    if out_path is None:
        out_path = os.path.splitext(csv_path)[0] + '.png'

    n = len(df)
    epochs = df['epoch']

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax_loss, ax_acc, ax_other = axes

    if 'train_loss' in df.columns:
        ax_loss.plot(epochs, df['train_loss'], marker='o', markersize=3, label='train_loss', color='tab:blue')
    if 'val_loss' in df.columns:
        ax_loss.plot(epochs, df['val_loss'], marker='o', markersize=3, label='val_loss', color='tab:red')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title(f'Imitation training log — {n} epochs')
    ax_loss.grid(True, alpha=0.3)
    if ax_loss.lines:
        ax_loss.legend(loc='upper right', fontsize=8)

    if _has_any(df, ('train_acc', 'val_acc')):
        if 'train_acc' in df.columns:
            ax_acc.plot(epochs, df['train_acc'], marker='o', markersize=3, label='train_acc', color='tab:blue')
        if 'val_acc' in df.columns:
            ax_acc.plot(epochs, df['val_acc'], marker='o', markersize=3, label='val_acc', color='tab:red')
        ax_acc.set_ylabel('accuracy')
        ax_acc.set_ylim(0.0, 1.0)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(loc='lower right', fontsize=8)

    # Bottom panel: lr (log scale) + grad_norm (twin axis).
    if 'lr' in df.columns:
        ax_other.semilogy(epochs, df['lr'], marker='o', markersize=3, label='lr', color='tab:green')
        ax_other.set_ylabel('learning rate (log)', color='tab:green')
        ax_other.tick_params(axis='y', labelcolor='tab:green')
    ax_other.grid(True, alpha=0.3)

    if 'train_grad_norm' in df.columns:
        ax_gn = ax_other.twinx()
        ax_gn.plot(epochs, df['train_grad_norm'], marker='o', markersize=3, label='train_grad_norm', color='tab:orange')
        ax_gn.set_ylabel('grad norm', color='tab:orange')
        ax_gn.tick_params(axis='y', labelcolor='tab:orange')
        h1, l1 = ax_other.get_legend_handles_labels()
        h2, l2 = ax_gn.get_legend_handles_labels()
        ax_other.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)
    elif ax_other.lines:
        ax_other.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('epoch')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out_path
