"""Greedy tracer baseline evaluation driver with per-dataset configs and trajectory visualizations.

Runs the greedy steepest-ascent tracer on the val/test splits at the RL agent's settings and
scores it through the shared scorer for direct comparability.
"""

import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOLERANCE
from data.centerline_extraction import CenterlineExtractor
from data.dataloader import OUTPUT_DIR as _OUTPUT_BASE
from data.dataloader import TEST_DATASETS, get_data, get_test_data
from evaluation.scoring import score_prediction, write_eval_csvs
from models.greedy_tracer import GreedyTracerBaseline

# Match the RL agent's eval (run_rl_tracing): same resolution, tolerance, FOV, shared scorer.
RESIZE = (512, 512)
_extractor = CenterlineExtractor()

# Per-dataset greedy-tracer parameters; DEFAULT_GREEDY_PARAMS is used for the combined val set.
GREEDY_PARAMS = {
    'DRIVE': dict(
        sigma_min=0.5,
        sigma_max=2.5,
        num_scales=5,
        gauss_sigma=1.0,
        seed_thresh=0.1727,
        step_thresh=0.0909,
        min_length=10.0,
        thin_output=True,
        min_obj_size=0,
    ),
    'DRHAGIS': dict(
        sigma_min=0.5,
        sigma_max=2.5,
        num_scales=5,
        gauss_sigma=0.8,
        seed_thresh=0.4,
        step_thresh=0.1891,
        min_length=10.0,
        thin_output=True,
        min_obj_size=50,
    ),
}

DEFAULT_GREEDY_PARAMS = dict(
    sigma_min=0.5,
    sigma_max=3.0,
    num_scales=5,
    gauss_sigma=1.5,
    seed_thresh=0.25,
    step_thresh=0.15,
    min_length=15,
    thin_output=True,
    min_obj_size=0,
)

FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_LEGEND = 10
TOP_N_ORDER = 50
DPI = 200


def save_standard_panel(img_rgb, vesselness, gt_skel_vis, pred_skel_vis, mask, res, image_id, panels_dir):
    """Save a 4-panel figure (image, vesselness, GT|Pred skeletons, overlay) for one image."""
    fov_bin = (mask > 0).astype(np.float32)
    vessel_vis = vesselness * fov_bin

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')

    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original Image (ID: {image_id})', fontweight='bold', fontsize=FONT_SIZE_TITLE)

    axes[1].imshow(vessel_vis, cmap='gray')
    axes[1].set_title('Vesselness Map', fontweight='bold', fontsize=FONT_SIZE_TITLE)

    side_by_side = np.concatenate([gt_skel_vis, pred_skel_vis], axis=1)
    axes[2].imshow(side_by_side, cmap='gray')
    axes[2].set_title('1px Skeletons\n(Left: GT | Right: Pred)', fontweight='bold', fontsize=FONT_SIZE_TITLE)

    overlay = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8)
    overlay[..., 1] = gt_skel_vis
    overlay[..., 0] = pred_skel_vis
    axes[3].imshow(overlay)
    axes[3].set_title(
        f'Overlay Analysis\nF1@2px: {res.get("f1@2px", 0):.3f} | clDice: {res.get("clDice", 0):.3f} | IoU: {res.get("iou", 0):.3f}',
        fontweight='bold',
        color='darkblue',
        fontsize=FONT_SIZE_TITLE,
    )

    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='GT'),
        Patch(facecolor='red', edgecolor='black', label='Pred'),
        Patch(facecolor='yellow', edgecolor='black', label='Match'),
    ]
    axes[3].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=FONT_SIZE_LEGEND)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(panels_dir, f'{image_id}_greedy_panel.png'), bbox_inches='tight', dpi=DPI)
    plt.close()


def save_trajectory_panel(vesselness, mask, traces, image_id, traj_dir, dataset_name=''):
    """Save a trajectory-analysis figure (seed scatter, top-N visit order, trace-length histogram).

    No-op when ``traces`` is empty.
    """
    if len(traces) == 0:
        return

    fov_bin = (mask > 0).astype(np.float32)
    vessel_bg = vesselness * fov_bin
    trace_lengths = np.array([len(p) for p in traces])
    seeds = np.array([p[0] for p in traces])

    BG = '#0d0d0d'
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis('off')

    axes[0].imshow(vessel_bg, cmap='gray', vmin=0, vmax=1)
    axes[0].scatter(seeds[:, 1], seeds[:, 0], c='cyan', s=12, alpha=0.8)
    axes[0].set_title(f'Vesselness + {len(traces)} Seeds', color='white', fontsize=FONT_SIZE_TITLE)

    n_show = min(TOP_N_ORDER, len(traces))
    cmap_order = plt.cm.plasma
    order_norm = mcolors.Normalize(vmin=0, vmax=max(n_show - 1, 1))
    axes[1].imshow(vessel_bg, cmap='gray', alpha=0.2)
    for idx in range(n_show):
        coords = np.array(traces[idx])
        axes[1].plot(coords[:, 1], coords[:, 0], color=cmap_order(order_norm(idx)), linewidth=1.2)
    axes[1].set_title(f'Top-{n_show} Visit Order', color='white', fontsize=FONT_SIZE_TITLE)

    sm = plt.cm.ScalarMappable(cmap=cmap_order, norm=order_norm)
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Visit Order (0 = First)', color='white', fontsize=FONT_SIZE_LABEL)
    cbar.ax.yaxis.set_tick_params(colors='white')

    axes[2].axis('on')
    axes[2].set_facecolor('#1a1a1a')
    log_bins = np.logspace(np.log10(max(trace_lengths.min(), 1)), np.log10(trace_lengths.max()), 40)
    axes[2].hist(trace_lengths, bins=log_bins, color='#f07f2a', alpha=0.85)
    axes[2].set_xscale('log')
    axes[2].set_title('Length Distribution (log x)', color='white', fontsize=FONT_SIZE_TITLE)
    axes[2].tick_params(colors='white')
    axes[2].set_xlabel('Trace Length (pixels)', color='white', fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel('Count', color='white', fontsize=FONT_SIZE_LABEL)

    plt.suptitle(
        f'Greedy Tracer Trajectory Analysis — {dataset_name} — {image_id}', color='white', fontsize=FONT_SIZE_TITLE + 4, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(os.path.join(traj_dir, f'{image_id}_trajectory.png'), facecolor=BG, dpi=DPI, bbox_inches='tight')
    plt.close()


def evaluate(split):
    """Run the greedy tracer on one split and write panels, trajectories, and metric CSVs; returns per-image metrics.

    Args:
        split: ``"val"`` for the combined val set, otherwise a test dataset name.
    """
    output_dir = str(_OUTPUT_BASE / 'greedy' / 'RL_tracing_e2e' / split)
    panels_dir = os.path.join(output_dir, 'panels')
    traj_dir = os.path.join(output_dir, 'trajectories')
    os.makedirs(panels_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    if split == 'val':
        dataset, _ = get_data('greedy_tracer', 'val', batch_size=1, resize=RESIZE, tolerance=TOLERANCE)
        params = DEFAULT_GREEDY_PARAMS
    else:
        dataset, _ = get_test_data(split, 'greedy_tracer', batch_size=1, resize=RESIZE, tolerance=TOLERANCE)
        params = GREEDY_PARAMS.get(split, DEFAULT_GREEDY_PARAMS)
    model = GreedyTracerBaseline(**params)

    print(f'[{split}]  {len(dataset)} images\n')
    all_metrics = []

    for i in tqdm(range(len(dataset)), desc=f'Greedy Tracer — {split}'):
        sample = dataset[i]
        (image_id, img_rgb, fov_mask, vessel_mask) = (sample['id'], sample['image'], sample['fov_mask'], sample['vessel_mask'])

        # Same GT (dataloader centerline) and shared scorer as the RL agent → comparable to v12.
        gt_skel = (sample['centerline'] > 0).astype(np.uint8)

        pred_skel, vesselness, traces = model.extract_centerline(sample['preprocessed'], fov_mask=fov_mask, return_vesselness=True)

        dt = _extractor.compute_distance_transform(gt_skel.astype(np.float32), TOLERANCE)
        res = score_prediction(pred_skel, centerline=gt_skel, vessel_mask=vessel_mask, fov_mask=fov_mask, distance_transform=dt, tolerance=TOLERANCE)

        res.update({'image_id': image_id, 'num_traces': len(traces), 'median_len': (float(np.median([len(t) for t in traces])) if traces else 0.0)})
        all_metrics.append(res)

        save_standard_panel(img_rgb, vesselness, (gt_skel > 0) * 255, (pred_skel > 0) * 255, fov_mask, res, image_id, panels_dir)
        save_trajectory_panel(vesselness, fov_mask, traces, image_id, traj_dir, dataset_name=split)

    write_eval_csvs(output_dir, all_metrics)
    f1 = np.mean([m['f1@2px'] for m in all_metrics]) if all_metrics else float('nan')
    print(f'\n[greedy/{split}] {len(all_metrics)} imgs  f1@2px={f1:.4f}  → {output_dir}')
    return all_metrics


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='store_true', help='Evaluate on the val set')
    ap.add_argument('--test', action='store_true', help='Evaluate on the test datasets')
    args = ap.parse_args()
    if not args.eval and not args.test:  # no flag → run both
        args.eval = args.test = True
    if args.eval:
        evaluate('val')
    if args.test:
        for name in TEST_DATASETS:
            evaluate(name)
