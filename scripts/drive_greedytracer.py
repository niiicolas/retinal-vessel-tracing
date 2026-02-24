"""
drive_greedytracer.py
=========================
Greedy Tracer
Processes all 20 images in the DRIVE training set with 1px, 2px, and 3px metrics.
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm
from skimage.morphology import skeletonize
import pandas as pd

from baselines.greedy_tracer_baseline import GreedyTracerBaseline
from evaluation.metrics import CenterlineMetrics


# ==========================================
# CONFIG
# ==========================================
DRIVE_ROOT = r"C:\ZHAW\BA\data\DRIVE\training"
OUTPUT_DIR = r"C:\ZHAW\BA\retinal-vessel-tracing\results\greedy_tracer"

MODEL_CFG = dict(
    sigma_min    = 0.5,
    sigma_max    = 3.0,
    num_scales   = 5,
    gauss_sigma  = 1.5,
    seed_thresh  = 0.25,
    step_thresh  = 0.15,
    min_length   = 15,
    thin_output  = True,
    min_obj_size = 0,
)

# Figure settings
FONT_SIZE_TITLE  = 14
FONT_SIZE_LABEL  = 12
FONT_SIZE_TICK   = 10
FONT_SIZE_LEGEND = 10
TOP_N_ORDER      = 50   
DPI              = 200


# ==========================================
# HELPERS
# ==========================================
def load_full_dataset(drive_root: str):
    """Loads ALL image, manual, and mask paths from the directory."""
    img_paths  = sorted(glob.glob(os.path.join(drive_root, "images",     "*.tif")))
    gt_paths   = sorted(glob.glob(os.path.join(drive_root, "1st_manual", "*.gif")))
    mask_paths = sorted(glob.glob(os.path.join(drive_root, "mask",       "*.gif")))
    return img_paths, gt_paths, mask_paths


# ==========================================
# STANDARD 4-PANEL FIGURE
# ==========================================
def save_standard_panel(
    img_rgb:       np.ndarray,
    vesselness:    np.ndarray,
    gt_skel_vis:   np.ndarray,
    pred_skel_vis: np.ndarray,
    mask:          np.ndarray,
    res:           dict,
    image_id:      str,
    panels_dir:    str,
):
    fov_bin    = (mask > 0).astype(np.float32)
    vessel_vis = vesselness * fov_bin

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original Image [{image_id}]", fontweight='bold', fontsize=FONT_SIZE_TITLE)

    axes[1].imshow(vessel_vis, cmap='gray')
    axes[1].set_title("Frangi Vesselness Map", fontweight='bold', fontsize=FONT_SIZE_TITLE)

    side_by_side = np.concatenate([gt_skel_vis, pred_skel_vis], axis=1)
    axes[2].imshow(side_by_side, cmap='gray')
    axes[2].set_title("1px Skeletons\n(Left: GT | Right: Pred)",
                      fontweight='bold', fontsize=FONT_SIZE_TITLE)

    overlay = np.zeros((*img_rgb.shape[:2], 3), dtype=np.uint8)
    overlay[..., 1] = gt_skel_vis  # Green: GT
    overlay[..., 0] = pred_skel_vis # Red: Pred
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Overlay Analysis\nF1@2px: {res['f1@2px']:.3f} | clDice: {res.get('clDice', 0):.3f}",
        fontweight='bold', color='darkblue', fontsize=FONT_SIZE_TITLE,
    )

    legend_elements = [
        Patch(facecolor='green',  edgecolor='black', label='GT Missing (FN)'),
        Patch(facecolor='red',    edgecolor='black', label='Pred (False Pos)'),
        Patch(facecolor='yellow', edgecolor='black', label='Match (True Pos)'),
    ]
    axes[3].legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.15), ncol=3,
                   frameon=False, fontsize=FONT_SIZE_LEGEND)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(panels_dir, f"{image_id}_greedy_panel.png"),
                bbox_inches='tight', dpi=DPI)
    plt.close()


# ==========================================
# TRAJECTORY FIGURE
# ==========================================
def save_trajectory_panel(vesselness, mask, traces, image_id, traj_dir):
    if len(traces) == 0:
        return

    fov_bin       = (mask > 0).astype(np.float32)
    vessel_bg     = vesselness * fov_bin
    trace_lengths = np.array([len(p) for p in traces])
    N             = len(traces)
    seeds         = np.array([p[0] for p in traces])

    BG = '#0d0d0d'
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis('off')

    # Panel A: Seeds
    axes[0].imshow(vessel_bg, cmap='gray', vmin=0, vmax=1)
    axes[0].scatter(seeds[:, 1], seeds[:, 0], c='cyan', s=12, alpha=0.8)
    axes[0].set_title(f"Vesselness + {N} Seeds", color='white', fontsize=FONT_SIZE_TITLE)

    # Panel B: Top Traces
    n_show     = min(TOP_N_ORDER, N)
    cmap_order = plt.cm.plasma
    order_norm = mcolors.Normalize(vmin=0, vmax=max(n_show - 1, 1))
    axes[1].imshow(vessel_bg, cmap='gray', alpha=0.2)
    for i in range(n_show):
        coords = np.array(traces[i])
        color  = cmap_order(order_norm(i))
        axes[1].plot(coords[:, 1], coords[:, 0], color=color, linewidth=1.2)

    axes[1].set_title(f"Top-{n_show} Visit Order", color='white', fontsize=FONT_SIZE_TITLE)

    sm = plt.cm.ScalarMappable(cmap=cmap_order, norm=order_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Visit Order (0 = First/Strongest)', color='white', fontsize=FONT_SIZE_LABEL)
    cbar.ax.yaxis.set_tick_params(colors='white')

    # Panel C: Histogram
    axes[2].axis('on')
    axes[2].set_facecolor('#1a1a1a')
    log_bins = np.logspace(np.log10(max(trace_lengths.min(), 1)), np.log10(trace_lengths.max()), 40)
    axes[2].hist(trace_lengths, bins=log_bins, color='#f07f2a', alpha=0.85)
    axes[2].set_xscale('log')
    axes[2].set_title("Length Distribution (log x)", color='white', fontsize=FONT_SIZE_TITLE)
    axes[2].tick_params(colors='white')
    
    axes[2].set_xlabel('Trace Length (pixels)', color='white', fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel('Count (Number of Traces)', color='white', fontsize=FONT_SIZE_LABEL)

    plt.suptitle(f"Greedy Tracer Trajectory Analysis — Image {image_id}", 
                 color='white', fontsize=FONT_SIZE_TITLE + 4, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(traj_dir, f"{image_id}_trajectory.png"), facecolor=BG, dpi=DPI, bbox_inches='tight')
    plt.close()


# ==========================================
# MAIN
# ==========================================
def main():
    print("\nFull Dataset Evaluation — Greedy Tracer Baseline")

    panels_dir = os.path.join(OUTPUT_DIR, "panels")
    traj_dir   = os.path.join(OUTPUT_DIR, "trajectories")
    os.makedirs(panels_dir, exist_ok=True)
    os.makedirs(traj_dir,   exist_ok=True)

    # Modified: Load all images instead of slicing by VAL_SIZE
    img_paths, gt_paths, mask_paths = load_full_dataset(DRIVE_ROOT)
    num_total = len(img_paths)
    print(f"Found {num_total} images for processing.")

    model      = GreedyTracerBaseline(**MODEL_CFG)
    metrics_fn = CenterlineMetrics(tolerance_levels=[1, 2, 3])

    all_metrics = []

    # Process every image in the dataset
    for img_path, gt_path, mask_path in tqdm(zip(img_paths, gt_paths, mask_paths), total=num_total):
        image_id = os.path.basename(img_path).split('_')[0]

        # Load data
        img_rgb = np.array(Image.open(img_path).convert('RGB'))
        gt      = np.array(Image.open(gt_path).convert('L'))
        mask    = np.array(Image.open(mask_path).convert('L'))

        # Precompute GT
        gt_skel     = (skeletonize(gt > 128) * 255).astype(np.uint8)
        vessel_mask = (gt > 128).astype(np.uint8) * 255

        # Execute Baseline
        pred_skel, vesselness, traces = model.extract_centerline(
            img_rgb, external_fov_mask=mask, return_vesselness=True
        )

        # Compute all required metrics (1, 2, 3px)
        res = metrics_fn.compute_all_metrics(pred_skel, gt_skel, vessel_mask)
        res.update({
            'image_id': image_id,
            'num_traces': len(traces),
            'median_len': float(np.median([len(t) for t in traces])) if traces else 0.0
        })
        all_metrics.append(res)

        # Generate per-image visuals
        save_standard_panel(img_rgb, vesselness, gt_skel, pred_skel, mask, res, image_id, panels_dir)
        save_trajectory_panel(vesselness, mask, traces, image_id, traj_dir)

    # ── GLOBAL SUMMARY ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_metrics)
    target_cols = [
        "clDice",
        "f1@1px", "precision@1px", "recall@1px",
        "f1@2px", "precision@2px", "recall@2px",
        "f1@3px", "precision@3px", "recall@3px"
    ]

    summary_rows = []
    for col in target_cols:
        if col in df.columns:
            summary_rows.append({
                "Metric": col, 
                "Mean ± Std": f"{df[col].mean():.4f} ± {df[col].std():.4f}"
            })

    summary_df = pd.DataFrame(summary_rows)
    
    print("\n" + "=" * 55)
    print(f"   GREEDY TRACER — FULL DRIVE DATASET (N={num_total})")
    print("=" * 55)
    print(summary_df.to_string(index=False))
    print("=" * 55)
    
    # Save results
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "full_summary_metrics.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "per_image_detailed_metrics.csv"), index=False)
    print(f"Full results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()