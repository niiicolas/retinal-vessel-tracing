"""
drive_frangi.py
=========================
Frangi Vesselness Baseline
Processes all 20 images in the DRIVE training set with 1px, 2px, and 3px metrics.
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
from matplotlib.lines import Line2D

from baselines.frangi_baseline import FrangiBaseline
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG
# ==========================================
image_dir  = r"C:\ZHAW\BA\data\DRIVE\training\images"
manual_dir = r"C:\ZHAW\BA\data\DRIVE\training\1st_manual"
mask_dir   = r"C:\ZHAW\BA\data\DRIVE\training\mask"
output_dir = r"C:\ZHAW\BA\retinal-vessel-tracing\results\frangi_training"

panels_dir = os.path.join(output_dir, "panels")
os.makedirs(panels_dir, exist_ok=True)

# ==========================================
# INITIALIZE MODEL & METRICS
# ==========================================
model = FrangiBaseline()
# We initialize with 1, 2, and 3px tolerances to see the full performance spectrum
metrics_calculator = CenterlineMetrics(tolerance_levels=[1, 2, 3])

all_metrics = []
mosaic_data = []

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
])

print(f"Found {len(image_files)} training images.\n")

# ==========================================
# MAIN LOOP
# ==========================================
for fname in image_files:
    image_id = fname.split('_')[0]
    print(f"Processing {fname} ...")

    # --- Load image ---
    img_bgr = cv2.imread(os.path.join(image_dir, fname))
    image   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Load DRIVE FOV mask ---
    mask_candidates = sorted([f for f in os.listdir(mask_dir) if f.startswith(image_id)])
    if not mask_candidates:
        raise RuntimeError(f"No external DRIVE FOV mask found for image {image_id}.")

    mask_path = os.path.join(mask_dir, mask_candidates[0])
    mask_pil  = Image.open(mask_path).convert('L')
    mask_np   = np.array(mask_pil)
    fov_mask_bool = mask_np > 128
    external_mask = mask_np.astype(np.uint8)

    # --- Load manual GT ---
    manual_candidates = sorted([f for f in os.listdir(manual_dir) if f.startswith(image_id)])
    if not manual_candidates:
        print(f"  WARNING: No manual annotation for {image_id}, skipping.\n")
        continue

    gt_pil = Image.open(os.path.join(manual_dir, manual_candidates[0])).convert('L')
    gt_binary = (np.array(gt_pil) > 128) & fov_mask_bool

    # --- Run Frangi baseline ---
    pred_skeleton, vesselness = model.extract_centerline(
        image,
        return_vesselness=True,
        external_fov_mask=external_mask
    )

    # --- Skeletonize GT (Create the 1px Ground Truth) ---
    gt_skeleton = skeletonize(gt_binary)

    # --- Compute metrics for all tolerances ---
    raw_metrics = metrics_calculator.compute_all_metrics(
        pred_skeleton,
        gt_skeleton,
        gt_vessel_mask=gt_binary
    )

    # --- Print 2px results to console as a standard reference ---
    f1_2px   = raw_metrics.get('f1@2px', 0.0)
    prec_2px = raw_metrics.get('precision@2px', 0.0)
    rec_2px  = raw_metrics.get('recall@2px', 0.0)
    cldice   = raw_metrics.get('clDice', 0.0)

    print(f"  Precision@2px: {prec_2px:.4f}")
    print(f"  Recall@2px:    {rec_2px:.4f}")
    print(f"  F1 Score@2px:  {f1_2px:.4f}")
    print(f"  clDice:        {cldice:.4f}\n")

    # --- Store all computed metrics ---
    metrics_entry = {"image_id": image_id}
    metrics_entry.update(raw_metrics)
    all_metrics.append(metrics_entry)

    mosaic_data.append({
        "image_id":      image_id,
        "gt_skeleton":   gt_skeleton,
        "pred_skeleton": pred_skeleton,
        "metrics":       metrics_entry,
    })

    # --- Panel visualization ---
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(vesselness, cmap='gray')
    axes[1].set_title("Frangi Vesselness", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    combined_skel = np.hstack((
        gt_skeleton.astype(np.uint8) * 255,
        pred_skeleton.astype(np.uint8) * 255
    ))
    axes[2].imshow(combined_skel, cmap='gray')
    axes[2].set_title("1px Skeletons\n(Left: GT | Right: Pred)", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    h, w = pred_skeleton.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = gt_skeleton.astype(np.uint8) * 255 # GT in Green
    overlay[:, :, 0] = pred_skeleton.astype(np.uint8) * 255 # Pred in Red

    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay Analysis\nF1@2px: {f1_2px:.3f} | clDice: {cldice:.3f}",
                      fontsize=14, fontweight='bold')
    axes[3].axis('off')

    legend_elements = [
        Line2D([0], [0], color='green', lw=4, label='GT (1px)'),
        Line2D([0], [0], color='red', lw=4, label='Pred (1px)'),
    ]
    axes[3].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    plt.tight_layout()
    panel_path = os.path.join(panels_dir, f"{image_id}_training_comparison.png")
    plt.savefig(panel_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# MOSAIC OVERVIEW
# ==========================================
if mosaic_data:
    n = len(mosaic_data)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5))
    axes = axes.flatten()

    for i, data in enumerate(mosaic_data):
        h, w = data['pred_skeleton'].shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 1] = data['gt_skeleton'].astype(np.uint8) * 255
        overlay[:, :, 0] = data['pred_skeleton'].astype(np.uint8) * 255

        axes[i].imshow(overlay)
        axes[i].set_title(
            f"[{data['image_id']}] clDice: {data['metrics']['clDice']:.3f}\n"
            f"F1@2px: {data['metrics']['f1@2px']:.3f}",
            fontsize=9, fontweight='bold'
        )
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    mosaic_path = os.path.join(output_dir, "mosaic_overview.png")
    plt.savefig(mosaic_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved mosaic → {mosaic_path}")

# ==========================================
# SUMMARY TABLE (Showing 1px, 2px, 3px)
# ==========================================
df = pd.DataFrame(all_metrics)

# Organize columns so they print in a logical order
metric_cols = [
    "clDice",
    "f1@1px", "precision@1px", "recall@1px",
    "f1@2px", "precision@2px", "recall@2px",
    "f1@3px", "precision@3px", "recall@3px"
]

summary_rows = []
for col in metric_cols:
    if col in df.columns:
        summary_rows.append({
            "Metric": col,
            "Mean ± Std": f"{df[col].mean():.4f} ± {df[col].std():.4f}"
        })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "="*45)
print("   FRANGI BASELINE — DRIVE TRAINING SET")
print("="*45)
print(summary_df.to_string(index=False))
print("="*45)

# Save to disk
df.to_csv(os.path.join(output_dir, "metrics_per_image.csv"), index=False)
summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
print(f"\nSaved all results → {output_dir}")