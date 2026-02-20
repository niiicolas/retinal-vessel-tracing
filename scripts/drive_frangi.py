import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.morphology import skeletonize
from matplotlib.lines import Line2D
from baselines.frangi_baseline import FrangiBaseline

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
# MAIN LOOP
# ==========================================
model = FrangiBaseline()
all_metrics = []
mosaic_data = []

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))
])

print(f"Found {len(image_files)} training images.\n")

for fname in image_files:
    image_id = fname.split('_')[0]  # e.g. "21" from "21_training.tif"
    print(f"Processing {fname} ...")

    # --- Load image ---
    img_bgr = cv2.imread(os.path.join(image_dir, fname))
    image   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Load FOV mask ---
    mask_candidates = sorted([f for f in os.listdir(mask_dir) if f.startswith(image_id)])
    if mask_candidates:
        mask_pil     = Image.open(os.path.join(mask_dir, mask_candidates[0])).convert('L')
        fov_mask     = np.array(mask_pil) > 128
        fov_mask_u8  = fov_mask.astype(np.uint8) * 255  # ← uint8 for preprocessor
    else:
        fov_mask     = np.ones(image.shape[:2], dtype=bool)
        fov_mask_u8  = None  # ← None → preprocessor generates its own

    # --- Load manual GT ---
    manual_candidates = sorted([f for f in os.listdir(manual_dir) if f.startswith(image_id)])
    if not manual_candidates:
        print(f"  WARNING: No manual annotation for {image_id}, skipping.")
        continue
    gt_pil    = Image.open(os.path.join(manual_dir, manual_candidates[0])).convert('L')
    gt_binary = (np.array(gt_pil) > 128) & fov_mask

    # --- Run Frangi WITH DRIVE FOV mask ---
    pred_skeleton, vesselness = model.extract_centerline(
        image,
        return_vesselness=True,
        external_fov_mask=fov_mask_u8   # ← DRIVE mask flows into preprocessor
    )
    pred_skeleton = (pred_skeleton > 0) & fov_mask

    # --- GT Skeleton ---
    gt_skeleton = skeletonize(gt_binary)

    # --- Compute metrics ---
    raw = model._compute_metrics(
    pred_skeleton,
    gt_skeleton,
    gt_mask=gt_binary
    )

    f1_val     = raw.get('f1', 0.0)
    cldice_val = raw.get('clDice', 0.0)
    prec_val   = raw.get('precision', 0.0)
    rec_val    = raw.get('recall', 0.0)

    print(f"  Precision: {prec_val:.4f}")
    print(f"  Recall:    {rec_val:.4f}")
    print(f"  F1 Score:  {f1_val:.4f}")
    print(f"  clDice:    {cldice_val:.4f}\n")

    all_metrics.append({
        "image_id":  image_id,
        "clDice":    cldice_val,
        "F1":        f1_val,
        "Precision": prec_val,
        "Recall":    rec_val,
    })
    mosaic_data.append({
        "image_id":      image_id,
        "gt_skeleton":   gt_skeleton,
        "pred_skeleton": pred_skeleton,
        "metrics":       all_metrics[-1],
    })

    # ==========================================
    # IMAGE 1: Full Comparison Panel (4 panels)
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))

    # 1. Original
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. Vesselness
    axes[1].imshow(vesselness, cmap='gray')
    axes[1].set_title("Frangi Vesselness", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. Skeletons side-by-side
    combined_skel = np.hstack((
        gt_skeleton.astype(np.uint8) * 255,
        pred_skeleton.astype(np.uint8) * 255
    ))
    axes[2].imshow(combined_skel, cmap='gray')
    axes[2].set_title("Skeletons\n(Left: GT | Right: Pred)", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # 4. Overlay: Green=GT, Red=Pred, Yellow=Match
    h, w = pred_skeleton.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = (gt_skeleton > 0) * 255
    overlay[:, :, 0] = (pred_skeleton > 0) * 255

    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Overlay Analysis\nF1: {f1_val:.3f} | clDice: {cldice_val:.3f}",
        fontsize=14, fontweight='bold', color='darkblue'
    )
    axes[3].axis('off')

    legend_elements = [
        Line2D([0], [0], color='green',  lw=4, label='GT (Missing)'),
        Line2D([0], [0], color='red',    lw=4, label='Pred (False Pos)'),
        Line2D([0], [0], color='yellow', lw=4, label='Match (True Pos)'),
    ]
    axes[3].legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=10, frameon=False)

    plt.tight_layout()
    panel_path = os.path.join(panels_dir, f"{image_id}_training_comparison.png")
    plt.savefig(panel_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved panel → {panel_path}")

# ==========================================
# MOSAIC: All overlays in one figure
# ==========================================
n = len(mosaic_data)
n_cols = 4
n_rows = int(np.ceil(n / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

for i, data in enumerate(mosaic_data):
    h, w = data['pred_skeleton'].shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 1] = (data['gt_skeleton'] > 0) * 255
    overlay[:, :, 0] = (data['pred_skeleton'] > 0) * 255

    axes[i].imshow(overlay)
    axes[i].set_title(
        f"[{data['image_id']}]  clDice: {data['metrics']['clDice']:.3f}\n"
        f"F1: {data['metrics']['F1']:.3f} | "
        f"Prec: {data['metrics']['Precision']:.3f} | "
        f"Rec: {data['metrics']['Recall']:.3f}",
        fontsize=9, fontweight='bold'
    )
    axes[i].axis('off')

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

legend_elements = [
    Line2D([0], [0], color='green',  lw=4, label='GT (Missing)'),
    Line2D([0], [0], color='red',    lw=4, label='Pred (False Pos)'),
    Line2D([0], [0], color='yellow', lw=4, label='Match (True Pos)'),
]
fig.legend(handles=legend_elements, loc='lower center',
           ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.suptitle("Frangi Baseline — DRIVE Training Set Overview",
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
mosaic_path = os.path.join(output_dir, "mosaic_overview.png")
plt.savefig(mosaic_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nSaved mosaic → {mosaic_path}")

# ==========================================
# SUMMARY TABLE: Mean ± Std
# ==========================================
df = pd.DataFrame(all_metrics)
metric_cols = ["clDice", "F1", "Precision", "Recall"]

summary_rows = []
for col in metric_cols:
    summary_rows.append({"Metric": col, "Mean ± Std": f"{df[col].mean():.4f} ± {df[col].std():.4f}"})

summary_df = pd.DataFrame(summary_rows)

print("\n" + "="*42)
print("  FRANGI BASELINE — DRIVE TRAINING SET")
print("="*42)
print(summary_df.to_string(index=False))
print("="*42)

# Save CSVs
df.to_csv(os.path.join(output_dir, "metrics_per_image.csv"), index=False)
summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

# Summary table figure
fig, ax = plt.subplots(figsize=(4, 2))
ax.axis('off')
table_data = [["Metric", "Mean ± Std"]] + [
    [row["Metric"], row["Mean ± Std"]] for _, row in summary_df.iterrows()
]
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.savefig(os.path.join(output_dir, "metrics_summary_table.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved all results → {output_dir}")