import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image 
from skimage.morphology import skeletonize
from baselines.frangi_baseline import FrangiBaseline

# ==========================================
# 1. SETUP & LOADING
# ==========================================
output_dir = r"C:\ZHAW\BA\retinal-vessel-tracing\results"
os.makedirs(output_dir, exist_ok=True)

image_path = r"C:\ZHAW\BA\data\DRIVE\training\images\21_training.tif"
manual_path = r"C:\ZHAW\BA\data\DRIVE\training\1st_manual\21_manual1.gif"

print(f"Loading image: {image_path}...")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image not found! Check path.")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Loading manual mask: {manual_path}...")
manual_pil = Image.open(manual_path).convert('L')
gt_mask = np.array(manual_pil) 
gt_binary = gt_mask > 128

# ==========================================
# 2. RUN MODEL
# ==========================================
print("Running Frangi Baseline...")
model = FrangiBaseline()
pred_skeleton, vesselness = model.extract_centerline(image, return_vesselness=True)

# ==========================================
# 3. CALCULATE METRICS
# ==========================================
gt_skeleton = skeletonize(gt_binary)
metrics = model.evaluate(image, gt_skeleton=gt_skeleton, gt_vessel_mask=gt_binary)

print("\n" + "="*30)
print(f" RESULTS (Saved to {output_dir})")
print("="*30)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics['f1']:.4f}")
print(f"clDice:    {metrics.get('clDice', 0.0):.4f}")
print("="*30)

# ==========================================
# 4. SAVE IMAGES
# ==========================================

# Extract metrics for easier access
f1_val = metrics.get('f1', 0.0)
cldice_val = metrics.get('clDice', 0.0)
prec_val = metrics.get('precision', 0.0)
rec_val = metrics.get('recall', 0.0)

# --- Image 1: Full Comparison Panel ---
fig, axes = plt.subplots(1, 4, figsize=(24, 7)) # Increased height slightly for better spacing

# 1. Original
axes[0].imshow(image)
axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
axes[0].axis('off')

# 2. Vesselness (Typo fixed: Vesselness)
axes[1].imshow(vesselness, cmap="gray")
axes[1].set_title("Frangi Vesselness", fontsize=14, fontweight='bold')
axes[1].axis('off')

# 3. Skeletons Side-by-Side
combined_skel = np.hstack((gt_skeleton, pred_skeleton))
axes[2].imshow(combined_skel, cmap="gray")
axes[2].set_title("Skeletons\n(Left: GT | Right: Pred)", fontsize=14, fontweight='bold')
axes[2].axis('off')

# 4. Overlay with dual metrics
# Creating the Yellow match: 
# Since Red + Green = Yellow in RGB, the current logic already handles this.
h, w = pred_skeleton.shape
overlay = np.zeros((h, w, 3), dtype=np.uint8)
overlay[:, :, 1] = (gt_skeleton > 0) * 255   # Green (Ground Truth)
overlay[:, :, 0] = (pred_skeleton > 0) * 255   # Red (Prediction)
# Where both are 255, the pixel becomes [255, 255, 0] (Yellow)

axes[3].imshow(overlay)
axes[3].set_title(f"Overlay Analysis\nF1: {f1_val:.3f} | clDice: {cldice_val:.3f}", 
                  fontsize=14, fontweight='bold', color='darkblue')
axes[3].axis('off')

# Adding a small legend manually to the plot for the BA
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='green', lw=4, label='GT (Missing)'),
                   Line2D([0], [0], color='red', lw=4, label='Pred (False Pos)'),
                   Line2D([0], [0], color='yellow', lw=4, label='Match (True Pos)')]
axes[3].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), 
               ncol=3, fontsize=10, frameon=False)

plt.tight_layout()
save_path_panel = os.path.join(output_dir, "21_training_comparison_final.png")
plt.savefig(save_path_panel, dpi=300, bbox_inches='tight')
print(f"Saved professional comparison to: {save_path_panel}")
plt.close()

# --- Image 2: Metric Summary Table (Optional but very "BA-conform") ---
# This creates a clean table image you can use in your results chapter.
fig_tab, ax_tab = plt.subplots(figsize=(4, 2))
ax_tab.axis('off')
table_data = [
    ["Metric", "Value"],
    ["Precision", f"{prec_val:.4f}"],
    ["Recall", f"{rec_val:.4f}"],
    ["F1 Score", f"{f1_val:.4f}"],
    ["clDice", f"{cldice_val:.4f}"]
]
table = ax_tab.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

save_path_table = os.path.join(output_dir, "21_training_metrics_table.png")
plt.savefig(save_path_table, dpi=300, bbox_inches='tight')
plt.close()