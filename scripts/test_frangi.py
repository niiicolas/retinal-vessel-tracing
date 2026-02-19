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
output_dir = r"C:\ZHAW\BA\skeleton_tracing_RBV\results"
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

# --- Image 1: Full Comparison Panel ---
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# Original
axes[0].imshow(image)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis('off')

# Vesselness
axes[1].imshow(vesselness, cmap="gray")
axes[1].set_title("Frangi Vesselness", fontsize=14)
axes[1].axis('off')

# Skeletons Side-by-Side
combined_skel = np.hstack((gt_skeleton, pred_skeleton))
axes[2].imshow(combined_skel, cmap="gray")
axes[2].set_title("GT Skeleton (Left) vs. Pred (Right)", fontsize=14)
axes[2].axis('off')

# Overlay
h, w = pred_skeleton.shape
overlay = np.zeros((h, w, 3), dtype=np.uint8)
overlay[:, :, 1] = (gt_skeleton > 0) * 255  # Green
overlay[:, :, 0] = (pred_skeleton > 0) * 255  # Red

axes[3].imshow(overlay)
axes[3].set_title(f"Overlay (F1: {metrics['f1']:.2f})\nGreen=GT, Red=Pred, Yellow=Match", fontsize=14)
axes[3].axis('off')

# FIX: Use tight_layout AND bbox_inches='tight'
plt.tight_layout()
save_path_panel = os.path.join(output_dir, "21_training_comparison.png")

# bbox_inches='tight' ensures titles are INCLUDED in the saved image
plt.savefig(save_path_panel, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved comparison to: {save_path_panel}")
plt.close() 

# --- Image 2: Just the Overlay (High Res) ---
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.axis('off')
plt.title(f"Skeleton Overlay (F1: {metrics['f1']:.3f})", fontsize=16)

save_path_overlay = os.path.join(output_dir, "21_training_overlay_only.png")
plt.savefig(save_path_overlay, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved overlay to:    {save_path_overlay}")
plt.close()