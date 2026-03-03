"""
test_cldice_skeletonization.py
================================
Quick visual check of the skeletons computed *inside* clDice
(morphological erosion/dilation proxy) vs the skimage skeletonization
used everywhere else.

Run on a single DRIVE image. No changes to existing pipeline needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize

# ── local imports ──────────────────────────────────────────────────────────
from baselines.frangi_baseline import FrangiBaseline
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG — change image_id to any DRIVE id
# ==========================================
IMAGE_ID   = "40"
IMAGE_PATH = rf"C:\ZHAW\BA\data\DRIVE\training\images\{IMAGE_ID}_training.tif"
GT_PATH    = rf"C:\ZHAW\BA\data\DRIVE\training\1st_manual\{IMAGE_ID}_manual1.gif"
MASK_PATH  = rf"C:\ZHAW\BA\data\DRIVE\training\mask\{IMAGE_ID}_training_mask.gif"
OUTPUT     = rf"C:\ZHAW\BA\retinal-vessel-tracing\results\cldice_skel_check_{IMAGE_ID}.png"

# ==========================================
# LOAD
# ==========================================
import cv2
img_bgr  = cv2.imread(IMAGE_PATH)
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gt       = np.array(Image.open(GT_PATH).convert('L'))
mask     = np.array(Image.open(MASK_PATH).convert('L'))

gt_bin        = (gt > 128) & (mask > 128)
gt_vessel     = gt_bin.astype(np.uint8)

# ==========================================
# GET PREDICTED MASK via Frangi
# ==========================================
model = FrangiBaseline()
_, _, pred_binary = model.extract_centerline(img_rgb, return_vesselness=False, external_fov_mask=mask)
pred_vessel = (pred_binary > 0).astype(np.uint8)

# ==========================================
# SKIMAGE SKELETON (used everywhere else)
# ==========================================
skel_gt_skimage   = skeletonize(gt_bin)
skel_pred_skimage = skeletonize(pred_vessel > 0)

# ==========================================
# CLDICE SKELETON (morphological proxy)
# Replicate exactly what cl_dice() does internally
# ==========================================
from skimage.morphology import skeletonize as _sk

# cl_dice in metrics.py calls skeletonize on the masks too —
# but the *soft* clDice loss uses the erosion/dilation proxy.
# For the hard metric in CenterlineMetrics.cl_dice() the same
# skimage skeletonize is used, so what we really want to check
# is the SOFT skeleton from the loss function.

import torch
import torch.nn.functional as F

def soft_erode(img):
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)

def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skeleton(img, num_iter=10):
    skel = F.relu(img - soft_open(img))
    for _ in range(num_iter):
        img  = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel  = skel + F.relu(delta - skel * delta)
    return skel

def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)

skel_gt_soft   = soft_skeleton(to_tensor(gt_vessel.astype(np.float32)))[0, 0].numpy()
skel_pred_soft = soft_skeleton(to_tensor(pred_vessel.astype(np.float32)))[0, 0].numpy()

# ==========================================
# PLOT
# ==========================================
fig, axes = plt.subplots(2, 4, figsize=(28, 14), facecolor='white')
fig.suptitle(f"clDice Skeletonization Check — Image {IMAGE_ID}", fontsize=16, fontweight='bold')

# Row 0: GT side
axes[0, 0].imshow(gt_vessel, cmap='gray');        axes[0, 0].set_title("GT Vessel Mask")
axes[0, 1].imshow(skel_gt_skimage, cmap='gray');  axes[0, 1].set_title("GT — skimage skeleton\n(used in F1 / hard clDice)")
axes[0, 2].imshow(skel_gt_soft, cmap='hot');      axes[0, 2].set_title("GT — soft skeleton\n(used in clDice loss)")

# Difference: skimage vs soft
diff_gt = np.abs(skel_gt_skimage.astype(np.float32) - (skel_gt_soft > 0.1).astype(np.float32))
axes[0, 3].imshow(diff_gt, cmap='bwr', vmin=-1, vmax=1)
axes[0, 3].set_title("GT diff (skimage vs soft)\nred=only skimage, blue=only soft")

# Row 1: Pred side
axes[1, 0].imshow(pred_vessel, cmap='gray');        axes[1, 0].set_title("Pred Vessel Mask")
axes[1, 1].imshow(skel_pred_skimage, cmap='gray');  axes[1, 1].set_title("Pred — skimage skeleton")
axes[1, 2].imshow(skel_pred_soft, cmap='hot');      axes[1, 2].set_title("Pred — soft skeleton")

diff_pred = np.abs(skel_pred_skimage.astype(np.float32) - (skel_pred_soft > 0.1).astype(np.float32))
axes[1, 3].imshow(diff_pred, cmap='bwr', vmin=-1, vmax=1)
axes[1, 3].set_title("Pred diff (skimage vs soft)")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved → {OUTPUT}")

# ==========================================
# QUICK STATS
# ==========================================
print("\n--- Skeleton pixel counts ---")
print(f"GT   skimage : {skel_gt_skimage.sum():>8,} px")
print(f"GT   soft    : {(skel_gt_soft > 0.1).sum():>8,} px")
print(f"Pred skimage : {skel_pred_skimage.sum():>8,} px")
print(f"Pred soft    : {(skel_pred_soft > 0.1).sum():>8,} px")
