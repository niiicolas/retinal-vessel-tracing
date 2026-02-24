"""
drive_cnn.py
=========================
Centerline UNet CNN Baseline
Processes all 20 images in the DRIVE training set with 1px, 2px, and 3px metrics.
"""

import os
import glob
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
from skimage.morphology import skeletonize
import pandas as pd
import albumentations as A
import warnings

# Import local modules
from baselines.centerline_unet_baseline import CenterlineUNet, CenterlineLoss, CenterlinePredictor
from data.fundus_preprocessor import FundusPreprocessor
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG
# ==========================================
DRIVE_ROOT   = r"C:\ZHAW\BA\data\DRIVE\training"
SAVE_PATH    = r"C:\ZHAW\BA\weights\centerline_unet.pt"
OUTPUT_DIR   = r"C:\ZHAW\BA\retinal-vessel-tracing\results\cnn_training"

EPOCHS       = 100
BATCH_SIZE   = 2
LR           = 1e-3
VAL_SIZE     = 4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CFG    = dict(in_channels=1, base_ch=16, depth=4)

# ==========================================
# AUGMENTATION PIPELINES
# ==========================================
def get_train_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=584),
        A.PadIfNeeded(min_height=584, min_width=584, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ], additional_targets={'fov': 'mask', 'thick_gt': 'mask'})

def get_val_transforms():
    return A.Compose([
        A.LongestMaxSize(max_size=584),
        A.PadIfNeeded(min_height=584, min_width=584, border_mode=0),
    ], additional_targets={'fov': 'mask', 'thick_gt': 'mask'})


# ==========================================
# MANUAL DATASET CLASS
# ==========================================
class ManualDriveDataset(Dataset):
    def __init__(self, root, indices, augment=False):
        self.img_paths  = sorted(glob.glob(os.path.join(root, "images", "*.tif")))
        self.gt_paths   = sorted(glob.glob(os.path.join(root, "1st_manual", "*.gif")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, "mask", "*.gif")))

        self.img_paths  = [self.img_paths[i]  for i in indices]
        self.gt_paths   = [self.gt_paths[i]   for i in indices]
        self.mask_paths = [self.mask_paths[i] for i in indices]

        self.augment      = augment
        self.transform    = get_train_transforms() if augment else get_val_transforms()
        self.preprocessor = FundusPreprocessor()

        print(f"Precomputing {len(self.img_paths)} images...")
        self.preprocessed = []   
        self.skeletons    = []   
        self.thick_gts    = []   
        self.masks        = []   

        for img_path, gt_path, mask_path in zip(self.img_paths, self.gt_paths, self.mask_paths):
            img_rgb = np.array(Image.open(img_path).convert('RGB'))
            gt      = np.array(Image.open(gt_path).convert('L'))
            mask    = np.array(Image.open(mask_path).convert('L'))

            preprocessed = self.preprocessor.preprocess(img_rgb, external_mask=mask)

            self.preprocessed.append(preprocessed)
            self.skeletons.append(skeletonize(gt > 128).astype(np.float32))
            self.thick_gts.append((gt > 128).astype(np.float32))
            self.masks.append(mask)

        print(f"  Done. {len(self.preprocessed)} images cached in RAM.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        preprocessed = self.preprocessed[idx]
        gt_skel      = self.skeletons[idx].copy()
        thick_gt     = self.thick_gts[idx].copy()
        mask         = self.masks[idx].copy()

        img_uint8 = (preprocessed * 255).clip(0, 255).astype(np.uint8)

        augmented = self.transform(
            image=img_uint8,
            mask=gt_skel,
            fov=mask,
            thick_gt=thick_gt,
        )

        img_aug      = augmented['image'].astype(np.float32) / 255.0
        gt_skel_aug  = augmented['mask']
        mask_aug     = augmented['fov']
        thick_gt_aug = augmented['thick_gt']

        img_t  = torch.from_numpy(img_aug).float().unsqueeze(0)
        gt_t   = torch.from_numpy(gt_skel_aug).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask_aug).float().unsqueeze(0) / 255.0

        return {
            'image':       img_t,
            'centerline':  gt_t,
            'mask':        mask_t,
            'vessel_mask': thick_gt_aug.astype(np.uint8),
            'path':        self.img_paths[idx],
        }


# ==========================================
# TRAINING LOGIC
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        img    = batch['image'].to(device)
        target = batch['centerline'].to(device)
        mask   = batch['mask'].to(device)

        optimizer.zero_grad()
        pred = model(img)
        loss, _ = criterion(pred, target, mask=mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ==========================================
# EVALUATION & VISUALIZATION
# ==========================================
@torch.no_grad()
def evaluate_and_visualize(checkpoint_path):
    print(f"\nFinal Evaluation & Visualization...")
    panels_dir = os.path.join(OUTPUT_DIR, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    val_indices = list(range(20))[-VAL_SIZE:]
    dataset     = ManualDriveDataset(DRIVE_ROOT, val_indices, augment=False)
    loader      = DataLoader(dataset, batch_size=1, num_workers=0)

    warnings.filterwarnings("ignore", category=FutureWarning)
    predictor   = CenterlinePredictor.from_checkpoint(checkpoint_path, device=DEVICE)
    metrics_fn  = CenterlineMetrics(tolerance_levels=[1, 2, 3])
    val_transform = get_val_transforms()

    all_metrics = []

    for batch in loader:
        img_np      = batch['image'][0, 0].numpy()
        mask_np     = (batch['mask'][0, 0].numpy() * 255).astype(np.uint8)
        gt_skel     = (batch['centerline'][0, 0].numpy() * 255).astype(np.uint8)
        vessel_mask = (batch['vessel_mask'][0].numpy() * 255).astype(np.uint8)
        img_path    = batch['path'][0]
        image_id    = os.path.basename(img_path).split('_')[0]

        orig_color   = np.array(Image.open(img_path).convert('RGB'))
        color_padded = val_transform(image=orig_color)['image']

        prob_map, pred_skel = predictor.predict(img_np, fov_mask=mask_np)

        # Compute metrics for all tolerances (1, 2, 3px)
        res = metrics_fn.compute_all_metrics(pred_skel, gt_skel, vessel_mask)
        res['image_id'] = image_id
        all_metrics.append(res)

        gt_skel_vis   = (gt_skel > 0).astype(np.uint8) * 255
        pred_skel_vis = (pred_skel > 0).astype(np.uint8) * 255
        fov_bin       = (mask_np > 0).astype(np.float32)
        prob_map_vis  = prob_map * fov_bin

        fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')
        axes[0].imshow(color_padded); axes[0].set_title("Original Image", fontweight='bold')
        axes[1].imshow(prob_map_vis, cmap='gray'); axes[1].set_title("CNN Probability", fontweight='bold')

        side_by_side = np.concatenate([gt_skel_vis, pred_skel_vis], axis=1)
        axes[2].imshow(side_by_side, cmap='gray')
        axes[2].set_title("Skeletons\n(Left: GT | Right: Pred)", fontweight='bold')

        overlay = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
        overlay[..., 1] = gt_skel_vis  
        overlay[..., 0] = pred_skel_vis
        axes[3].imshow(overlay)
        axes[3].set_title(f"Overlay Analysis\nF1@2px: {res['f1@2px']:.3f} | clDice: {res.get('clDice', 0):.3f}", fontweight='bold', color='darkblue')

        legend_elements = [
            Patch(facecolor='green',  edgecolor='black', label='GT Missing (FN)'),
            Patch(facecolor='red',    edgecolor='black', label='Pred (False Pos)'),
            Patch(facecolor='yellow', edgecolor='black', label='Match (True Pos)'),
        ]
        axes[3].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=12)

        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(panels_dir, f"{image_id}_cnn_panel.png"), bbox_inches='tight', dpi=150)
        plt.close()

    # ── CSV Export & FULL SUMMARY ──────────────────────────────────────────
    df = pd.DataFrame(all_metrics)

    # We now list all pixel levels for the final report
    target_metrics = [
        "clDice",
        "f1@1px", "precision@1px", "recall@1px",
        "f1@2px", "precision@2px", "recall@2px",
        "f1@3px", "precision@3px", "recall@3px"
    ]

    summary_rows = []
    for col in target_metrics:
        if col in df.columns:
            summary_rows.append({
                "Metric": col,
                "Mean ± Std": f"{df[col].mean():.4f} ± {df[col].std():.4f}"
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "metrics_per_image.csv"), index=False)

    print("\n" + "=" * 45)
    print("      CNN RESULTS — DRIVE VALIDATION SET")
    print("=" * 45)
    print(summary_df.to_string(index=False))
    print("=" * 45)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    indices = list(range(20))
    train_indices, val_indices = indices[:-VAL_SIZE], indices[-VAL_SIZE:]

    RUN_TRAINING = True

    if RUN_TRAINING:
        train_ds     = ManualDriveDataset(DRIVE_ROOT, train_indices, augment=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        model     = CenterlineUNet(**MODEL_CFG).to(DEVICE)
        criterion = CenterlineLoss(0.4, 0.6, pos_weight=10.0)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        print(f"Starting Training on {DEVICE} for {EPOCHS} epochs...")
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        loss_history = []
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            scheduler.step()
            loss_history.append(loss)
            print(f"Epoch {epoch:>3}/{EPOCHS} | Loss: {loss:.4f}")

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, EPOCHS + 1), loss_history, label='Training Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curve.png"))
        plt.close()

        torch.save({'model_state': model.state_dict(), 'model_cfg': MODEL_CFG}, SAVE_PATH)
        print(f"Model saved to: {SAVE_PATH}")

    evaluate_and_visualize(SAVE_PATH)