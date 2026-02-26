"""
drive_cnn.py
=========================
Centerline UNet CNN Baseline
Standardized: 16 Train / 4 Test split from the training directory.
This script follows the exact visual and metric reporting standards of the baselines.
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
VAL_SIZE     = 4  # Specifically held out for the Final Test Set
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

        self.transform    = get_train_transforms() if augment else get_val_transforms()
        self.preprocessor = FundusPreprocessor()

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

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        preprocessed = self.preprocessed[idx]
        gt_skel      = self.skeletons[idx].copy()
        thick_gt     = self.thick_gts[idx].copy()
        mask         = self.masks[idx].copy()

        img_uint8 = (preprocessed * 255).clip(0, 255).astype(np.uint8)
        augmented = self.transform(image=img_uint8, mask=gt_skel, fov=mask, thick_gt=thick_gt)

        img_aug = augmented['image'].astype(np.float32) / 255.0
        img_t  = torch.from_numpy(img_aug).float().unsqueeze(0)
        gt_t   = torch.from_numpy(augmented['mask']).float().unsqueeze(0)
        mask_t = torch.from_numpy(augmented['fov']).float().unsqueeze(0) / 255.0

        return {
            'image':       img_t,
            'centerline':  gt_t,
            'mask':        mask_t,
            'vessel_mask': augmented['thick_gt'].astype(np.uint8),
            'path':        self.img_paths[idx],
        }

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training Epoch", leave=False):
        img, target, mask = batch['image'].to(device), batch['centerline'].to(device), batch['mask'].to(device)
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
def evaluate_and_visualize(checkpoint_path, test_indices):
    print(f"\nEvaluating on Test Set")
    panels_dir = os.path.join(OUTPUT_DIR, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    dataset      = ManualDriveDataset(DRIVE_ROOT, test_indices, augment=False)
    loader       = DataLoader(dataset, batch_size=1, num_workers=0)

    warnings.filterwarnings("ignore", category=FutureWarning)
    predictor   = CenterlinePredictor.from_checkpoint(checkpoint_path, device=DEVICE)
    metrics_fn  = CenterlineMetrics(tolerance_levels=[1, 2, 3])
    val_transform = get_val_transforms()

    all_metrics, mosaic_data = [], []

    for batch in tqdm(loader, desc="Evaluating CNN"):
        img_np      = batch['image'][0, 0].numpy()
        mask_np     = (batch['mask'][0, 0].numpy() * 255).astype(np.uint8)
        gt_skel     = (batch['centerline'][0, 0].numpy() * 255).astype(np.uint8)
        vessel_mask = (batch['vessel_mask'][0].numpy() * 255).astype(np.uint8)
        img_path    = batch['path'][0]
        image_id    = os.path.basename(img_path).split('_')[0]

        prob_map, pred_skel = predictor.predict(img_np, fov_mask=mask_np)

        # Metrics
        res = metrics_fn.compute_all_metrics(pred_skel, gt_skel, vessel_mask)
        res['image_id'] = image_id
        all_metrics.append(res)

        # Viz Data
        gt_vis, pred_vis = (gt_skel > 0).astype(np.uint8) * 255, (pred_skel > 0).astype(np.uint8) * 255
        mosaic_data.append({"image_id": image_id, "gt_skeleton": gt_vis, "pred_skeleton": pred_vis, "metrics": res})

        # 4-Panel Viz
        fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')
        axes[0].imshow(val_transform(image=np.array(Image.open(img_path).convert('RGB')))['image'])
        axes[0].set_title(f"Original Image (ID: {image_id})", fontweight='bold')
        axes[1].imshow(prob_map * (mask_np > 0), cmap='gray'); axes[1].set_title("CNN Probability", fontweight='bold')
        side_by_side = np.concatenate([gt_vis, pred_vis], axis=1)
        axes[2].imshow(side_by_side, cmap='gray'); axes[2].set_title("Skeletons (GT | Pred)", fontweight='bold')
        overlay = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
        overlay[..., 1], overlay[..., 0] = gt_vis, pred_vis
        axes[3].imshow(overlay)
        axes[3].set_title(f"Overlay Analysis\nF1@2px: {res['f1@2px']:.3f} | clDice: {res.get('clDice', 0):.3f}", fontweight='bold', color='darkblue')

        legend_elements = [Patch(facecolor='green', label='GT'), Patch(facecolor='red', label='Pred'), Patch(facecolor='yellow', label='Match')]
        axes[3].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

        for ax in axes: ax.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(panels_dir, f"{image_id}_cnn_panel.png"), bbox_inches='tight', dpi=150); plt.close()

    # Mosaic Overview
    if mosaic_data:
        n_cols = 4
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols*6, 5))
        for i, data in enumerate(mosaic_data):
            h, w = data['gt_skeleton'].shape
            over = np.zeros((h, w, 3), dtype=np.uint8)
            over[..., 1], over[..., 0] = data['gt_skeleton'], data['pred_skeleton']
            axes[i].imshow(over); axes[i].set_title(f"[{data['image_id']}] F1@2px: {data['metrics']['f1@2px']:.3f}", fontweight='bold'); axes[i].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "mosaic_overview.png"), dpi=200); plt.close()

    # Reporting
    df = pd.DataFrame(all_metrics)
    summary_rows = [{"Metric": c, "Mean ± Std": f"{df[c].mean():.4f} ± {df[c].std():.4f}"} 
                    for c in ["clDice", "betti_0_error", "hd95", "f1@1px", "f1@2px", "f1@3px"] if c in df.columns]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "metrics_per_image.csv"), index=False)
    print("\n" + "="*45 + "\n   CNN RESULTS — HELD-OUT TEST SET (N=4)\n" + "="*45 + f"\n{summary_df.to_string(index=False)}\n" + "="*45)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    idx = list(range(20))
    train_idx, test_idx = idx[:16], idx[16:]

    RUN_TRAINING = False

    if RUN_TRAINING:
        train_loader = DataLoader(ManualDriveDataset(DRIVE_ROOT, train_idx, augment=True), batch_size=BATCH_SIZE, shuffle=True)
        model = CenterlineUNet(**MODEL_CFG).to(DEVICE)
        criterion = CenterlineLoss(0.4, 0.6, pos_weight=10.0)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            scheduler.step()
            if epoch % 10 == 0: print(f"Epoch {epoch:>3}/{EPOCHS} | Loss: {loss:.4f}")
        torch.save({'model_state': model.state_dict(), 'model_cfg': MODEL_CFG}, SAVE_PATH)

    evaluate_and_visualize(SAVE_PATH, test_idx)