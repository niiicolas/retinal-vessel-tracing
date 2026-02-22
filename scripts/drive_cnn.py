import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import skeletonize
import pandas as pd
import albumentations as A

# Import local modules
from baselines.centerline_unet_baseline import CenterlineUNet, CenterlineLoss, CenterlinePredictor
from evaluation.metrics import CenterlineMetrics

# ==========================================
# CONFIG
# ==========================================
DRIVE_ROOT  = r"C:\ZHAW\BA\data\DRIVE\training"
SAVE_PATH   = r"C:\ZHAW\BA\weights\centerline_unet.pt"
OUTPUT_DIR  = r"C:\ZHAW\BA\retinal-vessel-tracing\results\cnn_training"

EPOCHS      = 100   
BATCH_SIZE  = 2
LR          = 1e-3
VAL_SIZE    = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CFG   = dict(in_channels=1, base_ch=16, depth=4)

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
        A.CLAHE(clip_limit=2.0, p=0.2),
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
        self.img_paths = sorted(glob.glob(os.path.join(root, "images", "*.tif")))
        self.gt_paths  = sorted(glob.glob(os.path.join(root, "1st_manual", "*.gif")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, "mask", "*.gif")))
        
        self.img_paths = [self.img_paths[i] for i in indices]
        self.gt_paths  = [self.gt_paths[i] for i in indices]
        self.mask_paths = [self.mask_paths[i] for i in indices]
        
        self.augment = augment
        self.transform = get_train_transforms() if augment else get_val_transforms()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert('L'))
        gt  = np.array(Image.open(self.gt_paths[idx]).convert('L'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('L'))
        
        gt_skel = skeletonize(gt > 128).astype(np.float32)
        thick_gt = (gt > 128).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=img, mask=gt_skel, fov=mask, thick_gt=thick_gt)
            img = augmented['image']
            gt_skel = augmented['mask']
            mask = augmented['fov']
            thick_gt = augmented['thick_gt']

        img_t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        gt_t  = torch.from_numpy(gt_skel).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        return {
            'image': img_t, 
            'centerline': gt_t, 
            'mask': mask_t, 
            'vessel_mask': thick_gt.astype(np.uint8),
            'path': self.img_paths[idx]
        }

# ==========================================
# TRAINING LOGIC
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
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
def evaluate_and_visualize(checkpoint_path):
    print(f"\nFinal Evaluation & Visualization...")
    panels_dir = os.path.join(OUTPUT_DIR, "panels")
    os.makedirs(panels_dir, exist_ok=True)
    
    val_indices = list(range(20))[-VAL_SIZE:] 
    dataset = ManualDriveDataset(DRIVE_ROOT, val_indices, augment=False)
    loader = DataLoader(dataset, batch_size=1)
    
    # FIX: weights_only=True removes the FutureWarning
    predictor = CenterlinePredictor.from_checkpoint(checkpoint_path, device=DEVICE)
    metrics_fn = CenterlineMetrics(tolerance_levels=[1, 2, 3])
    
    all_metrics = []
    for i, batch in enumerate(loader):
        img_np, mask_np = batch['image'][0,0].numpy(), (batch['mask'][0,0].numpy() * 255).astype(np.uint8)
        gt_skel, vessel_mask = (batch['centerline'][0,0].numpy() * 255).astype(np.uint8), (batch['vessel_mask'][0].numpy() * 255).astype(np.uint8)
        image_id = os.path.basename(batch['path'][0]).split('_')[0]

        prob_map, pred_skel = predictor.predict(img_np, fov_mask=mask_np)
        res = metrics_fn.compute_all_metrics(pred_skel, gt_skel, vessel_mask)
        res['image_id'] = image_id
        all_metrics.append(res)
        
        fig, axes = plt.subplots(1, 4, figsize=(24, 7))
        axes[0].imshow(img_np, cmap='gray'); axes[0].set_title("Input")
        axes[1].imshow(prob_map, cmap='hot'); axes[1].set_title("CNN Probability")
        overlay = np.zeros((img_np.shape[0], img_np.shape[1], 3), dtype=np.uint8)
        overlay[..., 1], overlay[..., 0] = gt_skel, (pred_skel * 255).astype(np.uint8) 
        axes[2].imshow(overlay); axes[2].set_title("Overlay (Red:Pred, Green:GT)")
        axes[3].set_title(f"F1@2px: {res['f1@2px']:.3f}\nclDice: {res.get('clDice',0):.3f}"); axes[3].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(panels_dir, f"{image_id}_cnn_panel.png")); plt.close()

    df = pd.DataFrame(all_metrics)
    print("\n" + "="*45 + "\n      CNN RESULTS — DRIVE VALIDATION SET\n" + "="*45)
    # Added precision and recall to the summary table
    cols_to_print = ["f1@2px", "precision@2px", "recall@2px", "clDice"]
    for col in cols_to_print:
        if col in df.columns: 
            print(f"{col:<15}: {df[col].mean():.4f} ± {df[col].std():.4f}")
    print("="*45)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    indices = list(range(20))
    train_indices, val_indices = indices[:-VAL_SIZE], indices[-VAL_SIZE:]
    
    # Check if we should skip training (if weights already exist and we just want metrics)
    RUN_TRAINING = True 
    
    if RUN_TRAINING:
        train_ds = ManualDriveDataset(DRIVE_ROOT, train_indices, augment=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model = CenterlineUNet(**MODEL_CFG).to(DEVICE)
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
        
        plt.figure(figsize=(10, 5)); plt.plot(range(1, EPOCHS + 1), loss_history, label='Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, "training_loss_curve.png")); plt.close()
        
        torch.save({'model_state': model.state_dict(), 'model_cfg': MODEL_CFG}, SAVE_PATH)

    evaluate_and_visualize(SAVE_PATH)