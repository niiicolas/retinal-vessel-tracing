"""
Lightweight U-Net + clDice Loss + Greedy Tracing for Centerline Extraction
A simple baseline for vessel centerline prediction from vesselness maps.

Usage:
    python centerline_baseline.py --mode train --epochs 50
    python centerline_baseline.py --mode inference --input vesselness.npy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
from fundus_preprocessor import FundusPreprocessor


# ============================================================================
# MODEL: Lightweight U-Net
# ============================================================================

class UNet(nn.Module):
    """Lightweight U-Net for centerline probability prediction"""
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self.conv_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self.conv_block(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self.conv_block(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)


# ============================================================================
# LOSS: clDice (Topology-Aware Loss)
# ============================================================================

def soft_erode(img):
    """Soft erosion using min pooling"""
    return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)


def soft_dilate(img):
    """Soft dilation using max pooling"""
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_skeletonize(img, iterations=10):
    """Soft skeletonization via iterative thinning"""
    skeleton = img.clone()
    for _ in range(iterations):
        eroded = soft_erode(skeleton)
        temp = soft_dilate(soft_erode(skeleton))
        skeleton = skeleton - (skeleton - eroded) * temp
        skeleton = torch.clamp(skeleton, 0, 1)
    return skeleton


class clDiceLoss(nn.Module):
    """Centerline Dice loss for topology preservation"""
    
    def __init__(self, alpha=0.5, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
    
    def dice_score(self, pred, target):
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return dice.mean()
    
    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        
        # Standard Dice
        dice = self.dice_score(pred, target)
        
        # Skeleton Dice (topology)
        pred_skel = soft_skeletonize(pred, iterations=10)
        target_skel = soft_skeletonize(target, iterations=10)
        skel_dice = self.dice_score(pred_skel, target_skel)
        
        # Combined loss
        loss = (1 - self.alpha) * (1 - dice) + self.alpha * (1 - skel_dice)
        
        return loss, dice.item(), skel_dice.item()


# ============================================================================
# DATASET
# ============================================================================

class VesselDataset(Dataset):
    """Dataset for vesselness -> centerline pairs"""
    
    def __init__(self, vesselness_files, centerline_files, augment=False):
        self.vesselness_files = vesselness_files
        self.centerline_files = centerline_files
        self.augment = augment
    
    def __len__(self):
        return len(self.vesselness_files)
    
    def __getitem__(self, idx):
        # Load data
        vesselness = np.load(self.vesselness_files[idx]).astype(np.float32)
        centerline = np.load(self.centerline_files[idx]).astype(np.float32)
        
        # Ensure 2D
        if vesselness.ndim == 3:
            vesselness = vesselness[0]
        if centerline.ndim == 3:
            centerline = centerline[0]
        
        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            # Random flip
            if np.random.rand() > 0.5:
                vesselness = np.flip(vesselness, 0).copy()
                centerline = np.flip(centerline, 0).copy()
            if np.random.rand() > 0.5:
                vesselness = np.flip(vesselness, 1).copy()
                centerline = np.flip(centerline, 1).copy()
            # Random rotation
            k = np.random.randint(0, 4)
            vesselness = np.rot90(vesselness, k).copy()
            centerline = np.rot90(centerline, k).copy()
        
        # Add channel dimension and convert to tensor
        vesselness = torch.from_numpy(vesselness[None, ...])
        centerline = torch.from_numpy(centerline[None, ...])
        
        return vesselness, centerline


# ============================================================================
# GREEDY TRACING
# ============================================================================

class GreedyTracer:
    """Greedy centerline tracing on probability maps"""
    
    def __init__(self, min_prob=0.3, max_steps=1000):
        self.min_prob = min_prob
        self.max_steps = max_steps
        self.directions = np.array([
            [-1, 0], [-1, 1], [0, 1], [1, 1],
            [1, 0], [1, -1], [0, -1], [-1, -1]
        ])
    
    def trace_from_seed(self, prob_map, seed, visited):
        """Trace from a single seed point"""
        h, w = prob_map.shape
        trajectory = [seed]
        current = np.array(seed)
        
        for _ in range(self.max_steps):
            y, x = current
            visited[y, x] = True
            
            best_score = -1
            best_next = None
            
            for direction in self.directions:
                next_pos = current + direction
                ny, nx = next_pos
                
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx]:
                    continue
                
                score = prob_map[ny, nx]
                if score > best_score:
                    best_score = score
                    best_next = next_pos
            
            if best_next is None or prob_map[best_next[0], best_next[1]] < self.min_prob:
                break
            
            current = best_next
            trajectory.append(tuple(current))
        
        return trajectory
    
    def detect_seeds(self, prob_map, threshold=0.5):
        """Detect seed points using local maxima"""
        from scipy.ndimage import maximum_filter
        local_max = (prob_map == maximum_filter(prob_map, size=10))
        seeds = np.argwhere(local_max & (prob_map >= threshold))
        seeds = sorted(seeds, key=lambda s: prob_map[s[0], s[1]], reverse=True)
        return [(int(y), int(x)) for y, x in seeds]
    
    def trace(self, prob_map):
        """Trace complete centerline network"""
        h, w = prob_map.shape
        skeleton = np.zeros((h, w), dtype=np.float32)
        visited = np.zeros((h, w), dtype=bool)
        
        seeds = self.detect_seeds(prob_map)
        
        for seed in seeds:
            if visited[seed[0], seed[1]]:
                continue
            trajectory = self.trace_from_seed(prob_map, seed, visited)
            for y, x in trajectory:
                skeleton[y, x] = 1.0
        
        return skeleton


# ============================================================================
# TRAINING
# ============================================================================

def train(model, train_loader, val_loader, epochs=50, device='cuda'):
    """Train the U-Net model"""
    
    criterion = clDiceLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_cldice': []}
    
    Path('checkpoints').mkdir(exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for vesselness, centerline in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            vesselness = vesselness.to(device)
            centerline = centerline.to(device)
            
            optimizer.zero_grad()
            pred = model(vesselness)
            loss, _, _ = criterion(pred, centerline)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_cldice = 0
        
        with torch.no_grad():
            for vesselness, centerline in val_loader:
                vesselness = vesselness.to(device)
                centerline = centerline.to(device)
                
                pred = model(vesselness)
                loss, dice, cldice = criterion(pred, centerline)
                
                val_loss += loss.item()
                val_dice += dice
                val_cldice += cldice
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_cldice /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_cldice'].append(val_cldice)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
              f'Val Loss={val_loss:.4f}, Dice={val_dice:.3f}, clDice={val_cldice:.3f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pth')
            print(f'  → Saved best model (val_loss: {val_loss:.4f})')
    
    plot_history(history)
    return history


def plot_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['val_dice'], label='Dice Score')
    ax2.plot(epochs, history['val_cldice'], label='clDice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('checkpoints/training_history.png', dpi=150)
    plt.show()


# ============================================================================
# INFERENCE
# ============================================================================

@torch.no_grad()
def inference(model, vesselness, device='cuda'):
    """Run inference on vesselness map"""
    model.eval()
    
    if isinstance(vesselness, np.ndarray):
        vesselness = torch.from_numpy(vesselness).float()
    
    if vesselness.ndim == 2:
        vesselness = vesselness[None, None, ...]
    elif vesselness.ndim == 3:
        vesselness = vesselness[None, ...]
    
    vesselness = vesselness.to(device)
    
    logits = model(vesselness)
    prob = torch.sigmoid(logits)
    prob = prob.cpu().numpy()[0, 0]
    
    return prob


def visualize_results(vesselness, prob_map, skeleton):
    """Visualize extraction results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(vesselness, cmap='gray')
    axes[0, 0].set_title('Input: Vesselness')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(prob_map, cmap='hot')
    axes[0, 1].set_title('U-Net: Probability')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(skeleton, cmap='gray')
    axes[1, 0].set_title('Greedy: Centerline')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(vesselness, cmap='gray', alpha=0.6)
    axes[1, 1].imshow(skeleton, cmap='Reds', alpha=0.6)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png', dpi=150)
    plt.show()


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def create_synthetic_data(num_samples=100, size=256, save_dir='data'):
    """Create synthetic vessel dataset"""
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    
    save_dir = Path(save_dir)
    vessel_dir = save_dir / 'vesselness'
    center_dir = save_dir / 'centerline'
    vessel_dir.mkdir(parents=True, exist_ok=True)
    center_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Creating {num_samples} synthetic samples...')
    
    for i in tqdm(range(num_samples)):
        centerline = np.zeros((size, size), dtype=np.float32)
        
        num_vessels = np.random.randint(2, 4)
        for _ in range(num_vessels):
            t = np.linspace(0, 1, 150)
            x = (np.random.randint(20, size-20) + t * 120 * np.cos(np.random.rand() * 2 * np.pi) +
                 30 * np.sin(np.random.rand() * 3 * t * 2 * np.pi)).astype(int)
            y = (np.random.randint(20, size-20) + t * 120 * np.sin(np.random.rand() * 2 * np.pi) +
                 30 * np.cos(np.random.rand() * 3 * t * 2 * np.pi)).astype(int)
            
            for j in range(len(x)):
                if 0 <= x[j] < size and 0 <= y[j] < size:
                    centerline[y[j], x[j]] = 1.0
        
        dist = distance_transform_edt(1 - centerline)
        vesselness = np.exp(-dist ** 2 / 20.0)
        vesselness = gaussian_filter(vesselness, sigma=1.0)
        vesselness += np.random.randn(size, size) * 0.05
        vesselness = np.clip(vesselness, 0, 1).astype(np.float32)
        
        np.save(vessel_dir / f'sample_{i:04d}.npy', vesselness)
        np.save(center_dir / f'sample_{i:04d}.npy', centerline)
    
    print(f'Saved to {save_dir}')
    return vessel_dir, center_dir


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Centerline Extraction Baseline')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--input', type=str, help='Input RGB fundus image file for inference')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if args.mode == 'train':
        data_dir = Path(args.data_dir)
        if not (data_dir / 'vesselness').exists():
            print('Creating synthetic dataset...')
            create_synthetic_data(num_samples=100, save_dir=args.data_dir)
        
        vessel_files = sorted((data_dir / 'vesselness').glob('*.npy'))
        center_files = sorted((data_dir / 'centerline').glob('*.npy'))
        
        split = int(0.8 * len(vessel_files))
        train_dataset = VesselDataset(vessel_files[:split], center_files[:split], augment=True)
        val_dataset = VesselDataset(vessel_files[split:], center_files[split:], augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')
        
        model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
        print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
        
        train(model, train_loader, val_loader, epochs=args.epochs, device=device)
        
    elif args.mode == 'inference':
        # Load model
        model = UNet(in_channels=1, out_channels=1, base_channels=32).to(device)
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {args.model}')
        
        # Load + preprocess input
        preprocessor = FundusPreprocessor()

        if args.input:
            image = np.load(args.input)  # expects RGB fundus image
            _, _, clahe, mask = preprocessor.preprocess(image, return_intermediate=True)
            vesselness = clahe.astype(np.float32) / 255.0
            vesselness *= (mask > 0)
        else:
            # Demo data (synthetic, no preprocessing needed)
            print('No input provided, creating demo data...')
            from scipy.ndimage import gaussian_filter, distance_transform_edt
            size = 256
            centerline = np.zeros((size, size))
            t = np.linspace(0, 4 * np.pi, 200)
            x = np.linspace(20, size - 20, 200).astype(int)
            y = (size // 2 + 40 * np.sin(t)).astype(int)
            for i in range(len(x)):
                centerline[y[i], x[i]] = 1.0
            dist = distance_transform_edt(1 - centerline)
            vesselness = np.exp(-dist ** 2 / 20.0)
            vesselness = gaussian_filter(vesselness, sigma=1.0)
            vesselness = np.clip(vesselness + np.random.randn(size, size) * 0.05, 0, 1)
        
        # Predict probability
        print('Predicting centerline probability...')
        prob_map = inference(model, vesselness, device=device)
        
        # Trace centerlines
        print('Tracing centerlines...')
        tracer = GreedyTracer(min_prob=0.3)
        skeleton = tracer.trace(prob_map)
        
        # Visualize
        visualize_results(vesselness, prob_map, skeleton)
        
        print(f'Results saved to results_visualization.png')
        print(f'Centerline pixels: {skeleton.sum():.0f}')


if __name__ == '__main__':
    main()