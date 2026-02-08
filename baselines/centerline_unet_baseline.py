# baselines/centerline_unet_baseline.py
"""
Centerline UNet for direct vessel skeleton prediction.
Optimized for retinal vessel centerline extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
from skimage.morphology import skeletonize, remove_small_objects, binary_dilation, disk
from scipy import ndimage

# -------------------------
# Network Components
# -------------------------

class DoubleConv(nn.Module):
    """Two consecutive convolution blocks with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """Attention gate for focusing on relevant features"""
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_channels, inter_channels, 1)
        self.W_skip = nn.Conv2d(skip_channels, inter_channels, 1)
        self.psi = nn.Conv2d(inter_channels, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        
        # Ensure spatial dimensions match
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.sigmoid(self.psi(self.relu(g + s)))
        return skip * psi


# -------------------------
# Centerline UNet
# -------------------------

class CenterlineUNet(nn.Module):
    """
    UNet optimized for centerline/skeleton prediction.
    Features:
    - Attention gates to focus on thin vessel structures
    - Deep supervision for better gradient flow
    - Auxiliary vessel mask prediction for multi-task learning
    """
    
    def __init__(
        self, 
        in_channels=1, 
        features=(64, 128, 256, 512),
        dropout=0.1,
        attention=True,
        deep_supervision=True
    ):
        super().__init__()
        
        self.attention = attention
        self.deep_supervision = deep_supervision
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if attention else None
        self.pool = nn.MaxPool2d(2)

        # Down path (encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout=dropout))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        # Up path (decoder)
        for idx, feature in enumerate(reversed(features)):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            if attention:
                self.attention_gates.append(
                    AttentionGate(feature, feature, feature // 2)
                )
            self.ups.append(DoubleConv(feature * 2, feature, dropout=dropout))

        # Main output: centerline prediction
        self.centerline_head = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] // 2, 1, 1)
        )
        
        # Auxiliary output: vessel mask prediction (helps with training)
        self.vessel_head = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] // 2, 1, 1)
        )
        
        # Deep supervision outputs (if enabled)
        if deep_supervision:
            self.deep_outputs = nn.ModuleList([
                nn.Conv2d(feat, 1, 1) for feat in reversed(features[1:])
            ])

    def forward(self, x):
        skip_connections = []
        input_size = x.shape[2:]

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        deep_outputs = []
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip = skip_connections[idx // 2]

            # Apply attention gate if enabled
            if self.attention:
                skip = self.attention_gates[idx // 2](x, skip)

            # Ensure shapes match
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            # Concatenate and convolve
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))
            
            # Deep supervision outputs
            if self.deep_supervision and idx // 2 < len(self.deep_outputs):
                deep_out = self.deep_outputs[idx // 2](x)
                deep_out = F.interpolate(deep_out, size=input_size, mode='bilinear', align_corners=True)
                deep_outputs.append(torch.sigmoid(deep_out))

        # Final predictions
        centerline = torch.sigmoid(self.centerline_head(x))
        vessel_mask = torch.sigmoid(self.vessel_head(x))
        
        if self.training and self.deep_supervision:
            return {
                'centerline': centerline,
                'vessel_mask': vessel_mask,
                'deep_outputs': deep_outputs
            }
        else:
            return {
                'centerline': centerline,
                'vessel_mask': vessel_mask
            }


# -------------------------
# Preprocessing
# -------------------------

class CenterlinePreprocessor:
    """Preprocessing specifically for centerline extraction"""
    
    def __init__(self, use_clahe=True, normalize=True):
        self.use_clahe = use_clahe
        self.normalize = normalize
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess retinal image for centerline extraction.
        
        Args:
            image: RGB or grayscale retinal image
            
        Returns:
            Preprocessed image (H, W)
        """
        # Extract green channel (best contrast for vessels)
        if image.ndim == 3:
            image = image[:, :, 1]
        
        image = image.astype(np.float32)
        
        # CLAHE for better vessel visibility
        if self.use_clahe:
            from skimage import exposure
            image = exposure.equalize_adapthist(image, clip_limit=0.03)
        
        # Normalize
        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image


# -------------------------
# Postprocessing
# -------------------------

class CenterlinePostprocessor:
    """Postprocessing for predicted centerlines"""
    
    def __init__(
        self,
        centerline_threshold=0.5,
        vessel_threshold=0.5,
        min_size=20,
        use_vessel_mask=True,
        skeleton_iterations=1
    ):
        self.centerline_threshold = centerline_threshold
        self.vessel_threshold = vessel_threshold
        self.min_size = min_size
        self.use_vessel_mask = use_vessel_mask
        self.skeleton_iterations = skeleton_iterations
    
    def __call__(
        self, 
        centerline_prob: np.ndarray,
        vessel_prob: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess centerline predictions.
        
        Args:
            centerline_prob: Predicted centerline probability map
            vessel_prob: Predicted vessel probability map (optional)
            roi_mask: Region of interest mask (optional)
            
        Returns:
            (skeleton, vessel_mask): Binary skeleton and vessel mask
        """
        # Apply ROI mask
        if roi_mask is not None:
            centerline_prob = centerline_prob * roi_mask
            if vessel_prob is not None:
                vessel_prob = vessel_prob * roi_mask
        
        # Threshold centerline
        centerline_bin = centerline_prob > self.centerline_threshold
        
        # Use vessel mask to constrain centerline if available
        if self.use_vessel_mask and vessel_prob is not None:
            vessel_bin = vessel_prob > self.vessel_threshold
            vessel_bin = remove_small_objects(vessel_bin, self.min_size)
            vessel_bin = ndimage.binary_fill_holes(vessel_bin)
            
            # Centerline should be inside vessels
            centerline_bin = centerline_bin & vessel_bin
        else:
            vessel_bin = None
        
        # Clean up centerline
        centerline_bin = remove_small_objects(centerline_bin, self.min_size // 4)
        
        # Optional: refine skeleton
        if self.skeleton_iterations > 0:
            # Dilate slightly then re-skeletonize for smoother results
            selem = disk(1)
            centerline_dilated = binary_dilation(centerline_bin, selem)
            centerline_bin = skeletonize(centerline_dilated)
        
        return centerline_bin.astype(np.float32), vessel_bin


# -------------------------
# Inference Wrapper
# -------------------------

class CenterlineUNetPredictor:
    """
    Complete pipeline for centerline prediction.
    """
    
    def __init__(
        self,
        model: CenterlineUNet,
        device: str = "cuda",
        preprocessor: Optional[CenterlinePreprocessor] = None,
        postprocessor: Optional[CenterlinePostprocessor] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.preprocessor = preprocessor or CenterlinePreprocessor()
        self.postprocessor = postprocessor or CenterlinePostprocessor()
        self.model.eval()
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
        return_probabilities: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Predict centerline from retinal image.
        
        Args:
            image: Input retinal image (H, W, 3) or (H, W)
            roi_mask: Optional region of interest mask
            return_probabilities: Whether to return probability maps
            
        Returns:
            Dictionary containing:
                - 'skeleton': Binary centerline skeleton
                - 'vessel_mask': Binary vessel mask
                - 'centerline_prob': Centerline probability (if return_probabilities=True)
                - 'vessel_prob': Vessel probability (if return_probabilities=True)
        """
        # Preprocess
        image_processed = self.preprocessor(image)
        
        # To tensor
        x = torch.from_numpy(image_processed)[None, None].to(self.device)
        
        # Inference
        outputs = self.model(x)
        
        centerline_prob = outputs['centerline'][0, 0].cpu().numpy()
        vessel_prob = outputs['vessel_mask'][0, 0].cpu().numpy()
        
        # Postprocess
        skeleton, vessel_mask = self.postprocessor(
            centerline_prob, 
            vessel_prob, 
            roi_mask
        )
        
        result = {
            'skeleton': skeleton,
            'vessel_mask': vessel_mask
        }
        
        if return_probabilities:
            result['centerline_prob'] = centerline_prob
            result['vessel_prob'] = vessel_prob
        
        return result
    
    def evaluate(
        self,
        image: np.ndarray,
        gt_skeleton: np.ndarray,
        gt_vessel_mask: Optional[np.ndarray] = None,
        roi_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate centerline prediction.
        
        Args:
            image: Input retinal image
            gt_skeleton: Ground truth centerline skeleton
            gt_vessel_mask: Ground truth vessel mask (optional)
            roi_mask: Region of interest mask (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        from evaluation.metrics import CenterlineMetrics
        
        predictions = self.predict(image, roi_mask)
        
        metrics = CenterlineMetrics()
        return metrics.compute_all_metrics(
            predictions['skeleton'],
            gt_skeleton,
            gt_vessel_mask
        )


# -------------------------
# Loss Functions
# -------------------------

class CenterlineLoss(nn.Module):
    """
    Combined loss for centerline prediction.
    Balances centerline and vessel mask predictions.
    """
    
    def __init__(
        self,
        centerline_weight=2.0,
        vessel_weight=1.0,
        deep_supervision_weight=0.3,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()
        self.centerline_weight = centerline_weight
        self.vessel_weight = vessel_weight
        self.deep_supervision_weight = deep_supervision_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Dice loss for overlap maximization"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dict with 'centerline', 'vessel_mask', optional 'deep_outputs'
            targets: Dict with 'centerline', 'vessel_mask'
        """
        centerline_pred = outputs['centerline']
        vessel_pred = outputs['vessel_mask']
        centerline_gt = targets['centerline']
        vessel_gt = targets['vessel_mask']
        
        # Centerline loss (weighted more heavily)
        if self.use_focal_loss:
            cl_loss = self.focal_loss(centerline_pred, centerline_gt, self.focal_alpha, self.focal_gamma)
        else:
            cl_loss = F.binary_cross_entropy(centerline_pred, centerline_gt)
        
        cl_dice = self.dice_loss(centerline_pred, centerline_gt)
        centerline_loss = cl_loss + cl_dice
        
        # Vessel mask loss
        vessel_loss = F.binary_cross_entropy(vessel_pred, vessel_gt)
        vessel_dice = self.dice_loss(vessel_pred, vessel_gt)
        vessel_loss = vessel_loss + vessel_dice
        
        # Combined loss
        total_loss = (
            self.centerline_weight * centerline_loss + 
            self.vessel_weight * vessel_loss
        )
        
        # Deep supervision
        if 'deep_outputs' in outputs:
            deep_loss = 0
            for deep_out in outputs['deep_outputs']:
                deep_loss += F.binary_cross_entropy(deep_out, centerline_gt)
            total_loss += self.deep_supervision_weight * deep_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'centerline': centerline_loss.item(),
            'vessel': vessel_loss.item()
        }


# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    # Create model
    model = CenterlineUNet(
        in_channels=1,
        features=(64, 128, 256, 512),
        dropout=0.1,
        attention=True,
        deep_supervision=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 512, 512)
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Centerline output shape: {outputs['centerline'].shape}")
    print(f"Vessel mask output shape: {outputs['vessel_mask'].shape}")
    
    # Test predictor
    predictor = CenterlineUNetPredictor(
        model=model,
        device='cpu'
    )
    
    # Dummy image
    dummy_image = np.random.rand(512, 512, 3).astype(np.float32)
    result = predictor.predict(dummy_image, return_probabilities=True)
    
    print(f"\nPrediction keys: {result.keys()}")
    print(f"Skeleton unique values: {np.unique(result['skeleton'])}")