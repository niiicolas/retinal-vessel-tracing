# models/seed_detector.py
"""
Seed detection network for identifying starting points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torchvision.models as models


class SeedDetector(nn.Module):
    """
    CNN for detecting seed points (endpoints and junctions) on vessel centerlines.
    
    Outputs a heatmap of seed point probabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        seed_config = config.get('seed_detector', {})
        
        self.backbone_name = seed_config.get('backbone', 'resnet18')
        self.pretrained = seed_config.get('pretrained', True)
        self.nms_radius = seed_config.get('nms_radius', 10)
        self.confidence_threshold = seed_config.get('confidence_threshold', 0.5)
        self.top_k = seed_config.get('top_k_seeds', 50)
        
        # Build backbone
        self._build_backbone()
        
        # Decoder for upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _build_backbone(self):
        """Build the encoder backbone."""
        if self.backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=self.pretrained)
        elif self.backbone_name == 'resnet34':
            backbone = models.resnet34(pretrained=self.pretrained)
        elif self.backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")
        
        # Extract feature layers (without avgpool and fc)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            Seed heatmap (B, 1, H, W)
        """
        # Encode
        features = self.encoder(x)
        
        # Decode
        heatmap = self.decoder(features)
        
        # Resize to match input if needed
        if heatmap.shape[-2:] != x.shape[-2:]:
            heatmap = F.interpolate(
                heatmap, 
                size=x.shape[-2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        return heatmap
    
    def detect_seeds(self, 
                     image: torch.Tensor,
                     return_heatmap: bool = False
                     ) -> Tuple[List[List[Tuple[int, int, float]]], Optional[torch.Tensor]]:
        """
        Detect seed points in an image.
        
        Args:
            image: Input image tensor (B, 3, H, W)
            return_heatmap: Whether to also return the heatmap
            
        Returns:
            seeds: List of seed points per image [(y, x, confidence), ...]
            heatmap: Optional heatmap tensor
        """
        self.eval()
        with torch.no_grad():
            heatmap = self.forward(image)
        
        batch_seeds = []
        
        for b in range(heatmap.shape[0]):
            hmap = heatmap[b, 0].cpu().numpy()
            seeds = self._extract_seeds_nms(hmap)
            batch_seeds.append(seeds)
        
        if return_heatmap:
            return batch_seeds, heatmap
        return batch_seeds, None
    
    def _extract_seeds_nms(self, 
                           heatmap: np.ndarray
                           ) -> List[Tuple[int, int, float]]:
        """
        Extract seed points using non-maximum suppression.
        
        Args:
            heatmap: 2D heatmap array
            
        Returns:
            List of (y, x, confidence) tuples
        """
        from scipy import ndimage
        from skimage.feature import peak_local_max
        
        # Apply threshold
        mask = heatmap > self.confidence_threshold
        
        if not mask.any():
            # Return center of image if no seeds found
            h, w = heatmap.shape
            return [(h // 2, w // 2, 0.5)]
        
        # Find local maxima
        coordinates = peak_local_max(
            heatmap,
            min_distance=self.nms_radius,
            threshold_abs=self.confidence_threshold,
            num_peaks=self.top_k
        )
        
        # Extract seeds with confidence
        seeds = []
        for y, x in coordinates:
            confidence = float(heatmap[y, x])
            seeds.append((int(y), int(x), confidence))
        
        # Sort by confidence (descending)
        seeds.sort(key=lambda s: s[2], reverse=True)
        
        return seeds[:self.top_k]


class SeedDetectorTrainer:
    """Training utilities for seed detector."""
    
    def __init__(self, 
                 model: SeedDetector,
                 config: Dict[str, Any],
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('seed_detector', {}).get('learning_rate', 1e-4)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def create_seed_heatmap(self,
                            centerline: np.ndarray,
                            sigma: float = 3.0) -> np.ndarray:
        """
        Create ground truth seed heatmap from centerline.
        
        Args:
            centerline: Binary centerline mask
            sigma: Gaussian sigma for heatmap generation
            
        Returns:
            Seed heatmap with Gaussian blobs at endpoints and junctions
        """
        from data.centerline_extraction import CenterlineExtractor
        from scipy.ndimage import gaussian_filter
        
        extractor = CenterlineExtractor()
        
        # Find endpoints and junctions
        endpoints = extractor._find_endpoints(centerline)
        junctions = extractor._find_junctions(centerline)
        
        # Create heatmap
        heatmap = np.zeros_like(centerline, dtype=np.float32)
        
        # Place points
        for y, x in endpoints + junctions:
            if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                heatmap[y, x] = 1.0
        
        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def compute_loss(self,
                     pred_heatmap: torch.Tensor,
                     gt_heatmap: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy loss.
        
        Args:
            pred_heatmap: Predicted heatmap (B, 1, H, W)
            gt_heatmap: Ground truth heatmap (B, 1, H, W)
            mask: Optional FOV mask
            
        Returns:
            Loss value
        """
        # Focal loss for handling class imbalance
        alpha = 0.25
        gamma = 2.0
        
        bce = F.binary_cross_entropy(pred_heatmap, gt_heatmap, reduction='none')
        
        pt = torch.where(gt_heatmap > 0.5, pred_heatmap, 1 - pred_heatmap)
        focal_weight = alpha * (1 - pt) ** gamma
        
        loss = focal_weight * bce
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()
    
    def train_step(self,
                   images: torch.Tensor,
                   centerlines: torch.Tensor,
                   masks: Optional[torch.Tensor] = None) -> float:
        """
        Single training step.
        
        Args:
            images: Batch of images (B, 3, H, W)
            centerlines: Batch of centerlines (B, 1, H, W)
            masks: Optional FOV masks
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create GT heatmaps
        gt_heatmaps = []
        for i in range(centerlines.shape[0]):
            cl = centerlines[i, 0].cpu().numpy()
            hm = self.create_seed_heatmap(cl)
            gt_heatmaps.append(hm)
        gt_heatmaps = torch.tensor(np.stack(gt_heatmaps)).unsqueeze(1).to(self.device)
        
        # Forward pass
        pred_heatmaps = self.model(images)
        
        # Compute loss
        loss = self.compute_loss(pred_heatmaps, gt_heatmaps, masks)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
