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
        
        # Decoder
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
        
    def _build_backbone(self
