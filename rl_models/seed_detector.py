# models/seed_detector.py
"""
Seed detector — predicts a sparse heatmap of vessel endpoints and junctions.

Architecture: CenterlineUNet (same as baseline, see centerline_unet_baseline.py)
  - Depthwise-separable convolutions (~0.5M params)
  - 4-level encoder/decoder with skip connections  ← key upgrade vs ResNet+plain decoder
  - in_channels=3 (RGB with enhanced green)
  - Sigmoid output → heatmap in [0, 1]

Difference from centerline baseline:
  - Input: 3-channel RGB (not 1-channel greyscale)
  - GT targets: sparse Gaussian blobs at endpoints+junctions only (not full vessel mask)
  - Loss: focal loss (not BCE+clDice) — clDice is for connected segments, not sparse keypoints

Provides:
    SeedDetector  — UNet → (B,1,H,W) heatmap
                    call .detect_seeds() to get ranked (y, x, confidence) lists

Training logic lives in training/seed_detector_trainer.py.
Used at inference time by scripts/drive_rl_tracing.py via FrontierTracer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional


# ==========================================
# UNET BUILDING BLOCKS
# Copied from centerline_unet_baseline.py — identical implementation
# ==========================================

class DSConvBlock(nn.Module):
    """Depthwise-Separable Conv → BN → ReLU (x2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  in_ch,  3, padding=1, groups=in_ch,  bias=False),
            nn.Conv2d(in_ch,  out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool → DSConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DSConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Bilinear upsample → concat skip → DSConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DSConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ==========================================
# SEED DETECTOR  (UNet backbone)
# ==========================================

class SeedDetector(nn.Module):
    """
    UNet-based seed point heatmap predictor.

    Input : (B, 3, H, W)  full RGB fundus image, float32 in [0, 1]
    Output: (B, 1, H, W)  heatmap in [0, 1], peaks = endpoints / junctions

    Architecture is identical to CenterlineUNet(in_channels=3, base_ch=16)
    from centerline_unet_baseline.py. Skip connections give the decoder
    access to full-resolution spatial features at every level — critical
    for localising sparse keypoints precisely on thin vessels.

    Channel layout (base_ch=16):
        enc0 →  16   enc1 →  32   enc2 →  64   enc3 → 128
        bot  → 256
        up3  → 128   up2  →  64   up1  →  32   up0  →  16
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        seed_cfg = config.get('seed_detector', {})
        self.nms_radius           = seed_cfg.get('nms_radius',           10)
        self.confidence_threshold = seed_cfg.get('confidence_threshold', 0.3)
        self.top_k                = seed_cfg.get('top_k_seeds',          50)
        base_ch                   = seed_cfg.get('base_ch',              16)

        ch = [base_ch * (2 ** i) for i in range(5)]
        # ch = [16, 32, 64, 128, 256] for base_ch=16

        # Encoder — same as CenterlineUNet, in_channels=3 for RGB
        self.enc0 = DSConvBlock(3,     ch[0])
        self.enc1 = DownBlock(ch[0],   ch[1])
        self.enc2 = DownBlock(ch[1],   ch[2])
        self.enc3 = DownBlock(ch[2],   ch[3])

        # Bottleneck
        self.bot  = DownBlock(ch[3],   ch[4])

        # Decoder with skip connections — same as CenterlineUNet
        self.up3  = UpBlock(ch[4], ch[3], ch[3])   # 256 + 128 → 128
        self.up2  = UpBlock(ch[3], ch[2], ch[2])   # 128 +  64 →  64
        self.up1  = UpBlock(ch[2], ch[1], ch[1])   #  64 +  32 →  32
        self.up0  = UpBlock(ch[1], ch[0], ch[0])   #  32 +  16 →  16

        # Head — same as CenterlineUNet
        self.head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Same initialisation as CenterlineUNet."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — save skip tensors
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        # Bottleneck
        b = self.bot(s3)

        # Decoder — skip connections from encoder
        d3 = self.up3(b,  s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return torch.sigmoid(self.head(d0))   # (B, 1, H, W) in [0, 1]

    # ------------------------------------------------------------------
    # Inference helpers — identical to previous ResNet version
    # ------------------------------------------------------------------

    def detect_seeds(self,
                     image: torch.Tensor,
                     obs_half: int = 32,
                     return_heatmap: bool = False,
                     ) -> Tuple[List[List[Tuple[int, int, float]]], Optional[torch.Tensor]]:
        """
        Run inference and return ranked seed points per image.

        Args:
            image:          (B, 3, H, W) float32 tensor
            obs_half:       half of observation patch size — seeds within
                            this margin from the border are clipped inward
                            so the environment never crashes on out-of-bounds
            return_heatmap: also return the raw heatmap tensor

        Returns:
            batch_seeds:  list (len B) of [(y, x, confidence), ...] sorted
                          by confidence descending, top_k max per image
            heatmap:      (B,1,H,W) tensor if return_heatmap else None
        """
        self.eval()
        with torch.no_grad():
            heatmap = self.forward(image)

        h, w   = image.shape[-2], image.shape[-1]
        margin = obs_half + 5    # matches _pick_frontier_seed clip

        batch_seeds = []
        for b in range(heatmap.shape[0]):
            hmap  = heatmap[b, 0].cpu().numpy()
            seeds = self._extract_seeds_nms(hmap, h, w, margin)
            batch_seeds.append(seeds)

        return batch_seeds, (heatmap if return_heatmap else None)

    def _extract_seeds_nms(self,
                           heatmap: np.ndarray,
                           h: int, w: int,
                           margin: int,
                           ) -> List[Tuple[int, int, float]]:
        """
        Non-maximum suppression on heatmap → top-k seed list.

        Seeds outside the border margin are clipped inward rather than
        discarded, so we never lose coverage of near-border vessels.
        """
        from skimage.feature import peak_local_max

        if not (heatmap > self.confidence_threshold).any():
            # Fallback: image centre if no confident peaks found
            return [(h // 2, w // 2, 0.5)]

        coords = peak_local_max(
            heatmap,
            min_distance=self.nms_radius,
            threshold_abs=self.confidence_threshold,
            num_peaks=self.top_k,
        )

        seeds = []
        for y, x in coords:
            y = int(np.clip(y, margin, h - margin - 1))
            x = int(np.clip(x, margin, w - margin - 1))
            seeds.append((y, x, float(heatmap[y, x])))

        seeds.sort(key=lambda s: s[2], reverse=True)
        return seeds[:self.top_k]