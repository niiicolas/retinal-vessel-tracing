"""Shared UNet building blocks (depthwise-separable conv, down, up) for CenterlineUNet and SeedDetector."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvBlock(nn.Module):
    """Two depthwise-separable conv → BN → ReLU stages."""

    def __init__(self, in_ch: int, out_ch: int):
        """Build the two depthwise-separable conv stages (in_ch → out_ch → out_ch)."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-stage depthwise-separable conv block."""
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder stage: 2× max-pool then a DSConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        """Build the max-pool and DSConvBlock."""
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DSConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by 2 then convolve."""
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Decoder stage: bilinear upsample, concatenate the skip, then a DSConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        """Build the upsampler and the post-concat DSConvBlock."""
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DSConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample ``x``, align it to ``skip`` for odd dims, concat, then convolve."""
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))
