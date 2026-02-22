"""
centerline_unet.py
==================
Lightweight Centerline UNet for retinal vessel centerline probability estimation.

Architecture:
  - Depthwise-separable convolutions for efficiency
  - 4-level encoder/decoder with skip connections
  - Single-channel sigmoid output → centerline probability map

Loss:
  - clDice  (topology-aware, skeleton overlap)
  - Binary Cross-Entropy (pixel-level)
  - Combined: total = BCE_weight * BCE + clDice_weight * (1 - clDice)

Extras:
  - GreedyTracer: converts probability map → binary skeleton via
    seeded greedy traversal following local maxima
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import skeletonize
import numpy as np
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class DSConvBlock(nn.Module):
    """Depthwise-Separable Conv → BN → ReLU (x2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            # first DS conv
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # second DS conv
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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
    """Bilinear upsample → cat skip → DSConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DSConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd spatial dims
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────
# CENTERLINE UNET
# ─────────────────────────────────────────────────────────────

class CenterlineUNet(nn.Module):
    """
    Lightweight UNet for centerline probability estimation.

    Input:  (B, in_channels, H, W)  - float32, normalised 0-1
    Output: (B, 1, H, W)            - sigmoid probability map
    
    Default channel widths keep the model ~0.5 M parameters.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_ch: int = 16,           # multiply for wider network
        depth: int = 4,              # number of encoder levels (max 4)
    ):
        super().__init__()
        assert depth in (3, 4), "depth must be 3 or 4"

        ch = [base_ch * (2 ** i) for i in range(depth + 1)]
        # e.g. depth=4, base=16 → [16, 32, 64, 128, 256]

        # Encoder
        self.enc0 = DSConvBlock(in_channels, ch[0])
        self.enc1 = DownBlock(ch[0], ch[1])
        self.enc2 = DownBlock(ch[1], ch[2])
        self.enc3 = DownBlock(ch[2], ch[3])

        # Bottleneck (optional 4th level)
        self.has_bot = depth == 4
        if self.has_bot:
            self.bot = DownBlock(ch[3], ch[4])

        # Decoder
        if self.has_bot:
            self.up3 = UpBlock(ch[4], ch[3], ch[3])
        else:
            self.up3 = UpBlock(ch[3], ch[2], ch[2])

        self.up2 = UpBlock(ch[3] if self.has_bot else ch[2], ch[2] if self.has_bot else ch[1], ch[2] if self.has_bot else ch[1])
        self.up1 = UpBlock(ch[2] if self.has_bot else ch[1], ch[1] if self.has_bot else ch[0], ch[1] if self.has_bot else ch[0])
        self.up0 = UpBlock(ch[1] if self.has_bot else ch[0], ch[0], ch[0])

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        if self.has_bot:
            b = self.bot(s3)
            d3 = self.up3(b, s3)
        else:
            d3 = s3

        if self.has_bot:
            d2 = self.up2(d3, s2)
            d1 = self.up1(d2, s1)
            d0 = self.up0(d1, s0)
        else:
            d2 = self.up2(d3, s2)
            d1 = self.up1(d2, s1)
            d0 = self.up0(d1, s0)

        return torch.sigmoid(self.head(d0))


# ─────────────────────────────────────────────────────────────
# SOFT SKELETONISATION  (differentiable proxy for clDice)
# ─────────────────────────────────────────────────────────────

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Morphological min-pool (2-connectivity)."""
    if img.ndim == 4:                      # (B, 1, H, W)
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    raise ValueError("Expected 4-D tensor.")


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """
    Differentiable skeleton approximation via iterative soft-erosion.
    Reference: Shit et al., "clDice – a Novel Topology-Preserving Loss Function
               for Tubular Structure Segmentation", CVPR 2021.
    """
    skel = F.relu(img - _soft_open(img))
    for _ in range(num_iter):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ─────────────────────────────────────────────────────────────
# TOPOLOGY-AWARE LOSS
# ─────────────────────────────────────────────────────────────

class CenterlineLoss(nn.Module):
    """
    Combined loss:
        L = w_bce * BCE + w_cl * (1 - soft_clDice)

    soft_clDice uses a differentiable skeleton proxy so gradients
    flow back into the network through both terms.

    Args:
        bce_weight   : weight for BCE term
        cl_weight    : weight for clDice term
        skeleton_iter: soft-skeleton iterations (more → sharper, slower)
        pos_weight   : optional positive-class weight for BCE (handles imbalance)
    """

    def __init__(
        self,
        bce_weight: float = 0.4,
        cl_weight: float = 0.6,
        skeleton_iter: int = 10,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.cl_weight = cl_weight
        self.skeleton_iter = skeleton_iter
        self.pos_weight = (
            torch.tensor([pos_weight]) if pos_weight is not None else None
        )

    def _soft_cl_dice(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        soft clDice ∈ [0, 1], higher is better.
        pred, target: (B, 1, H, W) float, [0,1]
        """
        skel_pred   = soft_skeleton(pred,   self.skeleton_iter)
        skel_target = soft_skeleton(target, self.skeleton_iter)

        # Topology-Precision: how much of pred-skeleton lies on target
        tprec = (skel_pred * target).sum(dim=[1, 2, 3]) / (skel_pred.sum(dim=[1, 2, 3]) + 1e-8)
        # Topology-Sensitivity: how much of gt-skeleton is covered by pred
        tsens = (skel_target * pred).sum(dim=[1, 2, 3]) / (skel_target.sum(dim=[1, 2, 3]) + 1e-8)

        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
        return cl_dice.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred   : (B, 1, H, W) sigmoid output
            target : (B, 1, H, W) binary GT centerline, float
            mask   : (B, 1, H, W) optional FOV mask – loss only inside ROI

        Returns:
            total_loss, {'bce': ..., 'cl_dice': ..., 'total': ...}
        """
        if mask is not None:
            # flatten & select only ROI pixels for BCE
            p = pred[mask > 0]
            t = target[mask > 0]
        else:
            p = pred.reshape(-1)
            t = target.reshape(-1)

        pw = self.pos_weight.to(pred.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy(p, t, weight=None, reduction='mean')
        if pw is not None:
            # manual weighted BCE
            bce = -(pw * t * torch.log(p + 1e-8) + (1 - t) * torch.log(1 - p + 1e-8)).mean()

        cl_d = self._soft_cl_dice(pred, target)
        total = self.bce_weight * bce + self.cl_weight * (1.0 - cl_d)

        return total, {
            'bce': bce.item(),
            'cl_dice': cl_d.item(),
            'total': total.item(),
        }


# ─────────────────────────────────────────────────────────────
# GREEDY TRACER
# ─────────────────────────────────────────────────────────────

class GreedyTracer:
    """
    Converts a soft centerline probability map into a binary skeleton
    via greedy traversal from seed points.

    Algorithm:
      1. Threshold map at `seed_thresh` to find candidate seeds.
      2. Non-max suppress (keep local maxima in 3x3 neighbourhood).
      3. For each unvisited seed (highest prob first):
         a. Follow the steepest-ascent neighbour if prob > `step_thresh`.
         b. Trace until the path revisits a visited pixel or falls below threshold.
      4. Optionally post-process with morphological thinning.

    Args:
        seed_thresh  : minimum probability to start a trace
        step_thresh  : minimum probability to continue stepping
        min_length   : discard traces shorter than this (pixels)
        thin_output  : apply skimage skeletonize to the binary output
    """

    def __init__(
        self,
        seed_thresh: float = 0.5,
        step_thresh: float = 0.3,
        min_length: int = 5,
        thin_output: bool = True,
    ):
        self.seed_thresh = seed_thresh
        self.step_thresh = step_thresh
        self.min_length = min_length
        self.thin_output = thin_output

        # 8-connected neighbour offsets
        self._offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0,  -1),           (0,  1),
            (1,  -1),  (1, 0),  (1,  1),
        ]

    def _local_maxima(self, prob: np.ndarray) -> np.ndarray:
        """Return boolean mask of strict 8-neighbour local maxima."""
        padded = np.pad(prob, 1, mode='constant', constant_values=0)
        lm = np.ones_like(prob, dtype=bool)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[1 + dy:1 + dy + prob.shape[0],
                                 1 + dx:1 + dx + prob.shape[1]]
                lm &= prob >= shifted
        return lm

    def _trace_from(
        self,
        prob: np.ndarray,
        visited: np.ndarray,
        start_r: int,
        start_c: int,
    ) -> List[Tuple[int, int]]:
        """Greedy steepest-ascent trace. Returns list of (r, c) pixel coords."""
        H, W = prob.shape
        path = [(start_r, start_c)]
        visited[start_r, start_c] = True

        r, c = start_r, start_c
        while True:
            best_val = self.step_thresh
            best_rc = None
            for dr, dc in self._offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                    if prob[nr, nc] > best_val:
                        best_val = prob[nr, nc]
                        best_rc = (nr, nc)
            if best_rc is None:
                break
            r, c = best_rc
            visited[r, c] = True
            path.append((r, c))

        return path

    def trace(
        self,
        prob_map: np.ndarray,
        fov_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            prob_map : (H, W) float32 probability map
            fov_mask : (H, W) uint8/bool – trace only inside mask

        Returns:
            skeleton : (H, W) uint8 binary centerline image
        """
        prob = prob_map.copy().astype(np.float32)
        if fov_mask is not None:
            prob[fov_mask == 0] = 0.0

        H, W = prob.shape
        skeleton = np.zeros((H, W), dtype=np.uint8)
        visited = np.zeros((H, W), dtype=bool)

        # Candidate seeds: above threshold AND local maxima
        candidates = (prob >= self.seed_thresh) & self._local_maxima(prob)
        seed_coords = np.argwhere(candidates)

        # Sort seeds by descending probability (greedy: start from strongest)
        seed_probs = prob[seed_coords[:, 0], seed_coords[:, 1]]
        order = np.argsort(-seed_probs)
        seed_coords = seed_coords[order]

        for sr, sc in seed_coords:
            if visited[sr, sc]:
                continue
            path = self._trace_from(prob, visited, sr, sc)
            if len(path) >= self.min_length:
                for r, c in path:
                    skeleton[r, c] = 255

        if self.thin_output and skeleton.any():
            skeleton = (skeletonize(skeleton > 0) * 255).astype(np.uint8)

        return skeleton

    def trace_batch(
        self,
        prob_maps: np.ndarray,
        fov_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Args:
            prob_maps : (B, H, W) or (B, 1, H, W)
            fov_masks : (B, H, W) or (B, 1, H, W), optional

        Returns:
            skeletons : (B, H, W) uint8
        """
        if prob_maps.ndim == 4:
            prob_maps = prob_maps[:, 0]
        if fov_masks is not None and fov_masks.ndim == 4:
            fov_masks = fov_masks[:, 0]

        B = prob_maps.shape[0]
        results = []
        for i in range(B):
            m = fov_masks[i] if fov_masks is not None else None
            results.append(self.trace(prob_maps[i], m))
        return np.stack(results, axis=0)


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: FULL INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

class CenterlinePredictor:
    """
    Wraps model + tracer for end-to-end inference.

    Usage:
        predictor = CenterlinePredictor.from_checkpoint('weights.pt')
        skeleton  = predictor.predict(image_np, fov_mask_np)
    """

    def __init__(
        self,
        model: CenterlineUNet,
        tracer: Optional[GreedyTracer] = None,
        device: str = 'cpu',
        patch_size: Optional[int] = None,   # None → full image
        patch_stride: Optional[int] = None,
    ):
        self.model = model.to(device).eval()
        self.tracer = tracer or GreedyTracer()
        self.device = device
        self.patch_size = patch_size
        self.patch_stride = patch_stride or (patch_size // 2 if patch_size else None)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: str = 'cpu',
        **kwargs,
    ) -> 'CenterlinePredictor':
        ckpt = torch.load(path, map_location=device)
        cfg  = ckpt.get('model_cfg', {})
        model = CenterlineUNet(**cfg)
        model.load_state_dict(ckpt['model_state'])
        return cls(model, device=device, **kwargs)

    @torch.no_grad()
    def _infer_full(self, img_t: torch.Tensor) -> torch.Tensor:
        return self.model(img_t.unsqueeze(0).to(self.device))[0, 0].cpu()

    @torch.no_grad()
    def _infer_patched(self, img_t: torch.Tensor) -> torch.Tensor:
        """Sliding-window inference with Gaussian blend weights."""
        C, H, W = img_t.shape
        ps = self.patch_size
        st = self.patch_stride

        prob  = torch.zeros(H, W)
        count = torch.zeros(H, W)

        # Gaussian weight window
        lin   = torch.linspace(-1, 1, ps)
        gauss = torch.exp(-2 * (lin ** 2))
        win   = (gauss[:, None] * gauss[None, :])

        ys = list(range(0, H - ps + 1, st)) + [max(0, H - ps)]
        xs = list(range(0, W - ps + 1, st)) + [max(0, W - ps)]

        for y in set(ys):
            for x in set(xs):
                patch = img_t[:, y:y + ps, x:x + ps].unsqueeze(0).to(self.device)
                out = self.model(patch)[0, 0].cpu()
                prob[y:y + ps, x:x + ps]  += out * win
                count[y:y + ps, x:x + ps] += win

        return prob / (count + 1e-8)

    def predict(
        self,
        image: np.ndarray,           # (H, W) float32 pre-processed
        fov_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            prob_map : (H, W) float32  raw probability
            skeleton : (H, W) uint8    binarised centerline
        """
        img_t = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)

        if self.patch_size is not None:
            prob = self._infer_patched(img_t)
        else:
            prob = self._infer_full(img_t)

        prob_np = prob.numpy()
        skeleton = self.tracer.trace(prob_np, fov_mask)
        return prob_np, skeleton


# ─────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=== CenterlineUNet Sanity Check ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Model ──
    model = CenterlineUNet(in_channels=1, base_ch=16, depth=4).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total:,}  (~{total/1e6:.2f}M)")

    # ── Forward pass ──
    x      = torch.rand(2, 1, 512, 512, device=device)
    target = torch.zeros(2, 1, 512, 512, device=device)
    target[:, :, 100:400, 254:258] = 1.0   # thin vertical line

    pred = model(x)
    print(f"Input      : {tuple(x.shape)}  →  Output: {tuple(pred.shape)}")
    print(f"Pred range : [{pred.min():.3f}, {pred.max():.3f}]")

    # ── Loss ──
    criterion = CenterlineLoss(bce_weight=0.4, cl_weight=0.6, pos_weight=10.0)
    loss, breakdown = criterion(pred, target)
    print(f"Loss       : {loss.item():.4f}  |  {breakdown}")

    # ── Greedy Tracer ──
    tracer   = GreedyTracer(seed_thresh=0.5, step_thresh=0.3, min_length=5)
    prob_np  = pred[0, 0].detach().cpu().numpy()
    skeleton = tracer.trace(prob_np)
    print(f"Skeleton   : {skeleton.shape}, nonzero pixels: {skeleton.sum() // 255}")

    print("=== All OK ===")