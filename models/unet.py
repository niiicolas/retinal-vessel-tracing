"""Lightweight depthwise-separable Centerline UNet, its clDice loss, and an inference wrapper."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.greedy_tracer import GreedyTracer
from models.unet_blocks import DownBlock, DSConvBlock, UpBlock


class CenterlineUNet(nn.Module):
    """~0.5M-param 4-level depthwise-separable UNet mapping an image to a (B,1,H,W) centerline probability."""

    def __init__(self, in_channels: int = 1, base_ch: int = 16):
        """Build the encoder/bottleneck/decoder and 1-channel head; widths scale with ``base_ch``."""
        super().__init__()

        ch = [base_ch * (2**i) for i in range(5)]

        self.enc0 = DSConvBlock(in_channels, ch[0])
        self.enc1 = DownBlock(ch[0], ch[1])
        self.enc2 = DownBlock(ch[1], ch[2])
        self.enc3 = DownBlock(ch[2], ch[3])

        self.bot = DownBlock(ch[3], ch[4])

        # UpBlock(in_from_below, skip_from_encoder, out).
        self.up3 = UpBlock(ch[4], ch[3], ch[3])
        self.up2 = UpBlock(ch[3], ch[2], ch[2])
        self.up1 = UpBlock(ch[2], ch[1], ch[1])
        self.up0 = UpBlock(ch[1], ch[0], ch[0])

        self.head = nn.Sequential(nn.Conv2d(ch[0], ch[0] // 2, 1), nn.ReLU(inplace=True), nn.Conv2d(ch[0] // 2, 1, 1))

        self._init_weights()

    def _init_weights(self):
        """Kaiming-init conv weights (ReLU gain); ones/zeros for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, bottleneck, decode with skips; return a (B, 1, H, W) sigmoid probability map."""
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        b = self.bot(s3)

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return torch.sigmoid(self.head(d0))


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological erosion via 3×3 min-pool; requires a 4-D (B,1,H,W) tensor."""
    if img.ndim == 4:
        return -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
    raise ValueError('Expected 4-D tensor.')


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological dilation via 3×3 max-pool."""
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological opening (dilate ∘ erode)."""
    return _soft_dilate(_soft_erode(img))


def soft_skeleton(img: torch.Tensor, num_iter: int = 10) -> torch.Tensor:
    """Differentiable skeleton approximation via iterative soft erosion (Shit et al., clDice, CVPR 2021)."""
    skel = F.relu(img - _soft_open(img))
    for _ in range(num_iter):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CenterlineLoss(nn.Module):
    """Combined BCE + soft-clDice loss: ``w_bce·BCE + w_cl·(1 − soft_clDice)``.

    The differentiable soft-skeleton lets gradients flow through the topology term.
    """

    def __init__(self, bce_weight: float = 0.4, cl_weight: float = 0.6, skeleton_iter: int = 10, pos_weight: Optional[float] = None):
        """Store loss weights, soft-skeleton iteration count, and the optional BCE positive-class weight."""
        super().__init__()
        self.bce_weight = bce_weight
        self.cl_weight = cl_weight
        self.skeleton_iter = skeleton_iter
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def _soft_cl_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return mean soft clDice in [0, 1] (higher is better) for (B,1,H,W) probability maps."""
        skel_pred = soft_skeleton(pred, self.skeleton_iter)
        skel_target = soft_skeleton(target, self.skeleton_iter)

        # Topology precision (pred-skeleton on target) and sensitivity (gt-skeleton covered by pred).
        tprec = (skel_pred * target).sum(dim=[1, 2, 3]) / (skel_pred.sum(dim=[1, 2, 3]) + 1e-5)
        tsens = (skel_target * pred).sum(dim=[1, 2, 3]) / (skel_target.sum(dim=[1, 2, 3]) + 1e-5)

        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-5)
        return cl_dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Compute the combined loss for one batch.

        Args:
            pred: (B, 1, H, W) sigmoid output.
            target: (B, 1, H, W) binary GT centerline (float).
            mask: optional (B, 1, H, W) FOV mask restricting the BCE term to the ROI.

        Returns:
            ``(total_loss, {'bce', 'cl_dice', 'total'})``.
        """
        if mask is not None:
            p = pred[mask > 0]
            t = target[mask > 0]
        else:
            p = pred.reshape(-1)
            t = target.reshape(-1)

        pw = self.pos_weight.to(pred.device) if self.pos_weight is not None else None
        if pw is not None:
            # Weighted BCE up-weights false negatives on rare centerline pixels.
            bce = -(pw * t * torch.log(p + 1e-5) + (1 - t) * torch.log(1 - p + 1e-5)).mean()
        else:
            bce = F.binary_cross_entropy(p, t, reduction='mean')

        # clDice runs on the full maps (skeletonization is spatial), not the masked vectors.
        cl_d = self._soft_cl_dice(pred, target)
        total = self.bce_weight * bce + self.cl_weight * (1.0 - cl_d)

        return total, {'bce': bce.item(), 'cl_dice': cl_d.item(), 'total': total.item()}


class CenterlinePredictor:
    """End-to-end inference wrapper pairing a CenterlineUNet with a GreedyTracer."""

    def __init__(
        self,
        model: CenterlineUNet,
        tracer: Optional[GreedyTracer] = None,
        device: str = 'cpu',
        patch_size: Optional[int] = None,
        patch_stride: Optional[int] = None,
    ):
        """Hold an eval-mode model and tracer; setting ``patch_size`` enables sliding-window inference."""
        self.model = model.to(device).eval()
        self.tracer = tracer or GreedyTracer()
        self.device = device
        self.patch_size = patch_size
        self.patch_stride = patch_stride or (patch_size // 2 if patch_size else None)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = 'cpu', **kwargs) -> 'CenterlinePredictor':
        """Load weights + model config from a checkpoint and wrap them in a predictor."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get('model_cfg', {})
        model = CenterlineUNet(**cfg)
        model.load_state_dict(ckpt['model_state'])
        return cls(model, device=device, **kwargs)

    @torch.no_grad()
    def _infer_full(self, img_t: torch.Tensor) -> torch.Tensor:
        """Run the model on the whole image at once; return the (H, W) probability map."""
        return self.model(img_t.unsqueeze(0).to(self.device))[0, 0].cpu()

    @torch.no_grad()
    def _infer_patched(self, img_t: torch.Tensor) -> torch.Tensor:
        """Sliding-window inference with Gaussian-weighted blending to suppress patch-edge artifacts."""
        C, H, W = img_t.shape
        ps = self.patch_size
        st = self.patch_stride

        prob = torch.zeros(H, W)
        count = torch.zeros(H, W)

        # 2-D Gaussian window weights each patch's contribution by distance from its centre.
        lin = torch.linspace(-1, 1, ps)
        gauss = torch.exp(-2 * (lin**2))
        win = gauss[:, None] * gauss[None, :]

        ys = list(range(0, H - ps + 1, st)) + [max(0, H - ps)]
        xs = list(range(0, W - ps + 1, st)) + [max(0, W - ps)]

        for y in set(ys):
            for x in set(xs):
                patch = img_t[:, y : y + ps, x : x + ps].unsqueeze(0).to(self.device)
                out = self.model(patch)[0, 0].cpu()
                prob[y : y + ps, x : x + ps] += out * win
                count[y : y + ps, x : x + ps] += win

        return prob / (count + 1e-8)

    def predict(self, image: np.ndarray, fov_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the centerline probability map and traced skeleton for one image.

        Returns:
            ``(prob_map float32 (H, W), skeleton uint8 (H, W))``.
        """
        img_t = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)

        if self.patch_size is not None:
            prob = self._infer_patched(img_t)
        else:
            prob = self._infer_full(img_t)

        prob_np = prob.numpy()
        skeleton, _ = self.tracer.trace(prob_np, fov_mask)
        return prob_np, skeleton


if __name__ == '__main__':
    print('=== CenterlineUNet Sanity Check ===')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CenterlineUNet(in_channels=1, base_ch=16).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f'Parameters : {total:,}  (~{total / 1e6:.2f}M)')

    x = torch.rand(2, 1, 512, 512, device=device)
    target = torch.zeros(2, 1, 512, 512, device=device)
    target[:, :, 100:400, 254:258] = 1.0  # thin vertical line as a synthetic centerline

    pred = model(x)
    print(f'Input      : {tuple(x.shape)}  →  Output: {tuple(pred.shape)}')
    print(f'Pred range : [{pred.min():.3f}, {pred.max():.3f}]')

    criterion = CenterlineLoss(bce_weight=0.4, cl_weight=0.6, pos_weight=10.0)
    loss, breakdown = criterion(pred, target)
    print(f'Loss       : {loss.item():.4f}  |  {breakdown}')

    tracer = GreedyTracer(seed_thresh=0.5, step_thresh=0.3, min_length=5)
    prob_np = pred[0, 0].detach().cpu().numpy()
    skeleton, _ = tracer.trace(prob_np)
    print(f'Skeleton   : {skeleton.shape}, nonzero pixels: {skeleton.sum() // 255}')

    print('=== All OK ===')
