"""Multi-task Attention U-Net seed detector for the RL vessel-tracing pipeline.

Predicts vessel/centerline/radius/orientation maps (optionally with a Frangi prior and
MC-dropout confidence) and extracts seeds as ridge peaks. ``detect_seeds`` keeps the legacy
tuple API; ``detect_seeds_rich`` returns the per-seed dict. Trainer lives in training/.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.fundus_preprocessor import FOV_EROSION_FRAC, FOV_EROSION_MAX, FOV_EROSION_MIN
from models.unet_blocks import DSConvBlock, DownBlock


# FOV-scale-invariant seeding constants: seed spacing scales with FOV radius (floor 8 px)
# so small, tightly-cropped retinas aren't starved of peripheral seeds. See [[fov-scale-invariance]].
# FOV_EROSION_* live in data.fundus_preprocessor so the baselines reuse the same rim erosion.
SEED_REF_FOV_RADIUS = 250.0  # reference FOV radius where base spacing applies
SEED_BASE_SPACING = 22  # min seed spacing at the reference radius
SEED_FLOOR_SPACING = 8


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018) that suppresses irrelevant U-Net skip activations.

    Learns a gating mask from the coarser decoder feature ``g``: α = σ(ψ(ReLU(W_g g + W_x x))),
    x' = α ⊙ x.
    """

    def __init__(self, in_ch: int, gating_ch: int, inter_ch: Optional[int] = None):
        """Build the 1×1 projections and the gating head (defaults inter_ch to in_ch//2, min 8)."""
        super().__init__()
        inter = inter_ch or max(in_ch // 2, 8)
        self.W_x = nn.Conv2d(in_ch, inter, 1, bias=False)
        self.W_g = nn.Conv2d(gating_ch, inter, 1, bias=False)
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Gate skip features ``x`` by an attention mask learned from decoder feature ``g``."""
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        a = self.psi(self.W_x(x) + self.W_g(g))
        return x * a


class AttnUpBlock(nn.Module):
    """Decoder stage: bilinear upsample → attention-gated skip concat → DSConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        """Build the upsampler, attention gate, and post-concat DSConvBlock."""
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.gate = AttentionGate(skip_ch, in_ch)
        self.conv = DSConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample ``x``, gate ``skip`` by it, concatenate, and convolve."""
        x_up = self.up(x)
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(x_up, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        skip_a = self.gate(skip, x_up)
        return self.conv(torch.cat([x_up, skip_a], dim=1))


def frangi_vesselness(green_or_gray: np.ndarray, scales=(1.0, 1.6, 2.5, 4.0)) -> np.ndarray:
    """Return a multi-scale Frangi vesselness map (float32 (H, W) in [0, 1]) for a single-channel image.

    Intensities are inverted so dark vessels light up. Used as an optional input channel and
    an inference-time fallback gate.
    """
    from skimage.filters import frangi

    g = green_or_gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    inv = 1.0 - g
    v = frangi(inv, sigmas=scales, black_ridges=False)
    if v.max() > 0:
        v = v / v.max()
    return v.astype(np.float32)


def _soft_skeletonize(x: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Differentiable skeletonization via repeated soft erosion/opening."""
    pool = lambda t: F.max_pool2d(t, 3, stride=1, padding=1)
    neg_pool = lambda t: -F.max_pool2d(-t, 3, stride=1, padding=1)
    skel = F.relu(x - pool(neg_pool(x)))
    for _ in range(iters - 1):
        x = neg_pool(x)
        skel = skel + F.relu((1.0 - skel) * (x - pool(neg_pool(x))))
    return skel


def cl_dice_loss(pred_vessel: torch.Tensor, gt_vessel: torch.Tensor, iters: int = 8, eps: float = 1e-6) -> torch.Tensor:
    """Topology-preserving clDice loss (1 − clDice) on a soft vessel prediction."""
    s_p = _soft_skeletonize(pred_vessel, iters)
    s_g = _soft_skeletonize(gt_vessel, iters)
    t_prec = (s_p * gt_vessel).sum() / (s_p.sum() + eps)
    t_rec = (s_g * pred_vessel).sum() / (s_g.sum() + eps)
    cldice = 2.0 * t_prec * t_rec / (t_prec + t_rec + eps)
    return 1.0 - cldice


def soft_dice_bce(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, bce_w: float = 1.0, dice_w: float = 1.0, eps: float = 1e-6
) -> torch.Tensor:
    """FOV-masked weighted sum of binary cross-entropy and soft-Dice for a probability map."""
    pred = pred.clamp(eps, 1.0 - eps)
    m = mask if mask is not None else torch.ones_like(pred)
    bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    bce = (bce * m).sum() / m.sum().clamp(min=1.0)
    p = pred * m
    t = target * m
    inter = (p * t).sum(dim=[1, 2, 3])
    denom = (p * p).sum(dim=[1, 2, 3]) + (t * t).sum(dim=[1, 2, 3])
    dice = (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()
    return bce_w * bce + dice_w * dice


def build_centerline_tube(skeleton: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Build a soft tube target: 1.0 on the skeleton with Gaussian falloff perpendicular to it."""
    from scipy.ndimage import distance_transform_edt

    s = skeleton > 0
    if not s.any():
        return np.zeros_like(skeleton, dtype=np.float32)
    dist = distance_transform_edt(~s).astype(np.float32)
    tgt = np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float32)
    tgt[s] = 1.0
    return tgt


def build_radius_map(vessel_mask: np.ndarray) -> np.ndarray:
    """Build a per-pixel radius map (vessel-mask distance transform); on the centerline its value is the local radius."""
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(vessel_mask > 0).astype(np.float32)


def build_orientation_map(skeleton: np.ndarray) -> np.ndarray:
    """Build a (2, H, W) double-angle orientation target [cos(2θ), sin(2θ)] on the skeleton.

    Computed from the structure tensor of a blurred skeleton; double-angle encoding avoids the
    180° direction ambiguity.
    """
    from scipy.ndimage import gaussian_filter, sobel

    s = (skeleton > 0).astype(np.float32)
    if not s.any():
        return np.zeros((2, *skeleton.shape), dtype=np.float32)
    s_blur = gaussian_filter(s, sigma=1.0)
    gx = sobel(s_blur, axis=1)
    gy = sobel(s_blur, axis=0)
    Jxx = gaussian_filter(gx * gx, sigma=2.0)
    Jyy = gaussian_filter(gy * gy, sigma=2.0)
    Jxy = gaussian_filter(gx * gy, sigma=2.0)
    # theta is the gradient (principal-eigenvector) angle; the vessel tangent is the minor
    # eigenvector (θ + π/2), which under double-angle encoding negates both components.
    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    return np.stack([-np.cos(2.0 * theta), -np.sin(2.0 * theta)]).astype(np.float32)


def _farthest_point_subset(points: np.ndarray, k: int) -> np.ndarray:
    """Return indices of a farthest-point-sampled size-``k`` subset of ``points`` (N, 2) for even coverage."""
    n = len(points)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    pts = points.astype(np.float64)
    centroid = pts.mean(axis=0)
    start = int(np.argmin(((pts - centroid) ** 2).sum(axis=1)))
    selected = [start]
    min_d = ((pts - pts[start]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, ((pts - pts[nxt]) ** 2).sum(axis=1))
    return np.asarray(selected, dtype=np.int64)


def _snap_to_mask(points: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Snap each point to the nearest True pixel in ``mask`` within ``radius`` (out-of-range points unchanged).

    Closes the soft-target → discrete-vessel gap on thin-vessel datasets.
    """
    if radius <= 0 or not mask.any():
        return points
    H, W = mask.shape
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])  # search nearest offsets first
    out = points.copy()
    for i, (y, x) in enumerate(points):
        if mask[y, x]:
            continue
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx]:
                out[i] = (ny, nx)
                break
    return out


def _detect_optic_disc(image_rgb: np.ndarray, fov_mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Localise the optic disc as the brightest large blob inside the FOV.

    Returns ``(cy, cx, radius_px)`` or None. Used only for soft score down-weighting, so small
    localisation errors are harmless (vessels are never deleted).
    """
    import cv2

    if image_rgb.ndim != 3:
        return None
    g = (image_rgb[..., 1] * 255.0).astype(np.uint8) if image_rgb.dtype != np.uint8 else image_rgb[..., 1]
    fov = (fov_mask > 0).astype(np.uint8)
    if fov.sum() == 0:
        return None
    blurred = cv2.GaussianBlur(g, (31, 31), 0)
    blurred = blurred * fov
    _, _, _, maxLoc = cv2.minMaxLoc(blurred, mask=fov)
    cx, cy = int(maxLoc[0]), int(maxLoc[1])
    # OD radius ≈ 1/7 of the FOV radius.
    fov_r = math.sqrt(fov.sum() / math.pi)
    return cy, cx, max(20, int(0.15 * fov_r))


class SeedDetector(nn.Module):
    """Attention U-Net with multi-task heads (centerline, vessel, radius, orientation, log-variance).

    Input is (B, 3, H, W) RGB in [0, 1], or (B, 4, H, W) with a Frangi channel when
    ``use_frangi_input``. ``forward(return_aux=True)`` also emits 1/2- and 1/4-res deep-supervision maps.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Build the encoder/decoder, multi-task heads, and inference parameters from ``config``."""
        super().__init__()
        cfg = dict(config or {})
        self.base_ch = cfg.get('base_ch', 24)
        self.dropout_p = cfg.get('dropout', 0.10)
        self.use_frangi_input = cfg.get('use_frangi_input', True)
        in_ch = 4 if self.use_frangi_input else 3

        # Inference parameters (also overridable per detect_seeds call).
        self.confidence_threshold = cfg.get('confidence_threshold', 0.35)
        self.vessel_gate_threshold = cfg.get('vessel_gate_threshold', 0.25)
        self.top_k = cfg.get('top_k_seeds', 80)
        # Reference-radius base spacing; the real per-image NMS spacing is FOV-scaled in detect_seeds.
        self.nms_radius = SEED_BASE_SPACING
        self.mc_samples = cfg.get('mc_samples', 0)  # 0 → no MC dropout
        self.suppress_optic_disc = cfg.get('suppress_optic_disc', True)
        self.snap_radius = cfg.get('snap_radius', 2)

        ch = [self.base_ch * (2**i) for i in range(5)]

        self.enc0 = DSConvBlock(in_ch, ch[0])
        self.enc1 = DownBlock(ch[0], ch[1])
        self.enc2 = DownBlock(ch[1], ch[2])
        self.enc3 = DownBlock(ch[2], ch[3])
        self.bot = DownBlock(ch[3], ch[4])

        # Spatial dropout used for MC-dropout sampling at inference.
        self.drop_bot = nn.Dropout2d(self.dropout_p)

        self.up3 = AttnUpBlock(ch[4], ch[3], ch[3])
        self.up2 = AttnUpBlock(ch[3], ch[2], ch[2])
        self.up1 = AttnUpBlock(ch[2], ch[1], ch[1])
        self.up0 = AttnUpBlock(ch[1], ch[0], ch[0])

        def head(in_c: int, out_c: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(in_c, in_c // 2, 1), nn.ReLU(inplace=True), nn.Conv2d(in_c // 2, out_c, 1))

        self.head_centerline = head(ch[0], 1)
        self.head_vessel = head(ch[0], 1)
        self.head_radius = head(ch[0], 1)  # softplus output (pixels)
        self.head_orient = head(ch[0], 2)  # cos/sin of 2θ
        self.head_logvar = head(ch[0], 1)  # aleatoric log-variance

        # Deep-supervision heads (centerline only) at 1/2 and 1/4 resolution.
        self.head_aux_d1 = nn.Conv2d(ch[1], 1, 1)
        self.head_aux_d2 = nn.Conv2d(ch[2], 1, 1)

        self._init_weights()
        # Bias centerline/vessel logits to a ≈0.05 prior so the model doesn't start with a
        # saturated diffuse "carpet" (the v2 failure). See [[seed-detector-v3-diagnosis]].
        prior_p = 0.05
        b_init = -math.log((1.0 - prior_p) / prior_p)
        for h in (self.head_centerline, self.head_vessel):
            nn.init.zeros_(h[-1].weight)
            nn.init.constant_(h[-1].bias, b_init)

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

    @staticmethod
    def _maybe_add_frangi(image: torch.Tensor) -> torch.Tensor:
        """Append a per-image green-channel Frangi vesselness channel when the input has only 3 channels."""
        if image.shape[1] >= 4:
            return image
        out = []
        for b in range(image.shape[0]):
            g = image[b, 1].detach().cpu().numpy()
            v = frangi_vesselness(g)
            out.append(torch.from_numpy(v).to(image.device).unsqueeze(0))
        frangi_ch = torch.stack(out, dim=0)  # (B, 1, H, W)
        return torch.cat([image, frangi_ch], dim=1)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Tuple[torch.Tensor, ...]:
        """Run the attention U-Net and return the multi-task head outputs.

        Prepends a Frangi channel first when configured. Returns
        ``(centerline, vessel, radius, orient, logvar)``, plus the two deep-supervision aux
        maps when ``return_aux`` is True.
        """
        if self.use_frangi_input and x.shape[1] == 3:
            x = self._maybe_add_frangi(x)

        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.drop_bot(self.bot(s3))

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        centerline = torch.sigmoid(self.head_centerline(d0))
        vessel = torch.sigmoid(self.head_vessel(d0))
        radius = F.softplus(self.head_radius(d0))
        orient = torch.tanh(self.head_orient(d0))
        logvar = self.head_logvar(d0)

        if return_aux:
            aux_d1 = torch.sigmoid(self.head_aux_d1(d1))
            aux_d2 = torch.sigmoid(self.head_aux_d2(d2))
            return (centerline, vessel, radius, orient, logvar, aux_d1, aux_d2)
        return (centerline, vessel, radius, orient, logvar)

    @torch.no_grad()
    def _predict_with_mc(self, image: torch.Tensor, mc_samples: int):
        """Return mean ``(centerline, vessel, radius, orient, confidence)`` maps, optionally over MC-dropout samples.

        With ``mc_samples <= 0`` a single deterministic pass is used and confidence is the
        aleatoric exp(−σ²); otherwise dropout stays active (BN frozen) and confidence is the
        epistemic 1 − normalised centerline std.
        """
        if mc_samples <= 0:
            self.eval()
            c, v, r, o, lv = self.forward(image, return_aux=False)
            conf = torch.exp(-lv.clamp(min=-6.0, max=6.0)).clamp(0.0, 1.0)
            return c, v, r, o, conf

        # MC dropout: enable dropout layers but keep BatchNorm in eval mode.
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        cs, vs, rs, os_, conf_var = ([], [], [], [], [])
        for _ in range(mc_samples):
            c, v, r, o, _ = self.forward(image, return_aux=False)
            cs.append(c)
            vs.append(v)
            rs.append(r)
            os_.append(o)
        C = torch.stack(cs).mean(0)
        V = torch.stack(vs).mean(0)
        R = torch.stack(rs).mean(0)
        O = torch.stack(os_).mean(0)
        std = torch.stack(cs).std(0)
        conf = (1.0 - std.clamp(0.0, 0.5) / 0.5).clamp(0.0, 1.0)
        return C, V, R, O, conf

    @torch.no_grad()
    def detect_seeds(
        self,
        image: torch.Tensor,
        obs_half: int = 32,
        return_heatmap: bool = False,
        fov_mask: Optional[torch.Tensor] = None,
        n_seeds: Optional[int] = None,
        snap_mask: Optional[torch.Tensor] = None,
        snap_radius: Optional[int] = None,
        mc_samples: Optional[int] = None,
        debug_stages: Optional[Dict[str, int]] = None,
        traced_mask: Optional[np.ndarray] = None,
        difficulty: Optional[float] = None,
        training_mode: bool = False,
        stochastic_temperature: float = 2.0,
    ) -> Tuple[List[List[Tuple[int, int, float]]], Optional[torch.Tensor], torch.Tensor]:
        """Backward-compatible seed detection: forward → fused ridge map → NMS/coverage seeds.

        ``traced_mask`` / ``difficulty`` / ``training_mode`` / ``stochastic_temperature`` are
        accepted for legacy compatibility and ignored.

        Returns:
            ``(batch_seeds, heatmap or None, vessel_prob)`` — per-image (y, x, score) lists
            sorted by score, the post-processed centerline map if requested, and the raw
            vessel-probability map used downstream as the vessel gate.
        """
        n_seeds = n_seeds or self.top_k
        mc = mc_samples if mc_samples is not None else self.mc_samples
        snap_radius = snap_radius if snap_radius is not None else self.snap_radius

        (centerline, vessel, _radius, _orient, _conf) = self._predict_with_mc(image, mc)

        batch_seeds, fused = self._extract_from_maps(
            image,
            centerline,
            vessel,
            fov_mask=fov_mask,
            obs_half=obs_half,
            n_seeds=n_seeds,
            snap_mask=snap_mask,
            snap_radius=snap_radius,
            debug_stages=debug_stages,
        )
        return (batch_seeds, fused if return_heatmap else None, vessel)

    @torch.no_grad()
    def _extract_from_maps(
        self,
        image: torch.Tensor,
        centerline: torch.Tensor,
        vessel: torch.Tensor,
        fov_mask: Optional[torch.Tensor] = None,
        obs_half: int = 32,
        n_seeds: Optional[int] = None,
        snap_mask: Optional[torch.Tensor] = None,
        snap_radius: Optional[int] = None,
        debug_stages: Optional[Dict[str, int]] = None,
    ) -> Tuple[List[List[Tuple[int, int, float]]], torch.Tensor]:
        """Extract seeds from precomputed centerline + vessel maps; shared by both detect_seeds APIs.

        The aleatoric confidence head is deliberately excluded from the fused score: it is
        uncalibrated post-training and was effectively a no-op multiplier.

        Returns:
            ``(batch_seeds, fused)`` — per-image (y, x, score) lists and the fused ridge map.
        """
        n_seeds = n_seeds or self.top_k
        snap_radius = snap_radius if snap_radius is not None else self.snap_radius
        device = image.device
        h, w = image.shape[-2], image.shape[-1]
        margin = obs_half + 5

        if fov_mask is not None:
            area = fov_mask.flatten(1).sum(dim=1).clamp(min=1.0)
            fov_r = torch.sqrt(area / math.pi)
        else:
            fov_r = torch.full((image.shape[0],), SEED_REF_FOV_RADIUS, device=device)

        fused = (centerline * vessel).clone()

        # FOV-scaled rim erosion plus a hard border-zeroing of the obs margin.
        if fov_mask is not None:
            r_min = float(fov_r.min().item())
            erode_px = int(np.clip(round(FOV_EROSION_FRAC * r_min), FOV_EROSION_MIN, FOV_EROSION_MAX))
            k = 2 * erode_px + 1
            eroded = -F.max_pool2d(-fov_mask, kernel_size=k, stride=1, padding=k // 2)
            fused = fused * eroded
        fused[:, :, :margin, :] = 0.0
        fused[:, :, -margin:, :] = 0.0
        fused[:, :, :, :margin] = 0.0
        fused[:, :, :, -margin:] = 0.0

        batch_seeds: List[List[Tuple[int, int, float]]] = []
        for b in range(fused.shape[0]):
            score = fused[b, 0].cpu().numpy()
            vmap_np = vessel[b, 0].cpu().numpy()
            cmap_np = centerline[b, 0].cpu().numpy()
            r_b = float(fov_r[b].item())

            spacing = int(np.clip(round(SEED_BASE_SPACING * r_b / SEED_REF_FOV_RADIUS), SEED_FLOOR_SPACING, SEED_BASE_SPACING))

            if self.suppress_optic_disc and fov_mask is not None:
                rgb_np = image[b, :3].cpu().numpy().transpose(1, 2, 0)
                fov_np = fov_mask[b, 0].cpu().numpy()
                od = _detect_optic_disc(rgb_np, fov_np)
                if od is not None:
                    cy, cx, rad = od
                    yy, xx = np.ogrid[:h, :w]
                    od_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < rad * rad
                    score = score.copy()
                    score[od_mask] *= 0.4  # soft down-weight, not delete

            score = np.where(vmap_np > self.vessel_gate_threshold, score, 0.0)

            seeds = self._extract_seeds(score, vmap_np, cmap_np, n_seeds, spacing, self.confidence_threshold)

            if snap_mask is not None and len(seeds) > 0:
                m = snap_mask[b, 0].cpu().numpy() > 0
                yx = np.array([(y, x) for y, x, _ in seeds], dtype=np.int64)
                yx = _snap_to_mask(yx, m, snap_radius)
                seeds = [(int(yx[i, 0]), int(yx[i, 1]), float(seeds[i][2])) for i in range(len(seeds))]

            if debug_stages is not None:
                debug_stages['n_seeds_nms'] = len(seeds)
                debug_stages['n_seeds_after_vessel_gate'] = len(seeds)
                debug_stages['n_seeds_after_snap'] = len(seeds)
                debug_stages['n_seeds_after_frangi'] = len(seeds)
                debug_stages['n_seeds_after_fallback'] = len(seeds)
                debug_stages['n_seeds_final'] = len(seeds)

            batch_seeds.append(seeds)

        return batch_seeds, fused

    @torch.no_grad()
    def detect_seeds_rich(
        self,
        image: torch.Tensor,
        fov_mask: Optional[torch.Tensor] = None,
        obs_half: int = 32,
        n_seeds: Optional[int] = None,
        mc_samples: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Detect seeds and return the rich per-image dict the RL state representation consumes.

        Returns:
            One dict per image with ``seed_coords`` (N, 2 as (x, y)), ``seed_scores`` (N),
            ``seed_orientations`` (N radians in [-π/2, π/2)), and ``seed_radius`` (N pixels).
        """
        device = image.device
        n_seeds = n_seeds or self.top_k
        mc = mc_samples if mc_samples is not None else self.mc_samples

        # Single forward pass feeds both the seed extraction and the per-seed attribute lookups.
        c_full, v_full, r_full, o_full, _conf = self._predict_with_mc(image, mc)
        batch_seeds, _ = self._extract_from_maps(image, c_full, v_full, fov_mask=fov_mask, obs_half=obs_half, n_seeds=n_seeds)

        out: List[Dict[str, torch.Tensor]] = []
        for b, seeds in enumerate(batch_seeds):
            if not seeds:
                z = torch.zeros((0,), device=device)
                out.append({'seed_coords': torch.zeros((0, 2), device=device), 'seed_scores': z, 'seed_orientations': z, 'seed_radius': z})
                continue
            ys = torch.tensor([s[0] for s in seeds], device=device, dtype=torch.long)
            xs = torch.tensor([s[1] for s in seeds], device=device, dtype=torch.long)
            sc = torch.tensor([s[2] for s in seeds], device=device, dtype=torch.float32)
            cos2t = o_full[b, 0, ys, xs]
            sin2t = o_full[b, 1, ys, xs]
            theta = 0.5 * torch.atan2(sin2t, cos2t)  # decode double-angle → [-π/2, π/2)
            rad = r_full[b, 0, ys, xs].clamp(min=0.5)
            coords = torch.stack([xs.float(), ys.float()], dim=1)
            out.append({'seed_coords': coords, 'seed_scores': sc, 'seed_orientations': theta, 'seed_radius': rad})
        return out

    def _extract_seeds(
        self, score: np.ndarray, vmap: np.ndarray, cmap: np.ndarray, n_seeds: int, spacing: int, threshold: float
    ) -> List[Tuple[int, int, float]]:
        """Pick ridge-crest peaks on the fused score (NMS at ``spacing``) and FPS-subsample to ``n_seeds``.

        Returns (y, x, vessel·centerline score) tuples sorted by score descending.
        """
        from skimage.feature import peak_local_max

        if not (score > threshold).any():
            return []
        coords = peak_local_max(score, min_distance=spacing, threshold_abs=threshold, exclude_border=False)
        if len(coords) == 0:
            return []
        if len(coords) > n_seeds:
            keep = _farthest_point_subset(coords, n_seeds)
            coords = coords[keep]
        seeds = [(int(y), int(x), float(vmap[y, x] * cmap[y, x])) for y, x in coords]
        seeds.sort(key=lambda s: s[2], reverse=True)
        return seeds


def seed_detector_loss(
    preds: Tuple[torch.Tensor, ...],
    targets: Dict[str, torch.Tensor],
    fov_mask: Optional[torch.Tensor] = None,
    w_centerline: float = 1.0,
    w_vessel: float = 0.8,
    w_cldice: float = 0.4,
    w_radius: float = 0.2,
    w_orient: float = 0.2,
    w_uncert: float = 0.05,
    w_aux: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted multi-task training loss for the seed detector.

    Args:
        preds: the ``forward(return_aux=True)`` tuple
            ``(centerline, vessel, radius, orient, logvar, aux_d1, aux_d2)``.
        targets: dict with ``centerline_tube``, ``vessel_mask``, ``radius_map`` (used only on
            the centerline), and ``orient_map`` (cos/sin of 2θ on the centerline).

    Returns:
        ``(total_loss, breakdown)`` where breakdown holds per-component scalar values.
    """
    (centerline, vessel, radius, orient, logvar, aux_d1, aux_d2) = preds
    cl_t = targets['centerline_tube']
    ve_t = targets['vessel_mask']
    ra_t = targets['radius_map']
    or_t = targets['orient_map']

    # Centerline-tube and vessel-mask segmentation (Dice + BCE).
    loss_cl = soft_dice_bce(centerline, cl_t, mask=fov_mask)
    loss_ve = soft_dice_bce(vessel, ve_t, mask=fov_mask)

    # clDice topology loss split across both heads.
    skel_hard = (cl_t >= 0.999).float()
    loss_topo = 0.5 * cl_dice_loss(vessel, ve_t) + 0.5 * cl_dice_loss(centerline, skel_hard)

    # Radius and orientation regressions, supervised only on centerline pixels.
    cl_bin = (cl_t > 0.5).float()
    n_cl = cl_bin.sum().clamp(min=1.0)
    loss_r = (F.smooth_l1_loss(radius, ra_t, reduction='none') * cl_bin).sum() / n_cl
    loss_o = (F.smooth_l1_loss(orient, or_t, reduction='none') * cl_bin).sum() / (2.0 * n_cl)

    # Aleatoric uncertainty: heteroscedastic Laplace NLL on the centerline.
    logvar_c = logvar.clamp(min=-6.0, max=6.0)
    sigma = torch.exp(0.5 * logvar_c)
    nll = (centerline - cl_t).abs() / sigma + 0.5 * logvar_c
    n_cl_u = cl_bin.sum().clamp(min=1.0)
    loss_u = (nll * cl_bin).sum() / n_cl_u

    # Deep supervision: centerline target downsampled to each aux resolution.
    def _dsample(t, ref):
        return F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    loss_aux = (soft_dice_bce(aux_d1, _dsample(cl_t, aux_d1)) + soft_dice_bce(aux_d2, _dsample(cl_t, aux_d2))) * 0.5

    total = (
        w_centerline * loss_cl
        + w_vessel * loss_ve
        + w_cldice * loss_topo
        + w_radius * loss_r
        + w_orient * loss_o
        + w_uncert * loss_u
        + w_aux * loss_aux
    )

    return total, {
        'loss/centerline': float(loss_cl.item()),
        'loss/vessel': float(loss_ve.item()),
        'loss/cldice': float(loss_topo.item()),
        'loss/radius': float(loss_r.item()),
        'loss/orient': float(loss_o.item()),
        'loss/uncert': float(loss_u.item()),
        'loss/aux': float(loss_aux.item()),
        'loss/total': float(total.item()),
        # Legacy alias keys kept for older log consumers.
        'loss/focal': float(loss_cl.item()),
        'loss/vessel_bce': float(loss_ve.item()),
        'loss/fp_penalty': float(loss_topo.item()),
    }
