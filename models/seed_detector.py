"""Seed detector v4 — state-of-the-art research-grade module.

Designed from scratch for the RL retinal vessel tracing pipeline. Deliberately
ignores the previous v1/v2/v3 implementations and is a self-contained,
multi-task seed detection model that:

  1. Predicts a soft vessel mask, a soft centerline (ridge) probability, an
     estimated vessel radius and a (cos, sin) orientation field via a single
     Attention U-Net encoder–decoder.
  2. Optionally fuses a classical multi-scale Frangi vesselness prior as an
     extra input channel.
  3. Estimates aleatoric confidence via Monte-Carlo dropout at inference.
  4. Extracts seeds as ridge peaks on `vessel * centerline * confidence`
     (peak_local_max → FOV-scaled NMS → farthest-point coverage sub-sampling
     → optic-disc suppression).

Backward compatibility:
  * `detect_seeds(image, …) → (batch_seeds, heatmap, vessel_prob)` keeps the
    legacy tuple signature used by `scripts/run_rl_tracing.py` and
    `environment/seeding_utils.merge_seeds`. No call-site changes required.
  * `detect_seeds_rich(...)` returns the per-spec dict
    {seed_coords, seed_scores, seed_orientations, seed_radius} for direct RL
    consumption.

Training is in `training/seed_detector_trainer.py`. Entry script is
`scripts/train_seed_detector.py`.
"""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_blocks import (
    DSConvBlock,
    DownBlock,
)


# ---------------------------------------------------------------------------
# FOV-scale-invariant constants (see [[fov-scale-invariance]]).
# ---------------------------------------------------------------------------
SEED_REF_FOV_RADIUS = 250.0  # pixels at which the base spacing applies
SEED_BASE_SPACING = 22  # base min-distance between two seeds at the
# reference FOV radius. Scaled down for small
# FOVs (floor 8 px) so peripheral vessels in
# tightly-cropped retinas are not starved.
SEED_FLOOR_SPACING = 8
FOV_EROSION_FRAC = 0.04  # rim-erosion ≈4 % of FOV radius
FOV_EROSION_MIN = 4
FOV_EROSION_MAX = 17


# ===========================================================================
# Attention gate on skip connections (Oktay et al. 2018)
# ===========================================================================


class AttentionGate(nn.Module):
    """Additive attention gating for U-Net skip connections.

    Suppresses irrelevant skip activations (lesions, optic disc, background)
    by learning a gating mask from the coarser decoder feature `g`.

    α = σ(ψ(ReLU(W_g(g) + W_x(x))))     x' = α ⊙ x
    """

    def __init__(
        self,
        in_ch: int,
        gating_ch: int,
        inter_ch: Optional[int] = None,
    ):
        super().__init__()
        inter = inter_ch or max(in_ch // 2, 8)
        self.W_x = nn.Conv2d(in_ch, inter, 1, bias=False)
        self.W_g = nn.Conv2d(gating_ch, inter, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(
                g,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        a = self.psi(self.W_x(x) + self.W_g(g))
        return x * a


class AttnUpBlock(nn.Module):
    """Bilinear up-sample → attention-gated skip concat → DSConvBlock."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
    ):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False,
        )
        self.gate = AttentionGate(skip_ch, in_ch)
        self.conv = DSConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)
        if x_up.shape[-2:] != skip.shape[-2:]:
            x_up = F.interpolate(
                x_up,
                size=skip.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        skip_a = self.gate(skip, x_up)
        return self.conv(torch.cat([x_up, skip_a], dim=1))


# ===========================================================================
# Classical Frangi vesselness prior (vectorised, batchable on CPU/numpy)
# ===========================================================================


def frangi_vesselness(
    green_or_gray: np.ndarray,
    scales=(1.0, 1.6, 2.5, 4.0),
) -> np.ndarray:
    """Multi-scale Frangi vesselness on a single-channel image in [0, 1].

    Returns float32 (H, W) in [0, 1]. Used both as an optional input channel
    and as an inference-time fallback gate. Inverts intensities so dark
    vessels light up.
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


# ===========================================================================
# Topology-aware soft-skeleton loss (clDice — Shit et al. 2021)
# ===========================================================================


def _soft_skeletonize(x: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Differentiable skeletonisation via repeated soft erosion/dilation."""
    pool = lambda t: F.max_pool2d(t, 3, stride=1, padding=1)
    neg_pool = lambda t: -F.max_pool2d(-t, 3, stride=1, padding=1)
    skel = F.relu(x - pool(neg_pool(x)))
    for _ in range(iters - 1):
        x = neg_pool(x)
        skel = skel + F.relu((1.0 - skel) * (x - pool(neg_pool(x))))
    return skel


def cl_dice_loss(
    pred_vessel: torch.Tensor,
    gt_vessel: torch.Tensor,
    iters: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """clDice topology-preserving loss on a soft vessel prediction."""
    s_p = _soft_skeletonize(pred_vessel, iters)
    s_g = _soft_skeletonize(gt_vessel, iters)
    t_prec = (s_p * gt_vessel).sum() / (s_p.sum() + eps)
    t_rec = (s_g * pred_vessel).sum() / (s_g.sum() + eps)
    cldice = 2.0 * t_prec * t_rec / (t_prec + t_rec + eps)
    return 1.0 - cldice


def soft_dice_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    bce_w: float = 1.0,
    dice_w: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
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


# ===========================================================================
# Target builders (used by the trainer; live here so production + training
# share a single definition).
# ===========================================================================


def build_centerline_tube(skeleton: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Soft tube target: 1.0 on the skeleton, Gaussian falloff perpendicular."""
    from scipy.ndimage import (
        distance_transform_edt,
    )

    s = skeleton > 0
    if not s.any():
        return np.zeros_like(skeleton, dtype=np.float32)
    dist = distance_transform_edt(~s).astype(np.float32)
    tgt = np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float32)
    tgt[s] = 1.0
    return tgt


def build_radius_map(
    vessel_mask: np.ndarray,
) -> np.ndarray:
    """Per-pixel local vessel radius (px). Uses the distance transform of the
    vessel mask — the value at the centerline = local radius, decaying to 0
    at the boundary. The trainer supervises this only on the centerline."""
    from scipy.ndimage import (
        distance_transform_edt,
    )

    return distance_transform_edt(vessel_mask > 0).astype(np.float32)


def build_orientation_map(
    skeleton: np.ndarray,
) -> np.ndarray:
    """Per-pixel local orientation (radians), defined on the skeleton.

    Computed via the structure tensor of a slightly-blurred skeleton.
    Returns (2, H, W) float32: channel 0 = cos(2θ), channel 1 = sin(2θ).
    Double-angle encoding avoids the 180° ambiguity of vessel directions.
    """
    from scipy.ndimage import (
        gaussian_filter,
        sobel,
    )

    s = (skeleton > 0).astype(np.float32)
    if not s.any():
        return np.zeros((2, *skeleton.shape), dtype=np.float32)
    s_blur = gaussian_filter(s, sigma=1.0)
    gx = sobel(s_blur, axis=1)
    gy = sobel(s_blur, axis=0)
    Jxx = gaussian_filter(gx * gx, sigma=2.0)
    Jyy = gaussian_filter(gy * gy, sigma=2.0)
    Jxy = gaussian_filter(gx * gy, sigma=2.0)
    # 0.5·atan2(2Jxy, Jxx-Jyy) is the *principal* eigenvector angle
    # (i.e. the gradient direction). The vessel TANGENT is orthogonal to
    # that — the minor eigenvector — so we add π/2. Under double-angle
    # encoding (cos(2θ), sin(2θ)) that flips the sign of both components.
    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    return np.stack(
        [
            -np.cos(2.0 * theta),
            -np.sin(2.0 * theta),
        ]
    ).astype(np.float32)


# ===========================================================================
# Seed extraction helpers
# ===========================================================================


def _farthest_point_subset(points: np.ndarray, k: int) -> np.ndarray:
    """Return indices of an FPS-selected size-k subset of `points` (N, 2)."""
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
        min_d = np.minimum(
            min_d,
            ((pts - pts[nxt]) ** 2).sum(axis=1),
        )
    return np.asarray(selected, dtype=np.int64)


def _snap_to_mask(
    points: np.ndarray,
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Snap each point to the nearest True pixel in `mask` within `radius`.

    Out-of-range points are left in place. Used to close the soft-target →
    discrete-vessel gap on thin-vessel datasets.
    """
    if radius <= 0 or not mask.any():
        return points
    H, W = mask.shape
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])
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
    """Crude optic-disc localiser: brightest large blob inside the FOV.

    Returns (cy, cx, radius_px) or None. Used only as a soft suppression
    mask — small detection errors do not hurt because we only down-weight
    the score, never delete vessels.
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
    # FOV-scaled radius (typical OD ≈ 1/7 of FOV radius)
    fov_r = math.sqrt(fov.sum() / math.pi)
    return cy, cx, max(20, int(0.15 * fov_r))


# ===========================================================================
# Multi-task Attention U-Net seed detector
# ===========================================================================


class SeedDetector(nn.Module):
    """Attention U-Net seed detector with multi-task heads.

    Input
    -----
    image : (B, 3, H, W) float32 in [0, 1] RGB.

    Optional auxiliary input
    ------------------------
    If ``use_frangi_input=True`` (config flag), the model expects a 4-channel
    input (B, 4, H, W) where channel 3 is a precomputed Frangi vesselness
    map. The trainer + ``detect_seeds`` build this on the fly.

    Outputs (per pixel, full-resolution)
    ------------------------------------
    centerline_prob : (B, 1, H, W) in [0, 1]   — sigmoid
    vessel_prob     : (B, 1, H, W) in [0, 1]   — sigmoid
    radius_pred     : (B, 1, H, W) — softplus, in pixels
    orient_cos      : (B, 1, H, W) in [-1, 1] — tanh, cos(2θ)
    orient_sin      : (B, 1, H, W) in [-1, 1] — tanh, sin(2θ)
    log_var         : (B, 1, H, W) — aleatoric log-variance for centerline

    Deep supervision: a 1/2- and 1/4-resolution centerline head is also
    produced during training (returned in `forward(..., return_aux=True)`).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        cfg = dict(config or {})
        self.base_ch = cfg.get('base_ch', 24)
        self.dropout_p = cfg.get('dropout', 0.10)
        self.use_frangi_input = cfg.get('use_frangi_input', True)
        in_ch = 4 if self.use_frangi_input else 3

        # ---- Inference parameters (also configurable per call) ----
        self.confidence_threshold = cfg.get('confidence_threshold', 0.35)
        self.vessel_gate_threshold = cfg.get('vessel_gate_threshold', 0.25)
        self.top_k = cfg.get('top_k_seeds', 80)
        # The real per-image NMS spacing is computed FOV-radius-scaled inside
        # detect_seeds; this attribute is the reference-radius base value,
        # kept as a sensible scalar for any caller reading `seed_model.nms_radius`.
        self.nms_radius = SEED_BASE_SPACING
        self.mc_samples = cfg.get('mc_samples', 0)  # 0 → no MC dropout
        self.suppress_optic_disc = cfg.get('suppress_optic_disc', True)
        self.snap_radius = cfg.get('snap_radius', 2)

        ch = [self.base_ch * (2**i) for i in range(5)]

        # ---- Encoder ----
        self.enc0 = DSConvBlock(in_ch, ch[0])
        self.enc1 = DownBlock(ch[0], ch[1])
        self.enc2 = DownBlock(ch[1], ch[2])
        self.enc3 = DownBlock(ch[2], ch[3])
        self.bot = DownBlock(ch[3], ch[4])

        # Spatial dropout for MC sampling
        self.drop_bot = nn.Dropout2d(self.dropout_p)

        # ---- Decoder with attention-gated skips ----
        self.up3 = AttnUpBlock(ch[4], ch[3], ch[3])
        self.up2 = AttnUpBlock(ch[3], ch[2], ch[2])
        self.up1 = AttnUpBlock(ch[2], ch[1], ch[1])
        self.up0 = AttnUpBlock(ch[1], ch[0], ch[0])

        # ---- Heads ----
        def head(in_c: int, out_c: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c // 2, out_c, 1),
            )

        self.head_centerline = head(ch[0], 1)
        self.head_vessel = head(ch[0], 1)
        self.head_radius = head(ch[0], 1)  # softplus on output
        self.head_orient = head(ch[0], 2)  # cos/sin of 2θ
        self.head_logvar = head(ch[0], 1)  # aleatoric

        # Deep-supervision auxiliary heads (centerline only — cheapest signal)
        self.head_aux_d1 = nn.Conv2d(ch[1], 1, 1)  # 1/2 res
        self.head_aux_d2 = nn.Conv2d(ch[2], 1, 1)  # 1/4 res

        self._init_weights()
        # Bias-init the centerline + vessel logits to a prior of ≈0.05 so the
        # model does NOT start with a saturated diffuse "carpet" — the v2
        # failure mode. See [[seed-detector-v3-diagnosis]].
        prior_p = 0.05
        b_init = -math.log((1.0 - prior_p) / prior_p)
        for h in (
            self.head_centerline,
            self.head_vessel,
        ):
            nn.init.zeros_(h[-1].weight)
            nn.init.constant_(h[-1].bias, b_init)

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    @staticmethod
    def _maybe_add_frangi(
        image: torch.Tensor,
    ) -> torch.Tensor:
        """If input has 3 channels, append a per-image Frangi vesselness
        channel computed on the green channel. Done on CPU per-image — cheap
        compared to the forward pass."""
        if image.shape[1] >= 4:
            return image
        out = []
        for b in range(image.shape[0]):
            g = image[b, 1].detach().cpu().numpy()
            v = frangi_vesselness(g)
            out.append(torch.from_numpy(v).to(image.device).unsqueeze(0))
        frangi_ch = torch.stack(out, dim=0)  # (B, 1, H, W)
        return torch.cat([image, frangi_ch], dim=1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
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
            return (
                centerline,
                vessel,
                radius,
                orient,
                logvar,
                aux_d1,
                aux_d2,
            )
        return (
            centerline,
            vessel,
            radius,
            orient,
            logvar,
        )

    # ==================================================================
    # Inference API
    # ==================================================================

    @torch.no_grad()
    def _predict_with_mc(self, image: torch.Tensor, mc_samples: int):
        """Mean prediction over MC-dropout samples (or a single deterministic
        pass if mc_samples ≤ 0). Returns (centerline, vessel, radius, orient,
        confidence) — all (B, C, H, W) tensors on the input's device."""
        if mc_samples <= 0:
            self.eval()
            c, v, r, o, lv = self.forward(image, return_aux=False)
            # Aleatoric-only confidence: exp(-σ²) clamped to [0,1]
            conf = torch.exp(-lv.clamp(min=-6.0, max=6.0)).clamp(0.0, 1.0)
            return c, v, r, o, conf

        # MC dropout: keep dropout layers in train mode but BN frozen
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

        cs, vs, rs, os_, conf_var = (
            [],
            [],
            [],
            [],
            [],
        )
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
        # Epistemic confidence: 1 - normalised std on the centerline
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
        # legacy kwargs accepted for backward-compat; ignored:
        traced_mask: Optional[np.ndarray] = None,
        difficulty: Optional[float] = None,
        training_mode: bool = False,
        stochastic_temperature: float = 2.0,
    ) -> Tuple[
        List[List[Tuple[int, int, float]]],
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Backward-compatible inference: returns the legacy tuple.

        Pipeline:
            forward (+ MC dropout) → fused ridge map (vessel·centerline·conf)
                                  → FOV-radius-scaled erosion + border zeroing
                                  → optic-disc suppression
                                  → peak_local_max (FOV-scaled NMS)
                                  → farthest-point coverage subset
                                  → optional snap-to-vessel-mask

        Returns
        -------
        batch_seeds : list of (y, x, score) per image, sorted by score desc.
        heatmap     : post-processed centerline map if return_heatmap.
        vessel_prob : raw vessel-probability map (pre-suppression). Used as
                      vessel-gating signal by run_rl_tracing.
        """
        n_seeds = n_seeds or self.top_k
        mc = mc_samples if mc_samples is not None else self.mc_samples
        snap_radius = snap_radius if snap_radius is not None else self.snap_radius

        (
            centerline,
            vessel,
            _radius,
            _orient,
            _conf,
        ) = self._predict_with_mc(image, mc)

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
        return (
            batch_seeds,
            fused if return_heatmap else None,
            vessel,
        )

    # ------------------------------------------------------------------
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
    ) -> Tuple[
        List[List[Tuple[int, int, float]]],
        torch.Tensor,
    ]:
        """Seed extraction given precomputed centerline + vessel maps.

        Shared by detect_seeds and detect_seeds_rich so a single forward pass
        feeds both. The aleatoric `confidence` head is intentionally NOT used
        in the fused score: it is uncalibrated post-training (logvar drifts to
        near-zero where the L1 residual is small) and was effectively a no-op
        multiplier — see commit log. Kept as a debug-only output via
        _predict_with_mc.
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
            fov_r = torch.full(
                (image.shape[0],),
                SEED_REF_FOV_RADIUS,
                device=device,
            )

        fused = (centerline * vessel).clone()

        if fov_mask is not None:
            r_min = float(fov_r.min().item())
            erode_px = int(
                np.clip(
                    round(FOV_EROSION_FRAC * r_min),
                    FOV_EROSION_MIN,
                    FOV_EROSION_MAX,
                )
            )
            k = 2 * erode_px + 1
            eroded = -F.max_pool2d(
                -fov_mask,
                kernel_size=k,
                stride=1,
                padding=k // 2,
            )
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

            spacing = int(
                np.clip(
                    round(SEED_BASE_SPACING * r_b / SEED_REF_FOV_RADIUS),
                    SEED_FLOOR_SPACING,
                    SEED_BASE_SPACING,
                )
            )

            if self.suppress_optic_disc and fov_mask is not None:
                rgb_np = image[b, :3].cpu().numpy().transpose(1, 2, 0)
                fov_np = fov_mask[b, 0].cpu().numpy()
                od = _detect_optic_disc(rgb_np, fov_np)
                if od is not None:
                    cy, cx, rad = od
                    yy, xx = np.ogrid[:h, :w]
                    od_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < rad * rad
                    score = score.copy()
                    score[od_mask] *= 0.4

            score = np.where(
                vmap_np > self.vessel_gate_threshold,
                score,
                0.0,
            )

            seeds = self._extract_seeds(
                score,
                vmap_np,
                cmap_np,
                n_seeds,
                spacing,
                self.confidence_threshold,
            )

            if snap_mask is not None and len(seeds) > 0:
                m = snap_mask[b, 0].cpu().numpy() > 0
                yx = np.array(
                    [(y, x) for y, x, _ in seeds],
                    dtype=np.int64,
                )
                yx = _snap_to_mask(yx, m, snap_radius)
                seeds = [
                    (
                        int(yx[i, 0]),
                        int(yx[i, 1]),
                        float(seeds[i][2]),
                    )
                    for i in range(len(seeds))
                ]

            if debug_stages is not None:
                debug_stages['n_seeds_nms'] = len(seeds)
                debug_stages['n_seeds_after_vessel_gate'] = len(seeds)
                debug_stages['n_seeds_after_snap'] = len(seeds)
                debug_stages['n_seeds_after_frangi'] = len(seeds)
                debug_stages['n_seeds_after_fallback'] = len(seeds)
                debug_stages['n_seeds_final'] = len(seeds)

            batch_seeds.append(seeds)

        return batch_seeds, fused

    # ------------------------------------------------------------------
    @torch.no_grad()
    def detect_seeds_rich(
        self,
        image: torch.Tensor,
        fov_mask: Optional[torch.Tensor] = None,
        obs_half: int = 32,
        n_seeds: Optional[int] = None,
        mc_samples: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Return the per-spec dict the RL state representation consumes.

        Per image:
            {
              "seed_coords":       Tensor[N, 2]  (x, y) pixel coords (note the
                                                   (x, y) order, NOT (y, x)),
              "seed_scores":       Tensor[N],
              "seed_orientations": Tensor[N]   radians in [-π/2, π/2),
              "seed_radius":       Tensor[N]   pixels,
            }
        """
        device = image.device
        n_seeds = n_seeds or self.top_k
        mc = mc_samples if mc_samples is not None else self.mc_samples

        # SINGLE forward pass — extract seeds from the maps we just computed.
        c_full, v_full, r_full, o_full, _conf = self._predict_with_mc(image, mc)
        batch_seeds, _ = self._extract_from_maps(
            image,
            c_full,
            v_full,
            fov_mask=fov_mask,
            obs_half=obs_half,
            n_seeds=n_seeds,
        )

        out: List[Dict[str, torch.Tensor]] = []
        for b, seeds in enumerate(batch_seeds):
            if not seeds:
                z = torch.zeros((0,), device=device)
                out.append(
                    {
                        'seed_coords': torch.zeros((0, 2), device=device),
                        'seed_scores': z,
                        'seed_orientations': z,
                        'seed_radius': z,
                    }
                )
                continue
            ys = torch.tensor(
                [s[0] for s in seeds],
                device=device,
                dtype=torch.long,
            )
            xs = torch.tensor(
                [s[1] for s in seeds],
                device=device,
                dtype=torch.long,
            )
            sc = torch.tensor(
                [s[2] for s in seeds],
                device=device,
                dtype=torch.float32,
            )
            cos2t = o_full[b, 0, ys, xs]
            sin2t = o_full[b, 1, ys, xs]
            theta = 0.5 * torch.atan2(sin2t, cos2t)  # in [-π/2, π/2)
            rad = r_full[b, 0, ys, xs].clamp(min=0.5)
            coords = torch.stack([xs.float(), ys.float()], dim=1)
            out.append(
                {
                    'seed_coords': coords,
                    'seed_scores': sc,
                    'seed_orientations': theta,
                    'seed_radius': rad,
                }
            )
        return out

    # ------------------------------------------------------------------
    def _extract_seeds(
        self,
        score: np.ndarray,
        vmap: np.ndarray,
        cmap: np.ndarray,
        n_seeds: int,
        spacing: int,
        threshold: float,
    ) -> List[Tuple[int, int, float]]:
        """Ridge-peak extraction on the fused score map → FPS coverage subset.

        Peaks land on ridge crests, not on the gap between adjacent vessels —
        see [[seed-detector-v3-diagnosis]] for why this matters. The fused
        score is `vessel * centerline * confidence` so a peak both lies on a
        predicted centerline and inside a high-vessel region.
        """
        from skimage.feature import peak_local_max

        if not (score > threshold).any():
            return []
        coords = peak_local_max(
            score,
            min_distance=spacing,
            threshold_abs=threshold,
            exclude_border=False,
        )
        if len(coords) == 0:
            return []
        if len(coords) > n_seeds:
            keep = _farthest_point_subset(coords, n_seeds)
            coords = coords[keep]
        seeds = [
            (
                int(y),
                int(x),
                float(vmap[y, x] * cmap[y, x]),
            )
            for y, x in coords
        ]
        seeds.sort(key=lambda s: s[2], reverse=True)
        return seeds


# ===========================================================================
# Loss bundle (used by the trainer)
# ===========================================================================


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
    """Multi-task loss.

    `preds` is the tuple returned by `forward(..., return_aux=True)`:
        (centerline, vessel, radius, orient, logvar, aux_d1, aux_d2)
    `targets` keys:
        centerline_tube : (B, 1, H, W) float in [0, 1]
        vessel_mask     : (B, 1, H, W) float in {0, 1}
        radius_map      : (B, 1, H, W) float pixels (used only where
                          centerline_tube > 0.5)
        orient_map      : (B, 2, H, W) cos/sin of 2θ on the centerline
    """
    (
        centerline,
        vessel,
        radius,
        orient,
        logvar,
        aux_d1,
        aux_d2,
    ) = preds
    cl_t = targets['centerline_tube']
    ve_t = targets['vessel_mask']
    ra_t = targets['radius_map']
    or_t = targets['orient_map']

    # 1. Centerline tube segmentation (Dice + BCE)
    loss_cl = soft_dice_bce(centerline, cl_t, mask=fov_mask)

    # 2. Vessel mask segmentation
    loss_ve = soft_dice_bce(vessel, ve_t, mask=fov_mask)

    # 3. clDice topology loss on BOTH heads (50/50). Why both:
    #   - vessel head: clDice is designed for thick tubular masks and
    #     soft-skeletonises internally; this preserves tree topology in
    #     vessel_prob, which downstream consumers depend on (reward
    #     off-track penalty, obs vessel channel, seed-extraction
    #     vessel_gate_threshold). Dropping this pressure made vessel_prob
    #     locally noisier and degraded RL trace quality.
    #   - centerline head: applies clDice against the hard skeleton
    #     (cl_t == 1.0 exactly on skel pixels). The soft-skel inside
    #     cl_dice_loss is near-identity on the already-thin prediction, so
    #     this acts as a connectivity-preserving overlap loss on the ridge.
    # Net cost: a second _soft_skeletonize pass per step — cheap relative to
    # the U-Net forward.
    skel_hard = (cl_t >= 0.999).float()
    loss_topo = 0.5 * cl_dice_loss(vessel, ve_t) + 0.5 * cl_dice_loss(centerline, skel_hard)

    # 4. Radius (smooth-L1 only on centerline pixels)
    cl_bin = (cl_t > 0.5).float()
    n_cl = cl_bin.sum().clamp(min=1.0)
    loss_r = (F.smooth_l1_loss(radius, ra_t, reduction='none') * cl_bin).sum() / n_cl

    # 5. Orientation (cos/sin regression on the centerline)
    loss_o = (F.smooth_l1_loss(orient, or_t, reduction='none') * cl_bin).sum() / (2.0 * n_cl)

    # 6. Aleatoric uncertainty: heteroscedastic Laplace NLL on centerline.
    # Two corrections vs. the previous formulation:
    #   (a) `logvar` is clamped to the SAME range used at inference
    #       (_predict_with_mc, [-6, 6]). Without the clamp the bias term
    #       0.5·logvar is unbounded below, and on background pixels where the
    #       residual is ≈0 the optimiser drives logvar → -∞ to harvest free
    #       negative loss — symptom: train/val loss went negative (~-30).
    #   (b) The NLL is restricted to centerline pixels (cl_t > 0.5). On
    #       background pixels there is no residual to balance; including them
    #       gave the bias term an exploit surface, and the head has no useful
    #       signal off-vessel anyway.
    logvar_c = logvar.clamp(min=-6.0, max=6.0)
    sigma = torch.exp(0.5 * logvar_c)
    nll = (centerline - cl_t).abs() / sigma + 0.5 * logvar_c
    n_cl_u = cl_bin.sum().clamp(min=1.0)
    loss_u = (nll * cl_bin).sum() / n_cl_u

    # 7. Deep supervision (downsample target to match aux resolutions)
    def _dsample(t, ref):
        return F.interpolate(
            t,
            size=ref.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

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
        # legacy per-component loss keys (harmless extra entries)
        'loss/focal': float(loss_cl.item()),
        'loss/vessel_bce': float(loss_ve.item()),
        'loss/fp_penalty': float(loss_topo.item()),
    }
