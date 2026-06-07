"""Observation construction for the vessel-tracing environment."""

from typing import Any, Dict, Optional

import numpy as np


class ObservationBuilder:
    """Builds the (C, obs_size, obs_size) observation tensor for the RL agent.

    Layout: 10 base channels (RGB, visited, DT, DT-grad, centerline, tangent) followed by
    optional channels in ``_OPTIONAL_CHANNELS`` order — spatial channels first, broadcast
    scalar channels (prev_action, topology) last.
    """

    # Per-feature channel-count table — single source of truth, in emission order.
    # n_channels / n_scalar_channels and the policy network all read this so the obs
    # builder, gym space, and CNN input stay in lockstep.
    _BASE_CHANNELS = 10  # RGB(3) + visited(1) + DT(1) + DT-grad(2) + centerline(1) + tangent(2)
    # Invariant: spatial channels precede scalar (broadcast-constant) channels, which the
    # policy net slices off (obs[:, -n_scalar:, 0, 0]) for direct MLP-head injection.
    _OPTIONAL_CHANNELS = (
        ('use_curvature', True, 1),
        ('use_junction', True, 2),  # junction (binary) + endpoint (binary)
        ('use_vesselness', False, 1),
        ('use_unet_prior', False, 1),
        ('use_unet_uncertainty', False, 1),  # binary entropy of unet_prior
        ('use_global_visited', False, 1),
        ('use_prior_coverage', False, 1),
        ('use_covered_centerline', False, 1),  # crop of covered_centerline; pairs with F3 shaping
        ('use_multiscale', True, 5),  # wide RGB(3) + wide visited(1) + wide UNet prior(1)
        ('use_prev_action', False, 2),
        ('use_topology_memory', True, 2),
    )
    # Flags producing broadcast-constant (scalar) channels, kept last in the layout.
    _SCALAR_FLAGS = ('use_prev_action', 'use_topology_memory')

    @classmethod
    def n_channels(cls, config: Dict[str, Any]) -> int:
        """Return the observation channel count for ``config`` (base plus enabled optional).

        Single source consulted by ``__init__`` (buffer sizing), the gym space, and the
        policy network's CNN input-plane count.
        """
        env = config.get('environment', {})
        n = cls._BASE_CHANNELS
        for flag, default, count in cls._OPTIONAL_CHANNELS:
            if env.get(flag, default):
                n += count
        return n

    @classmethod
    def n_scalar_channels(cls, config: Dict[str, Any]) -> int:
        """Return the count of trailing broadcast-constant (scalar) channels.

        These (prev_action, topology_memory) are sliced out before the CNN and concatenated
        back at the MLP head — see ``ActorCriticNetwork.forward``.
        """
        env = config.get('environment', {})
        n = 0
        for flag, default, count in cls._OPTIONAL_CHANNELS:
            if flag in cls._SCALAR_FLAGS and env.get(flag, default):
                n += count
        return n

    @classmethod
    def n_spatial_channels(cls, config: Dict[str, Any]) -> int:
        """Return the channel count consumed by the CNN encoder (total minus scalars)."""
        return cls.n_channels(config) - cls.n_scalar_channels(config)

    @classmethod
    def junction_channel_idx(cls, config: Dict[str, Any]) -> Optional[int]:
        """Return the channel index of the junction map, or None if disabled.

        The PPO trainer reads this to pull junction supervision targets straight from the
        stored observation batch.
        """
        env = config.get('environment', {})
        if not env.get('use_junction', True):
            return None
        # Walk the table so any channel inserted before 'use_junction' is accounted for.
        n = cls._BASE_CHANNELS
        for flag, default, count in cls._OPTIONAL_CHANNELS:
            if flag == 'use_junction':
                return n
            if env.get(flag, default):
                n += count
        return None

    def __init__(self, config: Dict[str, Any]):
        """Read channel flags from config and pre-allocate the reusable obs buffer."""
        env_config = config.get('environment', {})
        self.obs_size = env_config.get('observation_size', 65)
        self.half_size = self.obs_size // 2
        self.use_vesselness = env_config.get('use_vesselness', False)
        self.use_curvature = env_config.get('use_curvature', True)
        self.use_junction = env_config.get('use_junction', True)
        self.use_unet_prior = env_config.get('use_unet_prior', False)
        # Per-pixel binary entropy of the UNet prob, peaking where the predictor is
        # uncertain (p≈0.5) — an exploration-under-uncertainty signal.
        self.use_unet_uncertainty = env_config.get('use_unet_uncertainty', False)
        self.use_global_visited = env_config.get('use_global_visited', False)
        self.use_prior_coverage = env_config.get('use_prior_coverage', False)
        # Local crop of covered_centerline so the encoder can act on where it has already
        # tracked (pairs with F3 uncovered-DT shaping).
        self.use_covered_centerline = env_config.get('use_covered_centerline', False)
        self.use_prev_action = env_config.get('use_prev_action', False)
        # Two broadcast scalars: distance-to-last-junction and fraction of its branches
        # still unvisited; supplied via build()'s topology_features.
        self.use_topology_memory = env_config.get('use_topology_memory', True)
        # Multi-scale wide-context crop (5 channels), area-pooled to obs_size so it shares
        # the local channels' spatial footprint.
        self.use_multiscale = env_config.get('use_multiscale', True)
        self.wide_crop_factor = int(env_config.get('wide_crop_factor', 4))
        self.tolerance = env_config.get('tolerance', 2.0)
        # DT-channel normaliser: the crop diagonal (~obs_size·√2) instead of tolerance, so
        # in-FOV distances stay below 1.0 and keep their long-range gradient.
        self.dt_norm_scale = float(env_config.get('dt_norm_scale', self.obs_size * 1.4142135623730951))

        # Ablation zero-masks: base geometry channels stay structurally present but are
        # zero-filled at build() when set, to test whether they mislead the policy.
        #   mask_dt → channels 4,5,6 ; mask_pred_centerline → 7 ; mask_tangent → 8,9
        self.mask_dt = bool(env_config.get('mask_dt', False))
        self.mask_pred_centerline = bool(env_config.get('mask_pred_centerline', False))
        self.mask_tangent = bool(env_config.get('mask_tangent', False))

        # Buffer sized from the class-level channel table (the SSoT).
        self._max_channels = self.n_channels(config)
        self._obs_buffer = np.zeros((self._max_channels, self.obs_size, self.obs_size), dtype=np.float32)
        self._stacked_sources: Optional[np.ndarray] = None  # (H, W, K)
        self._copy_on_build: bool = True  # set False for zero-copy inference
        # Full-image junction/endpoint map, set by prepare_stacked_sources() for the env's
        # per-position junction/endpoint bonus lookups.
        self.junction_map: Optional[np.ndarray] = None

        # Unit direction lookup for the 8 movement actions (N, NE, E, SE, S, SW, W, NW);
        # STOP has no direction.
        _RAW = np.array(
            [
                [-1, 0],
                [-1, 1],
                [0, 1],
                [1, 1],
                [1, 0],
                [1, -1],
                [0, -1],
                [-1, -1],
            ],
            dtype=np.float32,
        )
        self._action_dy_dx = _RAW / np.linalg.norm(_RAW, axis=1, keepdims=True)

    def prepare_stacked_sources(
        self,
        distance_transform: np.ndarray,
        dt_gradient: np.ndarray,
        centerline: np.ndarray,
        vessel_orientation: np.ndarray,
        unet_prior: Optional[np.ndarray] = None,
        vesselness: Optional[np.ndarray] = None,
    ) -> None:
        """Pre-stack the static per-episode maps into one (H, W, K) float32 array.

        Call once per episode in set_data(). Base layout (K=6): DT, grad_y, grad_x,
        centerline, tangent_y, tangent_x; optional maps appended in ``_OPTIONAL_CHANNELS``
        order. Dynamic (per-step) channels are emitted later in build(), not here.
        """
        H, W = distance_transform.shape[:2]
        # Channel count tracks config flags, not which optional maps were supplied, so a
        # missing map (e.g. unet_prior=None) yields a zero slot of the declared width.
        n_extra = (
            int(self.use_curvature)
            + 2 * int(self.use_junction)  # junction + endpoint
            + int(self.use_vesselness)
            + int(self.use_unet_prior)
            + int(self.use_unet_uncertainty)
        )
        s = np.empty((H, W, 6 + n_extra), dtype=np.float32)
        s[:, :, 0] = distance_transform
        s[:, :, 1] = dt_gradient[:, :, 0]
        s[:, :, 2] = dt_gradient[:, :, 1]
        s[:, :, 3] = (centerline > 0).astype(np.float32)
        s[:, :, 4] = vessel_orientation[:, :, 0]
        s[:, :, 5] = vessel_orientation[:, :, 1]
        idx = 6
        if self.use_curvature:
            s[:, :, idx] = self.compute_curvature(vessel_orientation)
            idx += 1
        if self.use_junction:
            jmap, emap = self.compute_junction_map(centerline)
            s[:, :, idx] = jmap
            s[:, :, idx + 1] = emap
            # Expose junction-only map for reward/junction-aux; endpoints stay obs-only.
            self.junction_map = jmap
            idx += 2
        if self.use_vesselness:
            if vesselness is not None:
                s[:, :, idx] = vesselness.astype(np.float32, copy=False)
            else:
                s[:, :, idx] = 0.0  # zero-fill preserves the declared obs shape
            idx += 1
        if self.use_unet_prior:
            if unet_prior is not None:
                s[:, :, idx] = unet_prior.astype(np.float32, copy=False)
            else:
                s[:, :, idx] = 0.0
            idx += 1
        if self.use_unet_uncertainty:
            if unet_prior is not None:
                s[:, :, idx] = self.compute_unet_entropy(unet_prior)
            else:
                s[:, :, idx] = 0.0
            idx += 1
        self._stacked_sources = s

    @staticmethod
    def compute_dt_gradient(distance_transform: np.ndarray) -> np.ndarray:
        """Return the (H, W, 2) [grad_y, grad_x] of the DT, negated/normalised to point toward the centerline.

        Call once per episode in set_data().
        """
        dt = distance_transform.astype(np.float32)
        grad_y, grad_x = np.gradient(dt)
        grad_y, grad_x = (-grad_y, -grad_x)  # negate so vectors point toward the centerline
        mag = np.sqrt(grad_y**2 + grad_x**2) + 1e-8
        grad_y = (grad_y / mag).astype(np.float32)
        grad_x = (grad_x / mag).astype(np.float32)
        return np.stack([grad_y, grad_x], axis=-1)

    def build(
        self,
        image: np.ndarray,
        visited_mask: np.ndarray,
        vesselness: Optional[np.ndarray],
        position: np.ndarray,
        prev_direction: Optional[int],
        distance_transform: Optional[np.ndarray] = None,
        centerline: Optional[np.ndarray] = None,
        vessel_orientation: Optional[np.ndarray] = None,
        dt_gradient: Optional[np.ndarray] = None,
        unet_prior: Optional[np.ndarray] = None,
        prior_coverage: Optional[np.ndarray] = None,
        covered_centerline: Optional[np.ndarray] = None,
        topology_features: Optional[tuple] = None,
    ) -> np.ndarray:
        """Assemble the (C, obs_size, obs_size) observation centred on ``position``.

        Crops the local RGB/visited window and the pre-stacked static maps, then appends the
        enabled optional channels in ``_OPTIONAL_CHANNELS`` order (spatial first, broadcast
        scalars last). Returns a copy of the reused buffer unless ``_copy_on_build`` is False.
        """
        y, x = int(position[0]), int(position[1])
        y_start = y - self.half_size
        y_end = y + self.half_size + 1
        x_start = x - self.half_size
        x_end = x + self.half_size + 1

        buf = self._obs_buffer

        # RGB (0-2) and visited mask (3).
        image_crop = self._crop(image, y_start, y_end, x_start, x_end)
        buf[0:3] = image_crop.transpose(2, 0, 1)

        buf[3] = self._crop(visited_mask, y_start, y_end, x_start, x_end)

        # Static channels (DT, grads, centerline, tangent, optional spatial maps).
        if self._stacked_sources is not None:
            n_static = self._stacked_sources.shape[2]
            static_crop = self._crop(self._stacked_sources, y_start, y_end, x_start, x_end)  # (obs, obs, n_static)
            buf[4 : 4 + n_static] = static_crop.transpose(2, 0, 1)
            # Normalise the DT channel (always static index 0) in place.
            buf[4] /= max(self.dt_norm_scale, 1e-6)
            np.clip(buf[4], 0.0, 1.0, out=buf[4])
            n = 4 + n_static
        else:
            # Fallback when prepare_stacked_sources() wasn't called: zero past RGB+visited
            # so optional slots don't leak stale data, then fill what we can per-step.
            buf[4:] = 0
            if distance_transform is not None:
                dt_crop = self._crop(distance_transform, y_start, y_end, x_start, x_end).astype(np.float32)
                dt_crop /= max(self.dt_norm_scale, 1e-6)
                np.clip(dt_crop, 0.0, 1.0, out=dt_crop)
                buf[4] = dt_crop
                if dt_gradient is not None:
                    grad_crop = self._crop(dt_gradient, y_start, y_end, x_start, x_end)
                    buf[5] = grad_crop[:, :, 0]
                    buf[6] = grad_crop[:, :, 1]
                else:
                    raw_dt = self._crop(distance_transform, y_start, y_end, x_start, x_end).astype(np.float32)
                    gy, gx = np.gradient(raw_dt)
                    gy, gx = -gy, -gx
                    mag = np.sqrt(gy**2 + gx**2) + 1e-8
                    buf[5] = gy / mag
                    buf[6] = gx / mag
            if centerline is not None:
                buf[7] = (self._crop(centerline, y_start, y_end, x_start, x_end) > 0).astype(np.float32)
            if vessel_orientation is not None:
                orient_crop = self._crop(vessel_orientation, y_start, y_end, x_start, x_end)
                buf[8] = orient_crop[:, :, 0]
                buf[9] = orient_crop[:, :, 1]
            # Advance n past the zero-filled curvature/junction/vesselness slots so the
            # layout still matches n_channels; the unet_prior slot is filled below.
            n = 10
            if self.use_curvature:
                n += 1
            if self.use_junction:
                n += 2
            if self.use_vesselness:
                if vesselness is not None:
                    buf[n] = self._crop(vesselness, y_start, y_end, x_start, x_end)
                n += 1

        # UNet-prior fallback — only when stacked sources are absent. Advance n whenever the
        # flag is on (zero-fill if the map is missing) so the channel count still matches.
        if self.use_unet_prior and self._stacked_sources is None:
            if unet_prior is not None:
                buf[n] = self._crop(unet_prior, y_start, y_end, x_start, x_end)
            else:
                buf[n] = 0.0
            n += 1

        # Global visited mask, area-pooled to obs_size² (preserves the coverage signature
        # that stride sampling would mostly drop).
        if self.use_global_visited:
            buf[n] = self._wide_to_obs(visited_mask)
            n += 1

        # Prior coverage from earlier traces, same area-pooling; zero in single-episode training.
        if self.use_prior_coverage:
            if prior_coverage is not None:
                buf[n] = self._wide_to_obs(prior_coverage)
            else:
                buf[n] = 0.0
            n += 1

        # Covered-centerline crop (binary); pairs with F3 shaping. Zero when not supplied.
        if self.use_covered_centerline:
            if covered_centerline is not None:
                cov_crop = self._crop(covered_centerline, y_start, y_end, x_start, x_end)
                buf[n] = (cov_crop > 0).astype(np.float32)
            else:
                buf[n] = 0.0
            n += 1

        # Multi-scale wide-context channels (spatial): a wide_crop_factor·obs_size crop
        # downsampled to obs_size² for coarse layout beyond the local receptive field.
        if self.use_multiscale:
            wide_half = self.half_size * self.wide_crop_factor
            wy_s = y - wide_half
            wy_e = y + wide_half + 1
            wx_s = x - wide_half
            wx_e = x + wide_half + 1
            wide_rgb_crop = self._crop(image, wy_s, wy_e, wx_s, wx_e)
            wide_vis_crop = self._crop(visited_mask, wy_s, wy_e, wx_s, wx_e)
            buf[n : n + 3] = self._wide_to_obs(wide_rgb_crop).transpose(2, 0, 1)
            buf[n + 3] = self._wide_to_obs(wide_vis_crop)
            if unet_prior is not None:
                wide_up_crop = self._crop(unet_prior, wy_s, wy_e, wx_s, wx_e)
                buf[n + 4] = self._wide_to_obs(wide_up_crop)
            else:
                buf[n + 4] = 0.0
            n += 5

        # Scalar (broadcast-constant) channels, kept last; the policy net slices these out
        # before the CNN and injects them at the MLP head.

        if self.use_prev_action:
            if prev_direction is not None and 0 <= prev_direction < 8:
                dy, dx = self._action_dy_dx[prev_direction]
            else:
                dy, dx = 0.0, 0.0
            buf[n].fill(dy)
            buf[n + 1].fill(dx)
            n += 2

        # Topology memory: (dist-to-last-junction/obs_size clipped to [0,1], fraction of that
        # junction's skeleton-neighbours still unvisited); (1.0, 0.0) before any junction.
        if self.use_topology_memory:
            if topology_features is not None:
                td, tb = (float(topology_features[0]), float(topology_features[1]))
            else:
                td, tb = 1.0, 0.0
            buf[n].fill(td)
            buf[n + 1].fill(tb)
            n += 2

        # Ablation zero-masks, applied last so they override whichever path filled the slot.
        if self.mask_dt:
            buf[4:7] = 0.0
        if self.mask_pred_centerline:
            buf[7] = 0.0
        if self.mask_tangent:
            buf[8:10] = 0.0

        # Buffer is reused across calls; copy out unless the caller opts into zero-copy.
        if self._copy_on_build:
            return buf[:n].copy()
        return buf[:n]

    def _crop(self, array: np.ndarray, y_start: int, y_end: int, x_start: int, x_end: int) -> np.ndarray:
        """Extract the [y_start:y_end, x_start:x_end] window, zero-padding out-of-bounds regions."""
        h, w = array.shape[:2]

        pad_top = max(0, -y_start)
        pad_bottom = max(0, y_end - h)
        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - w)

        ys = max(0, y_start)
        ye = min(h, y_end)
        xs = max(0, x_start)
        xe = min(w, x_end)

        crop = array[ys:ye, xs:xe]

        if pad_top or pad_bottom or pad_left or pad_right:
            pw = ((pad_top, pad_bottom), (pad_left, pad_right))
            if array.ndim == 3:
                pw = pw + ((0, 0),)
            crop = np.pad(crop, pw, mode='constant', constant_values=0)

        return crop

    def _wide_to_obs(self, wide_crop: np.ndarray) -> np.ndarray:
        """Area-average a wide crop down to (obs_size, obs_size[, C]) via cv2.INTER_AREA.

        INTER_AREA preserves intensity statistics under downsampling, unlike stride sampling
        which would drop most pixels at a 4× factor.
        """
        import cv2

        return cv2.resize(wide_crop.astype(np.float32, copy=False), (self.obs_size, self.obs_size), interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def compute_curvature(vessel_orientation: np.ndarray) -> np.ndarray:
        """Return per-pixel curvature (H, W) float32 in roughly [0, 1] from the vessel-tangent field.

        Uses the gradient magnitude of the structure-tensor tangent — peaks at bends, ~0 on
        straight segments.
        """
        ty = vessel_orientation[:, :, 0].astype(np.float32)
        tx = vessel_orientation[:, :, 1].astype(np.float32)
        gy_y, gy_x = np.gradient(ty)
        gx_y, gx_x = np.gradient(tx)
        curv = np.sqrt(gy_y**2 + gy_x**2 + gx_y**2 + gx_x**2)
        # Tangent components are bounded, so clip the gradient magnitude to a stable [0, 1].
        return np.clip(curv, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_unet_entropy(unet_prior: np.ndarray) -> np.ndarray:
        """Return per-pixel binary entropy of a probability map, normalised to [0, 1].

        ``H(p) = -p ln p - (1-p) ln(1-p)`` divided by ln(2): 1.0 at p=0.5 (max uncertainty),
        0.0 at p∈{0, 1}.
        """
        eps = 1e-7
        p = np.clip(unet_prior.astype(np.float32, copy=False), eps, 1.0 - eps)
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        return (h / float(np.log(2.0))).astype(np.float32)

    @staticmethod
    def compute_junction_map(centerline: np.ndarray, dilation_radius: int = 3) -> tuple:
        """Return ``(junction_map, endpoint_map)`` — dilated binary (H, W) float32 masks from the skeleton.

        By 8-neighbour count on the skeleton, per centerline pixel: >= 3 → junction, == 1 →
        endpoint. ``dilation_radius`` thickens both for visibility.
        """
        skel = (centerline > 0).astype(np.uint8)
        empty = np.zeros_like(skel, dtype=np.float32)
        if skel.sum() == 0:
            return empty, empty.copy()

        from scipy.ndimage import convolve, grey_dilation

        kernel = np.ones((3, 3), dtype=np.uint8)
        nbr_count = convolve(skel, kernel, mode='constant', cval=0) - skel

        junction_mask = np.zeros_like(skel, dtype=np.float32)
        endpoint_mask = np.zeros_like(skel, dtype=np.float32)
        junction_mask[(skel > 0) & (nbr_count >= 3)] = 1.0
        endpoint_mask[(skel > 0) & (nbr_count == 1)] = 1.0

        if dilation_radius > 0:
            size = 2 * dilation_radius + 1
            junction_mask = grey_dilation(junction_mask, size=(size, size))
            endpoint_mask = grey_dilation(endpoint_mask, size=(size, size))

        return junction_mask.astype(np.float32), endpoint_mask.astype(np.float32)

    @staticmethod
    def compute_vessel_orientation(image: np.ndarray) -> np.ndarray:
        """Return the (H, W, 2) normalised vessel tangent [tangent_y, tangent_x] from the structure tensor.

        Uses the green channel (best fundus vessel contrast). Call once per image in set_data().
        """
        if image.ndim == 3:
            gray = image[:, :, 1].astype(np.float64)
        else:
            gray = image.astype(np.float64)

        iy = np.gradient(gray, axis=0)
        ix = np.gradient(gray, axis=1)

        # Gaussian-weighted structure-tensor components.
        from scipy.ndimage import gaussian_filter

        sigma = 3.0  # integration scale ~ vessel width
        j_xx = gaussian_filter(ix * ix, sigma)
        j_xy = gaussian_filter(ix * iy, sigma)
        j_yy = gaussian_filter(iy * iy, sigma)

        # Dominant orientation of the 2x2 symmetric tensor; vessel tangent is θ rotated 90°.
        theta = 0.5 * np.arctan2(2.0 * j_xy, j_xx - j_yy + 1e-10)

        tangent_y = -np.sin(theta).astype(np.float32)
        tangent_x = np.cos(theta).astype(np.float32)

        orientation = np.stack([tangent_y, tangent_x], axis=-1)
        return orientation
