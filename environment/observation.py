# observation.py
"""Observation construction for vessel tracing environment."""

from typing import Any, Dict, Optional

import numpy as np


class ObservationBuilder:
    """Builds observation tensors for the RL agent.

    Base channels (always present):
    0-2 : RGB crop
    3   : visited mask crop
    4   : distance transform crop, normalised to [0, 1]
    5   : vessel gradient dy (from DT), normalised to [-1, 1]
    6   : vessel gradient dx (from DT), normalised to [-1, 1]
    7   : centerline binary mask
    8   : vessel tangent dy (along-vessel direction)
    9   : vessel tangent dx (along-vessel direction)

    Optional channels (in this order, gated by config flags — see
    ``_OPTIONAL_CHANNELS`` for the canonical table):
        curvature        — magnitude of the gradient of the vessel-tangent
                           field; peaks at bends. `use_curvature` (default True).
        junction+endpoint— two binary channels: junction (skeleton degree ≥ 3,
                           dilated) and endpoint (degree == 1, dilated). Splits
                           the previous magnitude-coded (1.0/0.5) channel into
                           two cleanly-interpretable indicators. `use_junction`
                           (default True, contributes 2 channels).
        vesselness       — Frangi vesselness map. `use_vesselness`.
        unet_prior       — Frozen Centerline-UNet probability map (local
                           crop). `use_unet_prior`.
        global_visited   — Full visited mask area-pooled to obs_size².
        prior_coverage   — Accumulated coverage from earlier traces, same
                           area-pooling.
        prev_action      — Two broadcast channels carrying (last_dy, last_dx)
                           of the most recently taken move. `use_prev_action`.
        topology_memory  — Two broadcast scalars: distance-from-last-junction
                           and branches-remaining-at-current-junction.
                           `use_topology_memory` (default True).
        multiscale       — Five channels (wide RGB ×3, wide visited, wide
                           UNet prior) covering ``wide_crop_factor * obs_size``
                           area-pooled to obs_size². `use_multiscale`.

    Channels 5-6 point TOWARD the centerline (perpendicular to vessel).
    Channels 8-9 point ALONG the vessel (tangent direction from structure tensor).
    """

    # ── Channel layout — single source of truth ───────────────────────────────
    #
    # Per-feature contribution to the observation channel count, in the
    # order channels are emitted by ``build()``. Optional features carry
    # ``(flag_name, default, channel_count)``. ``n_channels(config)`` and the
    # policy network both consult this table so the obs builder, gym space,
    # and CNN input stay in lockstep.
    _BASE_CHANNELS = 10  # RGB(3) + visited(1) + DT(1) + DT-grad(2) + centerline(1) + tangent(2)
    # Layout invariant: spatial channels precede scalar (broadcast-constant)
    # channels. The policy network slices ``obs[:, :-n_scalar]`` into the CNN
    # and reads ``obs[:, -n_scalar:, 0, 0]`` to recover the scalar values for
    # direct injection at the MLP head — see ``ActorCriticNetwork.forward``.
    _OPTIONAL_CHANNELS = (
        # (flag, default, count)            # ── spatial features ──────────
        ('use_curvature', True, 1),
        (
            'use_junction',
            True,
            2,
        ),  # junction (binary) + endpoint (binary)
        ('use_vesselness', False, 1),
        ('use_unet_prior', False, 1),
        (
            'use_unet_uncertainty',
            False,
            1,
        ),  # binary entropy of unet_prior
        ('use_global_visited', False, 1),
        ('use_prior_coverage', False, 1),
        # E3 — covered-centerline channel. Shows the agent where it has
        # already covered GT centerline pixels (tolerance-disk projected),
        # so the F3 shaping gradient toward uncovered work can be
        # *observed* and not only *felt*. Crop of the env's
        # covered_centerline mask, binary 0/1.
        ('use_covered_centerline', False, 1),
        (
            'use_multiscale',
            True,
            5,
        ),  # wide RGB(3) + wide visited(1) + wide UNet prior(1)
        # ── scalar (broadcast-constant) channels — kept last ──────────────
        ('use_prev_action', False, 2),
        ('use_topology_memory', True, 2),
    )
    # Flag names that produce broadcast-constant channels (i.e. scalars).
    # Used by ``n_scalar_channels`` / ``n_spatial_channels``.
    _SCALAR_FLAGS = (
        'use_prev_action',
        'use_topology_memory',
    )

    @classmethod
    def n_channels(cls, config: Dict[str, Any]) -> int:
        """Compute the observation channel count for a given config.

        Used by ``__init__`` (buffer sizing), ``vessel_env._setup_observation_space``
        (gym space shape), and ``models.policy_network._compute_in_channels``
        (CNN input plane count). The three previously maintained their own
        copies of the arithmetic, which drifted in past channel additions
        and required reading multiple files to add a new channel.
        """
        env = config.get('environment', {})
        n = cls._BASE_CHANNELS
        for (
            flag,
            default,
            count,
        ) in cls._OPTIONAL_CHANNELS:
            if env.get(flag, default):
                n += count
        return n

    @classmethod
    def n_scalar_channels(cls, config: Dict[str, Any]) -> int:
        """Count of broadcast-constant channels at the END of the obs.

        These channels (prev_action, topology_memory) carry agent-state
        scalars that don't need spatial filtering. The policy network
        slices them out before the CNN and concatenates them back at the
        MLP head — see ``ActorCriticNetwork.forward``.
        """
        env = config.get('environment', {})
        n = 0
        for (
            flag,
            default,
            count,
        ) in cls._OPTIONAL_CHANNELS:
            if flag in cls._SCALAR_FLAGS and env.get(flag, default):
                n += count
        return n

    @classmethod
    def n_spatial_channels(cls, config: Dict[str, Any]) -> int:
        """Channel count consumed by the CNN encoder (total minus scalars)."""
        return cls.n_channels(config) - cls.n_scalar_channels(config)

    @classmethod
    def junction_channel_idx(cls, config: Dict[str, Any]) -> Optional[int]:
        """Channel index of the junction map, or ``None`` if disabled.

        The PPO trainer reads this to pull per-step junction supervision
        targets directly from the stored observation batch without
        re-running the environment.
        """
        env = config.get('environment', {})
        if not env.get('use_junction', True):
            return None
        # Junction sits at _BASE_CHANNELS plus the always-preceding curvature
        # channel (when enabled). Any future channel inserted before
        # 'use_junction' in _OPTIONAL_CHANNELS will be picked up automatically.
        n = cls._BASE_CHANNELS
        for (
            flag,
            default,
            count,
        ) in cls._OPTIONAL_CHANNELS:
            if flag == 'use_junction':
                return n
            if env.get(flag, default):
                n += count
        return None

    def __init__(self, config: Dict[str, Any]):
        env_config = config.get('environment', {})
        self.obs_size = env_config.get('observation_size', 65)
        self.half_size = self.obs_size // 2
        self.use_vesselness = env_config.get('use_vesselness', False)
        self.use_curvature = env_config.get('use_curvature', True)
        self.use_junction = env_config.get('use_junction', True)
        self.use_unet_prior = env_config.get('use_unet_prior', False)
        # Per-pixel binary entropy of the UNet vessel probability, scaled to
        # [0, 1] by ln(2). Peaks where the predictor is uncertain (p ≈ 0.5)
        # — a natural "exploration under uncertainty" signal pairing with
        # the topology-memory channels. Cheap: reuses the prob map.
        self.use_unet_uncertainty = env_config.get('use_unet_uncertainty', False)
        self.use_global_visited = env_config.get('use_global_visited', False)
        self.use_prior_coverage = env_config.get('use_prior_coverage', False)
        # E3 — covered-centerline channel: local crop of the env's
        # covered_centerline mask so the agent can observe where it has
        # already tracked. Pairs with F3 (uncov-DT shaping) — F3 puts the
        # uncovered gradient in the reward, this channel lets the encoder
        # also act on it directly.
        self.use_covered_centerline = env_config.get('use_covered_centerline', False)
        self.use_prev_action = env_config.get('use_prev_action', False)
        # Topology-aware memory: two broadcast scalar channels carrying
        # (distance to last visited junction, fraction of its branches
        # still unvisited). Computed in vessel_env and threaded through
        # build() via the ``topology_features`` kwarg.
        self.use_topology_memory = env_config.get('use_topology_memory', True)
        # Multi-scale wide-context crop. Adds 5 channels — wide RGB (3),
        # wide visited (1), wide UNet prior (1, zero-filled when use_unet_prior
        # is off or the checkpoint is missing). The wider field of view is
        # downsampled to obs_size×obs_size via area averaging so it occupies
        # the same spatial footprint as the local channels.
        self.use_multiscale = env_config.get('use_multiscale', True)
        self.wide_crop_factor = int(env_config.get('wide_crop_factor', 4))
        self.tolerance = env_config.get('tolerance', 2.0)
        # DT-channel normalizer. Was self.tolerance, which saturated to 1.0
        # at every pixel >2.5 px from a centerline — flattening the channel
        # across most of the crop and removing long-range distance gradient.
        # Use the crop diagonal (~obs_size·√2) so distances within the agent's
        # field of view stay below 1.0 and retain magnitude information. The
        # unit DT-gradient channels (5, 6) carry direction independently.
        self.dt_norm_scale = float(
            env_config.get(
                'dt_norm_scale',
                self.obs_size * 1.4142135623730951,
            )
        )

        # ── Zero-mask flags for base geometry channels (ablation experiment) ──
        # These channels are structurally always present (the obs layout is
        # hardcoded at indices 4-9) but each flag, when True, zero-fills the
        # corresponding slice at build() time so the policy sees no signal
        # from it. Tests the "are these channels misleading the agent?"
        # hypothesis without refactoring the channel SSoT.
        #   mask_dt   → zeros channels 4, 5, 6  (DT magnitude + DT-grad y/x)
        #   mask_pred_centerline → zeros channel 7
        #   mask_tangent → zeros channels 8, 9
        self.mask_dt = bool(env_config.get('mask_dt', False))
        self.mask_pred_centerline = bool(env_config.get('mask_pred_centerline', False))
        self.mask_tangent = bool(env_config.get('mask_tangent', False))

        # Pre-allocate observation buffer — channel count comes from the
        # class-level table so adding a channel only requires editing
        # _OPTIONAL_CHANNELS and the build() emission code.
        self._max_channels = self.n_channels(config)
        self._obs_buffer = np.zeros(
            (
                self._max_channels,
                self.obs_size,
                self.obs_size,
            ),
            dtype=np.float32,
        )
        self._stacked_sources: Optional[np.ndarray] = None  # (H, W, K)
        self._copy_on_build: bool = True  # set False for zero-copy inference
        # Full-image junction/endpoint map (H, W) float32 — set by
        # prepare_stacked_sources() so VesselTracingEnv can look up the value
        # at the agent's current position for junction/endpoint bonuses.
        self.junction_map: Optional[np.ndarray] = None

        # Cached normalised direction lookup for prev-action channels.
        # 8 movement actions (N, NE, E, SE, S, SW, W, NW) — STOP has no direction.
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
        """Pre-stack static per-episode maps into one (H, W, K) float32 array.

        Call once per episode in set_data(), not per step.
        Base layout (K=6):
            0=DT  1=grad_y  2=grad_x  3=centerline  4=tangent_y  5=tangent_x
        Optional channels appended after, in the order set by
        ``_OPTIONAL_CHANNELS``:
            curvature (use_curvature),
            junction + endpoint (use_junction, 2 channels),
            vesselness (use_vesselness),
            unet_prior (use_unet_prior).

        Note: prev_action / topology_memory / multi-scale channels are *not*
        stacked here — they're dynamic (depend on the agent's per-step state)
        and emitted at build() time.
        """
        H, W = distance_transform.shape[:2]
        # Channel count is driven by config flags only — never by whether the
        # caller happened to supply an optional map. This keeps the observation
        # width in lockstep with ObservationBuilder.n_channels so fallback
        # paths (e.g. missing UNet checkpoint → unet_prior=None) still produce
        # an obs of the declared shape, just with a zero-filled slot.
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
            # Expose junction-only map for reward bonuses / junction-aux supervision.
            # Endpoints intentionally NOT exposed via this attribute — only the
            # observation channel — so existing reward shaping stays junction-only.
            self.junction_map = jmap
            idx += 2
        if self.use_vesselness:
            if vesselness is not None:
                s[:, :, idx] = vesselness.astype(np.float32, copy=False)
            else:
                # Frangi map not supplied — keep the slot zero-filled so the
                # declared obs shape is preserved.
                s[:, :, idx] = 0.0
            idx += 1
        if self.use_unet_prior:
            if unet_prior is not None:
                s[:, :, idx] = unet_prior.astype(np.float32, copy=False)
            else:
                # Predictor unavailable (checkpoint missing) — emit a zero
                # channel so obs dims still match n_channels.
                s[:, :, idx] = 0.0
            idx += 1
        if self.use_unet_uncertainty:
            if unet_prior is not None:
                s[:, :, idx] = self.compute_unet_entropy(unet_prior)
            else:
                # No predictor → no entropy signal; zero is the lowest-info
                # default, consistent with the unet_prior fallback above.
                s[:, :, idx] = 0.0
            idx += 1
        self._stacked_sources = s

    @staticmethod
    def compute_dt_gradient(
        distance_transform: np.ndarray,
    ) -> np.ndarray:
        """Precompute full-image DT gradient. Call once per episode in set_data().

        Returns (H, W, 2) array of [grad_y, grad_x], negated and normalised
        so vectors point TOWARD the centerline.
        """
        dt = distance_transform.astype(np.float32)
        grad_y, grad_x = np.gradient(dt)
        grad_y, grad_x = (
            -grad_y,
            -grad_x,
        )  # point toward centerline
        mag = np.sqrt(grad_y**2 + grad_x**2) + 1e-8
        grad_y = (grad_y / mag).astype(np.float32)
        grad_x = (grad_x / mag).astype(np.float32)
        return np.stack([grad_y, grad_x], axis=-1)  # (H, W, 2)

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
        y, x = int(position[0]), int(position[1])
        y_start = y - self.half_size
        y_end = y + self.half_size + 1
        x_start = x - self.half_size
        x_end = x + self.half_size + 1

        buf = self._obs_buffer

        # --- RGB (channels 0-2) ---
        image_crop = self._crop(image, y_start, y_end, x_start, x_end)
        buf[0:3] = image_crop.transpose(2, 0, 1)

        # --- Visited mask (channel 3) ---
        buf[3] = self._crop(
            visited_mask,
            y_start,
            y_end,
            x_start,
            x_end,
        )

        # --- Static channels (DT, grads, centerline, tangent, [curv], [junc], [unet]) ---
        if self._stacked_sources is not None:
            n_static = self._stacked_sources.shape[2]
            static_crop = self._crop(
                self._stacked_sources,
                y_start,
                y_end,
                x_start,
                x_end,
            )  # (obs, obs, n_static)
            buf[4 : 4 + n_static] = static_crop.transpose(2, 0, 1)
            # Normalise DT channel (always at static index 0) in-place
            buf[4] /= max(self.dt_norm_scale, 1e-6)
            np.clip(buf[4], 0.0, 1.0, out=buf[4])
            n = 4 + n_static
        else:
            # Fallback when prepare_stacked_sources() was not called.
            # Zero everything past the RGB+visited channels so optional
            # slots (curvature/junction/vesselness/unet) don't leak stale data.
            buf[4:] = 0
            if distance_transform is not None:
                dt_crop = self._crop(
                    distance_transform,
                    y_start,
                    y_end,
                    x_start,
                    x_end,
                ).astype(np.float32)
                dt_crop /= max(self.dt_norm_scale, 1e-6)
                np.clip(dt_crop, 0.0, 1.0, out=dt_crop)
                buf[4] = dt_crop
                if dt_gradient is not None:
                    grad_crop = self._crop(
                        dt_gradient,
                        y_start,
                        y_end,
                        x_start,
                        x_end,
                    )
                    buf[5] = grad_crop[:, :, 0]
                    buf[6] = grad_crop[:, :, 1]
                else:
                    raw_dt = self._crop(
                        distance_transform,
                        y_start,
                        y_end,
                        x_start,
                        x_end,
                    ).astype(np.float32)
                    gy, gx = np.gradient(raw_dt)
                    gy, gx = -gy, -gx
                    mag = np.sqrt(gy**2 + gx**2) + 1e-8
                    buf[5] = gy / mag
                    buf[6] = gx / mag
            if centerline is not None:
                buf[7] = (
                    self._crop(
                        centerline,
                        y_start,
                        y_end,
                        x_start,
                        x_end,
                    )
                    > 0
                ).astype(np.float32)
            if vessel_orientation is not None:
                orient_crop = self._crop(
                    vessel_orientation,
                    y_start,
                    y_end,
                    x_start,
                    x_end,
                )
                buf[8] = orient_crop[:, :, 0]
                buf[9] = orient_crop[:, :, 1]
            # Fallback channel layout still needs to match n_channels: the
            # buffer was zero-filled past channel 3 above, so we just advance
            # ``n`` over the curvature / junction(+endpoint) / vesselness
            # slots (left zero) and let the use_unet_prior branch below fill
            # its slot like normal.
            n = 10
            if self.use_curvature:
                n += 1
            if self.use_junction:
                n += 2
            if self.use_vesselness:
                if vesselness is not None:
                    buf[n] = self._crop(
                        vesselness,
                        y_start,
                        y_end,
                        x_start,
                        x_end,
                    )
                n += 1

        # Vesselness is now part of _stacked_sources (P2#10), so no separate
        # crop here. The ``vesselness`` kwarg below is retained only for the
        # fallback path when stacked sources are missing — see below.

        # --- UNet prior fallback (only used when stacked sources are absent) ---
        # Always advance n when the flag is on so the channel count matches
        # _compute_in_channels even if unet_prior was not provided (e.g. missing
        # checkpoint); zero-fill the slot in that case.
        if self.use_unet_prior and self._stacked_sources is None:
            if unet_prior is not None:
                buf[n] = self._crop(
                    unet_prior,
                    y_start,
                    y_end,
                    x_start,
                    x_end,
                )
            else:
                buf[n] = 0.0
            n += 1

        # --- Global downsampled visited mask ---
        # Area-pool the full visited_mask down to obs_size² via cv2.INTER_AREA.
        # The previous implementation used stride sampling, which on a
        # 1024×1024 image into a 65×65 crop dropped ~99.6% of visited pixels
        # — a thin trace would barely register in the downsample. Area
        # averaging counts every visited pixel and preserves the overall
        # coverage signature.
        if self.use_global_visited:
            buf[n] = self._wide_to_obs(visited_mask)
            n += 1

        # --- Prior coverage channel ---
        # Same area-pool treatment as the global visited mask. Zero during
        # single-episode training (prior_coverage=None).
        if self.use_prior_coverage:
            if prior_coverage is not None:
                buf[n] = self._wide_to_obs(prior_coverage)
            else:
                buf[n] = 0.0
            n += 1

        # E3 --- Covered-centerline channel ---
        # Local crop of the env's covered_centerline mask (the tolerance-disk
        # projected mask the reward uses). Binary 0/1, no normalization.
        # Pairs with F3 shaping: F3 puts the uncov-DT in the reward, this
        # channel lets the encoder act on it. Zero when env.covered_centerline
        # isn't supplied (early reset / fallback).
        if self.use_covered_centerline:
            if covered_centerline is not None:
                cov_crop = self._crop(
                    covered_centerline,
                    y_start,
                    y_end,
                    x_start,
                    x_end,
                )
                buf[n] = (cov_crop > 0).astype(np.float32)
            else:
                buf[n] = 0.0
            n += 1

        # --- Multi-scale wide-context channels (SPATIAL — emitted before scalars) ---
        # A centred ``wide_crop_factor * obs_size`` crop downsampled to
        # obs_size² gives the policy a coarse view of the vascular layout
        # beyond the local 65×65 receptive field. Helps with branch
        # continuation and reconnect-after-gap decisions.
        if self.use_multiscale:
            wide_half = self.half_size * self.wide_crop_factor
            wy_s = y - wide_half
            wy_e = y + wide_half + 1
            wx_s = x - wide_half
            wx_e = x + wide_half + 1
            wide_rgb_crop = self._crop(image, wy_s, wy_e, wx_s, wx_e)
            wide_vis_crop = self._crop(
                visited_mask,
                wy_s,
                wy_e,
                wx_s,
                wx_e,
            )
            buf[n : n + 3] = self._wide_to_obs(wide_rgb_crop).transpose(2, 0, 1)
            buf[n + 3] = self._wide_to_obs(wide_vis_crop)
            if unet_prior is not None:
                wide_up_crop = self._crop(
                    unet_prior,
                    wy_s,
                    wy_e,
                    wx_s,
                    wx_e,
                )
                buf[n + 4] = self._wide_to_obs(wide_up_crop)
            else:
                buf[n + 4] = 0.0
            n += 5

        # ── Scalar channels (broadcast constants — last in the layout) ────
        # These get sliced out and injected at the MLP head in the policy
        # network rather than being processed by the CNN. The buffer still
        # carries them as full obs_size² slices because the gym observation
        # space is a single Box(C, H, W) — the policy net handles the split.

        # --- Previous-action channels ---
        if self.use_prev_action:
            if prev_direction is not None and 0 <= prev_direction < 8:
                dy, dx = self._action_dy_dx[prev_direction]
            else:
                dy, dx = 0.0, 0.0
            buf[n].fill(dy)
            buf[n + 1].fill(dx)
            n += 2

        # --- Topology-memory channels ---
        # (distance to last visited junction / obs_size, clipped to [0, 1])
        # and (fraction of that junction's 8-skeleton-neighbours still
        # unvisited). Both fall back to (1.0, 0.0) before the agent has
        # reached any junction.
        if self.use_topology_memory:
            if topology_features is not None:
                td, tb = (
                    float(topology_features[0]),
                    float(topology_features[1]),
                )
            else:
                td, tb = 1.0, 0.0
            buf[n].fill(td)
            buf[n + 1].fill(tb)
            n += 2

        # Zero-mask hand-engineered geometry channels for ablation studies.
        # Applied last so the masks override whichever code path filled the
        # slot (stacked_sources fast path OR per-step fallback).
        if self.mask_dt:
            buf[4:7] = 0.0  # DT magnitude + DT-grad y + DT-grad x
        if self.mask_pred_centerline:
            buf[7] = 0.0
        if self.mask_tangent:
            buf[8:10] = 0.0

        # Copy out — buffer is reused across calls.
        # Callers that consume the observation immediately (e.g. inference)
        # can pass copy=False to avoid the allocation.
        if self._copy_on_build:
            return buf[:n].copy()
        return buf[:n]

    def _crop(
        self,
        array: np.ndarray,
        y_start: int,
        y_end: int,
        x_start: int,
        x_end: int,
    ) -> np.ndarray:
        """Extract a crop with zero-padding at boundaries."""
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
            pw = (
                (pad_top, pad_bottom),
                (pad_left, pad_right),
            )
            if array.ndim == 3:
                pw = pw + ((0, 0),)
            crop = np.pad(
                crop,
                pw,
                mode='constant',
                constant_values=0,
            )

        return crop

    def _wide_to_obs(self, wide_crop: np.ndarray) -> np.ndarray:
        """Area-average a ``(wide_size, wide_size[, C])`` crop down to
        ``(obs_size, obs_size[, C])``. cv2.INTER_AREA preserves intensity
        statistics (proper anti-aliasing for downsampling), unlike stride
        sampling which would drop ~99% of the pixels in a 4× factor crop.
        """
        import cv2  # already a project dependency (used in dataloader)

        return cv2.resize(
            wide_crop.astype(np.float32, copy=False),
            (self.obs_size, self.obs_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

    @staticmethod
    def compute_curvature(
        vessel_orientation: np.ndarray,
    ) -> np.ndarray:
        """Per-pixel curvature derived from the vessel-tangent field.

        The structure-tensor tangent already encodes vessel direction
        everywhere; the magnitude of its spatial gradient is a smooth
        proxy for local curvature (peaks at bends, ~0 on straight
        segments). Returns (H, W) float32 in roughly [0, 1].
        """
        ty = vessel_orientation[:, :, 0].astype(np.float32)
        tx = vessel_orientation[:, :, 1].astype(np.float32)
        gy_y, gy_x = np.gradient(ty)
        gx_y, gx_x = np.gradient(tx)
        curv = np.sqrt(gy_y**2 + gy_x**2 + gx_y**2 + gx_x**2)
        # Normalise: tangent components live in [-1, 1] so the gradient
        # magnitude is bounded; clip to [0, 1] for a stable input range.
        return np.clip(curv, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def compute_unet_entropy(
        unet_prior: np.ndarray,
    ) -> np.ndarray:
        """Per-pixel binary entropy of a probability map, normalised to [0, 1].

        ``H(p) = -p ln p - (1 - p) ln (1 - p)`` divided by ln(2). Peaks at
        1.0 when p == 0.5 (predictor is maximally uncertain) and is 0.0
        when p == 0 or p == 1. Gives the policy an explicit "the prior is
        unsure here" signal for exploration-under-uncertainty.
        """
        eps = 1e-7
        p = np.clip(
            unet_prior.astype(np.float32, copy=False),
            eps,
            1.0 - eps,
        )
        h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        return (h / float(np.log(2.0))).astype(np.float32)

    @staticmethod
    def compute_junction_map(
        centerline: np.ndarray,
        dilation_radius: int = 3,
    ) -> tuple:
        """Mark skeleton junctions and endpoints, dilated for visibility.

        Returns ``(junction_map, endpoint_map)`` — two binary (H, W) float32
        masks. Splitting the previous magnitude-coded (1.0 / 0.5 / 0.0)
        layout into two channels removes the "interpret a scalar code"
        burden from the policy; each channel is a clean binary indicator.

        Per centerline pixel, count 8-neighbours on the skeleton:
            >= 3 → junction_map = 1.0 (dilated)
            == 1 → endpoint_map = 1.0 (dilated)
        """
        skel = (centerline > 0).astype(np.uint8)
        empty = np.zeros_like(skel, dtype=np.float32)
        if skel.sum() == 0:
            return empty, empty.copy()

        from scipy.ndimage import (
            convolve,
            grey_dilation,
        )

        kernel = np.ones((3, 3), dtype=np.uint8)
        nbr_count = (
            convolve(
                skel,
                kernel,
                mode='constant',
                cval=0,
            )
            - skel
        )

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
    def compute_vessel_orientation(
        image: np.ndarray,
    ) -> np.ndarray:
        """Precompute vessel tangent direction from the image structure tensor.

        Uses the green channel (best vessel contrast in fundus images).
        Returns (H, W, 2) array of [tangent_y, tangent_x], normalised.

        Should be called once per image (in env.set_data), not per step.
        """
        # Use green channel for best vessel contrast
        if image.ndim == 3:
            gray = image[:, :, 1].astype(np.float64)
        else:
            gray = image.astype(np.float64)

        # Image gradients
        iy = np.gradient(gray, axis=0)
        ix = np.gradient(gray, axis=1)

        # Structure tensor components (Gaussian-weighted local averages)
        from scipy.ndimage import gaussian_filter

        sigma = 3.0  # integration scale — ~vessel width
        j_xx = gaussian_filter(ix * ix, sigma)
        j_xy = gaussian_filter(ix * iy, sigma)
        j_yy = gaussian_filter(iy * iy, sigma)

        # Eigendecomposition: smallest eigenvector = vessel tangent
        # For 2x2 symmetric matrix, analytic solution:
        # θ = 0.5 * atan2(2*Jxy, Jxx - Jyy)  gives the dominant orientation
        # The perpendicular direction (vessel tangent) is θ + π/2
        theta = 0.5 * np.arctan2(2.0 * j_xy, j_xx - j_yy + 1e-10)

        # Dominant eigenvector direction (perpendicular to vessel)
        # Rotate 90° to get vessel tangent
        tangent_y = -np.sin(theta).astype(np.float32)  # rotated by 90°
        tangent_x = np.cos(theta).astype(np.float32)

        orientation = np.stack([tangent_y, tangent_x], axis=-1)  # (H, W, 2)
        return orientation
