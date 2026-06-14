"""Gymnasium environment for retinal vessel tracing."""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .observation import ObservationBuilder
from .reward import RewardCalculator, RewardState


@dataclass
class EnvConfig:
    """Reference defaults for core env scalars; the live env reads ``config['environment']``."""

    observation_size: int = 65
    step_size: int = 1
    tolerance: float = 2.0
    max_off_track_streak: int = 5
    max_steps_per_episode: int = 2000
    use_vesselness: bool = False


class VesselTracingEnv(gym.Env):
    """Gym env where an agent walks one step at a time along a retinal vessel from a seed.

    Each step picks one of 8 directions (optionally tangent-relative) or STOP. Observation
    channels are derived entirely from UNet-predicted priors (no GT leakage); the GT
    centerline/DT feed only reward and coverage. Call ``set_data`` before stepping.
    """

    N_ACTIONS = 9  # 8 directional moves + STOP (index 8)
    STOP_ACTION = 8

    # Canonical 8-neighbour directions ("N"-relative frame). In tangent-relative mode
    # (default) step() rotates this grid so action 0 ("forward") follows the local tangent.
    DIRECTIONS = np.array(
        [
            [-1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
        ]
    )

    def __init__(self, config, image=None, centerline=None, distance_transform=None, vesselness=None, fov_mask=None):
        """Build action/observation spaces and reward/observation helpers, and zero episode state.

        Image data is optional here — call ``set_data`` before ``reset``.
        """
        super().__init__()

        self.config = config
        env_config = config.get('environment', {})

        self.obs_size = env_config.get('observation_size', 65)
        self.step_size = env_config.get('step_size', 1)
        self.tolerance = env_config.get('tolerance', 2.0)
        self.tangent_relative_actions = bool(env_config.get('tangent_relative_actions', True))
        self.max_off_track = env_config.get('max_off_track_streak', 3)
        self.max_steps = env_config.get('max_steps_per_episode', 2000)
        self._on_vessel_signal = env_config.get('on_vessel_signal', 'vesselness')
        self._vesselness_tau = float(env_config.get('vesselness_tau', 0.3))
        self.off_track_ramp = env_config.get('off_track_penalty_ramp', False)

        # Circular tolerance-disk template, reused every coverage update.
        tol_i = int(self.tolerance)
        r = np.arange(-tol_i - 1, tol_i + 2)
        self._cov_template = (r[:, None] ** 2 + r[None, :] ** 2) <= self.tolerance**2
        self._cov_half = tol_i + 1  # template half-size

        # Momentum blend: 0.0 = pure discrete steps, higher = smoother direction.
        self.momentum = env_config.get('momentum', 0.0)

        self.image = image
        self.centerline = centerline  # GT — reward path
        self.distance_transform = distance_transform  # GT — reward path
        self.vesselness = vesselness
        self.unet_prior = None  # (H, W) float32 in [0, 1], set in set_data()
        self.fov_mask = fov_mask

        self.vessel_orientation = None  # (H, W, 2), set in set_data()
        self.pred_centerline = None
        self.pred_distance_transform = None
        self.pred_dt_gradient = None
        self._offtrack_dt = None

        self.use_topology_memory = env_config.get('use_topology_memory', True)
        self._predicted_junction_pixels: list = []
        self._junction_neighbours: dict = {}
        self._visited_junctions: list = []

        if image is not None:
            self.height, self.width = image.shape[:2]
        else:
            self.height, self.width = 512, 512

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self._setup_observation_space()

        self.reward_calculator = RewardCalculator(config)
        self.observation_builder = ObservationBuilder(config)

        # Episode state.
        self.position = None
        self.visited_mask = None
        self.trajectory = None
        self.trajectory_mask = None  # all visited pixels (on + off vessel)
        self.step_count = 0
        self.off_track_streak = 0
        self.on_track_streak = 0
        self.prev_direction = None
        self._prev_world_vec: Optional[np.ndarray] = None
        self.covered_centerline = None
        self._covered_weight_sum = 0.0  # thickness-weighted coverage accumulator
        self.centerline_weight_map = None  # set in set_data()
        self.prior_coverage = None  # accumulated mask from earlier traces

        self._uncov_dt: Optional[np.ndarray] = None
        self._uncov_dt_refresh: int = int(env_config.get('uncov_dt_refresh_steps', 20))
        self._steps_since_uncov_refresh: int = 0
        self._shaping_uses_uncovered: bool = bool(config.get('reward', {}).get('shaping_uses_uncovered', False))

        self._progress_weight: float = float(config.get('reward', {}).get('progress_weight', 0.0))
        self._needs_uncov_dt: bool = self._shaping_uses_uncovered or self._progress_weight != 0.0

        self._frontier_mask = None
        self._momentum_vec: Optional[np.ndarray] = None  # running normalised direction

    def _setup_observation_space(self):
        """Define the Box observation space; channel count comes from the policy-net SSoT."""
        from models.policy_network import _compute_in_channels

        n_channels = _compute_in_channels(self.config)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(n_channels, self.obs_size, self.obs_size), dtype=np.float32)

    def set_data(
        self,
        image,
        centerline,
        distance_transform,
        vesselness=None,
        fov_mask=None,
        vessel_mask=None,
        vessel_orientation=None,
        dt_gradient=None,
        unet_prior=None,
        prior_coverage=None,
        pred_centerline=None,
        pred_distance_transform=None,
        pred_dt_gradient=None,
        vessel_width_px: Optional[float] = None,
        tolerance_px: Optional[float] = None,
    ):
        """Load one image and its derived maps for the next episode(s).

        Stores GT centerline/DT (reward path) and UNet-predicted priors (observation path),
        lazily computing vesselness, the UNet prior, and predicted priors when not supplied,
        then pre-stacks observation sources and the predicted-skeleton junction graph. Raises
        if predicted priors are needed but the seed-detector checkpoint is missing.
        ``vessel_width_px`` / ``tolerance_px`` are accepted but ignored (tolerance is absolute).
        """
        self.image = image
        # GT centerline / DT — reward + coverage paths only.
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.centerline_weight_map = self._build_centerline_weight_map(centerline, vessel_mask=None)
        env_cfg = self.config.get('environment', {})
        if vesselness is None and env_cfg.get('use_vesselness', False):
            from skimage.filters import frangi

            gray = image[:, :, 1] if image.ndim == 3 else image
            vesselness = frangi(gray.astype(np.float64), sigmas=np.linspace(1.0, 3.0, 5), black_ridges=True).astype(np.float32)
        self.vesselness = vesselness
        # Lazily compute the UNet prior once per sample (the predictor caches the model).
        if unet_prior is None and env_cfg.get('use_unet_prior', False):
            from data.dataloader import compute_unet_prior

            unet_prior = compute_unet_prior(image)
        self.unet_prior = unet_prior
        self.prior_coverage = prior_coverage  # accumulated mask from earlier traces
        self.fov_mask = fov_mask if fov_mask is not None else np.ones_like(centerline)
        self.height, self.width = image.shape[:2]

        self.vessel_orientation = vessel_orientation if vessel_orientation is not None else self.observation_builder.compute_vessel_orientation(image)

        if pred_centerline is None or pred_distance_transform is None or pred_dt_gradient is None:
            from data.dataloader import compute_predicted_priors

            bundle = compute_predicted_priors(image, self.tolerance)
            if bundle is None:
                raise RuntimeError(
                    'Predicted priors required but the seed-detector '
                    'checkpoint is missing. Train it via '
                    'scripts/train_seed_detector.py (writes '
                    'weights/seed_detector.pt) before running the RL agent. '
                    'The use_unet_prior flag does NOT make this optional — '
                    'the predicted-prior pipeline is always on post-P0 '
                    'GT-leakage removal.'
                )
            if pred_centerline is None:
                pred_centerline = bundle['centerline']
            if pred_distance_transform is None:
                pred_distance_transform = bundle['distance_transform']
            if pred_dt_gradient is None:
                pred_dt_gradient = bundle['dt_gradient']
            # Reuse the just-computed probability map for unet_prior — no second UNet pass.
            if self.unet_prior is None and env_cfg.get('use_unet_prior', False):
                self.unet_prior = bundle['unet_prior']

        self.pred_centerline = pred_centerline
        self.pred_distance_transform = pred_distance_transform
        self.pred_dt_gradient = pred_dt_gradient
        # Unclipped distance to the predicted ridge for leak-free off-track termination
        # (step()); computed once per sample.
        if pred_centerline is not None:
            from scipy.ndimage import distance_transform_edt

            self._offtrack_dt = distance_transform_edt(np.asarray(pred_centerline) <= 0).astype(np.float32)
        else:
            self._offtrack_dt = None
        # Alias for the predicted gradient so the legacy fallback in _get_observation stays
        # consistent with the non-leaking pipeline.
        self.dt_gradient = pred_dt_gradient

        self.observation_builder.prepare_stacked_sources(
            distance_transform=pred_distance_transform,
            dt_gradient=pred_dt_gradient,
            centerline=pred_centerline,
            vessel_orientation=self.vessel_orientation,
            unet_prior=self.unet_prior,
            vesselness=self.vesselness,
        )

        # Precompute predicted-skeleton junctions + their 8-neighbours once per sample so
        # per-step topology-memory lookups stay O(1).
        if self.use_topology_memory:
            self._predicted_junction_pixels = self._extract_junction_pixels(pred_centerline)
            self._junction_neighbours = {(y, x): self._skeleton_neighbours(pred_centerline, y, x) for (y, x) in self._predicted_junction_pixels}
        else:
            self._predicted_junction_pixels = []
            self._junction_neighbours = {}

    def reset(self, seed=None, start_position=None, **kwargs):
        """Start a new episode at ``start_position`` (or a sampled seed) and return ``(obs, info)``."""
        super().reset(seed=seed)

        if self.image is None:
            raise ValueError('No image data set. Call set_data() first.')

        self.visited_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectory_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectory = []
        self.step_count = 0
        self.off_track_streak = 0
        self.on_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = np.zeros_like(self.centerline, dtype=np.float32)
        self._covered_weight_sum = 0.0
        self._total_visited = 0
        self._total_visited_on_track = 0
        self._momentum_vec = None
        self._prev_world_vec = None
        self._visited_junctions = []

        if start_position is not None:
            self.position = np.array(start_position, dtype=np.int32)
        else:
            self.position = self._sample_start_position()

        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))
        self._update_coverage()
        self._update_frontier_mask()
        if self.use_topology_memory:
            self._maybe_register_junction()
        if self._needs_uncov_dt:
            self._refresh_uncov_dt()
        else:
            self._uncov_dt = None
        self._steps_since_uncov_refresh = 0

        return (self._get_observation(), self._get_info())

    def _refresh_uncov_dt(self) -> None:
        """Recompute distance to the nearest uncovered GT-centerline pixel.

        Uncovered = GT centerline ∧ ¬covered. When nothing is left, fills a large constant so
        min(D, τ) = τ everywhere and the shaping term cancels.
        """
        from scipy.ndimage import distance_transform_edt

        uncovered = (self.centerline > 0) & (self.covered_centerline == 0)
        if not uncovered.any():
            big = float(max(self.height, self.width))
            self._uncov_dt = np.full((self.height, self.width), big, dtype=np.float32)
            return
        # distance_transform_edt measures distance to the nearest zero, so invert the mask.
        self._uncov_dt = distance_transform_edt(~uncovered).astype(np.float32)

    def _compute_progress_cos(self, prev_pos, new_pos) -> Optional[float]:
        """Return the signed cosine between the step vector and the toward-uncovered tangent.

        The structure-tensor tangent at ``new_pos`` is sign-flipped (by probing uncov_dt at
        new_pos ± tangent) so positive means the agent committed toward uncovered work.
        Returns None when uncov_dt / vessel_orientation is unavailable or the step is zero.
        """
        if self._uncov_dt is None or self.vessel_orientation is None:
            return None
        dy = float(new_pos[0] - prev_pos[0])
        dx = float(new_pos[1] - prev_pos[1])
        step_mag = (dy * dy + dx * dx) ** 0.5
        if step_mag < 1e-6:
            return None
        y, x = int(new_pos[0]), int(new_pos[1])
        ty, tx = self.vessel_orientation[y, x]
        ty = float(ty)
        tx = float(tx)
        tmag = (ty * ty + tx * tx) ** 0.5
        if tmag < 1e-6:
            return 0.0  # degenerate tangent (e.g. low-contrast region)
        # Whichever sign of (ty, tx) lands on a smaller uncov_dt is toward-uncovered.
        h, w = self._uncov_dt.shape
        yp = max(0, min(h - 1, int(round(y + ty))))
        xp = max(0, min(w - 1, int(round(x + tx))))
        ym = max(0, min(h - 1, int(round(y - ty))))
        xm = max(0, min(w - 1, int(round(x - tx))))
        if self._uncov_dt[yp, xp] > self._uncov_dt[ym, xm]:
            ty, tx = -ty, -tx
        return (dy * ty + dx * tx) / (step_mag * tmag)

    def _sample_start_position(self) -> np.ndarray:
        """Pick an episode seed: ~50% GT endpoints, ~50% arbitrary centerline pixels.

        The mix matches the inference seed distribution so mid-vessel/junction starts aren't
        out-of-distribution. Falls back to an FOV pixel, then image centre, when no centerline.
        """
        centerline_points = np.argwhere(self.centerline > 0)
        if len(centerline_points) == 0:
            fov_points = np.argwhere(self.fov_mask > 0)
            if len(fov_points) == 0:
                return np.array([self.height // 2, self.width // 2])
            idx = self.np_random.integers(len(fov_points))
            return fov_points[idx]
        # ~50% endpoint starts (clean leaf-to-root traces), ~50% interior pixels (matches the
        # seed detector, which seeds segments and junctions, not just endpoints).
        from data.centerline_extraction import CenterlineExtractor

        extractor = CenterlineExtractor()
        endpoints = extractor._find_endpoints(self.centerline)
        if endpoints and self.np_random.random() < 0.5:
            idx = self.np_random.integers(len(endpoints))
            return np.array(endpoints[idx])
        idx = self.np_random.integers(len(centerline_points))
        return centerline_points[idx]

    def step(self, action: int):
        """Apply one action and return ``(obs, reward, terminated, truncated, info)``.

        Action 8 is STOP (scored by terminal F-β). Otherwise the agent moves one step_size
        step; the episode also ends on out-of-bounds, an off-track streak, or max-steps. One
        ``on_vessel`` decision (per ``on_vessel_signal``) drives both off-track termination
        and the reward's on/off-track gating.
        """
        self.step_count += 1

        # Explicit STOP action.
        if action == self.STOP_ACTION:
            f_beta = self._compute_fbeta()
            pos = np.array(self.position)
            dist = float(self.distance_transform[self.position[0], self.position[1]])
            cov_ratio = self.covered_centerline.sum() / max(self.centerline.sum(), 1.0)
            udist = float(self._uncov_dt[self.position[0], self.position[1]]) if self._uncov_dt is not None else None
            state = RewardState(
                is_terminal=True,
                terminal_reason='stop',
                new_coverage=0.0,
                is_on_track=dist <= self.tolerance,
                distance=dist,
                prev_distance=dist,
                coverage=cov_ratio,
                f_beta_score=f_beta,
                position=pos,
                step_number=self.step_count,
                junction_map_value=self._junction_val_at(self.position),
                uncovered_distance=udist,
                prev_uncovered_distance=udist,
            )
            reward, bd = self.reward_calculator.compute(state)
            info = self._get_info()
            info.update(bd)
            info['stopped'] = True
            info['episode_f1'] = float(f_beta)
            info['terminal_reason'] = 'stop'
            return (self._get_observation(), reward, True, False, info)

        # Movement.
        prev_pos = np.array(self.position)
        prev_distance = float(self.distance_transform[self.position[0], self.position[1]])
        # Snapshot uncovered-DT at the prev position before the coverage update, so the whole
        # step shapes against one frozen potential field (Ng & Russell consistency).
        prev_uncov_dist = float(self._uncov_dt[prev_pos[0], prev_pos[1]]) if self._uncov_dt is not None else None

        # Resolve the action to a world-frame displacement (rotated into the tangent frame
        # when tangent_relative_actions is on), then apply with optional momentum blending.
        raw_direction = self._action_to_world_displacement(action)

        if self.momentum > 0 and self._momentum_vec is not None:
            blended = (1.0 - self.momentum) * raw_direction + self.momentum * self._momentum_vec
            new_position = self.position + np.round(blended).astype(np.int32)
            if np.array_equal(new_position, self.position):
                new_position = self.position + np.round(raw_direction).astype(np.int32)
            self._momentum_vec = blended / (np.linalg.norm(blended) + 1e-8)
        else:
            new_position = self.position + np.round(raw_direction).astype(np.int32)
            norm = np.linalg.norm(raw_direction)
            self._momentum_vec = raw_direction / (norm + 1e-8) if norm > 0 else None

        # Cache the world-frame move direction for the next step's tangent sign alignment.
        rn = float(np.linalg.norm(raw_direction))
        if rn > 0:
            self._prev_world_vec = raw_direction / rn

        if not self._is_valid_position(new_position):
            state = RewardState(
                is_terminal=True,
                terminal_reason='oob',
                new_coverage=0.0,
                is_on_track=False,
                distance=float(self.distance_transform[prev_pos[0], prev_pos[1]]),
                prev_distance=prev_distance,
                coverage=self.covered_centerline.sum() / max(self.centerline.sum(), 1.0),
                f_beta_score=0.0,
                position=prev_pos,
                step_number=self.step_count,
                uncovered_distance=prev_uncov_dist,
                prev_uncovered_distance=prev_uncov_dist,
            )
            reward, bd = self.reward_calculator.compute(state)
            info = self._get_info()
            info.update(bd)
            info['terminal_reason'] = 'oob'
            return (self._get_observation(), reward, True, False, info)

        # Apply the move.
        self.position = new_position

        is_revisit = self.visited_mask[self.position[0], self.position[1]] > 0
        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))
        if self.use_topology_memory:
            self._maybe_register_junction()

        gt_distance = float(self.distance_transform[self.position[0], self.position[1]])
        # Unclipped predicted-ridge distance (leak-free) for centring and the
        # predicted_ridge on-vessel signal.
        pred_ridge_dist = float(self._offtrack_dt[self.position[0], self.position[1]]) if self._offtrack_dt is not None else gt_distance
        reward_prev_distance = float(self._offtrack_dt[prev_pos[0], prev_pos[1]]) if self._offtrack_dt is not None else prev_distance
        # Soft UNet vesselness at the current pixel (dense vessel evidence).
        vness = float(self.unet_prior[self.position[0], self.position[1]]) if self.unet_prior is not None else None

        if self._on_vessel_signal == 'vesselness' and vness is not None:
            on_vessel = vness >= self._vesselness_tau
        elif self._on_vessel_signal == 'gt':
            on_vessel = gt_distance <= self.tolerance
        else:  # "predicted_ridge" (also the vesselness fallback when unet_prior is absent)
            on_vessel = pred_ridge_dist <= self.tolerance

        # Gate reward on on_vessel, centre it on the predicted ridge.
        is_on_track = on_vessel
        reward_distance = pred_ridge_dist

        if not is_revisit:
            self._total_visited += 1
            if on_vessel:
                self._total_visited_on_track += 1

        if on_vessel:
            self.off_track_streak = 0
            self.on_track_streak += 1
        else:
            self.off_track_streak += 1
            self.on_track_streak = 0

        total_gt = max(float(self.centerline.sum()), 1.0)
        prev_coverage_sum = self.covered_centerline.sum()
        prev_coverage_ratio = prev_coverage_sum / total_gt
        prev_weighted_sum = self._covered_weight_sum

        is_on_frontier = bool(self._frontier_mask is not None and self._frontier_mask[self.position[0], self.position[1]])
        self._update_coverage()
        self._update_frontier_mask()
        new_coverage = self._covered_weight_sum - prev_weighted_sum
        current_coverage_ratio = self.covered_centerline.sum() / total_gt
        junction_val = self._junction_val_at(self.position)

        # Allow a longer off-track streak at junctions so the agent can probe a branch.
        effective_off_track = self.max_off_track * 2 if junction_val >= 0.8 else self.max_off_track
        terminated = self.off_track_streak >= effective_off_track
        truncated = self.step_count >= self.max_steps

        terminal_reason = ''
        f_beta = 0.0
        if terminated or truncated:
            terminal_reason = 'off_track' if terminated else 'max_steps'
            f_beta = self._compute_fbeta()

        new_uncov_dist = float(self._uncov_dt[self.position[0], self.position[1]]) if self._uncov_dt is not None else None

        progress_cos = self._compute_progress_cos(prev_pos, self.position)

        state = RewardState(
            is_terminal=terminated or truncated,
            terminal_reason=terminal_reason,
            new_coverage=new_coverage,
            is_on_track=is_on_track,
            distance=reward_distance,
            prev_distance=reward_prev_distance,
            coverage=current_coverage_ratio,
            f_beta_score=f_beta,
            position=np.array(self.position),
            step_number=self.step_count,
            junction_map_value=junction_val,
            is_revisit=is_revisit,
            is_on_frontier=is_on_frontier,
            uncovered_distance=new_uncov_dist,
            prev_uncovered_distance=prev_uncov_dist,
            progress_cos=progress_cos,
        )

        reward, bd = self.reward_calculator.compute(state)
        # Refresh _uncov_dt periodically to track coverage growth — after reward so this
        # step used a consistent potential field.
        if self._needs_uncov_dt and self._uncov_dt is not None:
            self._steps_since_uncov_refresh += 1
            if self._steps_since_uncov_refresh >= self._uncov_dt_refresh:
                self._refresh_uncov_dt()
                self._steps_since_uncov_refresh = 0
        self.prev_direction = action

        info = self._get_info()
        info.update(bd)
        if terminated or truncated:
            info['episode_f1'] = float(f_beta)
            info['terminal_reason'] = terminal_reason

        return (self._get_observation(), reward, terminated, truncated, info)

    def _action_to_world_displacement(self, action: int) -> np.ndarray:
        """Translate a discrete action into a world-frame displacement scaled by ``step_size``.

        With ``tangent_relative_actions`` (default), ``DIRECTIONS[action]`` is rotated so
        action 0 points along the local vessel tangent; otherwise it executes in pure world
        frame (action 0 = canonical N), matching the imitation expert's frame.
        """
        base = self.DIRECTIONS[action].astype(np.float64) * self.step_size
        if not self.tangent_relative_actions:
            return base
        ty, tx = self._tangent_aligned_at(int(self.position[0]), int(self.position[1]), reference=self._prev_world_vec)
        # Rotation R with R @ (-1, 0) = (ty, tx) maps the canonical frame to the tangent frame.
        dy, dx = float(base[0]), float(base[1])
        new_dy = -ty * dy + tx * dx
        new_dx = -tx * dy - ty * dx
        return np.array([new_dy, new_dx], dtype=np.float64)

    def _tangent_aligned_at(self, y: int, x: int, reference: Optional[np.ndarray] = None) -> tuple:
        """Return the local vessel tangent at (y, x), sign-aligned to ``reference``.

        The structure-tensor tangent is an undirected orientation (arbitrary sign); the first
        step aligns to image-up, later steps to the cached previous move. Falls back to the
        reference axis on a degenerate (zero-magnitude) tangent.
        """
        ty, tx = self.vessel_orientation[y, x]
        ty = float(ty)
        tx = float(tx)
        if reference is None:
            ref_y, ref_x = (-1.0, 0.0)  # canonical "N" / image-up
        else:
            ref_y, ref_x = (float(reference[0]), float(reference[1]))
        if ty * ref_y + tx * ref_x < 0.0:
            ty, tx = -ty, -tx
        mag = (ty * ty + tx * tx) ** 0.5
        if mag < 1e-6:
            return -1.0, 0.0
        return ty / mag, tx / mag

    @staticmethod
    def _extract_junction_pixels(skeleton: np.ndarray) -> list:
        """Return skeleton pixels with >= 3 skeleton 8-neighbours (junctions)."""
        from data.centerline_extraction import CenterlineExtractor

        return CenterlineExtractor()._find_junctions(skeleton)

    @staticmethod
    def _skeleton_neighbours(skeleton: np.ndarray, y: int, x: int) -> list:
        """Return the skeleton 8-neighbours of (y, x) (candidate branch entries at a junction)."""
        H, W = skeleton.shape
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skeleton[ny, nx] > 0:
                    out.append((ny, nx))
        return out

    def _maybe_register_junction(self, radius: int = 3) -> None:
        """Record the nearest predicted-skeleton junction within ``radius`` as most-recently visited."""
        if not self._predicted_junction_pixels:
            return
        y, x = (int(self.position[0]), int(self.position[1]))
        for jy, jx in self._predicted_junction_pixels:
            if abs(jy - y) <= radius and abs(jx - x) <= radius:
                key = (jy, jx)
                if key not in self._visited_junctions:
                    self._visited_junctions.append(key)
                # Promote to most-recent so topology channels track the active junction.
                elif self._visited_junctions[-1] != key:
                    self._visited_junctions.remove(key)
                    self._visited_junctions.append(key)
                break

    def _topology_features(self) -> tuple:
        """Return ``(normalised_distance, branches_remaining)`` topology-memory scalars.

        ``normalised_distance``: distance to the most recent junction / obs_size, clipped to
        [0, 1] (1.0 if none visited). ``branches_remaining``: that junction's unvisited
        skeleton-neighbour count / 8 (0.0 if none visited).
        """
        if not self.use_topology_memory or not self._visited_junctions:
            return 1.0, 0.0
        ly, lx = self._visited_junctions[-1]
        py, px = (float(self.position[0]), float(self.position[1]))
        dist = ((py - ly) ** 2 + (px - lx) ** 2) ** 0.5
        normalised = min(1.0, dist / max(float(self.obs_size), 1.0))
        nbrs = self._junction_neighbours.get((ly, lx), [])
        if not nbrs:
            return normalised, 0.0
        unvisited = sum(1 for (ny, nx) in nbrs if self.visited_mask[ny, nx] == 0.0)
        return normalised, unvisited / 8.0

    def _compute_fbeta(self) -> float:
        """Return the current episode's MARGINAL F-β contribution.

        ``f_beta(prior ∪ current) − f_beta(prior)`` when prior coverage exists, else plain
        ``f_beta(current)``. Marginal (not cumulative) so a STOP-immediately episode that adds
        nothing earns zero terminal reward — closing the cumulative "free-ride" loophole that
        previously let STOP-fast become optimal after the GT-DT signal was removed.
        """
        if self.prior_coverage is not None:
            cumulative = np.where((self.covered_centerline > 0) | (self.prior_coverage > 0), 1.0, 0.0).astype(np.float32)
            return self._fbeta_on(cumulative) - self._fbeta_on(self.prior_coverage)
        return self._fbeta_on(self.covered_centerline)

    def _fbeta_on(self, covered_mask: np.ndarray) -> float:
        """Return the tolerance-aware β-weighted F-score between ``covered_mask`` and the GT centerline.

        β² (``config['reward']['terminal_recall_beta_sq']``, default 4) weights recall over
        precision. Returns 0.0 for an empty mask so it can baseline marginal computations.
        """
        if covered_mask is None or not np.any(covered_mask):
            return 0.0
        from data.centerline_extraction import compute_centerline_f1

        rc = self.config.get('reward', {})
        beta_sq = float(rc.get('terminal_recall_beta_sq', 4.0))
        metrics = compute_centerline_f1(covered_mask, self.centerline, tolerance=self.tolerance)
        recall = metrics['recall']
        precision = metrics['precision']
        denom = beta_sq * precision + recall
        return (1.0 + beta_sq) * precision * recall / denom if denom > 0 else 0.0

    def _junction_val_at(self, position) -> float:
        """Return the junction-map value at ``position`` (0.0 if the map isn't built)."""
        if self.observation_builder.junction_map is not None:
            return float(self.observation_builder.junction_map[position[0], position[1]])
        return 0.0

    def _is_valid_position(self, position):
        """Return True if ``position`` is inside the FOV and leaves a full observation-window margin."""
        y, x = position
        half = self.obs_size // 2
        if y < half or y >= self.height - half:
            return False
        if x < half or x >= self.width - half:
            return False
        if self.fov_mask[y, x] == 0:
            return False
        return True

    def _build_centerline_weight_map(self, centerline, vessel_mask):
        """Build per-pixel thickness weights for centerline pixels (thin vessels weighted up).

        Each centerline pixel's weight is clip(1/sqrt(local_width / median_width), 0.7, 1.6),
        renormalised to mean 1.0 so total reward scale is preserved; off-centerline pixels get
        0. Falls back to uniform weight 1.0 when ``vessel_mask`` is unavailable.
        """
        H, W = centerline.shape
        weight_map = np.zeros((H, W), dtype=np.float32)
        cl_mask = centerline > 0
        if not cl_mask.any():
            return weight_map
        if vessel_mask is None:
            weight_map[cl_mask] = 1.0
            return weight_map

        from scipy.ndimage import distance_transform_edt

        vbin = (vessel_mask > 0).astype(np.uint8)
        inward = distance_transform_edt(vbin).astype(np.float32)
        local_width = 2.0 * np.maximum(inward[cl_mask], 1.0)
        if local_width.size == 0:
            return weight_map
        w_ref = float(np.median(local_width))
        if w_ref < 1.0:
            w_ref = 1.0
        raw_weight = 1.0 / np.sqrt(local_width / w_ref)
        raw_weight = np.clip(raw_weight, 0.7, 1.6)
        # Renormalise to mean 1.0 so clipping doesn't change total episode-reward scale.
        mean_w = float(raw_weight.mean())
        if mean_w > 0:
            raw_weight /= mean_w
        weight_map[cl_mask] = raw_weight.astype(np.float32)
        return weight_map

    def _update_coverage(self):
        """Mark GT centerline pixels within the tolerance disk of the position as covered.

        Updates both the binary ``covered_centerline`` (coverage ratio / F-β) and the
        thickness-weighted accumulator the reward's coverage term reads.
        """
        y, x = self.position
        h = self._cov_half

        y_min = max(0, y - h)
        y_max = min(self.height, y + h + 1)
        x_min = max(0, x - h)
        x_max = min(self.width, x + h + 1)

        patch = self.centerline[y_min:y_max, x_min:x_max]
        if not patch.any():
            return

        # Slice the disk template to the same boundary clipping as the image patch.
        ty_min = y_min - (y - h)
        ty_max = ty_min + (y_max - y_min)
        tx_min = x_min - (x - h)
        tx_max = tx_min + (x_max - x_min)
        within = self._cov_template[ty_min:ty_max, tx_min:tx_max]

        prev_patch = self.covered_centerline[y_min:y_max, x_min:x_max]
        newly_covered = within & (patch > 0) & (prev_patch == 0)

        self.covered_centerline[y_min:y_max, x_min:x_max] = np.where(within & (patch > 0), 1.0, prev_patch)

        # Each newly-covered pixel adds its thickness weight; the reward reads this
        # accumulator's delta so thin-vessel coverage earns ~1.6× thick-trunk coverage.
        if newly_covered.any():
            weight_patch = self.centerline_weight_map[y_min:y_max, x_min:x_max]
            self._covered_weight_sum += float(weight_patch[newly_covered].sum())

    def _update_frontier_mask(self):
        """Recompute the Stage B1 frontier: unvisited GT centerline pixels one step from a visited pixel.

        ``dilate(visited_mask, step_size) ∧ centerline ∧ ¬visited_mask``. Defined on the
        visit history (not the tolerance-disk coverage, inside which the agent always sits)
        and dilated by ``step_size`` so the band is exactly where the agent can land next.
        """
        from scipy.ndimage import binary_dilation

        vis_bool = self.visited_mask > 0
        if not vis_bool.any():
            self._frontier_mask = np.zeros_like(self.centerline, dtype=bool)
            return
        dilated = binary_dilation(vis_bool, structure=np.ones((3, 3), dtype=bool), iterations=max(1, int(round(self.step_size))))
        self._frontier_mask = dilated & (self.centerline > 0) & (~vis_bool)

    def _get_observation(self):
        """Build the current observation from predicted priors + agent state."""
        return self.observation_builder.build(
            image=self.image,
            visited_mask=self.visited_mask,
            vesselness=self.vesselness,
            position=self.position,
            prev_direction=self.prev_direction,
            # Observation uses PREDICTED priors; GT skeleton/DT stay on self.centerline /
            # self.distance_transform for the reward path only.
            distance_transform=self.pred_distance_transform,
            centerline=self.pred_centerline,
            vessel_orientation=self.vessel_orientation,
            dt_gradient=self.pred_dt_gradient,
            unet_prior=self.unet_prior,
            prior_coverage=self.prior_coverage,
            covered_centerline=self.covered_centerline,
            topology_features=(self._topology_features() if self.use_topology_memory else None),
        )

    def _get_info(self):
        """Assemble the per-step info dict (position, coverage ratio, precision, …)."""
        total = self.centerline.sum()
        covered = self.covered_centerline.sum()
        info = {
            'position': tuple(self.position),
            'step_count': self.step_count,
            'trajectory_length': len(self.trajectory),
            'off_track_streak': self.off_track_streak,
            'coverage_ratio': covered / max(total, 1),
            'covered_pixels': int(covered),
            'total_centerline_pixels': int(total),
        }
        # Precision: fraction of unique visited positions that were on-track.
        if self._total_visited > 0:
            info['precision'] = self._total_visited_on_track / self._total_visited
        else:
            info['precision'] = 0.0
        return info

    def render(self):
        """Return an RGB debug image: GT centerline (red), covered (green), trajectory (blue), agent (yellow)."""
        vis = (self.image.copy() * 255).astype(np.uint8)
        vis[self.centerline > 0] = [0, 0, 255]
        vis[self.covered_centerline > 0] = [0, 255, 0]
        for y, x in self.trajectory:
            vis[max(0, y - 1) : min(self.height, y + 2), max(0, x - 1) : min(self.width, x + 2)] = [255, 0, 0]
        y, x = self.position
        vis[max(0, y - 2) : min(self.height, y + 3), max(0, x - 2) : min(self.width, x + 3)] = [255, 255, 0]
        return vis


class VectorizedVesselEnv:
    """In-process vectorized wrapper running several VesselTracingEnv instances for training."""

    def __init__(self, config, num_envs=8, dataset=None):
        """Create ``num_envs`` in-process VesselTracingEnv instances over ``dataset``."""
        self.config = config
        self.num_envs = num_envs
        self.dataset = dataset
        self.envs = [VesselTracingEnv(config) for _ in range(num_envs)]
        self.current_samples = [None] * num_envs

    def _apply_sample(self, env, sample):
        """Unpack a dataset sample and call ``env.set_data()``."""
        env.set_data(
            image=sample['image'].permute(1, 2, 0).numpy(),
            centerline=sample['centerline'].squeeze().numpy(),
            distance_transform=sample['distance_transform'].squeeze().numpy(),
            fov_mask=sample['fov_mask'].squeeze().numpy(),
            vessel_mask=(sample['vessel_mask'].squeeze().numpy() if 'vessel_mask' in sample else None),
            vessel_orientation=(sample['vessel_orientation'].numpy() if 'vessel_orientation' in sample else None),
            unet_prior=(sample['unet_prior'].squeeze(0).numpy() if 'unet_prior' in sample else None),
            pred_centerline=(sample['pred_centerline'].squeeze().numpy() if 'pred_centerline' in sample else None),
            pred_distance_transform=(sample['pred_distance_transform'].squeeze().numpy() if 'pred_distance_transform' in sample else None),
            pred_dt_gradient=(sample['pred_dt_gradient'].numpy() if 'pred_dt_gradient' in sample else None),
        )

    def reset(self):
        """Load a random sample into every env, reset them, and return stacked obs + infos."""
        observations, infos = [], []
        for i, env in enumerate(self.envs):
            sample = self._get_random_sample()
            self.current_samples[i] = sample
            self._apply_sample(env, sample)
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(self, actions):
        """Step every env, auto-resetting finished ones with a fresh random sample.

        A finished env's final observation is exposed under ``info['terminal_observation']``.
        """
        (observations, rewards, terminateds, truncateds, infos) = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            (obs, reward, terminated, truncated, info) = env.step(action)
            if terminated or truncated:
                sample = self._get_random_sample()
                self.current_samples[i] = sample
                self._apply_sample(env, sample)
                obs, _ = env.reset()
                info['terminal_observation'] = obs
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return (np.stack(observations), np.array(rewards), np.array(terminateds), np.array(truncateds), infos)

    def _get_random_sample(self):
        """Draw a uniformly random sample from the dataset."""
        idx = np.random.randint(len(self.dataset))
        return self.dataset[idx]
