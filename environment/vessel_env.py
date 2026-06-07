"""RL Environment for vessel tracing."""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .observation import ObservationBuilder
from .reward import RewardCalculator, RewardState


@dataclass
class EnvConfig:
    observation_size: int = 65
    step_size: int = 1
    tolerance: float = 2.0
    max_off_track_streak: int = 5
    max_steps_per_episode: int = 2000
    use_vesselness: bool = False


class VesselTracingEnv(gym.Env):
    N_ACTIONS = 9  # 8 directional moves + STOP (index 8)
    STOP_ACTION = 8

    # Standard 8-neighbour direction set (canonical "N"-relative frame).
    # In tangent-relative mode (the default), step() rotates this grid so
    # action 0 ("forward") points along the local vessel tangent rather
    # than absolute image-up — see _rotate_into_tangent_frame.
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

    def __init__(
        self,
        config,
        image=None,
        centerline=None,
        distance_transform=None,
        vesselness=None,
        fov_mask=None,
    ):
        super().__init__()

        self.config = config
        env_config = config.get('environment', {})

        self.obs_size = env_config.get('observation_size', 65)
        self.step_size = env_config.get('step_size', 1)
        self.tolerance = env_config.get('tolerance', 2.0)
        # Whether DIRECTIONS[action] is rotated by the local vessel tangent
        # in _action_to_world_displacement. The imitation expert in
        # training/imitation.py generates action indices in WORLD frame
        # (direction_to_action(world_dy, world_dx)); if this flag is True
        # the env rotates them, creating a frame mismatch between the
        # imitation prior and PPO execution. Set False to make actions
        # world-frame (action 0 = canonical N regardless of tangent) so
        # imitation and env agree.
        self.tangent_relative_actions = bool(env_config.get('tangent_relative_actions', True))
        self.max_off_track = env_config.get('max_off_track_streak', 3)
        self.max_steps = env_config.get('max_steps_per_episode', 2000)
        # v10 — which signal decides "on a vessel" (drives BOTH off-track
        # termination AND the reward's off-vessel/near/progress gating, so the
        # two never conflict). All leak-free except "gt".
        #   "vesselness"     — soft UNet vessel-prob >= vesselness_tau (dense;
        #                      spans gaps in the thresholded skeleton → lifts the
        #                      recall/connectivity ceiling).
        #   "predicted_ridge"— distance to the predicted centerline <= tolerance.
        #   "gt"             — GT distance <= tolerance (NOT leak-free at inference).
        self._on_vessel_signal = env_config.get('on_vessel_signal', 'vesselness')
        self._vesselness_tau = float(env_config.get('vesselness_tau', 0.3))
        # Soft off-track tolerance — when True the per-step off-track
        # penalty ramps linearly with the streak instead of a flat penalty.
        # Toggled dynamically per curriculum stage via apply_overrides.
        self.off_track_ramp = env_config.get('off_track_penalty_ramp', False)

        # Precompute circular coverage template (reused every step)
        tol_i = int(self.tolerance)
        r = np.arange(-tol_i - 1, tol_i + 2)
        self._cov_template = (r[:, None] ** 2 + r[None, :] ** 2) <= self.tolerance**2
        self._cov_half = tol_i + 1  # half-size of template

        # Momentum blending
        self.momentum = env_config.get('momentum', 0.0)
        # 0.0 = no momentum (pure discrete), 0.3 = mild smoothing

        self.image = image
        self.centerline = centerline  # GT — reward path
        self.distance_transform = distance_transform  # GT — reward path
        self.vesselness = vesselness
        self.unet_prior = None  # (H, W) float32 in [0, 1], set in set_data()
        self.fov_mask = fov_mask

        self.vessel_orientation = None  # precomputed (H,W,2), set in set_data()
        # Predicted priors (UNet → skeleton → DT → DT-grad) — observation path.
        # GT versions live on self.centerline / self.distance_transform; these
        # parallel attributes feed ObservationBuilder so the agent never sees
        # the GT skeleton during training or inference.
        self.pred_centerline = None
        self.pred_distance_transform = None
        self.pred_dt_gradient = None
        # Unclipped distance to the PREDICTED centerline ridge. Drives off-track
        # TERMINATION without GT (the env's pred_distance_transform is clipped
        # at tolerance and so cannot distinguish on- vs far-off-track).
        self._offtrack_dt = None

        # Topology-aware memory (P1b). Per-episode graph of visited junctions
        # on the *predicted* skeleton, plus precomputed neighbour lists so the
        # "branches-remaining" feature is O(1) per step.
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

        # Episode state
        self.position = None
        self.visited_mask = None
        self.trajectory = None
        self.trajectory_mask = None  # all visited pixels (on + off vessel)
        self.step_count = 0
        self.off_track_streak = 0
        self.on_track_streak = 0
        self.prev_direction = None
        # World-frame unit displacement of the previous move. Disambiguates
        # the structure-tensor tangent's sign at the next step so "forward"
        # stays consistent across a trace.
        self._prev_world_vec: Optional[np.ndarray] = None
        self.covered_centerline = None
        self._covered_weight_sum = 0.0  # thickness-weighted accumulator
        self.centerline_weight_map = None  # set in set_data()
        self.prior_coverage = None  # accumulated mask from earlier traces
        # F3 — uncovered-DT shaping. Distance transform to the nearest GT
        # centerline pixel that is NOT yet in covered_centerline. Refreshed
        # every ``_uncov_dt_refresh`` steps (full-image DT is O(HW)
        # but cheap; refreshing per-step would be wasteful). Used only when
        # config['reward']['shaping_uses_uncovered'] is True.
        self._uncov_dt: Optional[np.ndarray] = None
        self._uncov_dt_refresh: int = int(env_config.get('uncov_dt_refresh_steps', 20))
        self._steps_since_uncov_refresh: int = 0
        self._shaping_uses_uncovered: bool = bool(config.get('reward', {}).get('shaping_uses_uncovered', False))
        # H6 — tangent-aligned progress reward needs _uncov_dt too (the
        # sign of forward_tangent is set by the uncov_dt gradient). So
        # populate _uncov_dt whenever EITHER feature is on, not just F3.
        self._progress_weight: float = float(config.get('reward', {}).get('progress_weight', 0.0))
        self._needs_uncov_dt: bool = self._shaping_uses_uncovered or self._progress_weight != 0.0
        # Stage B1 frontier mask: uncovered GT centerline pixels adjacent
        # (8-conn) to the currently-covered centerline.  Recomputed after
        # every coverage update so the next step's `is_on_frontier` query
        # reflects the latest coverage frontier.
        self._frontier_mask = None
        self._momentum_vec: Optional[np.ndarray] = None  # running direction

    def _setup_observation_space(self):
        from models.policy_network import (
            _compute_in_channels,
        )

        n_channels = _compute_in_channels(self.config)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                n_channels,
                self.obs_size,
                self.obs_size,
            ),
            dtype=np.float32,
        )

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
        # Accepted but ignored — kept so callers built against the
        # transitional width-scaling API don't break. Tolerance is
        # always the absolute value from config.
        vessel_width_px: Optional[float] = None,
        tolerance_px: Optional[float] = None,
    ):
        self.image = image
        # GT centerline / DT — used by reward + coverage paths only.
        self.centerline = centerline
        self.distance_transform = distance_transform
        # §2.1 reverted: thickness-weighted coverage caused a "stop-fast"
        # collapse (1000-iter run: ep_len shrank 43→14, stop_frac→0.97,
        # clDice 0.461→0.414, recall@2 0.589→0.460).  The reverse map is now
        # uniform (weight=1.0 at every centerline pixel) so the reward path
        # behaves identically to the pre-§2.1 baseline.  Helper retained but
        # called with vessel_mask=None to enforce uniformity.
        self.centerline_weight_map = self._build_centerline_weight_map(centerline, vessel_mask=None)
        # Lazily compute Frangi vesselness if it's enabled in config but the
        # caller didn't provide it. The dataloader doesn't supply this field,
        # so this keeps the env self-sufficient. One frangi() per sample load.
        env_cfg = self.config.get('environment', {})
        if vesselness is None and env_cfg.get('use_vesselness', False):
            from skimage.filters import frangi

            gray = image[:, :, 1] if image.ndim == 3 else image
            vesselness = frangi(
                gray.astype(np.float64),
                sigmas=np.linspace(1.0, 3.0, 5),
                black_ridges=True,
            ).astype(np.float32)
        self.vesselness = vesselness
        # Lazily compute UNet prior if enabled but not supplied. Called once
        # per sample load — the predictor caches the model itself.
        if unet_prior is None and env_cfg.get('use_unet_prior', False):
            from data.dataloader import (
                compute_unet_prior,
            )

            unet_prior = compute_unet_prior(image)
        self.unet_prior = unet_prior
        self.prior_coverage = prior_coverage  # accumulated mask from earlier traces
        self.fov_mask = fov_mask if fov_mask is not None else np.ones_like(centerline)
        self.height, self.width = image.shape[:2]

        # Use precomputed if provided, else fall back to computing
        self.vessel_orientation = (
            vessel_orientation
            if vessel_orientation is not None
            else self.observation_builder.compute_vessel_orientation(image)
        )

        # Predicted priors — single source of truth for the agent's
        # observation channels. Lazy-compute via the frozen UNet when the
        # dataloader didn't pre-supply them. Hard-fail if the UNet is
        # unavailable; falling back to GT here would silently re-introduce
        # the leakage this code path exists to prevent.
        if pred_centerline is None or pred_distance_transform is None or pred_dt_gradient is None:
            from data.dataloader import (
                compute_predicted_priors,
            )

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
            # If unet_prior was requested but not supplied separately, reuse
            # the probability map we just computed — no second UNet pass.
            if self.unet_prior is None and env_cfg.get('use_unet_prior', False):
                self.unet_prior = bundle['unet_prior']

        self.pred_centerline = pred_centerline
        self.pred_distance_transform = pred_distance_transform
        self.pred_dt_gradient = pred_dt_gradient
        # Unclipped distance to the predicted ridge for leak-free off-track
        # termination (see step()). Computed once per sample.
        if pred_centerline is not None:
            from scipy.ndimage import (
                distance_transform_edt,
            )

            self._offtrack_dt = distance_transform_edt(np.asarray(pred_centerline) <= 0).astype(np.float32)
        else:
            self._offtrack_dt = None
        # ``self.dt_gradient`` retained as an alias for the predicted gradient
        # so the legacy fallback path in ``_get_observation`` (used when
        # ``prepare_stacked_sources`` was not called) stays consistent with
        # the non-leaking pipeline.
        self.dt_gradient = pred_dt_gradient

        self.observation_builder.prepare_stacked_sources(
            distance_transform=pred_distance_transform,
            dt_gradient=pred_dt_gradient,
            centerline=pred_centerline,
            vessel_orientation=self.vessel_orientation,
            unet_prior=self.unet_prior,
            vesselness=self.vesselness,
        )

        # Precompute predicted-skeleton junction pixels + their skeleton
        # 8-neighbours for the topology-memory channels. Done once per
        # episode so per-step lookup stays O(1).
        if self.use_topology_memory:
            self._predicted_junction_pixels = self._extract_junction_pixels(pred_centerline)
            self._junction_neighbours = {
                (y, x): self._skeleton_neighbours(pred_centerline, y, x)
                for (
                    y,
                    x,
                ) in self._predicted_junction_pixels
            }
        else:
            self._predicted_junction_pixels = []
            self._junction_neighbours = {}

    def reset(
        self,
        seed=None,
        start_position=None,
        **kwargs,
    ):
        super().reset(seed=seed)

        if self.image is None:
            raise ValueError('No image data set. Call set_data() first.')

        self.visited_mask = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )
        self.trajectory_mask = np.zeros(
            (self.height, self.width),
            dtype=np.float32,
        )
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
        # F3 / H6 — initialise uncovered-centerline DT. Used by F3 shaping
        # (potential field) and H6 progress (tangent-sign disambiguation).
        # Refreshed every _uncov_dt_refresh steps in step().
        if self._needs_uncov_dt:
            self._refresh_uncov_dt()
        else:
            self._uncov_dt = None
        self._steps_since_uncov_refresh = 0

        return (
            self._get_observation(),
            self._get_info(),
        )

    def _refresh_uncov_dt(self) -> None:
        """Recompute distance-to-nearest-uncovered-centerline-pixel.

        Uncovered = GT centerline ∧ ¬covered_centerline. When everything is
        covered, returns a large constant so the potential function stays
        well-defined (shaping reward goes to ~0).
        """
        from scipy.ndimage import (
            distance_transform_edt,
        )

        uncovered = (self.centerline > 0) & (self.covered_centerline == 0)
        if not uncovered.any():
            # Nothing left to cover — set DT to a large constant so
            # min(D, τ) = τ everywhere and the shaping term cancels out.
            big = float(max(self.height, self.width))
            self._uncov_dt = np.full(
                (self.height, self.width),
                big,
                dtype=np.float32,
            )
            return
        # distance_transform_edt expects 1 = background, 0 = foreground;
        # the result at each pixel is the distance to the nearest
        # foreground (uncovered) pixel.
        self._uncov_dt = distance_transform_edt(~uncovered).astype(np.float32)

    def _compute_progress_cos(self, prev_pos, new_pos) -> Optional[float]:
        """Signed cosine between the agent's step vector and the local
        forward-tangent at its new position.

        forward_tangent = ``vessel_orientation[new_pos]`` (same field
        ``_action_to_world_displacement`` uses for tangent-relative actions,
        so "forward" here means the same thing action 0 means: world-frame
        displacement when the agent takes the canonical-forward action).
        We sign-flip the tangent if moving in its direction would
        increase distance to uncovered work, so progress is positive iff
        the agent committed to the toward-uncovered direction.

        Returns ``None`` when uncov_dt / vessel_orientation isn't available
        or the step vector is zero — callers treat None as no signal.
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
            return 0.0  # Degenerate tangent (e.g. low-contrast region)
        # Sign-align by probing uncov_dt at new_pos ± (ty, tx). This is
        # convention-agnostic w.r.t. how vessel_orientation labels its
        # components, because we use the tangent the same way the action
        # rotation does: as a displacement. Whichever sign of (ty, tx)
        # lands on a smaller uncov_dt is the toward-uncovered direction.
        h, w = self._uncov_dt.shape
        yp = max(0, min(h - 1, int(round(y + ty))))
        xp = max(0, min(w - 1, int(round(x + tx))))
        ym = max(0, min(h - 1, int(round(y - ty))))
        xm = max(0, min(w - 1, int(round(x - tx))))
        if self._uncov_dt[yp, xp] > self._uncov_dt[ym, xm]:
            ty, tx = -ty, -tx
        return (dy * ty + dx * tx) / (step_mag * tmag)

    def _sample_start_position(
        self,
    ) -> np.ndarray:
        centerline_points = np.argwhere(self.centerline > 0)
        if len(centerline_points) == 0:
            fov_points = np.argwhere(self.fov_mask > 0)
            if len(fov_points) == 0:
                return np.array(
                    [
                        self.height // 2,
                        self.width // 2,
                    ]
                )
            idx = self.np_random.integers(len(fov_points))
            return fov_points[idx]
        # Train/inference seed-distribution match.  Training previously
        # ALWAYS started at a centerline endpoint, but the seed detector
        # (frontier_tracer) seeds anywhere on the tree — interior segments
        # and junctions included.  A policy that only ever saw endpoint
        # starts treats a mid-vessel/junction seed as out-of-distribution
        # and (post-collapse) just emits STOP → empty trace at a valid
        # vessel pixel, exactly the observed failure.  Sample a mix:
        # ~50% endpoint starts (clean leaf-to-root traces, good for the
        # imitation-aligned behaviour) and ~50% arbitrary interior
        # centerline pixels (matches inference seeds).
        from data.centerline_extraction import (
            CenterlineExtractor,
        )

        extractor = CenterlineExtractor()
        endpoints = extractor._find_endpoints(self.centerline)
        if endpoints and self.np_random.random() < 0.5:
            idx = self.np_random.integers(len(endpoints))
            return np.array(endpoints[idx])
        idx = self.np_random.integers(len(centerline_points))
        return centerline_points[idx]

    def step(self, action: int):
        self.step_count += 1

        # ─── Explicit STOP action ─────────────────────────────────────────
        if action == self.STOP_ACTION:
            f_beta = self._compute_fbeta()
            pos = np.array(self.position)
            dist = float(
                self.distance_transform[
                    self.position[0],
                    self.position[1],
                ]
            )
            cov_ratio = self.covered_centerline.sum() / max(self.centerline.sum(), 1.0)
            udist = (
                float(
                    self._uncov_dt[
                        self.position[0],
                        self.position[1],
                    ]
                )
                if self._uncov_dt is not None
                else None
            )
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
            return (
                self._get_observation(),
                reward,
                True,
                False,
                info,
            )

        # ─── Movement ────────────────────────────────────────────────────────
        prev_pos = np.array(self.position)
        prev_distance = float(self.distance_transform[self.position[0], self.position[1]])
        # F3 — snapshot uncovered-DT at PREV position with the current
        # _uncov_dt; refresh after coverage update so within-step shaping
        # uses a single frozen potential field (Ng & Russell consistency).
        prev_uncov_dist = float(self._uncov_dt[prev_pos[0], prev_pos[1]]) if self._uncov_dt is not None else None

        # Tangent-relative action: rotate the canonical 8-dir grid so action
        # 0 ("forward") points along the local vessel tangent, with sign
        # disambiguated by the previous move. The resulting world-frame
        # displacement is then applied as a normal step (with optional
        # momentum blending).
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

        # Cache the world-frame movement direction for the *next* step's
        # tangent sign alignment.
        rn = float(np.linalg.norm(raw_direction))
        if rn > 0:
            self._prev_world_vec = raw_direction / rn

        # ─── Out of bounds ───────────────────────────────────────────────────
        # v11 REVERTED: crediting on-vessel boundary exits with their F-β made
        # OOB a cheap episode-ender and the policy bailed to the FOV edge
        # (term_oob_frac 0.28→0.43, recall ↓). Back to the flat-penalty form.
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
            return (
                self._get_observation(),
                reward,
                True,
                False,
                info,
            )

        # ─── Apply move ──────────────────────────────────────────────────────
        self.position = new_position

        is_revisit = self.visited_mask[self.position[0], self.position[1]] > 0
        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))
        if self.use_topology_memory:
            self._maybe_register_junction()

        gt_distance = float(self.distance_transform[self.position[0], self.position[1]])
        # Predicted-ridge distance (unclipped) — leak-free; used for centring
        # (r_near / shaping) and the predicted_ridge on-vessel signal.
        pred_ridge_dist = (
            float(
                self._offtrack_dt[
                    self.position[0],
                    self.position[1],
                ]
            )
            if self._offtrack_dt is not None
            else gt_distance
        )
        reward_prev_distance = (
            float(self._offtrack_dt[prev_pos[0], prev_pos[1]]) if self._offtrack_dt is not None else prev_distance
        )
        # Soft UNet vesselness at the current pixel (dense vessel evidence).
        vness = (
            float(
                self.unet_prior[
                    self.position[0],
                    self.position[1],
                ]
            )
            if self.unet_prior is not None
            else None
        )

        # ONE "on a vessel?" decision drives BOTH termination and reward gating
        # so they never conflict (the v9 failure). All leak-free except "gt".
        if self._on_vessel_signal == 'vesselness' and vness is not None:
            on_vessel = vness >= self._vesselness_tau
        elif self._on_vessel_signal == 'gt':
            on_vessel = gt_distance <= self.tolerance
        else:  # "predicted_ridge" (also the vesselness fallback if unet_prior absent)
            on_vessel = pred_ridge_dist <= self.tolerance

        # Reward geometry: gate on `on_vessel`, centre on the predicted ridge.
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
        # Track thickness-weighted accumulator for the reward signal; the
        # raw count (covered_centerline.sum()) is still used for the coverage
        # ratio reported in info / used by curriculum success criterion.
        prev_weighted_sum = self._covered_weight_sum
        # Stage B1: query frontier BEFORE coverage update — at this point
        # `_frontier_mask` reflects last step's frontier and we want to know
        # if the agent's NEW position lands on it.
        is_on_frontier = bool(
            self._frontier_mask is not None and self._frontier_mask[self.position[0], self.position[1]]
        )
        self._update_coverage()
        self._update_frontier_mask()
        new_coverage = self._covered_weight_sum - prev_weighted_sum
        current_coverage_ratio = self.covered_centerline.sum() / total_gt

        junction_val = self._junction_val_at(self.position)

        # Double off-track limit at junction pixels so the agent can try a
        # branch direction without immediate termination.
        effective_off_track = self.max_off_track * 2 if junction_val >= 0.8 else self.max_off_track
        terminated = self.off_track_streak >= effective_off_track
        truncated = self.step_count >= self.max_steps

        terminal_reason = ''
        f_beta = 0.0
        if terminated or truncated:
            terminal_reason = 'off_track' if terminated else 'max_steps'
            f_beta = self._compute_fbeta()

        # F3 — read NEW-position uncovered-DT against the same _uncov_dt
        # used for prev_uncov_dist. Refresh happens AFTER reward is built so
        # the within-step potential is consistent (Ng & Russell invariance).
        new_uncov_dist = (
            float(
                self._uncov_dt[
                    self.position[0],
                    self.position[1],
                ]
            )
            if self._uncov_dt is not None
            else None
        )

        # H6 — tangent-aligned progress signal. cos(step, forward_tangent)
        # where forward_tangent points toward uncovered work. Closes the
        # annulus-loiter exploit because perpendicular / reversing motion
        # earn 0 / negative even when the agent stays on-track.
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
        # F3 — refresh _uncov_dt periodically so it stays in sync with
        # coverage growth. Done AFTER reward computation so this step's
        # shaping used a consistent potential field.
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
            info['terminal_reason'] = terminal_reason  # "off_track" or "max_steps"

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _action_to_world_displacement(self, action: int) -> np.ndarray:
        """Translate a discrete action into a world-frame displacement vector
        (scaled by ``self.step_size``).

        When ``tangent_relative_actions`` is True (default), the canonical
        direction ``DIRECTIONS[action]`` is rotated so that action 0 points
        along the local vessel tangent. When False, the action is executed
        in pure world frame (action 0 = canonical N regardless of vessel
        orientation) — matches the imitation expert's frame.
        """
        base = self.DIRECTIONS[action].astype(np.float64) * self.step_size
        if not self.tangent_relative_actions:
            return base
        ty, tx = self._tangent_aligned_at(
            int(self.position[0]),
            int(self.position[1]),
            reference=self._prev_world_vec,
        )
        # Rotation matrix R with R @ (-1, 0) = (ty, tx) sends DIRECTIONS[action]
        # from the canonical frame to the tangent frame.
        dy, dx = float(base[0]), float(base[1])
        new_dy = -ty * dy + tx * dx
        new_dx = -tx * dy - ty * dx
        return np.array([new_dy, new_dx], dtype=np.float64)

    def _tangent_aligned_at(
        self,
        y: int,
        x: int,
        reference: Optional[np.ndarray] = None,
    ) -> tuple:
        """Return the local vessel tangent at (y, x), sign-aligned to a
        reference direction (typically the agent's previous move).

        The structure-tensor tangent encodes an undirected line orientation
        — its sign is arbitrary. The first step uses a default reference
        ("image up") so the agent has a stable forward at episode start;
        subsequent steps use the cached world-frame movement direction.
        """
        ty, tx = self.vessel_orientation[y, x]
        ty = float(ty)
        tx = float(tx)
        if reference is None:
            ref_y, ref_x = (
                -1.0,
                0.0,
            )  # canonical "N" / image-up
        else:
            ref_y, ref_x = (
                float(reference[0]),
                float(reference[1]),
            )
        if ty * ref_y + tx * ref_x < 0.0:
            ty, tx = -ty, -tx
        # Degenerate tangent (zero magnitude) — fall back to the reference
        # itself so the agent still has a usable forward axis.
        mag = (ty * ty + tx * tx) ** 0.5
        if mag < 1e-6:
            return -1.0, 0.0
        return ty / mag, tx / mag

    # ─── Topology memory (P1b) ──────────────────────────────────────────────

    @staticmethod
    def _extract_junction_pixels(
        skeleton: np.ndarray,
    ) -> list:
        """Skeleton pixels with >= 3 8-neighbours on the skeleton."""
        from data.centerline_extraction import (
            CenterlineExtractor,
        )

        return CenterlineExtractor()._find_junctions(skeleton)

    @staticmethod
    def _skeleton_neighbours(skeleton: np.ndarray, y: int, x: int) -> list:
        """8-neighbour skeleton pixels of (y, x) — used as candidate branch
        entry points when scoring "branches remaining at this junction".
        """
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
        """If the agent has landed within ``radius`` of a predicted-skeleton
        junction, append that junction to the visited list (deduped).
        """
        if not self._predicted_junction_pixels:
            return
        y, x = (
            int(self.position[0]),
            int(self.position[1]),
        )
        # Scan junctions in radius (cheap: typically ~50 junctions per image).
        for (
            jy,
            jx,
        ) in self._predicted_junction_pixels:
            if abs(jy - y) <= radius and abs(jx - x) <= radius:
                key = (jy, jx)
                if key not in self._visited_junctions:
                    self._visited_junctions.append(key)
                # Always promote to "most recent" so the topology channels
                # reflect the junction the agent is currently working on.
                elif self._visited_junctions[-1] != key:
                    self._visited_junctions.remove(key)
                    self._visited_junctions.append(key)
                break

    def _topology_features(self) -> tuple:
        """Return ``(normalised_distance, branches_remaining)`` scalars.

        ``normalised_distance``  : Euclidean distance from the current
            position to the most recently visited junction, divided by
            ``obs_size`` and clipped to [0, 1]. 1.0 if no junction visited.
        ``branches_remaining``   : count of the last junction's 8-skeleton
            neighbours that the agent has not yet stepped on, divided by 8
            so the channel sits in [0, 1]. 0.0 if no junction visited.
        """
        if not self.use_topology_memory or not self._visited_junctions:
            return 1.0, 0.0
        ly, lx = self._visited_junctions[-1]
        py, px = (
            float(self.position[0]),
            float(self.position[1]),
        )
        dist = ((py - ly) ** 2 + (px - lx) ** 2) ** 0.5
        normalised = min(
            1.0,
            dist / max(float(self.obs_size), 1.0),
        )
        nbrs = self._junction_neighbours.get((ly, lx), [])
        if not nbrs:
            return normalised, 0.0
        unvisited = sum(1 for (ny, nx) in nbrs if self.visited_mask[ny, nx] == 0.0)
        return normalised, unvisited / 8.0

    def _compute_fbeta(self) -> float:
        """Marginal F-β contribution of the *current* episode.

        Returns ``f_beta(prior ∪ current) − f_beta(prior_alone)`` when prior
        coverage is present; otherwise plain ``f_beta(current)``.

        Why marginal, not cumulative
        ----------------------------
        An earlier version of this method returned cumulative F-β on the
        union, with the rationale that a single trace covers <1% of the
        retinal tree so single-episode F-β is too small to provide a
        learning gradient. That rationale is right, but the implementation
        had a perverse-incentive bug: a STOP-immediately episode kept the
        cumulative high (it had been built up by previous episodes) and
        therefore got POSITIVE terminal reward despite contributing
        nothing — STOP-fast-and-free-ride became optimal. With the GT-DT
        signal removed (P0), per-step coverage gains shrank, the
        free-ride premium dominated, and the policy collapsed to STOP at
        step ~17 with 0.3% coverage.

        Marginal F-β fixes this: an episode that adds no coverage earns
        zero terminal reward (so the early_stop_penalty bites unopposed);
        an episode that legitimately extends the cumulative trace earns
        the delta. The agent still has gradient toward longer traces —
        more new coverage = bigger delta — while the free-ride loophole
        is closed.
        """
        if self.prior_coverage is not None:
            cumulative = np.where(
                (self.covered_centerline > 0) | (self.prior_coverage > 0),
                1.0,
                0.0,
            ).astype(np.float32)
            return self._fbeta_on(cumulative) - self._fbeta_on(self.prior_coverage)
        return self._fbeta_on(self.covered_centerline)

    def _fbeta_on(self, covered_mask: np.ndarray) -> float:
        """β-weighted F-score between ``covered_mask`` and ``self.centerline``.

        Tolerance-aware (matches ``compute_centerline_f1``); β² is read from
        ``config['reward']['terminal_recall_beta_sq']`` (default 4 → recall
        weighted 4× over precision). Returns 0.0 when ``covered_mask`` is
        empty so callers can use it as a baseline for marginal computations.
        """
        if covered_mask is None or not np.any(covered_mask):
            return 0.0
        from data.centerline_extraction import (
            compute_centerline_f1,
        )

        rc = self.config.get('reward', {})
        beta_sq = float(rc.get('terminal_recall_beta_sq', 4.0))
        metrics = compute_centerline_f1(
            covered_mask,
            self.centerline,
            tolerance=self.tolerance,
        )
        recall = metrics['recall']
        precision = metrics['precision']
        denom = beta_sq * precision + recall
        return (1.0 + beta_sq) * precision * recall / denom if denom > 0 else 0.0

    def _junction_val_at(self, position) -> float:
        """Return junction-map value at the given position (0.0 if not built)."""
        if self.observation_builder.junction_map is not None:
            return float(self.observation_builder.junction_map[position[0], position[1]])
        return 0.0

    def _is_valid_position(self, position):
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
        """Build per-pixel thickness weights for centerline pixels.

        For each centerline pixel p:
          inward_radius(p) = distance_transform_edt(vessel_mask)[p]
          local_width(p)   = 2 * max(inward_radius(p), 1.0)
          weight(p)        = clip(1.0 / sqrt(local_width(p) / W_REF), 0.7, 1.6)
                             where W_REF = median local_width across centerline
                             pixels.  Mean weight ≈ 1.0 → preserves total reward
                             scale.  Off-centerline pixels get weight 0.

        Falls back to uniform weight=1.0 if vessel_mask is unavailable.
        """
        H, W = centerline.shape
        weight_map = np.zeros((H, W), dtype=np.float32)
        cl_mask = centerline > 0
        if not cl_mask.any():
            return weight_map
        if vessel_mask is None:
            weight_map[cl_mask] = 1.0
            return weight_map

        from scipy.ndimage import (
            distance_transform_edt,
        )

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
        # Re-normalise so mean = 1.0 across centerline pixels (preserves
        # total episode-reward scale even after clipping)
        mean_w = float(raw_weight.mean())
        if mean_w > 0:
            raw_weight /= mean_w
        weight_map[cl_mask] = raw_weight.astype(np.float32)
        return weight_map

    def _update_coverage(self):
        y, x = self.position
        h = self._cov_half

        # Image-space bounds
        y_min = max(0, y - h)
        y_max = min(self.height, y + h + 1)
        x_min = max(0, x - h)
        x_max = min(self.width, x + h + 1)

        patch = self.centerline[y_min:y_max, x_min:x_max]
        if not patch.any():
            return

        # Slice the precomputed template to match boundary clipping
        ty_min = y_min - (y - h)
        ty_max = ty_min + (y_max - y_min)
        tx_min = x_min - (x - h)
        tx_max = tx_min + (x_max - x_min)
        within = self._cov_template[ty_min:ty_max, tx_min:tx_max]

        prev_patch = self.covered_centerline[y_min:y_max, x_min:x_max]
        newly_covered = within & (patch > 0) & (prev_patch == 0)

        # Update binary covered_centerline (used for F_β and coverage ratio)
        self.covered_centerline[y_min:y_max, x_min:x_max] = np.where(within & (patch > 0), 1.0, prev_patch)

        # Update weighted accumulator: each newly-covered centerline pixel
        # contributes its thickness weight. The reward uses the delta of this
        # accumulator (not the binary count) so thin-vessel coverage gets
        # ~1.6× the per-pixel reward of thick-trunk coverage.
        if newly_covered.any():
            weight_patch = self.centerline_weight_map[y_min:y_max, x_min:x_max]
            self._covered_weight_sum += float(weight_patch[newly_covered].sum())

    def _update_frontier_mask(self):
        """Recompute the Stage B1 frontier mask from current visit history.

        frontier = dilate(visited_mask, step_size, 8-conn) ∧ centerline ∧ ¬visited_mask

        i.e. unvisited GT centerline pixels REACHABLE IN ONE STEP from a pixel
        the agent has already stepped on.  Defined w.r.t. the actual visit
        history rather than the inflated ``covered_centerline`` (tolerance
        disk), so the agent can actually land on frontier pixels — under
        tolerance=2 coverage with step_size=1 the agent's position is
        always inside its own coverage disk, so a coverage-based frontier
        never fires.  ``visited_mask`` (rather than ``trajectory_mask``)
        is used because it includes the start position set in reset().

        The dilation radius MUST match ``step_size``: each step moves the
        agent up to ``step_size`` px per axis (DIRECTIONS · step_size), so a
        fixed 1-px band made ``is_on_frontier`` literally unsatisfiable at
        step_size=2 — the agent always lands ≥2 px from its last visited pixel,
        outside a 1-px band, so r_frontier was identically 0 for the whole v10
        run.  Dilating by ``step_size`` puts the band exactly where the agent
        can land next.
        """
        from scipy.ndimage import binary_dilation

        vis_bool = self.visited_mask > 0
        if not vis_bool.any():
            self._frontier_mask = np.zeros_like(self.centerline, dtype=bool)
            return
        dilated = binary_dilation(
            vis_bool,
            structure=np.ones((3, 3), dtype=bool),
            iterations=max(1, int(round(self.step_size))),
        )
        self._frontier_mask = dilated & (self.centerline > 0) & (~vis_bool)

    def _get_observation(self):
        return self.observation_builder.build(
            image=self.image,
            visited_mask=self.visited_mask,
            vesselness=self.vesselness,
            position=self.position,
            prev_direction=self.prev_direction,
            # Observation fallback path uses PREDICTED priors — GT skeleton
            # / DT stay on self.centerline / self.distance_transform for
            # the reward path only.
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
        # Precision: fraction of unique visited positions that were on-track
        if self._total_visited > 0:
            info['precision'] = self._total_visited_on_track / self._total_visited
        else:
            info['precision'] = 0.0
        return info

    def render(self):
        vis = (self.image.copy() * 255).astype(np.uint8)
        vis[self.centerline > 0] = [0, 0, 255]
        vis[self.covered_centerline > 0] = [
            0,
            255,
            0,
        ]
        for y, x in self.trajectory:
            vis[
                max(0, y - 1) : min(self.height, y + 2),
                max(0, x - 1) : min(self.width, x + 2),
            ] = [255, 0, 0]
        y, x = self.position
        vis[
            max(0, y - 2) : min(self.height, y + 3),
            max(0, x - 2) : min(self.width, x + 3),
        ] = [255, 255, 0]
        return vis


class VectorizedVesselEnv:
    """Vectorized environment for parallel training."""

    def __init__(self, config, num_envs=8, dataset=None):
        self.config = config
        self.num_envs = num_envs
        self.dataset = dataset
        self.envs = [VesselTracingEnv(config) for _ in range(num_envs)]
        self.current_samples = [None] * num_envs

    def _apply_sample(self, env, sample):
        """Unpack a dataset sample and call env.set_data()."""
        env.set_data(
            image=sample['image'].permute(1, 2, 0).numpy(),
            centerline=sample['centerline'].squeeze().numpy(),
            distance_transform=sample['distance_transform'].squeeze().numpy(),
            fov_mask=sample['fov_mask'].squeeze().numpy(),
            vessel_mask=(sample['vessel_mask'].squeeze().numpy() if 'vessel_mask' in sample else None),
            vessel_orientation=(sample['vessel_orientation'].numpy() if 'vessel_orientation' in sample else None),
            unet_prior=(sample['unet_prior'].squeeze(0).numpy() if 'unet_prior' in sample else None),
            pred_centerline=(sample['pred_centerline'].squeeze().numpy() if 'pred_centerline' in sample else None),
            pred_distance_transform=(
                sample['pred_distance_transform'].squeeze().numpy() if 'pred_distance_transform' in sample else None
            ),
            pred_dt_gradient=(sample['pred_dt_gradient'].numpy() if 'pred_dt_gradient' in sample else None),
        )

    def reset(self):
        observations, infos = [], []
        for i, env in enumerate(self.envs):
            sample = self._get_random_sample()
            self.current_samples[i] = sample
            self._apply_sample(env, sample)
            # env.set_data(
            #     image=sample["image"].permute(1, 2, 0).numpy(),
            #     centerline=sample["centerline"].squeeze().numpy(),
            #     distance_transform=sample["distance_transform"].squeeze().numpy(),
            #     fov_mask=sample["fov_mask"].squeeze().numpy(),
            # )
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.stack(observations), infos

    def step(self, actions):
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)
            if terminated or truncated:
                sample = self._get_random_sample()
                self.current_samples[i] = sample
                self._apply_sample(env, sample)
                # env.set_data(
                #     image=sample["image"].permute(1, 2, 0).numpy(),
                #     centerline=sample["centerline"].squeeze().numpy(),
                #     distance_transform=sample["distance_transform"].squeeze().numpy(),
                #     fov_mask=sample["fov_mask"].squeeze().numpy(),
                # )
                obs, _ = env.reset()
                info['terminal_observation'] = obs
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos,
        )

    def _get_random_sample(self):
        idx = np.random.randint(len(self.dataset))
        return self.dataset[idx]
