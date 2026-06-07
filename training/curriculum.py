# training/curriculum.py
"""Curriculum learning for progressive training difficulty."""

import logging
from collections import deque
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
)

import numpy as np
from scipy import ndimage

from data.centerline_extraction import (
    CenterlineExtractor,
)


@dataclass
class CurriculumStage:
    """A stage in the curriculum."""

    name: str
    difficulty: float
    min_success_rate: float
    min_episodes: int
    description: str = ''
    smoothness_weight: float = 0.0
    max_off_track_streak: int = 3
    max_steps_per_episode: int = 600
    entropy_coef: float = 0.05
    min_iterations: int = 0
    entropy_coef_end: Optional[float] = None
    entropy_anneal_iters: int = 0
    off_track_penalty_ramp: bool = False


# Convolution kernel used to count 8-connected neighbours on a binary
# skeleton. Pre-computed once at import time so compute_sample_difficulty
# does not rebuild it per call.
_NEIGHBOUR_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    dtype=np.int32,
)


class CurriculumManager:
    """Manages curriculum learning for vessel tracing.

    Progressively increases difficulty from easy cases (large, well-defined
    vessels) to hard cases (thin capillaries, pathologies, poor contrast).

    Three coupled difficulty notions are unified here:
      * ``stage.difficulty``  — discrete per-stage target.
      * warmup progress       — fractional ramp from start→end difficulty.
      * ``current_difficulty`` — the value sampled against by data filtering;
        always ``max(stage.difficulty, warmup_progress)`` so it can never
        regress below the active stage's floor.
    """

    def __init__(self, config: Dict[str, Any]):
        curriculum_config = config.get('curriculum', {})
        self._cfg = curriculum_config

        self.start_difficulty = float(curriculum_config.get('start_difficulty', 0.2))

        # Warmup is counted in episodes (CurriculumManager.step is invoked
        # once per finished episode by PPOTrainer).
        self.warmup_episodes = int(
            max(
                1,
                curriculum_config.get('warmup_episodes', 5_000),
            )
        )
        if self.warmup_episodes > 100_000:
            logging.warning(
                'CurriculumManager: warmup_episodes=%d is very large; the linear difficulty ramp is unlikely to complete during training.',
                self.warmup_episodes,
            )

        self.total_episodes = 0
        # Episodes observed since entering the current stage. Drives the
        # intra-stage difficulty ramp (see _compute_effective_difficulty).
        self._episodes_in_stage: int = 0

        # ------------------------------------------------------------------
        # Build CurriculumStage objects from the config dicts
        # ------------------------------------------------------------------
        self.stages: List[CurriculumStage] = self._build_stages(curriculum_config.get('stages', None))
        self._validate_stages()

        self.current_stage_idx = 0
        self._window_size = int(curriculum_config.get('advancement_window', 200))
        self._recent_successes: deque = deque(maxlen=self._window_size)
        self._warn_oversized_min_episodes()

        # Stage transitions are gated to AT MOST ONE per outer PPO
        # iteration. Inside one rollout the trainer may push thousands of
        # episodes through ``step()`` with the same ``stage_iter`` value;
        # without this guard a clear-and-refill of the success deque can
        # cascade advance→advance→advance in a single iteration (observed
        # on iter 1: easy → medium → full). Tracks the last ``stage_iter``
        # at which a transition fired.
        self._last_transition_iter: int = -1

        # Optional regression knobs. Disabled by default — a stage cannot
        # be un-advanced unless the user opts in, since regression has
        # historically been a footgun (oscillation between stages).
        self.enable_regression: bool = bool(curriculum_config.get('enable_regression', False))
        # Regress when rolling success rate falls below this fraction of
        # the *previous* stage's min_success_rate. Conservative default.
        self.regression_ratio: float = float(curriculum_config.get('regression_ratio', 0.5))
        # Minimum iterations in the current stage before regression can
        # fire. Mirrors min_iterations for advancement to avoid flapping.
        self.regression_grace_iters: int = int(curriculum_config.get('regression_grace_iters', 50))

        # Difficulty starts at the floor of the first stage so filtering
        # is sensible from step 0 — otherwise filter_samples returns a
        # near-empty set if start_difficulty < stage[0].difficulty.
        self.current_difficulty = self._compute_effective_difficulty()

    # ==================================================================
    # Construction helpers
    # ==================================================================

    @staticmethod
    def _build_stages(
        stage_dicts: Optional[List[Dict[str, Any]]],
    ) -> List[CurriculumStage]:
        if not stage_dicts:
            return [
                CurriculumStage(
                    name='default',
                    difficulty=1.0,
                    min_success_rate=0.3,
                    min_episodes=100,
                    description='Single default stage (no stages configured)',
                )
            ]

        valid_keys = {f.name for f in fields(CurriculumStage)}
        stages: List[CurriculumStage] = []
        for i, sd in enumerate(stage_dicts):
            if not isinstance(sd, dict):
                raise TypeError(f'curriculum.stages[{i}] must be a dict, got {type(sd).__name__}')
            unknown = set(sd.keys()) - valid_keys
            if unknown:
                logging.warning(
                    'CurriculumManager: stage %r contains unknown keys %s — ignoring. Valid keys: %s',
                    sd.get('name', f'#{i}'),
                    sorted(unknown),
                    sorted(valid_keys),
                )
                sd = {k: v for k, v in sd.items() if k in valid_keys}
            try:
                stages.append(CurriculumStage(**sd))
            except TypeError as e:
                raise TypeError(
                    f'curriculum.stages[{i}] (name={sd.get("name", "?")!r}) is missing a required field or has an invalid value: {e}'
                ) from e
        return stages

    def _validate_stages(self) -> None:
        """Sanity-check stage definitions; warn on non-monotonic difficulty."""
        for i, st in enumerate(self.stages):
            if not (0.0 <= st.difficulty <= 1.0):
                logging.warning(
                    'CurriculumManager: stage %r has difficulty=%.3f outside [0, 1]; the difficulty filter will behave oddly.',
                    st.name,
                    st.difficulty,
                )
            if not (0.0 <= st.min_success_rate <= 1.0):
                logging.warning(
                    'CurriculumManager: stage %r has min_success_rate=%.3f outside [0, 1].',
                    st.name,
                    st.min_success_rate,
                )
        for i in range(1, len(self.stages)):
            if self.stages[i].difficulty < self.stages[i - 1].difficulty:
                logging.warning(
                    'CurriculumManager: stage %r (difficulty=%.3f) is easier '
                    'than the previous stage %r (difficulty=%.3f). '
                    'Curriculum should be monotonically non-decreasing.',
                    self.stages[i].name,
                    self.stages[i].difficulty,
                    self.stages[i - 1].name,
                    self.stages[i - 1].difficulty,
                )

    def _warn_oversized_min_episodes(
        self,
    ) -> None:
        for st in self.stages:
            if st.min_episodes > self._window_size:
                logging.warning(
                    'CurriculumManager: stage %r requests min_episodes=%d '
                    'but the rolling window is only %d; advancement uses '
                    'min(min_episodes, window_size)=%d.',
                    st.name,
                    st.min_episodes,
                    self._window_size,
                    self._window_size,
                )

    # ==================================================================
    # Public API
    # ==================================================================

    def get_difficulty(self) -> float:
        """Current effective difficulty in ``[0, 1]``."""
        return self.current_difficulty

    def get_current_stage(
        self,
    ) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    def step(
        self,
        success: bool = False,
        stage_iter: int = 0,
    ) -> None:
        """Update curriculum state after one episode.

        Args:
            success: Whether the episode was successful.
            stage_iter: PPO iterations spent in the current stage.
        """
        self.total_episodes += 1
        self._episodes_in_stage += 1
        self._recent_successes.append(1 if success else 0)
        self.current_difficulty = self._compute_effective_difficulty()
        self._check_stage_transitions(stage_iter)

    def is_episode_successful(self, info: Dict[str, Any]) -> bool:
        """Decide whether an episode counts as a success.

        See module docstring for the (unchanged) criteria.
        """
        stage = self.get_current_stage()

        base = self._cfg.get('success_min_steps_base', 20)
        scale = self._cfg.get('success_min_steps_scale', 30)
        min_length = base + int(stage.difficulty * scale)
        min_precision = self._cfg.get('success_min_precision', 0.5)
        ep_len = info.get('step_count', 0)
        precision = info.get('precision', 0.0)

        ep_f1 = info.get('episode_f1', None)
        if ep_f1 is not None:
            min_f1 = (
                self._cfg.get('success_min_f1_base', 0.10)
                + self._cfg.get('success_min_f1_scale', 0.15) * stage.difficulty
            )
            return float(ep_f1) >= min_f1 and ep_len >= min_length and precision >= min_precision

        return ep_len >= min_length and precision >= min_precision

    def get_stage_overrides(
        self,
    ) -> Dict[str, Any]:
        """Return config overrides for the current stage.

        Includes the entropy annealing fields so downstream consumers can
        rely on a single source-of-truth instead of reading the stage
        object directly via ``getattr``.
        """
        stage = self.get_current_stage()
        return {
            'reward': {'smoothness_weight': stage.smoothness_weight},
            'environment': {
                'max_off_track_streak': stage.max_off_track_streak,
                'max_steps_per_episode': stage.max_steps_per_episode,
                'off_track_penalty_ramp': stage.off_track_penalty_ramp,
            },
            'training': {
                'entropy_coef': stage.entropy_coef,
                'entropy_coef_end': stage.entropy_coef_end,
                'entropy_anneal_iters': stage.entropy_anneal_iters,
            },
        }

    def filter_samples(
        self,
        samples: List[Dict],
        difficulty_fn: Callable[[Dict], float],
    ) -> List[Dict]:
        """Filter samples to those at or below the current difficulty.

        Falls back to the 10 *easiest* samples (not the first 10 in input
        order) when the threshold filter is too aggressive.
        """
        difficulty = self.get_difficulty()
        scored = [(difficulty_fn(s), s) for s in samples]
        filtered = [s for d, s in scored if d <= difficulty]

        if len(filtered) < 10:
            scored.sort(key=lambda item: item[0])
            filtered = [s for _, s in scored[:10]]

        return filtered

    def compute_sample_difficulty(
        self,
        centerline: np.ndarray,
        vessel_mask: np.ndarray,
    ) -> float:
        """Compute difficulty score for a sample in ``[0, 1]``.

        Empty-centerline samples are treated as maximally difficult and
        short-circuit before partial scores can be returned.
        """
        centerline_pixels = float(centerline.sum())
        vessel_pixels = float(vessel_mask.sum())

        if centerline_pixels <= 0 or vessel_pixels <= 0:
            return 1.0

        # --- average vessel width (thinner → harder) ---
        avg_width = vessel_pixels / centerline_pixels
        width_difficulty = 1.0 - min(avg_width / 10.0, 1.0)

        # --- junction density (more → harder) ---
        # Counted inline rather than via CenterlineExtractor._find_junctions
        # so we don't reach into private API and don't construct an
        # extractor on every call.
        binary_skel = (centerline > 0).astype(np.int32)
        neighbour_counts = ndimage.convolve(
            binary_skel,
            _NEIGHBOUR_KERNEL,
            mode='constant',
        )
        num_junctions = int(np.count_nonzero((binary_skel > 0) & (neighbour_counts > 2)))
        junction_density = num_junctions / centerline_pixels * 1000.0
        junction_difficulty = min(junction_density / 10.0, 1.0)

        # --- vessel pixel density (sparser → harder) ---
        total_pixels = float(centerline.shape[0] * centerline.shape[1])
        vessel_density = vessel_pixels / total_pixels
        density_difficulty = 1.0 - min(vessel_density * 20.0, 1.0)

        weights = self._cfg.get('difficulty_weights', [0.4, 0.3, 0.3])
        difficulty = weights[0] * width_difficulty + weights[1] * junction_difficulty + weights[2] * density_difficulty
        return float(np.clip(difficulty, 0.0, 1.0))

    @property
    def recent_success_rate(self) -> float:
        """Rolling success rate over the current window. 0.0 if empty."""
        if not self._recent_successes:
            return 0.0
        return sum(self._recent_successes) / len(self._recent_successes)

    # ==================================================================
    # Checkpointing
    # ==================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Serialise mutable state for checkpointing."""
        return {
            'version': 1,
            'current_stage_idx': self.current_stage_idx,
            'current_difficulty': self.current_difficulty,
            'total_episodes': self.total_episodes,
            'episodes_in_stage': self._episodes_in_stage,
            'recent_successes': list(self._recent_successes),
            'last_transition_iter': self._last_transition_iter,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore from a checkpoint produced by :meth:`state_dict`."""
        if not isinstance(state, dict):
            raise TypeError(f'CurriculumManager.load_state_dict expects a dict, got {type(state).__name__}')

        idx = int(state.get('current_stage_idx', 0))
        self.current_stage_idx = int(np.clip(idx, 0, len(self.stages) - 1))

        self.total_episodes = int(state.get('total_episodes', 0))
        self._episodes_in_stage = int(state.get('episodes_in_stage', 0))

        recent = state.get('recent_successes', [])
        self._recent_successes = deque(
            (int(bool(x)) for x in recent),
            maxlen=self._window_size,
        )
        self._last_transition_iter = int(state.get('last_transition_iter', -1))

        # Recompute difficulty rather than trusting the saved value so a
        # config change (e.g. tighter warmup) takes effect on resume.
        self.current_difficulty = self._compute_effective_difficulty()

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _compute_effective_difficulty(
        self,
    ) -> float:
        """The difficulty used by ``filter_samples`` and downstream code.

        Strictly stage-driven, with an intra-stage linear ramp:

          * ``target`` = current stage's ``difficulty``.
          * ``prev``   = previous stage's ``difficulty`` (or
            ``start_difficulty`` for stage 0).
          * Over the first ``warmup_episodes`` episodes spent inside the
            current stage, difficulty linearly interpolates ``prev → target``.
          * Afterwards difficulty is pinned to ``target`` until the next
            stage advancement.

        This couples the sample-filter threshold to the active stage so
        bug #2 (unsynchronised difficulty notions) cannot reappear: a
        global warmup can no longer race ahead of stage advancement and
        open the dataset filter while we are nominally still in ``easy``.
        """
        target = self.get_current_stage().difficulty
        if self.current_stage_idx == 0:
            prev = self.start_difficulty
        else:
            prev = self.stages[self.current_stage_idx - 1].difficulty

        if self._episodes_in_stage >= self.warmup_episodes:
            return float(target)
        progress = self._episodes_in_stage / self.warmup_episodes
        return float(prev + progress * (target - prev))

    def _check_stage_transitions(self, stage_iter: int) -> None:
        """Advance — or, if enabled, regress — based on rolling success.

        Hard cap of one transition per ``stage_iter`` value: once a stage
        changes, no further check fires until the trainer ticks
        ``stage_iter`` (i.e. starts a new outer PPO iteration). This
        prevents the chained-advance pathology where a single rollout's
        thousands of episodes refill the success deque after a clear()
        and immediately satisfy the next stage's gates.
        """
        if stage_iter <= self._last_transition_iter:
            return
        if self._maybe_advance(stage_iter):
            self._last_transition_iter = stage_iter
            return
        if self.enable_regression and self._maybe_regress(stage_iter):
            self._last_transition_iter = stage_iter

    def _maybe_advance(self, stage_iter: int) -> bool:
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        current_stage = self.stages[self.current_stage_idx]

        if stage_iter < current_stage.min_iterations:
            return False

        min_obs = min(
            current_stage.min_episodes,
            self._window_size,
        )
        if len(self._recent_successes) < min_obs:
            return False

        if self.recent_success_rate < current_stage.min_success_rate:
            return False

        self.current_stage_idx += 1
        self._recent_successes.clear()
        self._episodes_in_stage = 0
        self.current_difficulty = self._compute_effective_difficulty()
        logging.info(
            'Advancing to curriculum stage: %s',
            self.stages[self.current_stage_idx].name,
        )
        return True

    def _maybe_regress(self, stage_iter: int) -> bool:
        """Optionally step back one stage on sustained underperformance.

        Guarded by:
          * ``enable_regression`` (default False).
          * ``regression_grace_iters`` — minimum time in the stage.
          * Window must be full (same observation gate as advancement).
          * Rolling success rate must be below
            ``regression_ratio * previous_stage.min_success_rate``.
        """
        if self.current_stage_idx == 0:
            return False
        if stage_iter < self.regression_grace_iters:
            return False
        if len(self._recent_successes) < self._window_size:
            return False

        prev_stage = self.stages[self.current_stage_idx - 1]
        threshold = self.regression_ratio * prev_stage.min_success_rate
        if self.recent_success_rate >= threshold:
            return False

        self.current_stage_idx -= 1
        self._recent_successes.clear()
        self._episodes_in_stage = 0
        self.current_difficulty = self._compute_effective_difficulty()
        logging.warning(
            'Regressing to curriculum stage: %s (rolling success rate %.3f < %.3f)',
            self.stages[self.current_stage_idx].name,
            self.recent_success_rate,
            threshold,
        )
        return True
