"""Curriculum learning: progressively raises sample difficulty as the agent's success rate grows."""

import logging
from collections import deque
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import ndimage

from data.centerline_extraction import CenterlineExtractor


@dataclass
class CurriculumStage:
    """Config for one curriculum stage: difficulty target, advancement gates, and env/entropy overrides."""

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


# 8-neighbour count kernel, built once so compute_sample_difficulty doesn't rebuild it per call.
_NEIGHBOUR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int32)


class CurriculumManager:
    """Drives stage progression and the per-sample difficulty filter for vessel-tracing training.

    Unifies the per-stage difficulty target with an intra-stage warmup ramp into a single
    ``current_difficulty`` that never regresses below the active stage's floor.
    """

    def __init__(self, config: Dict[str, Any]):
        """Build stages from config and initialise stage/difficulty/success-tracking state."""
        curriculum_config = config.get('curriculum', {})
        self._cfg = curriculum_config

        self.start_difficulty = float(curriculum_config.get('start_difficulty', 0.2))

        # Warmup is counted in episodes; step() is called once per finished episode.
        self.warmup_episodes = int(max(1, curriculum_config.get('warmup_episodes', 5_000)))
        if self.warmup_episodes > 100_000:
            logging.warning(
                'CurriculumManager: warmup_episodes=%d is very large; the linear difficulty ramp is unlikely to complete during training.',
                self.warmup_episodes,
            )

        self.total_episodes = 0
        # Episodes since entering the current stage; drives the intra-stage ramp.
        self._episodes_in_stage: int = 0

        self.stages: List[CurriculumStage] = self._build_stages(curriculum_config.get('stages', None))
        self._validate_stages()

        self.current_stage_idx = 0
        self._window_size = int(curriculum_config.get('advancement_window', 200))
        self._recent_successes: deque = deque(maxlen=self._window_size)
        self._warn_oversized_min_episodes()

        # Cap transitions to one per outer PPO iteration: a single rollout pushes thousands of
        # episodes through step() with one stage_iter, which could otherwise cascade advances.
        self._last_transition_iter: int = -1

        # Optional regression (off by default — historically a footgun causing stage oscillation).
        self.enable_regression: bool = bool(curriculum_config.get('enable_regression', False))
        # Regress when success drops below this fraction of the previous stage's min_success_rate.
        self.regression_ratio: float = float(curriculum_config.get('regression_ratio', 0.5))
        # Minimum iterations in-stage before regression may fire (anti-flapping).
        self.regression_grace_iters: int = int(curriculum_config.get('regression_grace_iters', 50))

        # Seed difficulty at the first stage's floor so filtering is sensible from step 0.
        self.current_difficulty = self._compute_effective_difficulty()

    @staticmethod
    def _build_stages(stage_dicts: Optional[List[Dict[str, Any]]]) -> List[CurriculumStage]:
        """Build CurriculumStage objects from config dicts, dropping unknown keys and validating required ones.

        Returns a single default stage when no stages are configured.
        """
        if not stage_dicts:
            return [
                CurriculumStage(
                    name='default', difficulty=1.0, min_success_rate=0.3, min_episodes=100, description='Single default stage (no stages configured)'
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
        """Warn on out-of-range fields and non-monotonic (decreasing) stage difficulty."""
        for i, st in enumerate(self.stages):
            if not (0.0 <= st.difficulty <= 1.0):
                logging.warning(
                    'CurriculumManager: stage %r has difficulty=%.3f outside [0, 1]; the difficulty filter will behave oddly.', st.name, st.difficulty
                )
            if not (0.0 <= st.min_success_rate <= 1.0):
                logging.warning('CurriculumManager: stage %r has min_success_rate=%.3f outside [0, 1].', st.name, st.min_success_rate)
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

    def _warn_oversized_min_episodes(self) -> None:
        """Warn when a stage's ``min_episodes`` exceeds the rolling window (advancement is capped at the window)."""
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

    def get_difficulty(self) -> float:
        """Return the current effective difficulty in [0, 1]."""
        return self.current_difficulty

    def get_current_stage(self) -> CurriculumStage:
        """Return the currently active CurriculumStage."""
        return self.stages[self.current_stage_idx]

    def step(self, success: bool = False, stage_iter: int = 0) -> None:
        """Update curriculum state after one episode, then check for a stage transition.

        Args:
            success: whether the episode counted as a success.
            stage_iter: PPO iterations spent in the current stage (transition gate).
        """
        self.total_episodes += 1
        self._episodes_in_stage += 1
        self._recent_successes.append(1 if success else 0)
        self.current_difficulty = self._compute_effective_difficulty()
        self._check_stage_transitions(stage_iter)

    def is_episode_successful(self, info: Dict[str, Any]) -> bool:
        """Return whether an episode counts as a success under difficulty-scaled length/precision/F1 thresholds."""
        stage = self.get_current_stage()

        base = self._cfg.get('success_min_steps_base', 20)
        scale = self._cfg.get('success_min_steps_scale', 30)
        min_length = base + int(stage.difficulty * scale)
        min_precision = self._cfg.get('success_min_precision', 0.5)
        ep_len = info.get('step_count', 0)
        precision = info.get('precision', 0.0)

        # F1 gate applies only when the episode reported a terminal F1.
        ep_f1 = info.get('episode_f1', None)
        if ep_f1 is not None:
            min_f1 = self._cfg.get('success_min_f1_base', 0.10) + self._cfg.get('success_min_f1_scale', 0.15) * stage.difficulty
            return float(ep_f1) >= min_f1 and ep_len >= min_length and precision >= min_precision

        return ep_len >= min_length and precision >= min_precision

    def get_stage_overrides(self) -> Dict[str, Any]:
        """Return the current stage's reward/environment/training config overrides (incl. entropy annealing)."""
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

    def filter_samples(self, samples: List[Dict], difficulty_fn: Callable[[Dict], float]) -> List[Dict]:
        """Return samples at or below the current difficulty, falling back to the 10 easiest if too few pass.

        ``difficulty_fn`` maps a sample to its difficulty score.
        """
        difficulty = self.get_difficulty()
        scored = [(difficulty_fn(s), s) for s in samples]
        filtered = [s for d, s in scored if d <= difficulty]

        if len(filtered) < 10:
            scored.sort(key=lambda item: item[0])
            filtered = [s for _, s in scored[:10]]

        return filtered

    def compute_sample_difficulty(self, centerline: np.ndarray, vessel_mask: np.ndarray) -> float:
        """Score a sample's difficulty in [0, 1] from vessel thinness, junction density, and sparsity.

        Empty-centerline / empty-vessel samples short-circuit to maximal difficulty (1.0).
        """
        centerline_pixels = float(centerline.sum())
        vessel_pixels = float(vessel_mask.sum())

        if centerline_pixels <= 0 or vessel_pixels <= 0:
            return 1.0

        # Thinner average vessel width → harder.
        avg_width = vessel_pixels / centerline_pixels
        width_difficulty = 1.0 - min(avg_width / 10.0, 1.0)

        # Junction density (degree>2 skeleton pixels per 1000 centerline px) → harder.
        binary_skel = (centerline > 0).astype(np.int32)
        neighbour_counts = ndimage.convolve(binary_skel, _NEIGHBOUR_KERNEL, mode='constant')
        num_junctions = int(np.count_nonzero((binary_skel > 0) & (neighbour_counts > 2)))
        junction_density = num_junctions / centerline_pixels * 1000.0
        junction_difficulty = min(junction_density / 10.0, 1.0)

        # Sparser vessel coverage → harder.
        total_pixels = float(centerline.shape[0] * centerline.shape[1])
        vessel_density = vessel_pixels / total_pixels
        density_difficulty = 1.0 - min(vessel_density * 20.0, 1.0)

        weights = self._cfg.get('difficulty_weights', [0.4, 0.3, 0.3])
        difficulty = weights[0] * width_difficulty + weights[1] * junction_difficulty + weights[2] * density_difficulty
        return float(np.clip(difficulty, 0.0, 1.0))

    @property
    def recent_success_rate(self) -> float:
        """Rolling success rate over the current window (0.0 when empty)."""
        if not self._recent_successes:
            return 0.0
        return sum(self._recent_successes) / len(self._recent_successes)

    def state_dict(self) -> Dict[str, Any]:
        """Serialise mutable curriculum state for checkpointing."""
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
        """Restore curriculum state from a :meth:`state_dict` checkpoint, recomputing difficulty from config."""
        if not isinstance(state, dict):
            raise TypeError(f'CurriculumManager.load_state_dict expects a dict, got {type(state).__name__}')

        idx = int(state.get('current_stage_idx', 0))
        self.current_stage_idx = int(np.clip(idx, 0, len(self.stages) - 1))

        self.total_episodes = int(state.get('total_episodes', 0))
        self._episodes_in_stage = int(state.get('episodes_in_stage', 0))

        recent = state.get('recent_successes', [])
        self._recent_successes = deque((int(bool(x)) for x in recent), maxlen=self._window_size)
        self._last_transition_iter = int(state.get('last_transition_iter', -1))

        # Recompute (don't trust saved value) so config changes take effect on resume.
        self.current_difficulty = self._compute_effective_difficulty()

    def _compute_effective_difficulty(self) -> float:
        """Return the stage-driven difficulty: a linear prev→target ramp over the in-stage warmup, then pinned to target.

        Coupling the filter threshold to the active stage prevents a global warmup from racing
        ahead of stage advancement.
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
        """Try one advance (or, if enabled, one regress) per ``stage_iter``, gated to avoid chained transitions."""
        if stage_iter <= self._last_transition_iter:
            return
        if self._maybe_advance(stage_iter):
            self._last_transition_iter = stage_iter
            return
        if self.enable_regression and self._maybe_regress(stage_iter):
            self._last_transition_iter = stage_iter

    def _maybe_advance(self, stage_iter: int) -> bool:
        """Advance to the next stage when min-iterations, window-fill, and success-rate gates all pass; return True if it did."""
        if self.current_stage_idx >= len(self.stages) - 1:
            return False

        current_stage = self.stages[self.current_stage_idx]

        if stage_iter < current_stage.min_iterations:
            return False

        min_obs = min(current_stage.min_episodes, self._window_size)
        if len(self._recent_successes) < min_obs:
            return False

        if self.recent_success_rate < current_stage.min_success_rate:
            return False

        self.current_stage_idx += 1
        self._recent_successes.clear()
        self._episodes_in_stage = 0
        self.current_difficulty = self._compute_effective_difficulty()
        logging.info('Advancing to curriculum stage: %s', self.stages[self.current_stage_idx].name)
        return True

    def _maybe_regress(self, stage_iter: int) -> bool:
        """Step back one stage on sustained underperformance (grace period, full window, sub-threshold rate); return True if it did."""
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
