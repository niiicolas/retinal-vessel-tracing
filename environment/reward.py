"""Reward calculation for retinal vessel tracing.

Eight-component, tolerance-aware reward whose terminal F-β is computed on
``covered_centerline`` to align training with the clDice eval metric. All weights are
read from ``config['reward']`` / ``config['environment']``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class RewardState:
    """Snapshot of environment state for one reward computation, built in ``env.step()``."""

    is_terminal: bool
    terminal_reason: str  # "stop" | "off_track" | "max_steps" | "oob" | ""

    new_coverage: float  # GT centerline pixels newly covered this step
    is_on_track: bool  # dt[position] <= tolerance
    distance: float  # DT value at current position
    prev_distance: float  # DT value at previous position

    coverage: float  # episode coverage ratio [0, 1]
    f_beta_score: float  # pre-computed terminal F-β; 0.0 on non-terminal steps

    position: Optional[np.ndarray] = None  # (y, x), logging/debugging only
    step_number: int = 0
    junction_map_value: float = 0.0  # junction off-track tolerance, read by env
    is_revisit: bool = False

    # True iff the step landed on an unvisited GT centerline pixel 8-adjacent to a
    # covered one (Stage B1 frontier reward).
    is_on_frontier: bool = False

    # Distance to nearest UNCOVERED GT centerline pixel (F3 shaping); falls back to
    # distance / prev_distance when None.
    uncovered_distance: Optional[float] = None
    prev_uncovered_distance: Optional[float] = None

    # Signed cos(step_vec, forward_tangent), tangent sign-aligned toward uncovered work
    # (H6 progress reward); None is treated as 0.
    progress_cos: Optional[float] = None


class RewardCalculator:
    """Eight-component vessel-tracing reward.

    ``compute(state)`` returns ``(total_reward, breakdown)``; the breakdown always
    contains every key in :attr:`BREAKDOWN_KEYS`, so callers can accumulate per-component
    means without key-checking.
    """

    # Component names present in every breakdown dict, in order.
    BREAKDOWN_KEYS: Tuple[str, ...] = (
        'r_coverage',
        'r_frontier',
        'r_near',
        'r_off_vessel',
        'r_revisit',
        'r_step_cost',
        'r_shaping',
        'r_progress',
        'r_terminal',
    )

    def __init__(self, config: Dict[str, Any]) -> None:
        """Read all reward weights from ``config['reward']`` / ``config['environment']``."""
        rc = config.get('reward', {})
        ec = config.get('environment', {})

        # r_coverage: log-compressed new-pixel coverage (dominant dense signal).
        self.beta: float = rc.get('beta_coverage', 0.3)
        # Optional per-step cap on new_coverage before log1p; 0/negative disables it.
        self.coverage_per_step_cap: float = rc.get('coverage_per_step_cap', 0.0)

        # r_frontier: flat bonus for landing on a frontier pixel (Stage B1).
        self.beta_frontier: float = rc.get('beta_frontier', 0.05)

        # r_near: continuous proximity reward α·max(0, 1 − D/τ), peaking on the centerline.
        self.alpha_near: float = rc.get('alpha_near', 0.01)

        # r_off_vessel: flat penalty when off the vessel.
        self.gamma_off: float = rc.get('gamma_off', -0.2)

        # r_revisit: penalty for stepping onto an already-visited pixel.
        self.lambda_revisit: float = rc.get('lambda_revisit', 0.02)

        # r_step_cost: constant per-step cost.
        self.step_cost: float = rc.get('step_cost', -0.01)

        # r_shaping: potential-based shaping; shaping_gamma must equal training.ppo.gamma.
        self.shaping_weight: float = rc.get('shaping_weight', 1.0)
        self.shaping_gamma: float = rc.get('shaping_gamma', 0.99)
        self.tolerance: float = ec.get('tolerance', 2.0)

        # r_terminal: clDice-aligned F-β on covered_centerline, plus stop/OOB penalties.
        self.terminal_f1_weight: float = rc.get('terminal_f1_weight', 16.0)
        self.terminal_recall_beta_sq: float = float(rc.get('terminal_recall_beta_sq', 4.0))
        self.min_stop_coverage: float = rc.get('min_stop_coverage', 0.05)
        self.early_stop_penalty: float = rc.get('early_stop_penalty', -1.0)
        self.oob_penalty: float = rc.get('oob_penalty', -5.0)
        # F5: penalty for off_track/max_steps termination below min_stop_coverage, so
        # passive failure isn't free. Default 0 keeps backward compatibility.
        self.early_termination_penalty: float = rc.get('early_termination_penalty', 0.0)
        # F3: shape on distance-to-UNCOVERED centerline instead of any centerline; needs
        # RewardState.uncovered_distance, else falls back to distance / prev_distance.
        self.shaping_uses_uncovered: bool = bool(rc.get('shaping_uses_uncovered', False))
        # H6: tangent-aligned progress reward grading step DIRECTION (not position), to
        # close the "loiter in the tolerance band" loophole. Default 0 disables it.
        self.progress_weight: float = rc.get('progress_weight', 0.0)

    def compute(self, state: RewardState) -> Tuple[float, Dict[str, float]]:
        """Compute total reward and per-component breakdown for one step.

        Returns ``(total_reward, breakdown)`` where ``breakdown`` has exactly the keys in
        :attr:`BREAKDOWN_KEYS`, zero-filled for inactive components.
        """
        bd: Dict[str, float] = {k: 0.0 for k in self.BREAKDOWN_KEYS}

        # Out-of-bounds: flat penalty only, no step components. This penalty is
        # load-bearing — a penalty-free OOB turns "walk off the FOV edge" into a cheap
        # episode-ender and the policy bails to the boundary (v11 experiment, reverted).
        if state.terminal_reason == 'oob':
            bd['r_terminal'] = self.oob_penalty
            return bd['r_terminal'], bd

        # Step components — skipped for STOP (no movement).
        if state.terminal_reason != 'stop':
            # Log-compress new coverage so fat-trunk steps don't dwarf thin-vessel steps.
            nc = state.new_coverage
            if self.coverage_per_step_cap > 0.0:
                nc = min(nc, self.coverage_per_step_cap)
            bd['r_coverage'] = self.beta * float(np.log1p(nc))

            if self.beta_frontier != 0.0 and state.is_on_frontier:
                bd['r_frontier'] = self.beta_frontier

            if self.alpha_near != 0.0 and state.is_on_track:
                tol = max(self.tolerance, 1e-6)
                bd['r_near'] = self.alpha_near * max(0.0, 1.0 - state.distance / tol)

            if not state.is_on_track:
                bd['r_off_vessel'] = self.gamma_off

            if self.lambda_revisit != 0.0 and state.is_revisit:
                bd['r_revisit'] = -self.lambda_revisit

            bd['r_step_cost'] = self.step_cost

            # H6: the only term grading motion (step direction) rather than position.
            if self.progress_weight != 0.0 and state.is_on_track and state.progress_cos is not None:
                bd['r_progress'] = self.progress_weight * float(state.progress_cos)

            # Potential-based shaping Φ = −min(D, τ)/τ. With F3, D is distance to the
            # nearest UNCOVERED centerline so hugging already-covered ground stops paying.
            if self.shaping_weight != 0.0:
                tol = max(self.tolerance, 1e-6)
                if self.shaping_uses_uncovered and state.uncovered_distance is not None and state.prev_uncovered_distance is not None:
                    d_prev = state.prev_uncovered_distance
                    d_curr = state.uncovered_distance
                else:
                    d_prev = state.prev_distance
                    d_curr = state.distance
                phi_prev = -min(d_prev, self.tolerance) / tol
                phi_curr = -min(d_curr, self.tolerance) / tol
                bd['r_shaping'] = self.shaping_weight * (self.shaping_gamma * phi_curr - phi_prev)

        # Terminal component (every episode end).
        if state.is_terminal:
            r_t = self.terminal_f1_weight * state.f_beta_score
            # Linear ramp penalty for STOP below the coverage threshold (full at 0).
            if state.terminal_reason == 'stop' and state.coverage < self.min_stop_coverage:
                fraction = max(0.0, min(1.0, state.coverage / max(self.min_stop_coverage, 1e-6)))
                r_t += self.early_stop_penalty * (1.0 - fraction)
            # F5: same ramp for off_track/max_steps so passive failure isn't cheaper than STOP.
            if (
                state.terminal_reason in ('off_track', 'max_steps')
                and self.early_termination_penalty != 0.0
                and state.coverage < self.min_stop_coverage
            ):
                fraction = max(0.0, min(1.0, state.coverage / max(self.min_stop_coverage, 1e-6)))
                r_t += self.early_termination_penalty * (1.0 - fraction)
            bd['r_terminal'] = r_t

        return sum(bd.values()), bd
