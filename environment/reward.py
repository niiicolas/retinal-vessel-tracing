"""Reward calculation for retinal vessel tracing.

Eight components, matching the bachelor-thesis proposal's tolerance-aware
reward design.  Training signal is aligned with the eval metric (clDice) —
see ``VesselTracingEnv._compute_fbeta`` which operates on
``covered_centerline`` (the same mask the eval loop uses), not on
``trajectory_mask``.

Per-step components:
  r_coverage    — β × log1p(new GT centerline px covered)          [coverage]
  r_frontier    — β_frontier  if step lands on a frontier pixel    [B1 graph]
  r_near        — α × max(0, 1 − D(p)/τ)                           [proximity]
  r_off_vessel  — γ_off  if  D(p) > τ                              [off-track]
  r_revisit     — −λ     if  pixel was previously visited           [no-loops]
  r_step_cost   — c_step                                            [efficiency]
  r_shaping     — potential-based shaping on Φ(p) = −min(D,τ)/τ    [dense guidance]

Terminal:
  r_terminal    — w × F_β(covered_centerline, GT)                  [clDice proxy]
                  plus early_stop_penalty on premature STOP,
                  oob_penalty for out-of-bounds termination.

Design notes:
  • Terminal F_β uses ``covered_centerline`` (NOT ``trajectory_mask``) so the
    training signal aligns with clDice: Tsens ≈ recall(covered, GT),
    Tprec ≈ precision(covered, GT).
  • Shaping must use shaping_gamma == training.ppo.gamma (enforced in
    PPOTrainer.__init__) for policy invariance (Ng, Daswani & Russell 1999).

All weights are read from ``config["reward"]``; see MODEL_CONFIG in config.py.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── State dataclass ───────────────────────────────────────────────────────────


@dataclass
class RewardState:
    """Snapshot of environment state for a single reward computation.

    Build this in ``VesselTracingEnv.step()`` and pass it to
    ``RewardCalculator.compute()``.
    """

    # ── Episode routing ───────────────────────────────────────────────────
    is_terminal: bool
    terminal_reason: str  # "stop" | "off_track" | "max_steps" | "oob" | ""

    # ── Step-component inputs ─────────────────────────────────────────────
    new_coverage: float  # GT centreline pixels newly covered this step
    is_on_track: bool  # dt[position] ≤ tolerance
    distance: float  # distance-transform value at current position
    prev_distance: float  # distance-transform value at previous position

    # ── Terminal-component inputs ─────────────────────────────────────────
    coverage: float  # current episode coverage ratio [0, 1]
    f_beta_score: float  # pre-computed F-β; 0.0 for non-terminal steps

    # ── Context (not used in reward computation) ──────────────────────────
    position: Optional[np.ndarray] = None  # (y, x) — for logging / debugging
    step_number: int = 0
    junction_map_value: float = 0.0  # read by env for junction off-track tolerance
    is_revisit: bool = False  # True if current pixel was previously visited

    # Stage B1 frontier reward — True iff the step landed on a previously-
    # UNVISITED GT centerline pixel adjacent (8-conn) to a pixel the agent
    # has already stepped on.
    is_on_frontier: bool = False

    # F3: distance to nearest UNCOVERED GT centerline pixel — used by the
    # shaping potential when ``reward.shaping_uses_uncovered`` is True.
    # Falls back to ``distance`` / ``prev_distance`` when not supplied.
    uncovered_distance: Optional[float] = None
    prev_uncovered_distance: Optional[float] = None

    # H6: signed cos(step_vec, forward_tangent), where forward_tangent is
    # the structure-tensor tangent at the new position sign-aligned to
    # point toward uncovered work (via uncov_dt gradient). Range [-1, 1].
    # ``None`` when uncov_dt / vessel_orientation is unavailable; treated
    # as 0 by the reward.
    progress_cos: Optional[float] = None


# ── Reward calculator ─────────────────────────────────────────────────────────


class RewardCalculator:
    """Eight-component vessel-tracing reward.

    Usage::

        calc = RewardCalculator(config)
        reward, breakdown = calc.compute(state)

    ``breakdown`` always contains all :attr:`BREAKDOWN_KEYS` so callers can
    accumulate per-component means without key-checking.
    """

    #: Ordered tuple of component names returned in every breakdown dict.
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
        rc = config.get('reward', {})
        ec = config.get('environment', {})

        # r_coverage — raw new-pixel count, moderately scaled
        self.beta: float = rc.get('beta_coverage', 0.3)
        # Defensive per-step cap on new_coverage before log1p (see config).
        # 0 / negative disables the cap.
        self.coverage_per_step_cap: float = rc.get('coverage_per_step_cap', 0.0)

        # r_frontier — flat per-step bonus when the agent lands on a frontier
        # pixel (uncovered GT centerline adjacent to covered region).  Stage B1.
        self.beta_frontier: float = rc.get('beta_frontier', 0.05)

        # r_near — continuous proximity reward within tolerance.
        # α · max(0, 1 − D(p)/τ): peaks at α on the centerline, zero at/beyond τ.
        self.alpha_near: float = rc.get('alpha_near', 0.01)

        # r_off_vessel — flat, gentle penalty
        self.gamma_off: float = rc.get('gamma_off', -0.2)

        # r_revisit — explicit penalty for stepping onto an already-visited pixel.
        self.lambda_revisit: float = rc.get('lambda_revisit', 0.02)

        # r_step_cost
        self.step_cost: float = rc.get('step_cost', -0.01)

        # r_shaping
        self.shaping_weight: float = rc.get('shaping_weight', 1.0)
        self.shaping_gamma: float = rc.get('shaping_gamma', 0.99)
        self.tolerance: float = ec.get('tolerance', 2.0)

        # r_terminal — clDice-aligned (computed on covered_centerline)
        self.terminal_f1_weight: float = rc.get('terminal_f1_weight', 16.0)
        self.terminal_recall_beta_sq: float = float(rc.get('terminal_recall_beta_sq', 4.0))
        self.min_stop_coverage: float = rc.get('min_stop_coverage', 0.05)
        self.early_stop_penalty: float = rc.get('early_stop_penalty', -1.0)
        self.oob_penalty: float = rc.get('oob_penalty', -5.0)
        # F5: penalty applied to off_track / max_steps termination when the
        # episode covered less than ``min_stop_coverage``. Without this,
        # off_track death and max_steps timeout are FREE — the policy learns
        # to oscillate near vessels indefinitely (term_off_track_frac=0,
        # term_max_steps_frac ~0.6 observed in the lean-obs ablation).
        # Defaults to 0 for backward compatibility.
        self.early_termination_penalty: float = rc.get('early_termination_penalty', 0.0)
        # F3: when True, the shaping potential uses distance-to-UNCOVERED
        # centerline rather than distance-to-any-centerline. Requires the env
        # to populate ``RewardState.uncovered_distance`` /
        # ``prev_uncovered_distance``; falls back to ``distance`` /
        # ``prev_distance`` when those are None.
        self.shaping_uses_uncovered: bool = bool(rc.get('shaping_uses_uncovered', False))
        # H6 — tangent-aligned progress reward. The diagnosis from three
        # ablations (17 variants flat at F1@2px ≈ 0.18) was that every
        # position-based per-step term (coverage, near, shaping) lets the
        # policy "loiter inside the tolerance band" without committing to
        # a direction. ``progress_cos`` is signed cos(step_vec,
        # forward_tangent) where forward_tangent is the structure-tensor
        # tangent at the agent's new position, sign-aligned to point
        # toward uncovered work via the uncov_dt gradient. A perfect
        # forward step earns ``progress_weight``; pure perpendicular drift
        # earns 0; reversing earns negative. Closes the "annulus exploit"
        # at the reward-surface level. Off-track steps get no progress
        # credit. Default 0 → backward-compatible.
        self.progress_weight: float = rc.get('progress_weight', 0.0)

    # ── Public interface ──────────────────────────────────────────────────────

    def compute(self, state: RewardState) -> Tuple[float, Dict[str, float]]:
        """Compute total reward and per-component breakdown for one step.

        Returns ``(total_reward, breakdown)`` where ``breakdown`` is a dict
        with exactly the keys in :attr:`BREAKDOWN_KEYS`, all zero-filled for
        inactive components.
        """
        bd: Dict[str, float] = {k: 0.0 for k in self.BREAKDOWN_KEYS}

        # Out-of-bounds: flat penalty, no step components.
        # NOTE (v11, REVERTED): an experiment credited an ON-vessel FOV-boundary
        # exit with its F-β and dropped the penalty, on the theory that a
        # peripheral vessel traced to the FOV edge is a legitimate endpoint, not
        # a failure. It BACKFIRED: a positive, penalty-free OOB turned "walk off
        # the FOV edge" into a cheap episode-ender, so the policy bailed to the
        # boundary in short dashes (term_oob_frac 0.28→0.43, ep_len 56→49,
        # recall@2px 0.553→0.529, betti0_post 15→22). The flat penalty is
        # load-bearing — it keeps the agent tracing rather than exiting — so v12
        # restores it. (Crediting boundary endpoints needs a design that does
        # not make OOB more attractive than continuing; not worth the risk now.)
        if state.terminal_reason == 'oob':
            bd['r_terminal'] = self.oob_penalty
            return bd['r_terminal'], bd

        # ── Step components (skipped for the STOP action — no movement) ───
        if state.terminal_reason != 'stop':
            # Log-compressed new-pixel coverage — the dominant dense signal.
            # log1p(n) gives 0/0.69/1.10/1.39/1.61/1.79/.../2.64 for n=0..13,
            # so a fat-trunk step that covers 13 disk-px earns ~3.8× a thin-
            # vessel step (1 px) instead of 13× under the raw count.  Reduces
            # the policy's bias toward wide vessels without zeroing the
            # signal entirely.
            nc = state.new_coverage
            if self.coverage_per_step_cap > 0.0:
                nc = min(nc, self.coverage_per_step_cap)
            bd['r_coverage'] = self.beta * float(np.log1p(nc))

            # Frontier bonus (Stage B1): graph-aware credit assignment.
            if self.beta_frontier != 0.0 and state.is_on_frontier:
                bd['r_frontier'] = self.beta_frontier

            # Continuous proximity reward within tolerance.
            if self.alpha_near != 0.0 and state.is_on_track:
                tol = max(self.tolerance, 1e-6)
                bd['r_near'] = self.alpha_near * max(
                    0.0,
                    1.0 - state.distance / tol,
                )

            # Flat penalty for every step off the vessel.
            if not state.is_on_track:
                bd['r_off_vessel'] = self.gamma_off

            # Explicit revisit penalty.
            if self.lambda_revisit != 0.0 and state.is_revisit:
                bd['r_revisit'] = -self.lambda_revisit

            # Constant per-step cost.
            bd['r_step_cost'] = self.step_cost

            # H6 — tangent-aligned progress reward. The single per-step term
            # that grades MOTION (step direction) rather than position;
            # closes the "loiter in the tolerance band" loophole that
            # position-based rewards leave open.
            if self.progress_weight != 0.0 and state.is_on_track and state.progress_cos is not None:
                bd['r_progress'] = self.progress_weight * float(state.progress_cos)

            # Potential-based shaping: Φ(s) = −min(dt(s), ε) / ε
            # F3: when shaping_uses_uncovered is True and the env supplies
            # uncovered-DT readings, the potential is distance-to-UNCOVERED-
            # centerline (so the agent isn't rewarded for hugging already-
            # covered ground; oscillation near a covered branch stops paying).
            if self.shaping_weight != 0.0:
                tol = max(self.tolerance, 1e-6)
                if (
                    self.shaping_uses_uncovered
                    and state.uncovered_distance is not None
                    and state.prev_uncovered_distance is not None
                ):
                    d_prev = state.prev_uncovered_distance
                    d_curr = state.uncovered_distance
                else:
                    d_prev = state.prev_distance
                    d_curr = state.distance
                phi_prev = -min(d_prev, self.tolerance) / tol
                phi_curr = -min(d_curr, self.tolerance) / tol
                bd['r_shaping'] = self.shaping_weight * (self.shaping_gamma * phi_curr - phi_prev)

        # ── Terminal component (every episode end) ────────────────────────
        if state.is_terminal:
            r_t = self.terminal_f1_weight * state.f_beta_score
            # Penalise STOP before covering a meaningful fraction. Linear
            # ramp: full penalty at 0 coverage, zero at threshold.
            if state.terminal_reason == 'stop' and state.coverage < self.min_stop_coverage:
                fraction = max(
                    0.0,
                    min(
                        1.0,
                        state.coverage
                        / max(
                            self.min_stop_coverage,
                            1e-6,
                        ),
                    ),
                )
                r_t += self.early_stop_penalty * (1.0 - fraction)
            # F5: same ramp for off_track / max_steps termination so passive
            # failure is no longer cheaper than STOP. Off-track death and
            # timeout were previously free, which let the policy "oscillate
            # near vessels until max_steps" without bound.
            if (
                state.terminal_reason in ('off_track', 'max_steps')
                and self.early_termination_penalty != 0.0
                and state.coverage < self.min_stop_coverage
            ):
                fraction = max(
                    0.0,
                    min(
                        1.0,
                        state.coverage
                        / max(
                            self.min_stop_coverage,
                            1e-6,
                        ),
                    ),
                )
                r_t += self.early_termination_penalty * (1.0 - fraction)
            bd['r_terminal'] = r_t

        return sum(bd.values()), bd
