"""config.py — Unified project configuration.

Single source of truth: all hyperparameters live inside MODEL_CONFIG or
SEED_CONFIG.  Scripts unpack what they need at import time.
"""

import copy
import os as _os

import torch

from data.dataloader import (
    OUTPUT_DIR as OUTPUT_BASE,
)
from data.dataloader import WEIGHTS_DIR

# ═══════════════════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════════════════
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════════════
# WEIGHT / CHECKPOINT PATHS
# ═══════════════════════════════════════════════════════════════════════
PPO_WEIGHTS_PATH = str(WEIGHTS_DIR / 'ppo_policy.pt')
IMITATION_WEIGHTS_PATH = str(WEIGHTS_DIR / 'imitation_policy.pt')
SEED_WEIGHTS_PATH = str(WEIGHTS_DIR / 'seed_detector.pt')
PPO_LOG_PATH = str(WEIGHTS_DIR / 'ppo_log.csv')
IMITATION_LOG_PATH = str(WEIGHTS_DIR / 'imitation_log.csv')

# ═══════════════════════════════════════════════════════════════════════
# MASTER MODEL / ENVIRONMENT / REWARD CONFIG
# ═══════════════════════════════════════════════════════════════════════
MODEL_CONFIG = {
    'policy': {
        'hidden_dim': 256,
        'lstm_hidden': 256,
        'head_hidden': 128,
        'use_lstm': False,
        'use_junction_aux': True,
        'dropout': 0.05,
        'encoder_type': 'cnn',
    },
    'environment': {
        'observation_size': 65,
        # Fallback used only when per-sample vessel_width_px is not
        # available. The dataloader/env both compute the effective
        # tolerance as max(floor, k × vessel_width_px) per image.
        'tolerance': 2.5,
        'use_vesselness': False,
        # ── use_unet_prior is NARROW ──────────────────────────────────────
        # This flag ONLY controls whether the seed-detector's centerline
        # probability map is *also* included as a raw observation channel.
        # It does NOT control whether the seed-detector checkpoint is used:
        # since the P0 GT-leakage removal, the predicted-prior pipeline
        # (centerline / DT / DT-grad / junction) ALWAYS sources its
        # geometry from the seed detector's centerline-prob head via
        # data.dataloader.compute_unet_prior. The seed detector
        # (weights/seed_detector.pt, trained by scripts/train_seed_detector.py)
        # is therefore a hard prerequisite for any RL training/inference
        # regardless of this flag. Setting this False only drops one
        # observation channel (saves 1 channel of obs width); it does NOT
        # make the model optional.
        'use_unet_prior': True,
        # Binary entropy of the UNet probability map — peaks at p=0.5.
        # Pairs with use_topology_memory for "explore under uncertainty"
        # decisions at ambiguous branches. Reuses the existing UNet
        # probability so zero extra inference cost.
        'use_unet_uncertainty': False,
        'use_curvature': True,
        'use_junction': True,
        'use_prev_action': True,
        'use_global_visited': True,
        'use_prior_coverage': True,
        # E3 — local crop of covered_centerline as an obs channel so the
        # agent can SEE where it has already tracked (pairs with F3
        # uncov-DT shaping which is in the reward path only). Default off
        # to preserve channel layout for existing checkpoints.
        'use_covered_centerline': False,
        # In-episode graph of visited junctions on the predicted skeleton.
        # Emits 2 broadcast-scalar channels at the obs tail:
        #   dist-from-last-visited-junction (normalised to [0, 1])
        #   fraction of that junction's neighbours still unvisited
        'use_topology_memory': True,
        # Multi-scale wide-context crop. Emits 5 channels: wide RGB(3),
        # wide visited, wide UNet prior (zero-filled when use_unet_prior
        # is off). Wider field of view is area-pooled to obs_size².
        'use_multiscale': True,
        # Wide crop size = wide_crop_factor * observation_size. At
        # obs_size=65 and factor=4 the wide field is 257 px → covers the
        # optic disc / FOV boundary from anywhere inside the FOV.
        'wide_crop_factor': 4,
        'max_steps_per_episode': 500,
        # Fix A (D1 sweep, 2026-05-30): step_size 1 → 2. The BC val_acc sweep
        # over {world,tangent} x step{1..4} peaked at world-frame + step 2
        # (val_acc 0.733, loss 0.794). step 1 aliases more (staircase), steps
        # 3-4 lose info to np.sign over curved segments — 2 is the sweet spot.
        'step_size': 2,
        # When True, env rotates DIRECTIONS[action] by the local vessel
        # tangent (action 0 = "forward along tangent"). The imitation
        # expert in training/imitation.py generates action labels in
        # WORLD frame, which created a frame mismatch that destroyed the
        # imitation prior.
        # Fix A (D1 sweep): set to False (Option A, world-frame). World-frame
        # BC trounced tangent-frame (val_acc 0.73 vs 0.60; loss 0.79 vs 1.3+):
        # the tangent rotation added structure-tensor + sign-history noise that
        # made the label unlearnable. Keep imitation.tangent_aware=False to match.
        'tangent_relative_actions': False,
        'momentum': 0.0,
        # v10 — "on a vessel?" signal driving off-track termination AND the
        # reward's off-vessel/near/progress gating (one decision, no conflict).
        #   "vesselness" (default) — soft UNet vessel-prob >= vesselness_tau;
        #     dense, spans skeleton gaps → lifts the recall/connectivity ceiling.
        #   "predicted_ridge" — distance to predicted centerline <= tolerance (v9).
        #   "gt" — GT distance (NOT leak-free; debugging only).
        # All non-"gt" options keep the GT-ablation certification PASS.
        'on_vessel_signal': 'vesselness',
        'vesselness_tau': 0.3,
    },
    # "reward": {
    #     "beta_coverage": 0.30,
    #     "beta_frontier": 0,
    #     "alpha_near": 0.05, # proximity
    #     "gamma_off": 0, # off-vessel penalty
    #     "lambda_revisit": 0, # revisit penalty
    #     "step_cost": 0,
    #     "shaping_weight": 1.0, # potential based shaping
    #     "shaping_gamma": 0.99, # must equal training.ppo.gamma
    #     "terminal_f1_weight": 30.0,
    #     "terminal_recall_beta_sq": 8.0,
    #     "min_stop_coverage": 0,
    #     "early_stop_penalty": 0,
    #     "oob_penalty": -1.0, # out of bounds
    # },
    'reward': {
        # ── v10 calibration (2026-06-03) ──────────────────────────────────
        # v9 wandered: r_coverage (β=1.0·log1p over ~180 steps) swamped every
        # other term while predicted termination cut the traces → high reward,
        # flat val F1, fragmented output. v10 rebalances: the PRIMARY per-step
        # signal is directed PROGRESS (progress_weight, against the predicted
        # vessel), coverage is heavily downweighted, and off-vessel is a real
        # penalty consistent with the vesselness on-vessel signal
        # (environment.on_vessel_signal). All weights are RVT_REWARD_*-
        # overridable for a ratio sweep; these are starting points.
        'beta_coverage': 0.10,
        'beta_frontier': 0.10,
        'alpha_near': 0.10,
        'gamma_off': -0.30,
        # r_revisit: penalty MAGNITUDE (positive). reward.py applies it as
        # -lambda_revisit, so a positive value here = a penalty.  Was -0.2,
        # which (double-negated) silently *rewarded* revisiting in place.
        'lambda_revisit': 0.10,
        # Defensive cap on new GT-centerline px credited in one step before
        # the log1p compression.  Stops a single pathological disk-sweep step
        # from dominating the running reward-std; normal on-vessel steps
        # cover only a few px so this never bites them.
        'coverage_per_step_cap': 12.0,
        'step_cost': -0.01,
        'shaping_weight': 0.30,
        'shaping_gamma': 0.99,
        'terminal_f1_weight': 10.0,
        'terminal_recall_beta_sq': 4.0,
        # ``min_stop_coverage`` is the per-trace coverage (fraction of the
        # WHOLE-image GT centerline) below which a STOP is ramped toward
        # ``early_stop_penalty``.  A single seed traces ONE branch ≈ 3% of the
        # tree (measured mean_cov_at_stop ≈ 0.034), so 0.10 is unreachable and
        # r_terminal is net-NEGATIVE on every stop.  v11 lowered this to 0.02 to
        # make the terminal net-positive — but that REGRESSED recall (0.553→
        # 0.529) and connectivity (betti 15→22): the persistent stop-penalty was
        # load-bearing, keeping the agent tracing rather than ending early.
        # Cheapening the clean STOP (together with the v11 OOB credit) made the
        # policy end episodes sooner.  Reverted to 0.10.  Sweep via
        # RVT_REWARD_MIN_STOP_COVERAGE.
        'min_stop_coverage': 0.10,
        'early_stop_penalty': -2.0,
        'oob_penalty': -1.0,
        # F5 — penalty applied to off_track / max_steps termination when
        # coverage < min_stop_coverage. 0.0 = backward compat (passive
        # failure is free); set to ~-2.0 to discourage "wander forever".
        'early_termination_penalty': 0.0,
        # F3 — switch the shaping potential from distance-to-any-centerline
        # to distance-to-UNCOVERED-centerline so the agent isn't paid for
        # hugging ground it already covered.
        'shaping_uses_uncovered': False,
        # H6 — tangent-aligned progress reward. Per-step credit for
        # cos(step_vec, forward_tangent) where forward_tangent points
        # toward uncovered work. Closes the annulus-loiter exploit that
        # every position-based reward leaves open. 0 = off (backward
        # compatible); 0.5 is a reasonable starting magnitude given the
        # rest of the reward stack. v10: this is now the PRIMARY per-step term.
        'progress_weight': 0.30,
    },
    'training': {
        'patience': 100,
        'reward_norm_clip': 10.0,
        'terminal_norm_clip': 20.0,
        'lr_end_factor': 0.1,
        'value_clamp': 10.0,
        'ppo': {
            'lr': 5e-5,
            'lr_warmup_iters': 30,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.05,
            'value_coef': 0.5,
            'max_grad_norm': 2.0,
            'epochs': 5,
            'mini_batch_size': 512,
            'steps_per_iter': 8192,
            # v12 canonical: 600 (was 400). The clDice-gated entropy anneal
            # (full stage, anneal_iters=400) needs the longer budget to reach
            # its floor; v10 stopped at 400 with entropy still 0.52 and reward
            # still rising. This default reproduces the recorded v12 model with
            # no env-var overrides; RVT_PPO_NUM_ITERATIONS still overrides it.
            'num_iterations': 600,
            'eval_every': 20,
            'save_every': 50,
            'lstm_chunk_length': 256,
            # R2D2-style burn-in: each chunk reaches back by this many steps
            # before its training region and runs the LSTM forward without
            # contributing to the loss. Lets the recurrent state thaw from the
            # rollout-time snapshot into one consistent with the *current*
            # encoder before policy/value gradients are taken.
            'lstm_burn_in': 16,
            'n_envs': 16,
            'target_kl': 0.05,
        },
        'imitation': {
            # Option B: invert env's tangent rotation at expert-gen time so
            # action labels match what the env will execute. Independent of
            # the env's tangent_relative_actions flag — set both to match
            # for "frames-agree" experiments, or set them differently to
            # deliberately test the frame-mismatch state.
            # Fix A (D1 sweep): False to match environment.tangent_relative_actions
            # = False (Option A, world-frame won the BC sweep decisively).
            'tangent_aware': False,
            'lr': 3e-4,
            'batch_size': 512,
            'lstm_batch_size': 16,
            'num_epochs': 15,
            'max_grad_norm': 1.0,
            'use_augment': False,
            'lr_step_size': 5,
            'lr_gamma': 0.5,
            'num_workers': 16,
        },
    },
    'curriculum': {
        'start_difficulty': 0.3,
        'warmup_episodes': 5_000,
        'advancement_window': 200,
        # v10 — lower the advancement gates so a working policy progresses past
        # `easy` (it never has across the whole project). These feed
        # CurriculumManager.is_episode_successful (single-trace episode metrics
        # are inherently small, so the old f1>=0.145 / precision>=0.5 gate was
        # ~unreachable).
        'success_min_coverage_base': 0.10,
        'success_min_precision': 0.4,
        'success_min_f1_base': 0.06,
        'stages': [
            {
                'name': 'easy',
                'difficulty': 0.3,
                # Lowered 0.3 → 0.2 so a working tracer can actually clear the
                # gate and advance. The prior runs sat in `easy` for all 200
                # iterations because the policy never reached 0.3 success
                # (val_coverage ≈ 0.0035), so it only ever saw the easiest 30%
                # of images and never generalised.
                'min_success_rate': 0.15,
                'min_episodes': 50,
                'min_iterations': 10,
                'max_off_track_streak': 15,
                'max_steps_per_episode': 300,
                # CRITICAL FIX: `easy` previously defined only `entropy_coef`
                # (no end/iters), so the annealing branch in
                # PPOTrainer._get_curriculum_overrides_dict was skipped and
                # entropy_coef stayed pinned at 0.05 forever. Because the agent
                # never left `easy`, entropy was NEVER annealed and the policy
                # diffused back toward uniform (entropy rose 1.05 → 1.76 of a
                # 2.20 max). Giving `easy` a real anneal schedule lets the
                # policy commit to the (now stronger) imitation prior.
                'entropy_coef': 0.03,
                'entropy_coef_end': 0.008,
                'entropy_anneal_iters': 200,
            },
            {
                'name': 'medium',
                'difficulty': 0.6,
                'min_success_rate': 0.15,
                'min_episodes': 100,
                'min_iterations': 20,
                'max_off_track_streak': 12,
                'max_steps_per_episode': 500,
                'entropy_coef': 0.04,
                'entropy_coef_end': 0.02,
                'entropy_anneal_iters': 600,
            },
            {
                'name': 'full',
                'difficulty': 1.0,
                'min_success_rate': 0.1,
                'min_episodes': 200,
                'min_iterations': 30,
                'max_off_track_streak': 10,
                'max_steps_per_episode': 700,
                'entropy_coef': 0.03,
                'entropy_coef_end': 0.02,
                # v11 — was 600. The anneal is performance-GATED (the timer only
                # advances on clDice improvement; see PPOTrainer freeze logic),
                # so over a 400-iter run that entered `full` at ~iter 100 the
                # coef only reached ~0.025 and entropy plateaued at 0.52 — the
                # policy never sharpened, leaving residual off-vessel/OOB drift.
                # 400 lets the coef reach its 0.02 floor within the longer v11
                # run (num_iterations=600) even with some freezing.
                'entropy_anneal_iters': 400,
            },
        ],
    },
    'inference': {
        'mode': 'e2e',
        'max_traces': 80,
        'min_cov_gain': 0.0001,
        'dilation_radius': 5,
        'n_ring_seeds': 8,
        'ring_inset_px': 40,
        # Inference-time centerline SNAP (no retrain). The policy traces inside
        # vessels but ~2-3px off the GT centerline (f1 jumps 0.26→0.36 from 2px
        # →3px tolerance). The UNet predicted centerline is well-aligned to GT
        # (high UNet F1), so snapping each traced path point onto the nearest
        # predicted-centerline ridge pixel (within snap_radius_px) sharpens
        # localization. Set False to A/B against the raw traced skeleton.
        'snap_to_centerline': True,
        'snap_radius_px': 3.5,
        # Vessel gate (no retrain): DROP traced points farther than
        # snap_radius_px from any predicted-centerline ridge pixel and break
        # the polyline there. The policy has a residual straight-line bias and
        # from off-vessel/grid/ring seeds paints straight strokes across
        # background; those points have no predicted vessel nearby, so gating
        # removes the false-positive web. Set False to A/B.
        'vessel_gate': True,
        # GT-ABLATION certification (no retrain). When True, the FrontierTracer
        # feeds the env GARBAGE ground truth (zeroed centerline, large-constant
        # distance transform) while the predicted inputs, the policy, and the
        # GT used for METRIC SCORING are untouched. If the metrics are
        # unchanged vs corrupt_gt=False, the prediction provably does not depend
        # on GT → leak-free. Any change exposes a remaining GT leak.
        'corrupt_gt': False,
        # Environment overrides applied by get_inference_config()
        'max_steps_per_episode': 700,
        'max_off_track_streak': 10,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# SEED DETECTOR CONFIG
# ═══════════════════════════════════════════════════════════════════════
SEED_CONFIG = {
    'seed_detector': {
        'base_ch': 24,
        'dropout': 0.10,
        'use_frangi_input': True,
        'confidence_threshold': 0.35,
        'vessel_gate_threshold': 0.25,
        'top_k_seeds': 80,
        'mc_samples': 0,  # >0 enables MC-dropout at inference
        'suppress_optic_disc': True,
        'snap_radius': 2,
    },
    'training': {
        'sigma': 1.5,
        'num_epochs': 200,
        'warmup_epochs': 5,
        'batch_size': 4,
        'lr': 1e-3,
        'num_workers': 2,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLE OVERRIDES (for HPC ablation sweeps)
# ═══════════════════════════════════════════════════════════════════════
# Each ablation flag / reward weight can be overridden from the shell so
# Slurm array tasks can sweep configurations without touching this file.
# Unset variables leave the hard-coded defaults above untouched.


def _env_bool(name: str, default: bool) -> bool:
    v = _os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def _env_float(name: str, default):
    v = _os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


# Observation channels & policy heads
_env_cfg = MODEL_CONFIG['environment']
_env_cfg['use_vesselness'] = _env_bool(
    'RVT_USE_VESSELNESS',
    _env_cfg['use_vesselness'],
)
_env_cfg['use_unet_prior'] = _env_bool(
    'RVT_USE_UNET_PRIOR',
    _env_cfg['use_unet_prior'],
)
_env_cfg['use_unet_uncertainty'] = _env_bool(
    'RVT_USE_UNET_UNCERTAINTY',
    _env_cfg['use_unet_uncertainty'],
)
_env_cfg['use_curvature'] = _env_bool('RVT_USE_CURVATURE', _env_cfg['use_curvature'])
_env_cfg['use_junction'] = _env_bool('RVT_USE_JUNCTION', _env_cfg['use_junction'])
_env_cfg['use_prev_action'] = _env_bool(
    'RVT_USE_PREV_ACTION',
    _env_cfg['use_prev_action'],
)
_env_cfg['use_global_visited'] = _env_bool(
    'RVT_USE_GLOBAL_VISITED',
    _env_cfg['use_global_visited'],
)
_env_cfg['use_prior_coverage'] = _env_bool(
    'RVT_USE_PRIOR_COVERAGE',
    _env_cfg['use_prior_coverage'],
)
_env_cfg['use_topology_memory'] = _env_bool(
    'RVT_USE_TOPOLOGY_MEMORY',
    _env_cfg['use_topology_memory'],
)
_env_cfg['use_covered_centerline'] = _env_bool(
    'RVT_USE_COVERED_CENTERLINE',
    _env_cfg['use_covered_centerline'],
)
# E2 — agent step size in pixels (env env.step + imitation expert stride).
_env_cfg['step_size'] = int(_env_float('RVT_STEP_SIZE', _env_cfg['step_size']))
_env_cfg['tangent_relative_actions'] = _env_bool(
    'RVT_TANGENT_RELATIVE_ACTIONS',
    _env_cfg['tangent_relative_actions'],
)
# v10 on-vessel signal (string) + threshold.
_env_cfg['on_vessel_signal'] = _os.environ.get(
    'RVT_ON_VESSEL_SIGNAL',
    _env_cfg['on_vessel_signal'],
).strip()
_env_cfg['vesselness_tau'] = _env_float(
    'RVT_VESSELNESS_TAU',
    _env_cfg['vesselness_tau'],
)
# Decouple imitation-side flag from env-side. Set both for "frames agree"
# experiments; set them differently to test the frame-mismatch state.
_imi_cfg = MODEL_CONFIG['training']['imitation']
_imi_cfg['tangent_aware'] = _env_bool(
    'RVT_IMITATION_TANGENT_AWARE',
    _imi_cfg['tangent_aware'],
)
# E4 — FrontierTracer inference knobs: fewer/longer traces.
_inf_cfg = MODEL_CONFIG['inference']
_inf_cfg['max_traces'] = int(
    _env_float(
        'RVT_INFERENCE_MAX_TRACES',
        _inf_cfg['max_traces'],
    )
)
_inf_cfg['max_steps_per_episode'] = int(
    _env_float(
        'RVT_INFERENCE_MAX_STEPS',
        _inf_cfg['max_steps_per_episode'],
    )
)
_inf_cfg['min_cov_gain'] = _env_float(
    'RVT_INFERENCE_MIN_COV_GAIN',
    _inf_cfg['min_cov_gain'],
)
_inf_cfg['snap_to_centerline'] = _env_bool(
    'RVT_INFERENCE_SNAP',
    _inf_cfg['snap_to_centerline'],
)
_inf_cfg['snap_radius_px'] = _env_float(
    'RVT_INFERENCE_SNAP_RADIUS',
    _inf_cfg['snap_radius_px'],
)
_inf_cfg['vessel_gate'] = _env_bool(
    'RVT_INFERENCE_VESSEL_GATE',
    _inf_cfg['vessel_gate'],
)
_inf_cfg['corrupt_gt'] = _env_bool(
    'RVT_INFERENCE_CORRUPT_GT',
    _inf_cfg['corrupt_gt'],
)
_env_cfg['use_multiscale'] = _env_bool(
    'RVT_USE_MULTISCALE',
    _env_cfg['use_multiscale'],
)
# Zero-mask flags for the base geometry channels (ablation experiment) — see
# ObservationBuilder.__init__. Defaults False so production runs are unchanged.
_env_cfg['mask_dt'] = _env_bool('RVT_MASK_DT', _env_cfg.get('mask_dt', False))
_env_cfg['mask_pred_centerline'] = _env_bool(
    'RVT_MASK_PRED_CENTERLINE',
    _env_cfg.get('mask_pred_centerline', False),
)
_env_cfg['mask_tangent'] = _env_bool(
    'RVT_MASK_TANGENT',
    _env_cfg.get('mask_tangent', False),
)
_pol_cfg = MODEL_CONFIG['policy']
_pol_cfg['use_junction_aux'] = _env_bool(
    'RVT_USE_JUNCTION_AUX',
    _pol_cfg['use_junction_aux'],
)
_pol_cfg['use_lstm'] = _env_bool('RVT_USE_LSTM', _pol_cfg['use_lstm'])

# Reward weights (shaping_gamma intentionally omitted — must equal ppo.gamma)
_rw_cfg = MODEL_CONFIG['reward']
for _key in (
    'beta_coverage',
    'beta_frontier',
    'alpha_near',
    'gamma_off',
    'lambda_revisit',
    'coverage_per_step_cap',
    'step_cost',
    'shaping_weight',
    'terminal_f1_weight',
    'terminal_recall_beta_sq',
    'min_stop_coverage',
    'early_stop_penalty',
    'oob_penalty',
    'early_termination_penalty',
    'progress_weight',
):
    _rw_cfg[_key] = _env_float(
        f'RVT_REWARD_{_key.upper()}',
        _rw_cfg[_key],
    )
# F3 — bool flag (not a float weight)
_rw_cfg['shaping_uses_uncovered'] = _env_bool(
    'RVT_REWARD_SHAPING_USES_UNCOVERED',
    _rw_cfg['shaping_uses_uncovered'],
)

# PPO knobs commonly swept (shorter runs for ablations, etc.)
_ppo_cfg = MODEL_CONFIG['training']['ppo']
_ppo_cfg['num_iterations'] = int(
    _env_float(
        'RVT_PPO_NUM_ITERATIONS',
        _ppo_cfg['num_iterations'],
    )
)


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE ALIASES
# ═══════════════════════════════════════════════════════════════════════
TOLERANCE = MODEL_CONFIG['environment']['tolerance']
OBS_SIZE = MODEL_CONFIG['environment']['observation_size']


def get_config() -> dict:
    """Return a deep copy of MODEL_CONFIG for safe mutation (e.g. sweeps)."""
    return copy.deepcopy(MODEL_CONFIG)


# ═══════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════


def get_inference_config() -> dict:
    """Return a policy config tuned for inference (no dropout, longer episodes).

    Differences from the training config:
    - ``policy.dropout`` set to 0.0 — disables regularisation at test time.
    - ``environment.max_steps_per_episode`` raised to the inference value so
      the agent can complete long vessel paths without early truncation.
    - ``environment.max_off_track_streak`` set to the inference value.
    - ``environment.step_size`` INHERITS the trained value. It used to be
      hard-coded to 1, which silently reverted the stride a policy was
      trained with (e.g. step_size=3): the agent's action displacements at
      deployment no longer matched what it learned, degrading tracing. The
      stride must match training, so we leave the deep-copied value as-is.

    Used by ``scripts/run_rl_tracing.py``.
    """
    cfg = copy.deepcopy(MODEL_CONFIG)
    inf = cfg['inference']
    cfg['environment']['max_steps_per_episode'] = inf['max_steps_per_episode']
    cfg['environment']['max_off_track_streak'] = inf['max_off_track_streak']
    # NOTE: step_size intentionally inherited from MODEL_CONFIG (which already
    # reflects any RVT_STEP_SIZE override) — see docstring above.
    cfg['policy']['dropout'] = 0.0
    return cfg


def get_seed_inference_config() -> dict:
    """Return a seed-detector config tuned for inference.

    Differences from the training config:
    - ``nms_radius`` = 10 (was 15 "for cleaner selection", but that filter was too
      aggressive: two valid seeds on parallel vessels 15px apart were merged into
      one, leaving some images with only 20-30 seeds).  10 px still suppresses
      duplicate peaks but keeps seeds on nearby branches distinct.
    - ``frangi_spacing`` = 12 (was 20): denser auxiliary seeds on long unbranched
      segments.  A 100 px branch now gets ~8 supplementary seeds instead of ~5.
    - ``top_k_seeds`` set to ``max_traces`` — matches the inference budget.
    - ``confidence_threshold`` = 0.08: catches low-confidence thin-vessel seeds.

    Used by ``scripts/run_rl_tracing.py``.
    """
    cfg = copy.deepcopy(SEED_CONFIG)
    cfg['seed_detector'].update(
        {
            'nms_radius': 10,
            'frangi_spacing': 12,
            'top_k_seeds': MODEL_CONFIG['inference']['max_traces'],
            'confidence_threshold': 0.08,
        }
    )
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION METRIC COLUMNS
# ═══════════════════════════════════════════════════════════════════════
METRIC_COLS = [
    'iou',
    'clDice',
    'betti_0_error_raw',
    'betti_0_error_postproc',
    'betti_0_covered',
    'gt_edge_cov80_frac',
    'hd95',
    'f1@1px',
    'precision@1px',
    'recall@1px',
    'f1@2px',
    'precision@2px',
    'recall@2px',
    'f1@3px',
    'precision@3px',
    'recall@3px',
]
CSV_COLUMNS = ['image_id'] + METRIC_COLS
