"""Unified project configuration: single source of truth for all hyperparameters."""

import copy

import torch

from data.dataloader import (OUTPUT_DIR as OUTPUT_BASE)
from data.dataloader import WEIGHTS_DIR

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Weight / checkpoint / log paths.
PPO_WEIGHTS_PATH = str(WEIGHTS_DIR / 'ppo_policy.pt')
IMITATION_WEIGHTS_PATH = str(WEIGHTS_DIR / 'imitation_policy.pt')
SEED_WEIGHTS_PATH = str(WEIGHTS_DIR / 'seed_detector.pt')
PPO_LOG_PATH = str(WEIGHTS_DIR / 'ppo_log.csv')
IMITATION_LOG_PATH = str(WEIGHTS_DIR / 'imitation_log.csv')

# Master policy / environment / reward / training / curriculum / inference config.
MODEL_CONFIG = {
    'policy': {
        'hidden_dim': 256,
        'lstm_hidden': 256,  # hidden width of the LSTM (only used when use_lstm)
        'head_hidden': 128,  # hidden width of the actor/critic heads
        'use_lstm': False,
        'use_junction_aux': True,  # train an auxiliary junction-prediction head
        'dropout': 0.05,
        'encoder_type': 'cnn',  # observation encoder backbone
    },
    'environment': {
        'observation_size': 65,  # local observation crop side length (px)
        'tolerance': 2.5,  # GT-centerline match tolerance (px); per-image value is width-scaled
        'use_vesselness': False,
        'use_unet_prior': True,  # add the UNet centerline-prob map as an obs channel
        'use_unet_uncertainty': False,  # add the UNet-prob binary entropy (uncertainty) as an obs channel
        'use_curvature': True,
        'use_junction': True,
        'use_prev_action': True,
        'use_global_visited': True,
        'use_prior_coverage': True,
        'use_covered_centerline': False,  # add a local covered-centerline crop as an obs channel
        'use_topology_memory': True,  # add 2 scalar channels: dist to last-visited junction + fraction of its neighbours unvisited
        'use_multiscale': True,  # add 5 wide-context channels (RGB, visited, UNet prior) area-pooled to obs_size
        'wide_crop_factor': 4,  # wide-context crop side = this × observation_size
        'max_steps_per_episode': 500,
        'step_size': 2,  # pixels the agent advances per action step
        'tangent_relative_actions': False,  # action frame: False = world, True = relative to current tangent
        'momentum': 0.0,  # action-direction smoothing factor (0 = disabled)
        'on_vessel_signal': 'vesselness',  # on-vessel test for off-track + reward gating: 'vesselness'|'predicted_ridge'|'gt'
        'vesselness_tau': 0.3,  # UNet-prob threshold for the 'vesselness' on-vessel test
        'mask_dt': False,  # ablation: zero distance-transform obs channels 4,5,6
        'mask_pred_centerline': False,  # ablation: zero predicted-centerline obs channel 7
        'mask_tangent': False,  # ablation: zero tangent obs channels 8,9
    },
    'reward': {
        'beta_coverage': 0.10,  # weight on new GT-centerline pixels covered per step
        'beta_frontier': 0.10,  # weight on frontier (reach) expansion per step
        'alpha_near': 0.10,  # weight on proximity to the centerline
        'gamma_off': -0.30,  # penalty per off-vessel step
        'lambda_revisit': 0.10,  # penalty for revisiting covered pixels (applied as negative)
        'coverage_per_step_cap': 12.0,  # cap on new GT-centerline px credited per step (pre-log1p)
        'step_cost': -0.01,  # constant per-step cost
        'shaping_weight': 0.30,  # weight on potential-based shaping (potential Φ = distance to centerline)
        'shaping_gamma': 0.99,  # discount for reward shaping; must equal training.ppo.gamma
        'terminal_f1_weight': 10.0,  # weight on the terminal F-β reward
        'terminal_recall_beta_sq': 4.0,  # β² of the terminal F-β (>1 weights recall over precision)
        'min_stop_coverage': 0.10,  # per-trace coverage below which STOP incurs early_stop_penalty
        'early_stop_penalty': -2.0,  # penalty for stopping below min_stop_coverage
        'oob_penalty': -1.0,  # penalty for stepping out of image bounds
        'early_termination_penalty': 0.0,  # penalty for off-track/max-steps termination below min_stop_coverage
        'shaping_uses_uncovered': False,  # shape on distance to UNCOVERED centerline (else any centerline)
        'progress_weight': 0.30,  # weight on directed progress: cos(step, tangent toward uncovered work)
    },
    'training': {
        'patience': 100,  # early-stop patience (eval rounds without improvement)
        'reward_norm_clip': 10.0,  # clip range for normalized per-step rewards
        'terminal_norm_clip': 20.0,  # clip range for normalized terminal rewards
        'lr_end_factor': 0.1,  # final LR as a fraction of the initial LR
        'value_clamp': 10.0,  # clamp range for value-function targets
        'ppo': {
            'lr': 5e-5,
            'lr_warmup_iters': 30,  # linear LR warmup length (iterations)
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_eps': 0.2,
            'entropy_coef': 0.05,
            'value_coef': 0.5,
            'max_grad_norm': 2.0,
            'epochs': 5,
            'mini_batch_size': 512,
            'steps_per_iter': 8192,
            'num_iterations': 600,
            'eval_every': 20,
            'save_every': 50,
            'lstm_chunk_length': 256,  # BPTT chunk length (only used when policy.use_lstm)
            'lstm_burn_in': 16,  # BPTT burn-in steps before gradients accumulate
            'n_envs': 8,  # parallel rollout environments
            'target_kl': 0.05,  # KL threshold for early-stopping a PPO update
        },
        'imitation': {
            'tangent_aware': False,  # BC action frame; must match environment.tangent_relative_actions
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
        'advancement_window': 200,  # episodes averaged for the stage-advancement check
        'success_min_coverage_base': 0.10,  # min per-episode coverage for success (CurriculumManager.is_episode_successful)
        'success_min_precision': 0.4,  # min per-episode precision for success
        'success_min_f1_base': 0.06,  # min per-episode F1 for success
        'stages': [
            {
                'name': 'easy',
                'difficulty': 0.3,
                'min_success_rate': 0.15,  # success-rate gate to advance past this stage
                'min_episodes': 50,
                'min_iterations': 10,
                'max_off_track_streak': 15,
                'max_steps_per_episode': 300,
                'entropy_coef': 0.03,  # entropy coefficient at stage start
                'entropy_coef_end': 0.008,  # entropy coefficient after annealing
                'entropy_anneal_iters': 200,  # iterations over which entropy_coef anneals to its end value
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
                'entropy_anneal_iters': 400,
            },
        ],
    },
    'inference': {
        'mode': 'e2e',  # inference pipeline mode (end-to-end: seed detection then tracing)
        'max_traces': 80,  # max number of seeded traces per image
        'min_cov_gain': 0.0001,  # discard a trace that adds less than this coverage fraction
        'dilation_radius': 5,  # px radius for dilating the predicted skeleton
        'n_ring_seeds': 8,  # number of fallback seeds placed around the FOV ring
        'ring_inset_px': 40,  # inset (px) of the seed ring from the FOV edge
        'snap_to_centerline': True,  # snap traced points onto the predicted centerline
        'snap_radius_px': 3.5,  # max snap distance (px)
        'vessel_gate': True,  # gate steps on the on-vessel signal
        'corrupt_gt': False,  # corrupt GT to certify the pipeline is leak-free (debug)
        'max_steps_per_episode': 700,
        'max_off_track_streak': 10,
    },
}


# Convenience aliases.
TOLERANCE = MODEL_CONFIG['environment']['tolerance']
OBS_SIZE = MODEL_CONFIG['environment']['observation_size']


def get_config() -> dict:
    """Return a deep copy of MODEL_CONFIG for safe mutation (e.g. sweeps)."""
    return copy.deepcopy(MODEL_CONFIG)


def get_inference_config() -> dict:
    """Return a deep-copied policy config tuned for inference (no dropout, longer episodes).

    Raises max_steps_per_episode / max_off_track_streak to their inference values and zeroes
    dropout; ``step_size`` is intentionally inherited from MODEL_CONFIG so the deployed stride
    matches training. Used by scripts/run_rl_tracing.py.
    """
    cfg = copy.deepcopy(MODEL_CONFIG)
    inf = cfg['inference']
    cfg['environment']['max_steps_per_episode'] = inf['max_steps_per_episode']
    cfg['environment']['max_off_track_streak'] = inf['max_off_track_streak']
    cfg['policy']['dropout'] = 0.0
    return cfg


# Evaluation metric columns (CSV schema).
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
