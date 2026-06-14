"""PPO with GAE for vessel tracing: rollout buffer, greedy evaluation, and the training loop.

Supports feedforward (random mini-batch) and recurrent LSTM (chunked, hidden-state-aware) policies.
Driven by scripts/train_ppo.py.
"""

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import csv

from environment.reward import RewardCalculator
from training.curriculum import CurriculumManager


class RolloutBuffer:
    """Per-environment store of rollout transitions (plus pre-action LSTM states) with GAE computation."""

    def __init__(self):
        """Initialise empty transition lists."""
        self.reset()

    def reset(self):
        """Clear all stored transitions."""
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []
        # LSTM states are kept on CPU to save GPU memory.
        self.lstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: float,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Append one transition, moving any LSTM state to CPU."""
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        if lstm_state is not None:
            self.lstm_states.append((lstm_state[0].detach().cpu(), lstm_state[1].detach().cpu()))
        else:
            self.lstm_states.append(None)

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns for this buffer, bootstrapped from ``last_value``.

        Advantages are NOT normalised here — a single global normalisation happens in the PPO
        update after all buffers are concatenated, to preserve cross-env advantage ordering.
        """
        n = len(self.rewards)
        advantages = np.empty(n, dtype=np.float32)

        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in range(n - 1, -1, -1):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * not_done - values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            advantages[t] = gae
            next_value = values[t]

        advantages_t = torch.from_numpy(advantages)
        returns = advantages_t + torch.from_numpy(values)
        return returns, advantages_t

    def get_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack stored obs/actions/log_probs into tensors."""
        obs = torch.tensor(np.array(self.obs), dtype=torch.float32)
        actions = torch.tensor(np.array(self.actions), dtype=torch.long)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
        return obs, actions, log_probs


def evaluate(
    model: nn.Module, val_samples: List[Dict], config: dict, device: torch.device, tolerance: float, n_episodes: int = 1, n_parallel: int = 32
) -> Dict[str, float]:
    """Run greedy episodes per val sample with batched GPU inference and return mean coverage/F1/clDice.

    Up to ``n_parallel`` envs step together (one batched forward per step); LSTM hidden state is
    tracked per slot. clDice is computed on the multi-episode coverage union per sample, since a
    single-episode clDice is a ~constant artifact. Also returns width-stratified recall.
    """
    from data.centerline_extraction import compute_centerline_f1
    from environment.vessel_env import VesselTracingEnv

    model.eval()
    use_lstm = getattr(model, 'use_lstm', False)
    coverages = []
    f1_scores = []
    cldice_scores = []
    # Union coverage per sample (clDice computed after all its episodes finish).
    per_sample_coverage: Dict[int, np.ndarray] = {}
    per_sample_vessel: Dict[int, np.ndarray] = {}

    # Work queue of (sample, start_position) pairs.
    work_queue: deque = deque()
    for sample in val_samples:
        cl_points = np.argwhere(sample['centerline'] > 0)
        if len(cl_points) == 0:
            continue
        for _ in range(n_episodes):
            idx = np.random.randint(len(cl_points))
            work_queue.append((sample, tuple(cl_points[idx])))

    if not work_queue:
        model.train()
        return {'mean_coverage': 0.0, 'mean_f1': 0.0, 'mean_cldice': 0.0}

    # Parallel slot arrays.
    n_slots = min(n_parallel, len(work_queue))
    envs: List[Optional[object]] = [None] * n_slots
    samples_ref: List[Optional[Dict]] = [None] * n_slots
    obs_list: List[Optional[np.ndarray]] = [None] * n_slots
    lstm_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_slots
    active = [False] * n_slots

    def _start_slot(slot, sample, start_pos):
        """Spin up a fresh env for a parallel eval slot, seeded at ``start_pos``."""
        env = VesselTracingEnv(config)
        env.set_data(
            image=sample['image'],
            centerline=sample['centerline'],
            distance_transform=sample['distance_transform'],
            fov_mask=sample['fov_mask'],
            vessel_orientation=sample.get('vessel_orientation'),
            vesselness=sample.get('vesselness'),
            unet_prior=sample.get('unet_prior'),
            pred_centerline=sample.get('pred_centerline'),
            pred_distance_transform=sample.get('pred_distance_transform'),
            pred_dt_gradient=sample.get('pred_dt_gradient'),
        )
        obs, _ = env.reset(start_position=start_pos)
        envs[slot] = env
        samples_ref[slot] = sample
        obs_list[slot] = obs
        lstm_states[slot] = model.init_hidden(batch_size=1, device=device)
        active[slot] = True

    for i in range(n_slots):
        if work_queue:
            sample, start = work_queue.popleft()
            _start_slot(i, sample, start)

    with torch.no_grad():
        while any(active):
            active_idx = [i for i in range(n_slots) if active[i]]
            if not active_idx:
                break

            obs_batch = torch.from_numpy(np.stack([obs_list[i] for i in active_idx])).float().to(device)

            if use_lstm:
                h_cat = torch.cat([lstm_states[i][0].to(device) for i in active_idx], dim=0)
                c_cat = torch.cat([lstm_states[i][1].to(device) for i in active_idx], dim=0)
                batched_lstm = (h_cat, c_cat)
            else:
                batched_lstm = None

            logits, _, new_lstm = model(obs_batch, batched_lstm)
            actions = logits.argmax(dim=-1)  # greedy

            for j, i in enumerate(active_idx):
                (obs, _, terminated, truncated, info) = envs[i].step(actions[j].item())
                obs_list[i] = obs

                if use_lstm and new_lstm is not None:
                    lstm_states[i] = (new_lstm[0][j : j + 1, :].detach().cpu(), new_lstm[1][j : j + 1, :].detach().cpu())

                if terminated or truncated:
                    coverages.append(info['coverage_ratio'])
                    metrics = compute_centerline_f1(envs[i].covered_centerline, samples_ref[i]['centerline'], tolerance=tolerance)
                    f1_scores.append(metrics['f1'])

                    # Union this episode's coverage into the per-sample accumulator.
                    cov = envs[i].covered_centerline
                    if cov is not None:
                        s_key = id(samples_ref[i])
                        if s_key not in per_sample_coverage:
                            per_sample_coverage[s_key] = (cov > 0).astype(np.float32)
                            per_sample_vessel[s_key] = samples_ref[i].get('vessel_mask', samples_ref[i]['centerline'])
                        else:
                            per_sample_coverage[s_key] = np.where(cov > 0, 1.0, per_sample_coverage[s_key])

                    # Refill the slot from the queue, or retire it.
                    if work_queue:
                        sample, start = work_queue.popleft()
                        _start_slot(i, sample, start)
                    else:
                        active[i] = False

    # clDice + width-stratified recall on the per-sample coverage union.
    from evaluation.metrics import CenterlineMetrics

    _cm = CenterlineMetrics()
    rec_thin: List[float] = []
    rec_med: List[float] = []
    rec_thick: List[float] = []
    for s_key, cov in per_sample_coverage.items():
        cldice_scores.append(_cm.cl_dice(cov, per_sample_vessel[s_key]))
        # Skip width stratification when only a centerline-as-mask fallback is available (uniform widths).
        vmask = per_sample_vessel[s_key]
        cl_for_sample = None
        for s in val_samples:
            if id(s) == s_key:
                cl_for_sample = s['centerline']
                if 'vessel_mask' not in s:
                    vmask = None
                break
        if vmask is not None and cl_for_sample is not None:
            rb = _cm.recall_by_width(
                pred=(cov > 0).astype(np.uint8),
                gt_skeleton=(cl_for_sample > 0).astype(np.uint8),
                gt_vessel_mask=(vmask > 0).astype(np.uint8),
                tolerance=int(round(tolerance)),
            )
            tol_i = int(round(tolerance))
            if rb[f'n_centerline_thin'] > 0:
                rec_thin.append(rb[f'recall@{tol_i}px_thin'])
            if rb[f'n_centerline_med'] > 0:
                rec_med.append(rb[f'recall@{tol_i}px_med'])
            if rb[f'n_centerline_thick'] > 0:
                rec_thick.append(rb[f'recall@{tol_i}px_thick'])

    model.train()
    return {
        'mean_coverage': float(np.mean(coverages)) if coverages else 0.0,
        'mean_f1': float(np.mean(f1_scores)) if f1_scores else 0.0,
        'mean_cldice': float(np.mean(cldice_scores)) if cldice_scores else 0.0,
        'mean_recall_thin': float(np.mean(rec_thin)) if rec_thin else 0.0,
        'mean_recall_med': float(np.mean(rec_med)) if rec_med else 0.0,
        'mean_recall_thick': float(np.mean(rec_thick)) if rec_thick else 0.0,
    }


class RunningRewardNormalizer:
    """Scales rewards by their running std without subtracting the mean (SB3 VecNormalize convention).

    Mean-centering is omitted on purpose: subtracting the mean would flip ordinary on-vessel steps
    negative and collapse the near-constant early-stop penalty to ~0. The running mean is tracked only
    for an unbiased Welford variance, never subtracted.
    """

    def __init__(self, clip: float = 10.0, gamma: float = 0.99, update_clip: float = 5.0):
        """Configure output clip, (unused-here) gamma, and the update-time outlier clip; init running stats."""
        self.clip = clip
        self.gamma = gamma
        self.update_clip = update_clip
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-4

    def update(self, reward: float):
        """Update running mean/variance with one reward (Welford), clipping outliers before the variance update."""
        # Clip only the value fed into the variance so outliers don't permanently inflate the scale.
        r = float(np.clip(reward, -self.update_clip, self.update_clip))
        self.count += 1
        delta = r - self.running_mean
        self.running_mean += delta / self.count
        delta2 = r - self.running_mean
        self.running_var += (delta * delta2 - self.running_var) / self.count

    def normalize(self, reward: float) -> float:
        """Return the reward scaled by the running std (no mean subtraction) and clipped to ±clip."""
        std = max(np.sqrt(self.running_var), 1e-8)
        return np.clip(reward / std, -self.clip, self.clip)


class PPOTrainer:
    """PPO trainer with GAE, curriculum, reward normalisation, and a junction auxiliary loss.

    Handles both feedforward and LSTM policies; LSTM rollouts pass hidden state step-by-step and
    train on contiguous chunks via ``forward_sequence`` with done-mask resets.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.1,
        entropy_coef: float = 0.05,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
        mini_batch_size: int = 256,
        steps_per_iter: int = 4096,
        num_iterations: int = 1000,
        eval_every: int = 25,
        save_every: int = 50,
        tolerance: float = 2.0,
        lstm_chunk_length: int = 32,
    ):
        """Wire up the model, hyperparameters, curriculum, reward normaliser, optimizer, and LR schedule."""
        self.model = model
        self.config = config
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.steps_per_iter = steps_per_iter
        self.num_iterations = num_iterations
        self.eval_every = eval_every
        self.save_every = save_every
        self.tolerance = tolerance
        self.lstm_chunk_length = lstm_chunk_length
        # Burn-in length comes from config so call sites need no change; 0 disables.
        self.lstm_burn_in = int(config.get('training', {}).get('ppo', {}).get('lstm_burn_in', 0))
        self.use_lstm = getattr(model, 'use_lstm', False)
        self.value_clamp = config.get('training', {}).get('value_clamp', 10.0)

        # Adaptive KL early-stopping target (None/0 disables).
        ppo_cfg = config.get('training', {}).get('ppo', {})
        target_kl = ppo_cfg.get('target_kl', None)
        self.target_kl: Optional[float] = float(target_kl) if target_kl else None

        # Per-stage iteration counter (drives intra-stage entropy annealing; reset on stage change).
        self._stage_iter: int = 0
        # Rolling eval-clDice window gating entropy annealing (only anneal while improving).
        self._eval_cldice_window: deque = deque(maxlen=3)
        self._entropy_frozen: bool = False
        self._last_stage_idx: int = 0

        # Potential-based shaping requires shaping_gamma == ppo.gamma to stay policy-invariant (Ng 1999).
        shaping_gamma = config.get('reward', {}).get('shaping_gamma', gamma)
        if abs(shaping_gamma - gamma) > 1e-6:
            raise ValueError(
                f'reward.shaping_gamma ({shaping_gamma}) must equal training.ppo.gamma ({gamma}) for potential-based shaping to be policy-invariant.'
            )

        # Junction-channel index for extracting aux-head supervision targets from the obs tensor.
        from models.policy_network import _junction_channel_idx

        self._junction_ch_idx: Optional[int] = _junction_channel_idx(config)
        obs_size = config.get('environment', {}).get('observation_size', 65)
        self._obs_center: int = obs_size // 2  # center pixel index for the junction GT lookup

        self.curriculum = CurriculumManager(config)
        # Single std-only normaliser for ALL rewards: a separate mean-centering terminal normaliser
        # used to collapse the near-constant early-stop penalty and even flip OOB positive.
        self.reward_normalizer = RunningRewardNormalizer(clip=config.get('training', {}).get('reward_norm_clip', 10.0))

        self.patience = config.get('training', {}).get('patience', 100)
        self.no_improve_count = 0

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Optional linear warmup then linear decay.
        lr_end_factor = config.get('training', {}).get('lr_end_factor', 0.1)
        warmup_iters = ppo_cfg.get('lr_warmup_iters', 0)

        if warmup_iters > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters)
            decay_sched = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=max(num_iterations - warmup_iters, 1)
            )
            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_iters])
        else:
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=lr_end_factor, total_iters=num_iterations)

    def _get_curriculum_overrides_dict(self) -> dict:
        """Return the current stage's env overrides as a flat dict and apply training-side overrides (entropy).

        The entropy coefficient is linearly annealed within a stage toward ``entropy_coef_end`` over
        ``entropy_anneal_iters``, driven by ``self._stage_iter`` and gated by the freeze flag.
        """
        overrides = self.curriculum.get_stage_overrides()
        result = {}
        env_ov = overrides.get('environment', {})
        if 'max_off_track_streak' in env_ov:
            result['max_off_track'] = env_ov['max_off_track_streak']
        if 'max_steps_per_episode' in env_ov:
            result['max_steps'] = env_ov['max_steps_per_episode']
        if 'off_track_penalty_ramp' in env_ov:
            result['off_track_ramp'] = env_ov['off_track_penalty_ramp']
        reward_ov = overrides.get('reward', {})
        if 'smoothness_weight' in reward_ov:
            result['smoothness_weight'] = reward_ov['smoothness_weight']

        # Entropy coefficient is a trainer-side override (not sent to the env).
        train_ov = overrides.get('training', {})
        if 'entropy_coef' in train_ov:
            stage = self.curriculum.get_current_stage()
            ec_start = float(train_ov['entropy_coef'])
            ec_end = getattr(stage, 'entropy_coef_end', None)
            ec_iters = getattr(stage, 'entropy_anneal_iters', 0) or 0
            if ec_end is not None and ec_iters > 0:
                # A short stage-start warmup window bypasses the freeze so annealing can begin.
                stage_warmup_iters = 200
                bypass_freeze = self._stage_iter < stage_warmup_iters
                if (not self._entropy_frozen) or bypass_freeze:
                    t = min(self._stage_iter, ec_iters) / float(ec_iters)
                    self.entropy_coef = ec_start + t * (float(ec_end) - ec_start)
            else:
                self.entropy_coef = ec_start

        return result

    def _ppo_update_ff(self, buffers: List[RolloutBuffer], last_values: List[float]) -> Dict[str, float]:
        """Run the feedforward PPO update with random mini-batches over per-env GAE.

        Computes GAE per env, episode-length-weights advantages, globally re-normalises them, then
        optimizes clipped policy + clamped value + entropy (and the junction aux loss). Returns a
        stats dict (losses, KL, grad norm, explained variance).
        """
        all_returns, all_advantages = [], []
        (all_obs, all_actions, all_old_log_probs) = [], [], []
        all_values_raw = []  # for explained variance

        for buf, lv in zip(buffers, last_values):
            if len(buf.rewards) == 0:
                continue
            ret, adv = buf.compute_returns_and_advantages(lv, self.gamma, self.gae_lambda)
            # Up-weight long-episode advantages before the global normalisation.
            ep_w = torch.from_numpy(self._ep_length_weights(buf.dones))
            adv = adv * ep_w

            obs, actions, log_probs = buf.get_tensors()
            all_returns.append(ret)
            all_advantages.append(adv)
            all_obs.append(obs)
            all_actions.append(actions)
            all_old_log_probs.append(log_probs)
            all_values_raw.extend(buf.values)

        returns = torch.cat(all_returns).to(self.device)
        # Global advantage normalisation across the whole concatenated dataset.
        advantages = torch.cat(all_advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        obs = torch.cat(all_obs).to(self.device)
        actions = torch.cat(all_actions).to(self.device)
        old_log_probs = torch.cat(all_old_log_probs).to(self.device)

        (total_p, total_v, total_e, total_kl, total_gn, n) = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        max_gn = 0.0
        nan_skips = 0
        epochs_run = 0
        dataset_size = len(obs)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(dataset_size)
            epoch_kl_sum = 0.0
            epoch_kl_n = 0
            for start in range(0, dataset_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]

                logits, values, _ = self.model(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_prob - old_log_probs[idx])

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (log_prob - old_log_probs[idx])).mean().item()
                    total_kl += approx_kl
                    epoch_kl_sum += approx_kl
                    epoch_kl_n += 1

                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                p_loss = -torch.min(surr1, surr2).mean()

                v_loss = nn.functional.mse_loss(values, torch.clamp(returns[idx], -self.value_clamp, self.value_clamp))

                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                # Junction aux: classify center-pixel junction/endpoint from a junction-channel-masked
                # encoder pass so the head can't trivially read the label from its own input.
                if self.model.junction_head is not None and self._junction_ch_idx is not None:
                    obs_batch = obs[idx]
                    j_vals = obs_batch[:, self._junction_ch_idx, self._obs_center, self._obs_center]  # (B,)
                    j_class = torch.zeros(len(idx), dtype=torch.long, device=self.device)
                    j_class[j_vals > 0.7] = 2  # junction  (~1.0)
                    j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1  # endpoint (~0.5)
                    obs_masked = obs_batch.clone()
                    obs_masked[:, self._junction_ch_idx, :, :] = 0.0
                    enc_feats = self.model.encode(obs_masked)  # (B, hidden_dim)
                    j_logits = self.model.junction_head(enc_feats)  # (B, 3)
                    j_loss = nn.functional.cross_entropy(j_logits, j_class)
                    loss = loss + 0.1 * j_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                gn_val = grad_norm.item()
                # NaN guard: clip_grad_norm_ doesn't fix NaN/Inf grads — skip the step instead of
                # corrupting every parameter.
                if not np.isfinite(gn_val):
                    self.optimizer.zero_grad(set_to_none=True)
                    nan_skips += 1
                    continue
                total_gn += gn_val
                if gn_val > max_gn:
                    max_gn = gn_val
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += entropy.item()
                n += 1

            epochs_run += 1

            # Adaptive KL early-stop: bail on further passes once the epoch's mean KL exceeds target.
            if self.target_kl is not None and epoch_kl_n > 0:
                epoch_kl_mean = epoch_kl_sum / epoch_kl_n
                if epoch_kl_mean > self.target_kl:
                    break

        with torch.no_grad():
            values_all = torch.tensor(all_values_raw, dtype=torch.float32)
            ev = (1 - (returns.cpu() - values_all).var() / (returns.cpu().var() + 1e-8)).item()

        return {
            'policy_loss': total_p / max(n, 1),
            'value_loss': total_v / max(n, 1),
            'entropy': total_e / max(n, 1),
            'approx_kl': total_kl / max(n, 1),
            'grad_norm': total_gn / max(n, 1),
            'grad_norm_max': max_gn,
            'nan_skips': nan_skips,
            'epochs_run': epochs_run,
            'explained_variance': ev,
        }

    def _ppo_update_lstm(self, buffers: List[RolloutBuffer], last_values: List[float]) -> Dict[str, float]:
        """Run the recurrent PPO update over batched fixed-length chunks with masked, optionally burned-in losses.

        Each chunk is a contiguous ``lstm_chunk_length`` slice (right-padded), stacked along the batch
        dim; a valid mask zeroes padding (and, when ``lstm_burn_in>0``, the forward-only burn-in prefix
        that thaws the recurrent state). Returns the same stats dict as the FF path.
        """
        T_train = self.lstm_chunk_length
        T_burn = self.lstm_burn_in
        T_chunk = T_train + T_burn  # total tensor length per chunk

        # GAE per buffer + global advantage normalisation (matches the FF path).
        per_buf_returns: List[Optional[torch.Tensor]] = []
        per_buf_advantages: List[Optional[torch.Tensor]] = []
        all_values_raw: List[float] = []

        for buf, lv in zip(buffers, last_values):
            if len(buf.rewards) == 0:
                per_buf_returns.append(None)
                per_buf_advantages.append(None)
                continue
            ret, adv = buf.compute_returns_and_advantages(lv, self.gamma, self.gae_lambda)
            ep_w = torch.from_numpy(self._ep_length_weights(buf.dones))
            adv = adv * ep_w

            per_buf_returns.append(ret)
            per_buf_advantages.append(adv)
            all_values_raw.extend(buf.values)

        non_empty = [a for a in per_buf_advantages if a is not None]
        if non_empty:
            cat = torch.cat(non_empty)
            mean = cat.mean()
            std = cat.std() + 1e-8
            per_buf_advantages = [None if a is None else (a - mean) / std for a in per_buf_advantages]

        # Build chunk specs (buf_idx, actual_s, burn_in_actual, total_len): tensor starts at actual_s
        # = logical_s − burn_in; training stride is T_train so chunks tile the buffer even with overlap.
        chunk_specs: List[Tuple[int, int, int, int]] = []
        for buf_idx, buf in enumerate(buffers):
            T = len(buf.obs)
            for logical_s in range(0, T, T_train):
                burn_in_actual = min(T_burn, logical_s)
                actual_s = logical_s - burn_in_actual
                total_len = min(T_chunk, T - actual_s)
                # Keep only chunks with at least 2 training steps.
                if (total_len - burn_in_actual) >= 2:
                    chunk_specs.append((buf_idx, actual_s, burn_in_actual, total_len))

        if not chunk_specs:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'grad_norm': 0.0,
                'grad_norm_max': 0.0,
                'nan_skips': 0,
                'epochs_run': 0,
                'explained_variance': 0.0,
            }

        # Batch by training length so each step sees ~mini_batch_size effective transitions.
        chunks_per_batch = max(1, self.mini_batch_size // T_train)
        total_p = total_v = total_e = total_kl = total_gn = 0.0
        max_gn = 0.0
        nan_skips = 0
        n_updates = 0
        epochs_run = 0

        obs_shape = buffers[0].obs[0].shape  # (C, H, W)
        C, H, W = obs_shape

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(len(chunk_specs)).tolist()
            epoch_kl_sum = 0.0
            epoch_kl_n = 0

            for batch_start in range(0, len(perm), chunks_per_batch):
                batch_idx = perm[batch_start : batch_start + chunks_per_batch]
                B = len(batch_idx)

                # (T, B, ...) tensors; padding is zero and zeroed out by the valid mask.
                obs_np = np.zeros((T_chunk, B, C, H, W), dtype=np.float32)
                actions_np = np.zeros((T_chunk, B), dtype=np.int64)
                old_lp_np = np.zeros((T_chunk, B), dtype=np.float32)
                returns_np = np.zeros((T_chunk, B), dtype=np.float32)
                advs_np = np.zeros((T_chunk, B), dtype=np.float32)
                dones_np = np.zeros((T_chunk, B), dtype=np.float32)
                mask_np = np.zeros((T_chunk, B), dtype=np.float32)

                init_h_list: List[torch.Tensor] = []
                init_c_list: List[torch.Tensor] = []
                # reset_flags[j] is True when the chunk starts right after a done: its stored init
                # state is a stale, gradient-severed snapshot, so we swap in the live init params below.
                reset_flags: List[bool] = []

                for j, ci in enumerate(batch_idx):
                    (buf_idx, actual_s, burn_in_actual, total_len) = chunk_specs[ci]
                    buf = buffers[buf_idx]
                    s = actual_s  # tensor begins here in the buffer

                    obs_np[:total_len, j] = np.asarray(buf.obs[s : s + total_len], dtype=np.float32)
                    actions_np[:total_len, j] = np.asarray(buf.actions[s : s + total_len], dtype=np.int64)
                    old_lp_np[:total_len, j] = np.asarray(buf.log_probs[s : s + total_len], dtype=np.float32)
                    returns_np[:total_len, j] = per_buf_returns[buf_idx][s : s + total_len].numpy()
                    advs_np[:total_len, j] = per_buf_advantages[buf_idx][s : s + total_len].numpy()
                    dones_np[:total_len, j] = np.asarray(buf.dones[s : s + total_len], dtype=np.float32)
                    # Mask: burn-in prefix forward-only (0), training region (1), trailing pad (0).
                    mask_np[burn_in_actual:total_len, j] = 1.0

                    init = buf.lstm_states[s]
                    if init is not None:
                        init_h_list.append(init[0].squeeze(0))  # (hidden,)
                        init_c_list.append(init[1].squeeze(0))
                    else:
                        zero = self.model.init_hidden(batch_size=1, device='cpu')
                        init_h_list.append(zero[0].squeeze(0))
                        init_c_list.append(zero[1].squeeze(0))
                    # Buffer-boundary chunks (actual_s==0) are treated as continuation (cheap drift).
                    reset_flags.append(s > 0 and buf.dones[s - 1] > 0)

                obs_t = torch.from_numpy(obs_np).to(self.device)
                actions_t = torch.from_numpy(actions_np).to(self.device)
                old_lp_t = torch.from_numpy(old_lp_np).to(self.device)
                returns_t = torch.from_numpy(returns_np).to(self.device)
                advs_t = torch.from_numpy(advs_np).to(self.device)
                dones_t = torch.from_numpy(dones_np).to(self.device)
                mask_t = torch.from_numpy(mask_np).to(self.device)

                init_h = torch.stack(init_h_list, dim=0).to(self.device)  # (B, hidden)
                init_c = torch.stack(init_c_list, dim=0).to(self.device)
                # For reset rows, substitute the live learnable init params so gradients reach
                # init_h/init_c; non-reset rows keep their rollout-time state.
                if any(reset_flags):
                    reset_b = torch.tensor(reset_flags, dtype=torch.float32, device=self.device).unsqueeze(-1)  # (B, 1)
                    live_h = self.model.lstm_head.init_h.unsqueeze(0)  # (1, hidden)
                    live_c = self.model.lstm_head.init_c.unsqueeze(0)
                    init_h = init_h * (1.0 - reset_b) + live_h * reset_b
                    init_c = init_c * (1.0 - reset_b) + live_c * reset_b
                init_state = (init_h, init_c)

                # Main forward over (T, B); the junction aux runs a separate masked pass below.
                use_junction_aux = self.model.junction_head is not None and self._junction_ch_idx is not None
                logits_seq, values_seq = self.model.forward_sequence(obs_t, init_state, dones_t)

                dist = torch.distributions.Categorical(logits=logits_seq)
                log_prob = dist.log_prob(actions_t)  # (T, B)
                entropy = dist.entropy()  # (T, B)

                ratio = torch.exp(log_prob - old_lp_t)

                mask_sum = mask_t.sum().clamp(min=1.0)

                with torch.no_grad():
                    kl_per = ((ratio - 1) - (log_prob - old_lp_t)) * mask_t
                    approx_kl = (kl_per.sum() / mask_sum).item()
                    total_kl += approx_kl
                    epoch_kl_sum += approx_kl
                    epoch_kl_n += 1

                surr1 = ratio * advs_t
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advs_t
                per_step_p = -torch.min(surr1, surr2)
                p_loss = (per_step_p * mask_t).sum() / mask_sum

                # Clamp targets only (matches the FF path).
                target_t = torch.clamp(returns_t, -self.value_clamp, self.value_clamp)
                per_step_v = (values_seq - target_t).pow(2)
                v_loss = (per_step_v * mask_t).sum() / mask_sum

                ent_loss = (entropy * mask_t).sum() / mask_sum

                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * ent_loss

                # Junction aux on a junction-channel-masked encoder pass (mask zeroes padded steps).
                if use_junction_aux:
                    obs_flat = obs_t.reshape(T_chunk * B, *obs_t.shape[2:])
                    mask_flat = mask_t.reshape(-1)
                    j_vals = obs_flat[:, self._junction_ch_idx, self._obs_center, self._obs_center]  # (T*B,)
                    j_class = torch.zeros(T_chunk * B, dtype=torch.long, device=self.device)
                    j_class[j_vals > 0.7] = 2  # junction
                    j_class[(j_vals > 0.3) & (j_vals <= 0.7)] = 1  # endpoint
                    obs_masked = obs_flat.clone()
                    obs_masked[:, self._junction_ch_idx, :, :] = 0.0
                    enc_feats_aux = self.model.encode(obs_masked)  # (T*B, hidden_dim)
                    j_logits = self.model.junction_head(enc_feats_aux)  # (T*B, 3)
                    j_loss_per = nn.functional.cross_entropy(j_logits, j_class, reduction='none')
                    j_loss = (j_loss_per * mask_flat).sum() / mask_sum
                    loss = loss + 0.1 * j_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                gn_val = grad_norm.item()
                # NaN guard — see _ppo_update_ff for rationale.
                if not np.isfinite(gn_val):
                    self.optimizer.zero_grad(set_to_none=True)
                    nan_skips += 1
                    continue
                total_gn += gn_val
                if gn_val > max_gn:
                    max_gn = gn_val
                self.optimizer.step()

                total_p += p_loss.item()
                total_v += v_loss.item()
                total_e += ent_loss.item()
                n_updates += 1

            epochs_run += 1

            # Adaptive KL early-stop (mirrors the FF path).
            if self.target_kl is not None and epoch_kl_n > 0:
                epoch_kl_mean = epoch_kl_sum / epoch_kl_n
                if epoch_kl_mean > self.target_kl:
                    break

        with torch.no_grad():
            all_returns = torch.cat([r for r in per_buf_returns if r is not None])
            values_all = torch.tensor(all_values_raw, dtype=torch.float32)
            ev = (1 - (all_returns.cpu() - values_all).var() / (all_returns.cpu().var() + 1e-8)).item()

        return {
            'policy_loss': total_p / max(n_updates, 1),
            'value_loss': total_v / max(n_updates, 1),
            'entropy': total_e / max(n_updates, 1),
            'approx_kl': total_kl / max(n_updates, 1),
            'grad_norm': total_gn / max(n_updates, 1),
            'grad_norm_max': max_gn,
            'nan_skips': nan_skips,
            'epochs_run': epochs_run,
            'explained_variance': ev,
        }

    @staticmethod
    def _ep_length_weights(dones: list) -> np.ndarray:
        """Return per-step weights ``sqrt(episode_length / mean_episode_length)`` to offset short-episode frequency bias.

        Incomplete trailing episodes use their observed length so they are still up-weighted.
        """
        dones_arr = np.asarray(dones, dtype=np.float32)
        n = len(dones_arr)
        if n == 0:
            return np.ones(0, dtype=np.float32)

        weights = np.ones(n, dtype=np.float32)
        ep_lengths = []
        start = 0
        for t in range(n):
            if dones_arr[t] > 0 or t == n - 1:
                ep_len = t - start + 1
                ep_lengths.append(ep_len)
                weights[start : t + 1] = float(ep_len)
                start = t + 1

        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 1.0
        return np.sqrt(np.maximum(weights / max(mean_len, 1.0), 0.0)).astype(np.float32)

    def _ppo_update(self, buffers: List[RolloutBuffer], last_values: List[float]) -> Dict[str, float]:
        """Dispatch to the LSTM or feedforward PPO update based on the policy type."""
        if self.use_lstm:
            return self._ppo_update_lstm(buffers, last_values)
        return self._ppo_update_ff(buffers, last_values)

    def load_checkpoint(self, save_path: str, imitation_path: str) -> Tuple[int, float]:
        """Resume from the PPO checkpoint if present, else warm-start from imitation weights.

        Returns ``(start_iteration, best_cldice)``. When warm-starting, shape-mismatched heads are
        dropped and the value head (untrained in imitation) is re-initialised.
        """
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except (KeyError, TypeError, ValueError):
                    print('  Scheduler state incompatible, starting fresh.')

            if 'curriculum_state' in ckpt:
                self.curriculum.load_state_dict(ckpt['curriculum_state'])
                print(f'  Restored curriculum stage: {self.curriculum.get_current_stage().name}')

            start = ckpt.get('iteration', 0) + 1
            best = ckpt.get('best_cldice', ckpt.get('best_f1', 0.0))
            print(f'Resumed from PPO checkpoint  iter={start - 1}  best_clDice={best:.3f}')
            return start, best

        if os.path.exists(imitation_path):
            ckpt = torch.load(imitation_path, map_location=self.device, weights_only=True)
            # Drop shape-mismatched tensors (e.g. an old N_ACTIONS=8 actor) so a stale imitation
            # checkpoint doesn't block PPO startup.
            state = dict(ckpt['model_state_dict'])
            model_state = self.model.state_dict()
            stripped = []
            for k in list(state.keys()):
                if k in model_state and state[k].shape != model_state[k].shape:
                    stripped.append((k, tuple(state[k].shape), tuple(model_state[k].shape)))
                    del state[k]
            # strict=False: the imitation checkpoint may lack LSTM / value-head weights.
            self.model.load_state_dict(state, strict=False)
            print(f'Loaded imitation weights  val_acc={ckpt.get("val_acc", 0):.3f}')
            if stripped:
                print('  Skipped mismatched tensors (will be re-initialised): ' + ', '.join(f'{k} ckpt={a} model={b}' for k, a, b in stripped))
            # Value head was never trained during imitation — re-initialise it.
            for layer in self.model.value_head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)
            print('Value head re-initialized.')
            # If the actor's last layer was stripped, re-init it as at construction (near-uniform +
            # slight negative STOP bias).
            actor_last_keys = {'actor_head.3.weight', 'actor_head.3.bias'}
            if any(k in actor_last_keys for k, _, _ in stripped):
                last = self.model.actor_head[-1]
                nn.init.orthogonal_(last.weight, gain=0.01)
                nn.init.zeros_(last.bias)
                if self.model.N_ACTIONS == 9:
                    with torch.no_grad():
                        last.bias[8] = -1.0
                print(f'Actor head last layer re-initialised for N_ACTIONS={self.model.N_ACTIONS}.')
            return 1, 0.0

        print('WARNING: No weights found, training from scratch.')
        return 1, 0.0

    def _pick_sample_index(self, train_samples) -> int:
        """Return a random training-sample index at or below the current curriculum difficulty.

        Per-sample difficulties are computed once and cached; falls back to the first 10 samples if
        too few qualify.
        """
        difficulty = self.curriculum.get_difficulty()
        if not hasattr(self, '_sample_difficulties'):
            # Score every sample once, evicting as we go so the pass doesn't pin the
            # whole dataset in the main process's cache.
            self._sample_difficulties = []
            _cache = getattr(train_samples, '_cache', None)
            for i in range(len(train_samples)):
                s = train_samples[i]
                d = self.curriculum.compute_sample_difficulty(s['centerline'], s.get('vessel_mask', s['centerline']))
                self._sample_difficulties.append(d)
                if _cache is not None:
                    _cache.pop(i, None)

        valid = [i for i, d in enumerate(self._sample_difficulties) if d <= difficulty]
        if len(valid) < 10:
            valid = list(range(min(10, len(train_samples))))
        return int(np.random.choice(valid))

    def _collect_rollout_vec(
        self,
        vec_env,
        buffers: List[RolloutBuffer],
        train_samples,
        obs_list,
        lstm_states_list,
        ep_rewards,
        ep_lengths,
        episode_rewards: deque,
        episode_lengths: deque,
        current_sample_ids: List[int],
        accumulated_coverage: dict,
    ):
        """Collect ``steps_per_iter`` steps across parallel envs into per-env buffers; return last values for GAE.

        Steps all envs with batched inference, normalises rewards, stores transitions with the
        pre-action LSTM state, and on episode end advances the curriculum, accumulates per-image
        coverage (so later episodes get prior_coverage), and reseeds the env.
        """
        n_envs = vec_env.n_envs
        steps_collected = 0

        while steps_collected < self.steps_per_iter:
            obs_batch = torch.from_numpy(np.stack(obs_list)).float().to(self.device)  # (n_envs, C, H, W)

            if self.use_lstm:
                h_cat = torch.cat([s[0].to(self.device) for s in lstm_states_list], dim=0)
                h_c_cat = torch.cat([s[1].to(self.device) for s in lstm_states_list], dim=0)
                batched_lstm = (h_cat, h_c_cat)
            else:
                batched_lstm = None

            with torch.no_grad():
                (actions, log_probs, _, values, new_lstm) = self.model.get_action_and_value(obs_batch, batched_lstm)

            action_list = [actions[i].item() for i in range(n_envs)]
            (all_obs, all_rewards, all_terminated, all_truncated, all_infos) = vec_env.step(action_list)

            for i in range(n_envs):
                action_i = action_list[i]
                log_prob_i = log_probs[i].item()
                value_i = values[i].item()
                next_obs = all_obs[i]
                reward = all_rewards[i]
                done = all_terminated[i] or all_truncated[i]

                # Capture the LSTM state BEFORE this action (what the buffer/training needs).
                if self.use_lstm:
                    lstm_state_i = (lstm_states_list[i][0], lstm_states_list[i][1])
                else:
                    lstm_state_i = None

                # Single shared std-only normaliser keeps the terminal early-stop penalty negative.
                self.reward_normalizer.update(reward)
                norm_reward = self.reward_normalizer.normalize(reward)

                buffers[i].add(obs_list[i], action_i, log_prob_i, norm_reward, value_i, float(done), lstm_state_i)

                ep_rewards[i] += reward
                ep_lengths[i] += 1
                steps_collected += 1

                # Accumulate per-component reward means for logging.
                info_i = all_infos[i] if isinstance(all_infos, (list, tuple)) else {}
                if isinstance(info_i, dict):
                    for _key in RewardCalculator.BREAKDOWN_KEYS:
                        _val = info_i.get(_key)
                        if _val is not None:
                            self._rwrd_sums[_key] = self._rwrd_sums.get(_key, 0.0) + _val
                            self._rwrd_counts[_key] = self._rwrd_counts.get(_key, 0) + 1

                # Advance the LSTM state to the post-action state.
                if self.use_lstm:
                    lstm_states_list[i] = (new_lstm[0][i : i + 1, :].detach().cpu(), new_lstm[1][i : i + 1, :].detach().cpu())

                if done:
                    episode_rewards.append(ep_rewards[i])
                    episode_lengths.append(ep_lengths[i])

                    # Tally termination cause for per-iteration instrumentation.
                    _tr = all_infos[i].get('terminal_reason') if isinstance(all_infos[i], dict) else None
                    if _tr:
                        self._term_counts[_tr] = self._term_counts.get(_tr, 0) + 1

                    # Reward-independent behaviour signals at episode end.
                    info_done = all_infos[i] if isinstance(all_infos[i], dict) else {}
                    if 'precision' in info_done:
                        self._ep_on_track.append(float(info_done['precision']))
                    if 'coverage_ratio' in info_done:
                        self._ep_coverage.append(float(info_done['coverage_ratio']))
                    if _tr == 'stop':
                        self._stop_steps.append(int(info_done.get('step_count', 0)))
                        self._stop_cov.append(float(info_done.get('coverage_ratio', 0.0)))

                    success = self.curriculum.is_episode_successful(all_infos[i])
                    prev_stage = self.curriculum.current_stage_idx
                    self.curriculum.step(success=success, stage_iter=self._stage_iter)

                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

                    if self.use_lstm:
                        fresh = self.model.init_hidden(batch_size=1, device=self.device)
                        lstm_states_list[i] = (fresh[0].detach().cpu(), fresh[1].detach().cpu())

                    # Accumulate per-image coverage so the next episode on this image gets
                    # prior_coverage — closing the train/eval gap for the gated connectivity bonus.
                    finished_sample_id = current_sample_ids[i]
                    cov_mask = vec_env.get_coverage_mask(i)
                    if cov_mask is not None and finished_sample_id >= 0:
                        prev = accumulated_coverage.get(finished_sample_id)
                        if prev is None:
                            accumulated_coverage[finished_sample_id] = (cov_mask > 0).astype(np.float32)
                        else:
                            accumulated_coverage[finished_sample_id] = np.where(cov_mask > 0, 1.0, prev)
                        # Bound memory by evicting the oldest entry.
                        if len(accumulated_coverage) > 512:
                            accumulated_coverage.pop(next(iter(accumulated_coverage)))

                    # Reseed via the curriculum, sending only the index over IPC.
                    sample_idx = self._pick_sample_index(train_samples)
                    current_sample_ids[i] = sample_idx
                    prior_cov = accumulated_coverage.get(sample_idx)
                    vec_env.set_sample(i, sample_idx, prior_coverage=prior_cov)
                    overrides = self._get_curriculum_overrides_dict()
                    vec_env.apply_overrides(i, overrides)
                    next_obs = vec_env.reset(i)

                    if self.curriculum.current_stage_idx != prev_stage:
                        stage = self.curriculum.get_current_stage()
                        print(f'  → Curriculum stage: {stage.name}')
                        # Propagate the new stage to every env.
                        new_overrides = self._get_curriculum_overrides_dict()
                        for j in range(n_envs):
                            vec_env.apply_overrides(j, new_overrides)

                obs_list[i] = next_obs

        # Last values for GAE bootstrap.
        obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            if self.use_lstm:
                h_cat = torch.cat([s[0] for s in lstm_states_list], dim=0).to(self.device)
                h_c_cat = torch.cat([s[1] for s in lstm_states_list], dim=0).to(self.device)
                last_values = self.model.get_value(obs_batch, (h_cat, h_c_cat))
            else:
                last_values = self.model.get_value(obs_batch, None)
        return [last_values[i].item() for i in range(n_envs)]

    def train(self, train_samples, val_samples, save_path: str, log_path: str, imitation_path: str = '') -> None:
        """Run the full PPO loop: warm-start, then per-iteration rollout, update, curriculum, eval, checkpoint, log.

        Resumes from ``save_path`` or warm-starts from ``imitation_path``, saves the best-by-clDice
        checkpoint plus periodic snapshots, writes a per-iteration CSV, and early-stops on patience.
        """
        from environment.vec_env import SubprocVecEnv

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        start_iteration, best_cldice = self.load_checkpoint(save_path, imitation_path)

        N_ENVS = self.config.get('training', {}).get('ppo', {}).get('n_envs', 8)
        vec_env = SubprocVecEnv(self.config, n_envs=N_ENVS)

        # Track the current sample per env to update its accumulated coverage after each episode.
        current_sample_ids: List[int] = [-1] * N_ENVS
        # Per-sample accumulated coverage enabling multi-episode coverage training.
        accumulated_coverage: dict = {}

        obs_list = []
        lstm_states_list = []
        ep_rewards = [0.0] * N_ENVS
        ep_lengths = [0] * N_ENVS

        overrides = self._get_curriculum_overrides_dict()
        for i in range(N_ENVS):
            sample_idx = self._pick_sample_index(train_samples)
            current_sample_ids[i] = sample_idx
            vec_env.set_sample(i, sample_idx)
            vec_env.apply_overrides(i, overrides)
            obs = vec_env.reset(i)
            obs_list.append(obs)
            hidden = self.model.init_hidden(batch_size=1, device=self.device)
            lstm_states_list.append(tuple(t.detach().cpu() for t in hidden) if hidden is not None else None)

        buffers = [RolloutBuffer() for _ in range(N_ENVS)]
        episode_rewards: deque = deque(maxlen=50)
        episode_lengths: deque = deque(maxlen=50)

        _csv_fields = (
            [
                'iteration',
                'mean_reward',
                'mean_ep_length',
                'policy_loss',
                'value_loss',
                'entropy',
                'approx_kl',
                'explained_variance',
                'grad_norm',
                'lr',
                'stage',
            ]
            + list(RewardCalculator.BREAKDOWN_KEYS)
            + [
                'val_coverage',
                'val_f1',
                'val_cldice',
                'val_recall_thin',
                'val_recall_med',
                'val_recall_thick',
                'term_stop_frac',
                'term_off_track_frac',
                'term_max_steps_frac',
                'term_oob_frac',
                'on_track_frac',
                'mean_cov_at_done',
                'mean_step_at_stop',
                'mean_cov_at_stop',
            ]
        )
        _csv_file = open(log_path, 'w', newline='', encoding='utf-8')
        _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields, extrasaction='ignore')
        _csv_writer.writeheader()

        print(f'Starting curriculum stage: {self.curriculum.get_current_stage().name}')
        print(
            f'\nStarting PPO — iters {start_iteration}–{self.num_iterations} '
            f'× {self.steps_per_iter} steps  {N_ENVS} envs'
            f'  LSTM={"ON chunk_len=" + str(self.lstm_chunk_length) + " burn_in=" + str(self.lstm_burn_in) if self.use_lstm else "OFF"}\n'
        )

        # Sync the per-stage counter in case the checkpoint resumed mid-curriculum.
        self._last_stage_idx = self.curriculum.current_stage_idx

        for iteration in range(start_iteration, self.num_iterations + 1):
            # Reset the per-stage counter (and entropy-anneal state) on stage change.
            if self.curriculum.current_stage_idx != self._last_stage_idx:
                self._stage_iter = 0
                self._last_stage_idx = self.curriculum.current_stage_idx
                self._entropy_frozen = False
                self._eval_cldice_window.clear()
            self._stage_iter += 1

            for buf in buffers:
                buf.reset()
            self._rwrd_sums: dict = {}
            self._rwrd_counts: dict = {}
            self._term_counts: dict = {}
            self._ep_on_track: list = []  # info["precision"] at done
            self._ep_coverage: list = []  # info["coverage_ratio"] at done
            self._stop_steps: list = []  # info["step_count"] for STOP episodes
            self._stop_cov: list = []  # info["coverage_ratio"] for STOP episodes
            self.model.eval()

            last_values = self._collect_rollout_vec(
                vec_env,
                buffers,
                train_samples,
                obs_list,
                lstm_states_list,
                ep_rewards,
                ep_lengths,
                episode_rewards,
                episode_lengths,
                current_sample_ids=current_sample_ids,
                accumulated_coverage=accumulated_coverage,
            )

            self.model.train()
            stats = self._ppo_update(buffers, last_values)
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()

            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
            log = (
                f'Iter {iteration:4d}/{self.num_iterations}  '
                f'reward={mean_reward:7.3f}  ep_len={mean_length:6.1f}  '
                f'p_loss={stats["policy_loss"]:7.4f}  '
                f'v_loss={stats["value_loss"]:6.4f}  '
                f'entropy={stats["entropy"]:.3f}'
            )

            did_eval = False
            if iteration % self.eval_every == 0 and val_samples:
                ev = evaluate(self.model, val_samples, self.config, self.device, self.tolerance, n_episodes=4)
                did_eval = True
                stage = self.curriculum.get_current_stage()
                log += (
                    f'  |  val_cov={ev["mean_coverage"]:.3f}'
                    f'  val_f1={ev["mean_f1"]:.3f}'
                    f'  val_clDice={ev["mean_cldice"]:.3f}'
                    f'  R[t/m/T]={ev.get("mean_recall_thin", 0):.2f}/'
                    f'{ev.get("mean_recall_med", 0):.2f}/'
                    f'{ev.get("mean_recall_thick", 0):.2f}'
                    f'  stage={stage.name}'
                    f'  ent_c={self.entropy_coef:.3f}'
                )

                if ev['mean_cldice'] > best_cldice:
                    best_cldice = ev['mean_cldice']
                    self.no_improve_count = 0
                    torch.save(
                        {
                            'iteration': iteration,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_cldice': best_cldice,
                            'config': self.config,
                            'curriculum_state': self.curriculum.state_dict(),
                        },
                        save_path,
                    )
                    log += f'  ✓ saved (best clDice={best_cldice:.3f})'
                else:
                    self.no_improve_count += 1

                # Performance-gated entropy anneal: freeze when val_clDice stops improving over the window.
                self._eval_cldice_window.append(ev['mean_cldice'])
                if len(self._eval_cldice_window) >= 2:
                    recent_improvement = max(self._eval_cldice_window) - min(self._eval_cldice_window)
                    was_frozen = self._entropy_frozen
                    self._entropy_frozen = recent_improvement < 0.005
                    if self._entropy_frozen and not was_frozen:
                        log += f'  [entropy frozen @ {self.entropy_coef:.4f}]'
                    elif not self._entropy_frozen and was_frozen:
                        log += '  [entropy unfrozen]'
                if self.no_improve_count >= self.patience:
                    print(log)
                    print(f'\nEarly stopping: no improvement for {self.patience} eval cycles.')
                    break

            print(log)
            _csv_row = {
                'iteration': iteration,
                'mean_reward': mean_reward,
                'mean_ep_length': mean_length,
                'policy_loss': stats['policy_loss'],
                'value_loss': stats['value_loss'],
                'entropy': stats['entropy'],
                'approx_kl': stats['approx_kl'],
                'explained_variance': stats['explained_variance'],
                'grad_norm': stats['grad_norm'],
                'lr': current_lr,
                'stage': self.curriculum.get_current_stage().name,
            }
            for _k in RewardCalculator.BREAKDOWN_KEYS:
                _csv_row[_k] = self._rwrd_sums.get(_k, 0.0) / max(self._rwrd_counts.get(_k, 1), 1)
            # Per-iteration termination-reason fractions.
            _term_total = max(sum(self._term_counts.values()), 1)
            _csv_row['term_stop_frac'] = self._term_counts.get('stop', 0) / _term_total
            _csv_row['term_off_track_frac'] = self._term_counts.get('off_track', 0) / _term_total
            _csv_row['term_max_steps_frac'] = self._term_counts.get('max_steps', 0) / _term_total
            _csv_row['term_oob_frac'] = self._term_counts.get('oob', 0) / _term_total
            _csv_row['on_track_frac'] = float(np.mean(self._ep_on_track)) if self._ep_on_track else 0.0
            _csv_row['mean_cov_at_done'] = float(np.mean(self._ep_coverage)) if self._ep_coverage else 0.0
            _csv_row['mean_step_at_stop'] = float(np.mean(self._stop_steps)) if self._stop_steps else 0.0
            _csv_row['mean_cov_at_stop'] = float(np.mean(self._stop_cov)) if self._stop_cov else 0.0
            if did_eval:
                _csv_row.update(
                    {
                        'val_coverage': ev['mean_coverage'],
                        'val_f1': ev['mean_f1'],
                        'val_cldice': ev['mean_cldice'],
                        'val_recall_thin': ev.get('mean_recall_thin', 0.0),
                        'val_recall_med': ev.get('mean_recall_med', 0.0),
                        'val_recall_thick': ev.get('mean_recall_thick', 0.0),
                    }
                )
            _csv_writer.writerow(_csv_row)
            _csv_file.flush()

            if iteration % self.save_every == 0:
                ckpt_path = save_path.replace('.pt', f'_iter{iteration}.pt')
                torch.save(
                    {
                        'iteration': iteration,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'config': self.config,
                    },
                    ckpt_path,
                )

        vec_env.close()
        _csv_file.close()

        print(f'\nDone. Best clDice: {best_cldice:.3f}')
        print(f'Weights: {save_path}')
        print(f'Log:     {log_path}')

        try:
            from training.plots import plot_ppo_log

            png = plot_ppo_log(log_path)
            if png:
                print(f'Plot:    {png}')
        except Exception as e:
            print(f'[plot_ppo_log] skipped: {e}')
