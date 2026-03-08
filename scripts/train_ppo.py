# scripts/train_ppo.py
"""
PPO Training Loop — Step 2 of 2
Loads imitation-pretrained weights and fine-tunes with RL.

Run AFTER train_imitation.py.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from collections import deque
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_network import ActorCriticNetwork
from environment.vessel_env import VesselTracingEnv
from data.centerline_extraction import CenterlineExtractor
from data.fundus_preprocessor import FundusPreprocessor

_preprocessor = FundusPreprocessor()

# ==========================================
# CONFIG
# ==========================================
DRIVE_ROOT        = r"C:\ZHAW\BA\data\DRIVE\training"
IMITATION_WEIGHTS = r"C:\ZHAW\BA\weights\imitation_policy.pt"
SAVE_PATH         = r"C:\ZHAW\BA\weights\ppo_policy.pt"
LOG_PATH          = r"C:\ZHAW\BA\weights\ppo_log.txt"

IMAGES_DIR  = os.path.join(DRIVE_ROOT, "images")
VESSELS_DIR = os.path.join(DRIVE_ROOT, "1st_manual")
MASKS_DIR   = os.path.join(DRIVE_ROOT, "mask")

# PPO hyperparameters
LR              = 1e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.1
ENTROPY_COEF    = 0.05
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 1.0
PPO_EPOCHS      = 4
MINI_BATCH_SIZE = 256

# Training schedule
NUM_ITERATIONS  = 1000
STEPS_PER_ITER  = 4096
EVAL_EVERY      = 25
SAVE_EVERY      = 50

# Environment
TOLERANCE      = 2.0
OBS_SIZE       = 65
MAX_STEPS      = 2000
USE_VESSELNESS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'policy': {
        'hidden_dim': 128,
        'lstm_hidden': 128,
        'use_lstm': False,
        'dropout': 0.05,
        'encoder_type': 'cnn',
    },
    'environment': {
        'observation_size': OBS_SIZE,
        'tolerance': TOLERANCE,
        'use_vesselness': USE_VESSELNESS,
        'max_steps_per_episode': MAX_STEPS,
        'max_off_track_streak': 8,   # increased — less aggressive termination on thin vessels
        'step_size': 1,
    },
    'reward': {
        'alpha_near': 0.1,
        'beta_coverage': 1.0,
        'gamma_off': -0.5,
        'lambda_revisit': -1.0,
        'step_cost': -0.01,
        'direction_bonus': 0.05,
        'terminal_f1_weight': 5.0,
        'use_potential_shaping': False,
    },
    'training': {
        'ppo': {'gamma': GAMMA}
    }
}


# ==========================================
# LOAD DRIVE SAMPLES
# ==========================================

def load_sample(img_id: str) -> Optional[Dict]:
    img_path    = os.path.join(IMAGES_DIR, f"{img_id}_training.tif")
    vessel_path = os.path.join(VESSELS_DIR, f"{img_id}_manual1.gif")
    mask_path   = os.path.join(MASKS_DIR,   f"{img_id}_training_mask.gif")

    if not os.path.exists(img_path):
        return None

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    vessel     = np.array(Image.open(vessel_path).convert('L'))
    vessel_bin = (vessel > 128).astype(np.uint8)

    fov_raw = np.array(Image.open(mask_path).convert('L'))
    fov_bin = (fov_raw > 128).astype(np.uint8)

    # Enhance green channel with full preprocessing pipeline:
    # green extraction → gamma → median blur → FOV mask → CLAHE → ROI normalise.
    # R and B channels are kept as raw normalised values.
    enhanced_green = _preprocessor.preprocess(
        image         = (img_rgb * 255).astype(np.uint8),
        external_mask = fov_raw,
    )
    img_rgb[:, :, 1] = enhanced_green  # replace green channel only

    extractor  = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    centerline = extractor.extract_centerline(vessel_bin)
    dist_tf    = extractor.compute_distance_transform(centerline, tolerance=TOLERANCE)

    return {
        'id':             img_id,
        'image':          img_rgb,
        'centerline':     centerline,
        'dist_transform': dist_tf,
        'fov_mask':       fov_bin,
        'vessel_mask':    vessel_bin,
    }


def load_all_samples() -> List[Dict]:
    # Clean data split:
    #   Train : 21-35 (15 images) — matches imitation training set
    #   Val   : 36-37 (2 images)  — monitored during PPO training
    #   Test  : 38-40 (3 images)  — held out, only used for final evaluation
    print("Loading DRIVE training samples (IDs 21-35)...")
    samples = []
    for i in range(21, 36):
        s = load_sample(str(i))
        if s:
            samples.append(s)
            print(f"  [{s['id']}] centerline px: {int(s['centerline'].sum())}")
    print(f"Loaded {len(samples)} training samples.\n")
    return samples


def load_val_samples() -> List[Dict]:
    # Val set: 36-37
    samples = []
    for i in range(36, 38):
        s = load_sample(str(i))
        if s:
            samples.append(s)
    return samples


# ==========================================
# ROLLOUT BUFFER
# ==========================================

class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs       = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.values    = []
        self.dones     = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = self.rewards
        values  = self.values + [last_value]
        dones   = self.dones

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t+1] * (1 - dones[t]) - values[t]
            gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns    = advantages + torch.tensor(self.values, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def get_tensors(self):
        obs       = torch.tensor(np.array(self.obs),       dtype=torch.float32)
        actions   = torch.tensor(np.array(self.actions),   dtype=torch.long)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32)
        return obs, actions, log_probs


# ==========================================
# PPO UPDATE
# ==========================================

def ppo_update(model: ActorCriticNetwork,
               optimizer: optim.Optimizer,
               buffer: RolloutBuffer,
               last_value: float) -> Dict[str, float]:

    returns, advantages = buffer.compute_returns_and_advantages(last_value)
    obs, actions, old_log_probs = buffer.get_tensors()

    returns       = returns.to(DEVICE)
    advantages    = advantages.to(DEVICE)
    obs           = obs.to(DEVICE)
    actions       = actions.to(DEVICE)
    old_log_probs = old_log_probs.to(DEVICE)

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    n_updates         = 0
    dataset_size      = len(obs)

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, MINI_BATCH_SIZE):
            idx = indices[start:start + MINI_BATCH_SIZE]

            mb_obs        = obs[idx]
            mb_actions    = actions[idx]
            mb_old_lp     = old_log_probs[idx]
            mb_returns    = returns[idx]
            mb_advantages = advantages[idx]

            logits, values, _ = model(mb_obs)
            dist     = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(mb_actions)
            entropy  = dist.entropy().mean()

            ratio       = torch.exp(log_prob - mb_old_lp)
            surr1       = ratio * mb_advantages
            surr2       = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clamp both values and returns to prevent exploding value loss
            value_loss = nn.functional.mse_loss(
                torch.clamp(values,      -10.0, 10.0),
                torch.clamp(mb_returns,  -10.0, 10.0)
            )

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()
            n_updates         += 1

    return {
        'policy_loss': total_policy_loss / n_updates,
        'value_loss':  total_value_loss  / n_updates,
        'entropy':     total_entropy     / n_updates,
    }


# ==========================================
# EVALUATION
# ==========================================

def evaluate(model: ActorCriticNetwork,
             val_samples: List[Dict],
             n_episodes: int = 5) -> Dict[str, float]:
    model.eval()
    coverages = []
    f1_scores = []

    with torch.no_grad():
        for sample in val_samples:
            env = VesselTracingEnv(CONFIG)
            env.set_data(
                image=sample['image'],
                centerline=sample['centerline'],
                distance_transform=sample['dist_transform'],
                fov_mask=sample['fov_mask'],
            )

            cl_points = np.argwhere(sample['centerline'] > 0)
            if len(cl_points) == 0:
                continue

            for _ in range(n_episodes):
                idx    = np.random.randint(len(cl_points))
                start  = tuple(cl_points[idx])
                obs, _ = env.reset(start_position=start)

                done = False
                while not done:
                    obs_t  = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    logits, _, _ = model(obs_t)
                    action = logits.argmax(dim=-1).item()
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                coverages.append(info['coverage_ratio'])

                from data.centerline_extraction import compute_centerline_f1
                metrics = compute_centerline_f1(
                    env.covered_centerline,
                    sample['centerline'],
                    tolerance=TOLERANCE
                )
                f1_scores.append(metrics['f1'])

    model.train()
    return {
        'mean_coverage': float(np.mean(coverages)) if coverages else 0.0,
        'mean_f1':       float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


# ==========================================
# MAIN TRAINING LOOP
# ==========================================

def train_ppo():
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # --- Load data ---
    train_samples = load_all_samples()
    val_samples   = load_val_samples()

    if not train_samples:
        print("ERROR: No training samples loaded.")
        return

    # --- Build model ---
    model     = ActorCriticNetwork(CONFIG).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=NUM_ITERATIONS
    )
    start_iteration = 1
    best_f1         = 0.0

    # Resume from PPO checkpoint if it exists, otherwise load imitation weights
    if os.path.exists(SAVE_PATH):
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iteration = checkpoint.get('iteration', 0) + 1
        best_f1         = checkpoint.get('best_f1', 0.0)
        print(f"Resumed from PPO checkpoint  iter={start_iteration-1}  best_F1={best_f1:.3f}")
    elif os.path.exists(IMITATION_WEIGHTS):
        checkpoint = torch.load(IMITATION_WEIGHTS, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded imitation weights  val_acc={checkpoint.get('val_acc', '?'):.3f}")
        # Reset value head — never trained during imitation
        for layer in model.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)
        print("Value head re-initialized.")
    else:
        print("WARNING: No weights found, training from scratch.")

    # --- Setup ---
    env             = VesselTracingEnv(CONFIG)
    buffer          = RolloutBuffer()
    episode_rewards = deque(maxlen=50)
    episode_lengths = deque(maxlen=50)
    log_lines       = []

    # Initial episode
    current_sample = np.random.choice(train_samples)
    env.set_data(
        image=current_sample['image'],
        centerline=current_sample['centerline'],
        distance_transform=current_sample['dist_transform'],
        fov_mask=current_sample['fov_mask'],
    )
    obs, _    = env.reset()
    ep_reward = 0.0
    ep_length = 0

    print(f"\nStarting PPO training — iterations {start_iteration}–{NUM_ITERATIONS} × {STEPS_PER_ITER} steps\n")

    for iteration in range(start_iteration, NUM_ITERATIONS + 1):
        buffer.reset()
        model.eval()

        # --- Collect rollout ---
        for _ in range(STEPS_PER_ITER):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, log_prob, _, value, _ = model.get_action_and_value(obs_t)

            action_int = action.item()
            log_prob_f = log_prob.item()
            value_f    = value.item()

            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            # Clip reward to [-1, 1] — standard PPO stabilization
            clipped_reward = np.clip(reward, -1.0, 1.0)
            buffer.add(obs, action_int, log_prob_f, clipped_reward, value_f, float(done))

            ep_reward += reward  # log raw reward
            ep_length += 1
            obs = next_obs

            if done:
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                ep_reward = 0.0
                ep_length = 0

                current_sample = np.random.choice(train_samples)
                env.set_data(
                    image=current_sample['image'],
                    centerline=current_sample['centerline'],
                    distance_transform=current_sample['dist_transform'],
                    fov_mask=current_sample['fov_mask'],
                )
                obs, _ = env.reset()

        # Bootstrap last value
        with torch.no_grad():
            obs_t      = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            last_value = model.get_value(obs_t).item()

        # --- PPO update ---
        model.train()
        update_stats = ppo_update(model, optimizer, buffer, last_value)
        scheduler.step()

        # --- Logging ---
        mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0

        log = (f"Iter {iteration:4d}/{NUM_ITERATIONS}  "
               f"reward={mean_reward:7.3f}  "
               f"ep_len={mean_length:6.1f}  "
               f"p_loss={update_stats['policy_loss']:7.4f}  "
               f"v_loss={update_stats['value_loss']:6.4f}  "
               f"entropy={update_stats['entropy']:.3f}")

        # --- Evaluation ---
        if iteration % EVAL_EVERY == 0 and val_samples:
            eval_stats = evaluate(model, val_samples)
            log += (f"  |  val_coverage={eval_stats['mean_coverage']:.3f}"
                    f"  val_f1={eval_stats['mean_f1']:.3f}")

            if eval_stats['mean_f1'] > best_f1:
                best_f1 = eval_stats['mean_f1']
                torch.save({
                    'iteration':            iteration,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1':              best_f1,
                    'config':               CONFIG,
                }, SAVE_PATH)
                log += f"  ✓ saved (best F1={best_f1:.3f})"

        print(log)
        log_lines.append(log)

        # Periodic checkpoint
        if iteration % SAVE_EVERY == 0:
            ckpt_path = SAVE_PATH.replace('.pt', f'_iter{iteration}.pt')
            torch.save({
                'iteration':            iteration,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config':               CONFIG,
            }, ckpt_path)

    # Save log
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

    print(f"\nDone. Best F1: {best_f1:.3f}")
    print(f"Weights: {SAVE_PATH}")
    print(f"Log:     {LOG_PATH}")


if __name__ == "__main__":
    train_ppo()