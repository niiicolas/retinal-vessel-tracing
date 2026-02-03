# training/ppo.py
"""
Proximal Policy Optimization implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    values: List[float]
    log_probs: List[float]
    dones: List[bool]
    
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    """
    PPO trainer with GAE for vessel tracing.
    """
    
    def __init__(self,
                 policy: nn.Module,
                 config: Dict[str, Any],
                 device: torch.device):
        self.policy = policy
        self.config = config
        self.device = device
        
        ppo_config = config.get('training', {}).get('ppo', {})
        
        self.lr = ppo_config.get('learning_rate', 3e-4)
        self.gamma = ppo_config.get('gamma', 0.99)
        self.gae_lambda = ppo_config.get('gae_lambda', 0.95)
        self.clip_epsilon = ppo_config.get('clip_epsilon', 0.2)
        self.value_loss_coef = ppo_config.get('value_loss_coef', 0.5)
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)
        self.max_grad_norm = ppo_config.get('max_grad_norm', 0.5)
        self.num_epochs = ppo_config.get('num_epochs', 10)
        self.batch_size = ppo_config.get('batch_size', 64)
        self.rollout_length = ppo_config.get('rollout_length', 256)
        self.total_timesteps = ppo_config.get('total_timesteps', 5_000_000)
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_timesteps // self.rollout_length
        )
        
        # Logging
        self.use_wandb = config.get('logging', {}).get('use_wandb', False) and WANDB_AVAILABLE
        self.log_interval = config.get('logging', {}).get('log_interval', 100)
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.coverage_ratios = deque(maxlen=100)
        
    def compute_gae(self,
                    rewards: np.ndarray,
                    values: np.ndarray,
                    dones: np.ndarray,
                    next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            next_value: Value estimate for final state
            
        Returns:
            advantages: GAE advantages
            returns: Target returns
        """
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + values
        return advantages, returns
    
    def update(self, buffer: RolloutBuffer, next_value: float) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            buffer: Rollout buffer with collected experience
            next_value: Value estimate for final state
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to numpy arrays
        observations = np.array(buffer.observations)
        actions = np.array(buffer.actions)
        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        log_probs = np.array(buffer.log_probs)
        dones = np.array(buffer.dones)
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # Multiple epochs of updates
        dataset_size = len(buffer)
        
        for epoch in range(self.num_epochs):
            # Random permutation for mini-batches
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                _, new_log_probs, entropy, new_values, _ = self.policy.get_action_and_value(
                    batch_obs, action=batch_actions
                )
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_loss_coef * value_loss 
                    + self.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def collect_rollout(self, 
                        env,
                        buffer: RolloutBuffer,
                        hidden: Optional[Tuple] = None
                        ) -> Tuple[float, Optional[Tuple]]:
        """
        Collect rollout data from environment.
        
        Args:
            env: Vectorized environment
            buffer: Buffer to store experience
            hidden: Optional LSTM hidden state
            
        Returns:
            next_value: Value estimate for final state
            hidden: Updated hidden state
        """
        self.policy.eval()
        buffer.clear()
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        with torch.no_grad():
            for _ in range(self.rollout_length):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                
                if len(obs_tensor.shape) == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                action, log_prob, _, value, hidden = self.policy.get_action_and_value(
                    obs_tensor, hidden
                )
                
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()
                
                # Handle single env
                if isinstance(action, np.ndarray) and len(action.shape) == 0:
                    action = int(action)
                elif isinstance(action, np.ndarray):
                    action = action[0]
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                buffer.add(
                    obs=obs if len(obs.shape) == 3 else obs[0],
                    action=action,
                    reward=reward if np.isscalar(reward) else reward[0],
                    value=value if np.isscalar(value) else value[0],
                    log_prob=log_prob if np.isscalar(log_prob) else log_prob[0],
                    done=done
                )
                
                episode_reward += reward if np.isscalar(reward) else reward[0]
                episode_length += 1
                
                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    if 'coverage_ratio' in info:
                        self.coverage_ratios.append(info['coverage_ratio'])
                    
                    obs, _ = env.reset()
                    episode_reward = 0
                    episode_length = 0
                    if self.policy.use_lstm:
                        hidden = None
                else:
                    obs = next_obs
            
            # Get value for final state
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            next_value = self.policy.get_value(obs_tensor, hidden)
            next_value = next_value.cpu().numpy()
            if not np.isscalar(next_value):
                next_value = next_value[0]
        
        return next_value, hidden
    
    def train(self, 
              env,
              num_timesteps: Optional[int] = None,
              callback=None) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            env: Training environment
            num_timesteps: Total timesteps to train
            callback: Optional callback function
            
        Returns:
            Training history
        """
        if num_timesteps is None:
            num_timesteps = self.total_timesteps
            
        buffer = RolloutBuffer()
        hidden = None
        
        num_updates = num_timesteps // self.rollout_length
        
        history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episode_reward': [],
            'episode_length': [],
            'coverage_ratio': []
        }
        
        pbar = tqdm(range(num_updates), desc="Training PPO")
        
        for update in pbar:
            # Collect rollout
            next_value, hidden = self.collect_rollout(env, buffer, hidden)
            
            # Update policy
            self.policy.train()
            metrics = self.update(buffer, next_value)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            history['policy_loss'].append(metrics['policy_loss'])
            history['value_loss'].append(metrics['value_loss'])
            history['entropy'].append(metrics['entropy'])
            
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths)
                avg_coverage = np.mean(self.coverage_ratios) if self.coverage_ratios else 0
                
                history['episode_reward'].append(avg_reward)
                history['episode_length'].append(avg_length)
                history['coverage_ratio'].append(avg_coverage)
                
                pbar.set_postfix({
                    'reward': f'{avg_reward:.2f}',
                    'coverage': f'{avg_coverage:.2%}',
                    'policy_loss': f'{metrics["policy_loss"]:.4f}'
                })
            
            # Wandb logging
            if self.use_wandb and update % self.log_interval == 0:
                log_dict = {
                    'policy_loss': metrics['policy_loss'],
                    'value_loss': metrics['value_loss'],
                    'entropy': metrics['entropy'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                if len(self.episode_rewards) > 0:
                    log_dict['episode_reward'] = np.mean(self.episode_rewards)
                    log_dict['episode_length'] = np.mean(self.episode_lengths)
                if self.coverage_ratios:
                    log_dict['coverage_ratio'] = np.mean(self.coverage_ratios)
                wandb.log(log_dict, step=update * self.rollout_length)
            
            # Callback
            if callback is not None:
                callback(update, metrics, self.policy)
        
        return history
    
    def save_checkpoint(self, path: str, extra_info: Dict = None):
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        if extra_info:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
