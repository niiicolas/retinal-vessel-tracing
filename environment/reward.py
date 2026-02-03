# environment/reward.py
"""
Reward calculation for vessel tracing.
"""

import numpy as np
from typing import Dict, Any, Optional


class RewardCalculator:
    """
    Calculates rewards for the vessel tracing agent.
    
    Implements the tolerance-aware reward system from the proposal:
    - Proximity reward (staying near centerline)
    - Coverage bonus (discovering new centerline segments)
    - Off-track penalty
    - Revisit penalty
    - Step cost
    - Direction smoothness bonus
    - Potential-based shaping
    """
    
    def __init__(self, config: Dict[str, Any]):
        reward_config = config.get('reward', {})
        
        self.alpha = reward_config.get('alpha_near', 1.0)
        self.beta = reward_config.get('beta_coverage', 2.0)
        self.gamma_off = reward_config.get('gamma_off', -1.0)
        self.lambda_revisit = reward_config.get('lambda_revisit', -0.5)
        self.step_cost = reward_config.get('step_cost', -0.01)
        self.direction_bonus = reward_config.get('direction_bonus', 0.1)
        self.terminal_f1_weight = reward_config.get('terminal_f1_weight', 10.0)
        self.use_potential_shaping = reward_config.get('use_potential_shaping', True)
        
        self.tolerance = config.get('environment', {}).get('tolerance', 2.0)
        self.gamma = config.get('training', {}).get('ppo', {}).get('gamma', 0.99)
        
        # Large penalties
        self.out_of_bounds_penalty = -10.0
        self.off_track_termination_penalty = -5.0
        
    def compute_step_reward(self,
                            distance: float,
                            is_revisit: bool,
                            is_on_track: bool,
                            new_coverage: float,
                            prev_distance: float,
                            action: int,
                            prev_action: Optional[int]) -> float:
        """
        Compute the reward for a single step.
        
        Args:
            distance: Current distance to nearest centerline pixel
            is_revisit: Whether current position was already visited
            is_on_track: Whether within tolerance of centerline
            new_coverage: Number of new centerline pixels covered
            prev_distance: Previous distance to centerline
            action: Current action taken
            prev_action: Previous action (for smoothness)
            
        Returns:
            Total reward for this step
        """
        reward = 0.0
        
        # 1. Proximity reward
        r_near = self.alpha * max(0, 1.0 - distance / self.tolerance)
        reward += r_near
        
        # 2. Coverage bonus
        if new_coverage > 0:
            r_cov = self.beta * new_coverage
            reward += r_cov
        
        # 3. Off-track penalty
        if not is_on_track:
            reward += self.gamma_off
            
        # 4. Revisit penalty
        if is_revisit:
            reward += self.lambda_revisit
            
        # 5. Step cost
        reward += self.step_cost
        
        # 6. Direction smoothness bonus
        if prev_action is not None and action == prev_action:
            reward += self.direction_bonus
            
        # 7. Potential-based shaping
        if self.use_potential_shaping:
            phi_curr = -distance
            phi_prev = -prev_distance
            shaping = self.gamma * phi_curr - phi_prev
            reward += shaping
            
        return reward
    
    def compute_terminal_reward(self,
                                covered_centerline: np.ndarray,
                                gt_centerline: np.ndarray) -> float:
        """
        Compute terminal reward based on F1 score.
        
        Args:
            covered_centerline: Binary mask of covered centerline
            gt_centerline: Ground truth centerline
            
        Returns:
            Terminal bonus proportional to F1 score
        """
        # Compute precision and recall
        covered = covered_centerline > 0
        gt = gt_centerline > 0
        
        true_positive = np.logical_and(covered, gt).sum()
        predicted_positive = covered.sum()
        actual_positive = gt.sum()
        
        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(actual_positive, 1)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        return self.terminal_f1_weight * f1
    
    def compute_out_of_bounds_penalty(self) -> float:
        """Return penalty for going out of bounds."""
        return self.out_of_bounds_penalty
    
    def compute_off_track_termination_penalty(self) -> float:
        """Return penalty for terminating due to off-track streak."""
        return self.off_track_termination_penalty
