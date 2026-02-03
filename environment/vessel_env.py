# environment/vessel_env.py
"""
RL Environment for vessel tracing.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque

from .reward import RewardCalculator
from .observation import ObservationBuilder


@dataclass
class EnvConfig:
    """Environment configuration."""
    observation_size: int = 65
    step_size: int = 1
    tolerance: float = 2.0
    max_off_track_streak: int = 5
    max_steps_per_episode: int = 2000
    use_vesselness: bool = True
    

class VesselTracingEnv(gym.Env):
    """
    RL Environment for tracing vessel centerlines.
    
    The agent navigates through the image, making 8-directional moves
    to trace vessel centerlines while avoiding off-centerline regions.
    """
    
    # 8-directional moves: N, NE, E, SE, S, SW, W, NW
    DIRECTIONS = np.array([
        [-1, 0],   # N
        [-1, 1],   # NE
        [0, 1],    # E
        [1, 1],    # SE
        [1, 0],    # S
        [1, -1],   # SW
        [0, -1],   # W
        [-1, -1],  # NW
    ])
    
    def __init__(self, 
                 config: Dict[str, Any],
                 image: Optional[np.ndarray] = None,
                 centerline: Optional[np.ndarray] = None,
                 distance_transform: Optional[np.ndarray] = None,
                 vesselness: Optional[np.ndarray] = None,
                 fov_mask: Optional[np.ndarray] = None):
        """
        Initialize the vessel tracing environment.
        
        Args:
            config: Configuration dictionary
            image: RGB image (H, W, 3) normalized to [0, 1]
            centerline: Binary centerline mask (H, W)
            distance_transform: Distance to centerline, clipped at tolerance (H, W)
            vesselness: Vesselness filter response (H, W), optional
            fov_mask: Field of view mask (H, W)
        """
        super().__init__()
        
        self.config = config
        env_config = config.get('environment', {})
        
        self.obs_size = env_config.get('observation_size', 65)
        self.step_size = env_config.get('step_size', 1)
        self.tolerance = env_config.get('tolerance', 2.0)
        self.max_off_track = env_config.get('max_off_track_streak', 5)
        self.max_steps = env_config.get('max_steps_per_episode', 2000)
        
        # Image data (will be set by reset or set_image)
        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.vesselness = vesselness
        self.fov_mask = fov_mask
        
        # Determine image size from provided data
        if image is not None:
            self.height, self.width = image.shape[:2]
        else:
            self.height, self.width = 512, 512  # Default
        
        # Action space: 8 directions + STOP
        self.action_space = spaces.Discrete(9)
        
        # Observation space
        self._setup_observation_space()
        
        # Initialize components
        self.reward_calculator = RewardCalculator(config)
        self.observation_builder = ObservationBuilder(config)
        
        # State variables
        self.position = None
        self.visited_mask = None
        self.trajectory = None
        self.step_count = 0
        self.off_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = None
        
    def _setup_observation_space(self):
        """Setup the observation space based on configuration."""
        # Channels: RGB (3) + visited mask (1) + prev direction (8) + optional vesselness (1)
        n_channels = 3 + 1 + 8
        if self.vesselness is not None or self.config.get('environment', {}).get('use_vesselness', True):
            n_channels += 1
            
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_channels, self.obs_size, self.obs_size),
            dtype=np.float32
        )
        
    def set_data(self, 
                 image: np.ndarray,
                 centerline: np.ndarray,
                 distance_transform: np.ndarray,
                 vesselness: Optional[np.ndarray] = None,
                 fov_mask: Optional[np.ndarray] = None):
        """Set the image data for the environment."""
        self.image = image
        self.centerline = centerline
        self.distance_transform = distance_transform
        self.vesselness = vesselness
        self.fov_mask = fov_mask if fov_mask is not None else np.ones_like(centerline)
        self.height, self.width = image.shape[:2]
        
    def reset(self, 
              seed: Optional[int] = None,
              start_position: Optional[Tuple[int, int]] = None,
              **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            start_position: Optional starting position (y, x)
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        if self.image is None:
            raise ValueError("No image data set. Call set_data() first.")
        
        # Reset state
        self.visited_mask = np.zeros((self.height, self.width), dtype=np.float32)
        self.trajectory = []
        self.step_count = 0
        self.off_track_streak = 0
        self.prev_direction = None
        self.covered_centerline = np.zeros_like(self.centerline)
        
        # Set starting position
        if start_position is not None:
            self.position = np.array(start_position, dtype=np.int32)
        else:
            self.position = self._sample_start_position()
            
        # Mark starting position as visited
        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))
        
        # Update covered centerline
        self._update_coverage()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _sample_start_position(self) -> np.ndarray:
        """Sample a random starting position on or near the centerline."""
        # Find valid centerline positions
        centerline_points = np.argwhere(self.centerline > 0)
        
        if len(centerline_points) == 0:
            # Fallback to random position in FOV
            fov_points = np.argwhere(self.fov_mask > 0)
            if len(fov_points) == 0:
                return np.array([self.height // 2, self.width // 2])
            idx = self.np_random.integers(len(fov_points))
            return centerline_points[idx]
            
        # Prefer endpoints for starting
        from data.centerline_extraction import CenterlineExtractor
        extractor = CenterlineExtractor()
        endpoints = extractor._find_endpoints(self.centerline)
        
        if endpoints:
            idx = self.np_random.integers(len(endpoints))
            return np.array(endpoints[idx])
        
        idx = self.np_random.integers(len(centerline_points))
        return centerline_points[idx]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-7 for directions, 8 for STOP)
            
        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode ended (goal reached or failure)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        self.step_count += 1
        
        # Handle STOP action
        if action == 8:
            reward = self.reward_calculator.compute_terminal_reward(
                self.covered_centerline, self.centerline
            )
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, True, False, info
        
        # Compute new position
        direction = self.DIRECTIONS[action] * self.step_size
        new_position = self.position + direction
        
        # Check bounds
        if not self._is_valid_position(new_position):
            reward = self.reward_calculator.compute_out_of_bounds_penalty()
            observation = self._get_observation()
            info = self._get_info()
            return observation, reward, True, False, info
        
        # Update position
        old_position = self.position.copy()
        self.position = new_position
        
        # Check if revisiting
        is_revisit = self.visited_mask[self.position[0], self.position[1]] > 0
        
        # Update visited mask
        self.visited_mask[self.position[0], self.position[1]] = 1.0
        self.trajectory.append(tuple(self.position))
        
        # Get distance to centerline
        distance = self.distance_transform[self.position[0], self.position[1]]
        
        # Check if on track
        is_on_track = distance <= self.tolerance
        
        # Update off-track streak
        if is_on_track:
            self.off_track_streak = 0
        else:
            self.off_track_streak += 1
        
        # Update coverage
        prev_coverage = self.covered_centerline.copy()
        self._update_coverage()
        new_coverage = self.covered_centerline.sum() - prev_coverage.sum()
        
        # Compute reward
        reward = self.reward_calculator.compute_step_reward(
            distance=distance,
            is_revisit=is_revisit,
            is_on_track=is_on_track,
            new_coverage=new_coverage,
            prev_distance=self.distance_transform[old_position[0], old_position[1]],
            action=action,
            prev_action=self.prev_direction
        )
        
        # Update previous direction
        self.prev_direction = action
        
        # Check termination
        terminated = self.off_track_streak >= self.max_off_track
        truncated = self.step_count >= self.max_steps
        
        if terminated:
            reward += self.reward_calculator.compute_off_track_termination_penalty()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is within valid bounds."""
        y, x = position
        half = self.obs_size // 2
        
        if y < half or y >= self.height - half:
            return False
        if x < half or x >= self.width - half:
            return False
        if self.fov_mask[y, x] == 0:
            return False
            
        return True
    
    def _update_coverage(self):
        """Update the covered centerline mask."""
        y, x = self.position
        
        # Mark centerline pixels within tolerance as covered
        y_min = max(0, y - int(self.tolerance) - 1)
        y_max = min(self.height, y + int(self.tolerance) + 2)
        x_min = max(0, x - int(self.tolerance) - 1)
        x_max = min(self.width, x + int(self.tolerance) + 2)
        
        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                if self.centerline[py, px] > 0:
                    dist = np.sqrt((py - y) ** 2 + (px - x) ** 2)
                    if dist <= self.tolerance:
                        self.covered_centerline[py, px] = 1.0
    
    def _get_observation(self) -> np.ndarray:
        """Construct the observation tensor."""
        return self.observation_builder.build(
            image=self.image,
            visited_mask=self.visited_mask,
            vesselness=self.vesselness,
            position=self.position,
            prev_direction=self.prev_direction
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state."""
        total_centerline = self.centerline.sum()
        covered = self.covered_centerline.sum()
        coverage_ratio = covered / max(total_centerline, 1)
        
        return {
            'position': tuple(self.position),
            'step_count': self.step_count,
            'trajectory_length': len(self.trajectory),
            'off_track_streak': self.off_track_streak,
            'coverage_ratio': coverage_ratio,
            'covered_pixels': int(covered),
            'total_centerline_pixels': int(total_centerline)
        }
    
    def render(self) -> np.ndarray:
        """Render the current state as an image."""
        # Create visualization
        vis = (self.image.copy() * 255).astype(np.uint8)
        
        # Draw centerline in blue
        centerline_mask = self.centerline > 0
        vis[centerline_mask] = [0, 0, 255]
        
        # Draw covered centerline in green
        covered_mask = self.covered_centerline > 0
        vis[covered_mask] = [0, 255, 0]
        
        # Draw trajectory in red
        for y, x in self.trajectory:
            vis[max(0, y-1):min(self.height, y+2), 
                max(0, x-1):min(self.width, x+2)] = [255, 0, 0]
        
        # Draw current position in yellow
        y, x = self.position
        vis[max(0, y-2):min(self.height, y+3), 
            max(0, x-2):min(self.width, x+3)] = [255, 255, 0]
        
        return vis


class VectorizedVesselEnv:
    """Vectorized environment for parallel training."""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 num_envs: int = 8,
                 dataset=None):
        self.config = config
        self.num_envs = num_envs
        self.dataset = dataset
        
        # Create individual environments
        self.envs = [VesselTracingEnv(config) for _ in range(num_envs)]
        
        # Track which sample each env is using
        self.current_samples = [None] * num_envs
        
    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all environments with new samples."""
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            # Get new sample from dataset
            sample = self._get_random_sample()
            self.current_samples[i] = sample
            
            # Set data in environment
            env.set_data(
                image=sample['image'].permute(1, 2, 0).numpy(),
                centerline=sample['centerline'].squeeze().numpy(),
                distance_transform=sample['distance_transform'].squeeze().numpy(),
                fov_mask=sample['fov_mask'].squeeze().numpy()
            )
            
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
            
        return np.stack(observations), infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Auto-reset if episode ended
            if terminated or truncated:
                sample = self._get_random_sample()
                self.current_samples[i] = sample
                
                env.set_data(
                    image=sample['image'].permute(1, 2, 0).numpy(),
                    centerline=sample['centerline'].squeeze().numpy(),
                    distance_transform=sample['distance_transform'].squeeze().numpy(),
                    fov_mask=sample['fov_mask'].squeeze().numpy()
                )
                
                obs, _ = env.reset()
                info['terminal_observation'] = obs
            
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            
        return (
            np.stack(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )
    
    def _get_random_sample(self):
        """Get a random sample from the dataset."""
        idx = np.random.randint(len(self.dataset))
        return self.dataset[idx]
