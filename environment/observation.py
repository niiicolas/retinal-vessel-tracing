# environment/observation.py
"""
Observation construction for vessel tracing environment.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class ObservationBuilder:
    """
    Builds observation tensors for the RL agent.
    
    Observation includes:
    - Local RGB crop around current position
    - Local visited mask crop
    - Previous move direction (one-hot encoded)
    - Optional vesselness filter response
    """
    
    def __init__(self, config: Dict[str, Any]):
        env_config = config.get('environment', {})
        self.obs_size = env_config.get('observation_size', 65)
        self.half_size = self.obs_size // 2
        self.use_vesselness = env_config.get('use_vesselness', True)
        
    def build(self,
              image: np.ndarray,
              visited_mask: np.ndarray,
              vesselness: Optional[np.ndarray],
              position: np.ndarray,
              prev_direction: Optional[int]) -> np.ndarray:
        """
        Build observation tensor.
        
        Args:
            image: Full RGB image (H, W, 3)
            visited_mask: Full visited mask (H, W)
            vesselness: Optional vesselness response (H, W)
            position: Current position (y, x)
            prev_direction: Previous action direction (0-7) or None
            
        Returns:
            Observation tensor (C, obs_size, obs_size)
        """
        y, x = position
        h, w = image.shape[:2]
        
        # Extract local crops
        y_start = y - self.half_size
        y_end = y + self.half_size + 1
        x_start = x - self.half_size
        x_end = x + self.half_size + 1
        
        # Handle boundary conditions with padding
        image_crop = self._extract_crop_with_padding(image, y_start, y_end, x_start, x_end)
        visited_crop = self._extract_crop_with_padding(
            visited_mask[:, :, np.newaxis], y_start, y_end, x_start, x_end
        )[:, :, 0]
        
        # Build observation channels
        channels = []
        
        # RGB channels (3)
        channels.append(image_crop.transpose(2, 0, 1))  # (3, H, W)
        
        # Visited mask (1)
        channels.append(visited_crop[np.newaxis])  # (1, H, W)
        
        # Previous direction one-hot (8)
        direction_map = np.zeros((8, self.obs_size, self.obs_size), dtype=np.float32)
        if prev_direction is not None:
            direction_map[prev_direction] = 1.0
        channels.append(direction_map)
        
        # Optional vesselness (1)
        if self.use_vesselness and vesselness is not None:
            vesselness_crop = self._extract_crop_with_padding(
                vesselness[:, :, np.newaxis], y_start, y_end, x_start, x_end
            )[:, :, 0]
            channels.append(vesselness_crop[np.newaxis])
        
        # Concatenate all channels
        observation = np.concatenate(channels, axis=0).astype(np.float32)
        
        return observation
    
    def _extract_crop_with_padding(self,
                                    array: np.ndarray,
                                    y_start: int,
                                    y_end: int,
                                    x_start: int,
                                    x_end: int) -> np.ndarray:
        """Extract crop with zero-padding for boundary cases."""
        h, w = array.shape[:2]
        
        # Compute padding
        pad_top = max(0, -y_start)
        pad_bottom = max(0, y_end - h)
        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - w)
        
        # Clip indices
        y_start_clipped = max(0, y_start)
        y_end_clipped = min(h, y_end)
        x_start_clipped = max(0, x_start)
        x_end_clipped = min(w, x_end)
        
        # Extract valid region
        crop = array[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]
        
        # Apply padding if needed
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            if len(array.shape) == 3:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            else:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
            crop = np.pad(crop, pad_width, mode='constant', constant_values=0)
            
        return crop
