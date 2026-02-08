# baselines/greedy_tracer_baseline.py
"""
Greedy tracing baseline that follows maximum vesselness.
"""

import numpy as np
from typing import Tuple, Optional


class GreedyTracer:
    DIRECTIONS = np.array([
        [-1, 0], [-1, 1], [0, 1], [1, 1],
        [1, 0], [1, -1], [0, -1], [-1, -1]
    ])
    
    def __init__(self, step_size: int = 1, max_steps: int = 1000,
                 min_vesselness: float = 0.05, momentum: float = 0.3):
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_vesselness = min_vesselness
        self.momentum = momentum  # Weight for continuing previous direction
        
    def trace(self, vesselness: np.ndarray, start_point: Tuple[int, int],
              mask: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = vesselness.shape
        visited = np.zeros((h, w), dtype=bool)
        trajectory = [start_point]
        current = np.array(start_point)
        prev_direction = None
        
        for _ in range(self.max_steps):
            y, x = current
            visited[y, x] = True
            best_score = -1
            best_direction = None
            best_next = None
            
            for i, direction in enumerate(self.DIRECTIONS):
                next_pos = current + direction * self.step_size
                ny, nx = next_pos
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if mask is not None and mask[ny, nx] == 0:
                    continue
                if visited[ny, nx]:
                    continue
                
                score = vesselness[ny, nx]
                if prev_direction is not None:
                    prev_vec = self.DIRECTIONS[prev_direction]
                    similarity = np.dot(direction, prev_vec) / (
                        np.linalg.norm(direction) * np.linalg.norm(prev_vec) + 1e-8
                    )
                    score += self.momentum * max(0, similarity)
                
                if score > best_score:
                    best_score = score
                    best_direction = i
                    best_next = next_pos
            
            if best_next is None or vesselness[best_next[0], best_next[1]] < self.min_vesselness:
                break
            
            current = best_next
            prev_direction = best_direction
            trajectory.append(tuple(current))
        
        return np.array(trajectory)
    
    def trace_from_seeds(self, vesselness: np.ndarray, seeds: list,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        h, w = vesselness.shape
        skeleton = np.zeros((h, w), dtype=np.float32)
        for seed in seeds:
            trajectory = self.trace(vesselness, seed, mask)
            for y, x in trajectory:
                if 0 <= y < h and 0 <= x < w:
                    skeleton[y, x] = 1.0
        return skeleton
