# baselines/frangi_baseline.py
"""
Classical Frangi vesselness baseline for comparison.
"""

import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import Tuple, Dict, Optional
import cv2


class FrangiBaseline:
    """
    Classical vessel segmentation using Frangi filter.
    
    Pipeline:
    1. Frangi vesselness filter
    2. Thresholding
    3. Morphological operations
    4. Skeletonization
    5. Graph pruning
    """
    
    def __init__(self,
                 sigma_min: float = 1.0,
                 sigma_max: float = 5.0,
                 num_scales: int = 5,
                 threshold: float = 0.1,
                 min_size: int = 50,
                 prune_length: int = 10):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.threshold = threshold
        self.min_size = min_size
        self.prune_length = prune_length
        
    def extract_centerline(self, image: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract vessel centerline from image.
        
        Args:
            image: RGB image (H, W, 3) or grayscale (H, W)
            mask: Optional FOV mask
            
        Returns:
            skeleton: Binary skeleton
            vesselness: Vesselness response
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Use green channel (best vessel contrast in fundus images)
            gray = image[:, :, 1].astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # Apply CLAHE for contrast enhancement
        gray_uint8 = (gray * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_uint8).astype(np.float64) / 255.0
        
        # Compute Frangi vesselness
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(enhanced, sigmas=sigmas, black_ridges=False)
        
        # Normalize vesselness
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        
        # Apply mask if provided
        if mask is not None:
            vesselness = vesselness * mask
        
        # Threshold
        binary = vesselness > self.threshold
        
        # Remove small objects
        binary = remove_small_objects(binary, min_size=self.min_size)
        
        # Morphological closing to connect nearby vessels
        binary = morphology.binary_closing(binary, morphology.disk(2))
        
        # Fill small holes
        binary = ndimage.binary_fill_holes(binary)
        
        # Skeletonize
        skeleton = skeletonize(binary)
        
        # Prune short branches
        skeleton = self._prune_skeleton(skeleton)
        
        return skeleton.astype(np.float32), vesselness
    
    def _prune_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Remove spurious short branches."""
        from data.centerline_extraction import CenterlineExtractor
        
        extractor = CenterlineExtractor(
            min_branch_length=self.prune_length,
            prune_iterations=3
        )
        
        return extractor._prune_skeleton(skeleton)
    
    def evaluate(self, image: np.ndarray, 
                 gt_skeleton: np.ndarray,
                 gt_vessel_mask: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate on a single image.
        
        Args:
            image: Input image
            gt_skeleton: Ground truth skeleton
            gt_vessel_mask: Ground truth vessel mask
            mask: FOV mask
            
        Returns:
            Dictionary of metrics
        """
        from evaluation.metrics import CenterlineMetrics
        
        pred_skeleton, _ = self.extract_centerline(image, mask)
        
        metrics = CenterlineMetrics()
        return metrics.compute_all_metrics(pred_skeleton, gt_skeleton, gt_vessel_mask)


class GreedyTracer:
    """
    Greedy tracing baseline that follows maximum vesselness.
    """
    
    DIRECTIONS = np.array([
        [-1, 0], [-1, 1], [0, 1], [1, 1],
        [1, 0], [1, -1], [0, -1], [-1, -1]
    ])
    
    def __init__(self, 
                 step_size: int = 1,
                 max_steps: int = 1000,
                 min_vesselness: float = 0.05,
                 momentum: float = 0.3):
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_vesselness = min_vesselness
        self.momentum = momentum  # Weight for continuing previous direction
        
    def trace(self, 
              vesselness: np.ndarray,
              start_point: Tuple[int, int],
              mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Trace from a starting point following max vesselness.
        
        Args:
            vesselness: Vesselness response map
            start_point: (y, x) starting position
            mask: Optional FOV mask
            
        Returns:
            trajectory: Array of (y, x) positions
        """
        h, w = vesselness.shape
        visited = np.zeros((h, w), dtype=bool)
        
        trajectory = [start_point]
        current = np.array(start_point)
        prev_direction = None
        
        for _ in range(self.max_steps):
            y, x = current
            visited[y, x] = True
            
            # Score each direction
            best_score = -1
            best_direction = None
            best_next = None
            
            for i, direction in enumerate(self.DIRECTIONS):
                next_pos = current + direction * self.step_size
                ny, nx = next_pos
                
                # Check bounds
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                
                # Check mask
                if mask is not None and mask[ny, nx] == 0:
                    continue
                
                # Check if visited
                if visited[ny, nx]:
                    continue
                
                # Compute score
                score = vesselness[ny, nx]
                
                # Add momentum bonus
                if prev_direction is not None:
                    # Cosine similarity with previous direction
                    prev_vec = self.DIRECTIONS[prev_direction]
                    similarity = np.dot(direction, prev_vec) / (
                        np.linalg.norm(direction) * np.linalg.norm(prev_vec) + 1e-8
                    )
                    score += self.momentum * max(0, similarity)
                
                if score > best_score:
                    best_score = score
                    best_direction = i
                    best_next = next_pos
            
            # Stop if no valid direction or vesselness too low
            if best_next is None or vesselness[best_next[0], best_next[1]] < self.min_vesselness:
                break
            
            current = best_next
            prev_direction = best_direction
            trajectory.append(tuple(current))
        
        return np.array(trajectory)
    
    def trace_from_seeds(self,
                         vesselness: np.ndarray,
                         seeds: list,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Trace from multiple seeds.
        
        Args:
            vesselness: Vesselness response
            seeds: List of (y, x) seed points
            mask: Optional FOV mask
            
        Returns:
            skeleton: Combined skeleton from all traces
        """
        h, w = vesselness.shape
        skeleton = np.zeros((h, w), dtype=np.float32)
        
        for seed in seeds:
            trajectory = self.trace(vesselness, seed, mask)
            
            for y, x in trajectory:
                if 0 <= y < h and 0 <= x < w:
                    skeleton[y, x] = 1.0
        
        return skeleton
