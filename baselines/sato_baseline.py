# baselines/sato_baseline.py
"""
Classical Sato vesselness baseline for comparison.
"""

import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import Tuple, Optional
import cv2


class SatoBaseline:
    """
    Classical vessel segmentation using Sato filter.
    
    Pipeline:
    1. Sato vesselness filter
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
        # Grayscale / green channel
        if len(image.shape) == 3:
            gray = image[:, :, 1].astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        # Normalize
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        
        # CLAHE
        gray_uint8 = (gray * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_uint8).astype(np.float64) / 255.0
        
        # Sato vesselness
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.sato(enhanced, sigmas=sigmas, black_ridges=False)
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        
        if mask is not None:
            vesselness *= mask
        
        # Threshold + morphological ops
        binary = vesselness > self.threshold
        binary = remove_small_objects(binary, min_size=self.min_size)
        binary = morphology.binary_closing(binary, morphology.disk(2))
        binary = ndimage.binary_fill_holes(binary)
        
        # Skeletonize + prune
        skeleton = skeletonize(binary)
        skeleton = self._prune_skeleton(skeleton)
        
        return skeleton.astype(np.float32), vesselness
    
    def _prune_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        from data.centerline_extraction import CenterlineExtractor
        extractor = CenterlineExtractor(min_branch_length=self.prune_length, prune_iterations=3)
        return extractor._prune_skeleton(skeleton)
    
    def evaluate(self, image: np.ndarray, gt_skeleton: np.ndarray,
                 gt_vessel_mask: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None):
        from evaluation.metrics import CenterlineMetrics
        pred_skeleton, _ = self.extract_centerline(image, mask)
        metrics = CenterlineMetrics()
        return metrics.compute_all_metrics(pred_skeleton, gt_skeleton, gt_vessel_mask)
