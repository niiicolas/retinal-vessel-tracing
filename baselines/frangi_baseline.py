# frangi_baseline.py
"""
==================
Frangi Vesselness Baseline with Topological Pruning.

Workflow:
  1. Multi-scale Frangi filter (Hessian-based enhancement)
  2. Morphological cleanup (Binary closing + size filtering)
  3. Skeletonization (1-pixel centerline extraction)
  4. Skan Pruning: Removes 'Type 1' spurs (Tip-to-Junction) below prune_length
==================
"""

import numpy as np
from skimage import filters, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import Optional, Tuple
from skan import Skeleton as SkanSkeleton, summarize

# Local imports
from data.fundus_preprocessor import FundusPreprocessor

class FrangiBaseline:
    """
    Classical vessel segmentation using Frangi filter, with graph-based pruning.

    Pipeline:
    1. Preprocessing (CLAHE, Green channel, Masking)
    2. Frangi vesselness filter
    3. Binary Thresholding
    4. Small object removal
    5. Skeletonization (Source of Truth)
    6. SKAN Pruning (Graph-based removal of spurious tips)
    """

    def __init__(self,
                 sigma_min: float = 0.5,
                 sigma_max: float = 3.0,
                 num_scales: int = 5,
                 threshold: float = 0.08,
                 min_size: int = 50,
                 prune_length: int = 10):

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.threshold = threshold
        self.min_size = min_size
        self.prune_length = prune_length

        self.preprocessor = FundusPreprocessor()

    def extract_centerline(self,
                           image: np.ndarray,
                           return_vesselness: bool = False,
                           external_fov_mask: Optional[np.ndarray] = None
                           ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Extract a 1-pixel skeleton from a fundus image.

        Returns:
            skeleton (np.ndarray): 1-pixel centerline of vessels
            vesselness (np.ndarray or None): Frangi vesselness map if requested
            binary_mask (np.ndarray): Binary vessel mask after thresholding and cleanup
        """
        preprocessed, _, _, _, mask = self.preprocessor.preprocess(
            image,
            external_mask=external_fov_mask,
            return_intermediate=True
        )

        # 1. Frangi vesselness
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(preprocessed.astype(np.float64), sigmas=sigmas, black_ridges=True)
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        vesselness *= (mask > 0)

        # 2. Binary segmentation + morphological cleanup
        binary = vesselness > self.threshold
        binary = morphology.binary_closing(binary, morphology.disk(1))
        binary = remove_small_objects(binary.astype(bool), min_size=self.min_size)

        # 3. Skeletonization
        skeleton = skeletonize(binary)

        # 4. Graph-based pruning with SKAN
        if np.any(skeleton):
            skeleton = self._prune_with_skan(skeleton)

        # 5. Return tuple (skeleton, vesselness, binary)
        if not return_vesselness:
            vesselness = None

        return skeleton.astype(np.float32), vesselness, binary.astype(np.uint8)

    def _prune_with_skan(self, skeleton_img: np.ndarray) -> np.ndarray:
        """
        Remove short spur branches ending in tips using SKAN graph.
        """
        try:
            skel = SkanSkeleton(skeleton_img)
            stats = summarize(skel, separator='_')

            short_tips = stats[(stats['branch-type'] == 1) &
                               (stats['branch-distance'] < self.prune_length)]
            pruned_skeleton = skeleton_img.copy()

            for edge_idx in short_tips.index:
                coords = skel.path_coordinates(edge_idx)
                for r, c in coords.astype(int):
                    pruned_skeleton[r, c] = 0

            return skeletonize(pruned_skeleton)

        except Exception:
            # fallback if SKAN fails
            return skeleton_img