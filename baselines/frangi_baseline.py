"""
Classical Frangi vesselness baseline for comparison.
Integrated with skan for graph-based pruning and optimized for centerline extraction.
"""
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import Optional
from skan import Skeleton, summarize

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

    def extract_centerline(self, image: np.ndarray,
                           return_vesselness: bool = False,
                           external_fov_mask: Optional[np.ndarray] = None):
        """
        Extracts a 1-pixel wide skeleton.
        """
        preprocessed, _, _, _, mask = self.preprocessor.preprocess(
            image,
            external_mask=external_fov_mask,
            return_intermediate=True
        )

        # 1. Frangi vesselness
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(preprocessed.astype(np.float64), sigmas=sigmas, black_ridges=True)

        # Normalize to [0, 1]
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)
        vesselness *= (mask > 0)

        # 2. Binary segmentation
        binary = vesselness > self.threshold
        # Morphological clean up of the mask
        binary = morphology.binary_closing(binary, morphology.disk(1))
        binary = remove_small_objects(binary.astype(bool), min_size=self.min_size)

        # 3. Skeletonization (Model-internal)
        skeleton = skeletonize(binary)

        # 4. Graph-based Pruning (skan)
        if np.any(skeleton):
            skeleton = self._prune_with_skan(skeleton)

        if return_vesselness:
            return skeleton.astype(np.float32), vesselness
        return skeleton.astype(np.float32), None

    def _prune_with_skan(self, skeleton_img: np.ndarray) -> np.ndarray:
        """
        Uses graph theory to remove short branches that end in a leaf node (tip).
        """
        try:
            # Create graph from skeleton
            skel = Skeleton(skeleton_img)
            stats = summarize(skel)

            # Identification of branch types:
            # Type 1: Tip-to-Junction (The spurs we want to remove)
            # Type 2: Junction-to-Junction (Main vessel segments)
            
            # Find indices of short tip branches
            short_tips = stats[(stats['branch-type'] == 1) & 
                               (stats['branch-distance'] < self.prune_length)]
            
            pruned_skeleton = skeleton_img.copy()
            
            # Remove pixels belonging to short branches
            for edge_idx in short_tips.index:
                coords = skel.path_coordinates(edge_idx)
                for r, c in coords.astype(int):
                    pruned_skeleton[r, c] = 0
            
            # Re-run skeletonize once to ensure 1-pixel thickness at former junctions
            return skeletonize(pruned_skeleton)
            
        except Exception:
            # Fallback if the graph is too small to be analyzed
            return skeleton_img