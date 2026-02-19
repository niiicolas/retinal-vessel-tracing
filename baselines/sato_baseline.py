# baselines/sato_baseline.py
"""
Classical Sato vesselness baseline using FundusPreprocessor for preprocessing.
"""
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import Tuple, Dict, Optional
from fundus_preprocessor import FundusPreprocessor


class SatoBaseline:
    """
    Classical vessel segmentation using Sato filter, with preprocessing.
    
    Pipeline:
    1. FundusPreprocessor (green channel + CLAHE + FOV mask)
    2. Sato vesselness filter
    3. Thresholding
    4. Morphological operations
    5. Skeletonization
    6. Graph pruning
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
        self.preprocessor = FundusPreprocessor()

    def extract_centerline(self, image: np.ndarray,
                           return_vesselness: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract vessel centerline from image using preprocessor.

        Args:
            image: RGB fundus image
            return_vesselness: If True, also return the vesselness map

        Returns:
            skeleton: Binary skeleton
            vesselness: Vesselness response (or None)
        """
        # --- Preprocessing ---
        _, _, clahe, mask = self.preprocessor.preprocess(
            image, return_intermediate=True
        )

        # --- Sato vesselness ---
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.sato(clahe.astype(np.float64) / 255.0, sigmas=sigmas, black_ridges=False)

        # Normalize vesselness
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)

        # Apply FOV mask
        vesselness *= (mask > 0)

        # Threshold + morphological ops
        binary = vesselness > self.threshold
        binary = remove_small_objects(binary, min_size=self.min_size)
        binary = morphology.binary_closing(binary, morphology.disk(2))
        binary = ndimage.binary_fill_holes(binary)

        # Skeletonize + prune
        skeleton = skeletonize(binary)
        skeleton = self._prune_skeleton(skeleton)

        if return_vesselness:
            return skeleton.astype(np.float32), vesselness
        return skeleton.astype(np.float32), None

    # ------------------------------------------------------------------
    # Skeleton pruning
    # ------------------------------------------------------------------

    def _prune_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Remove short spurious branches from skeleton iteratively."""
        for _ in range(3):
            skeleton = self._prune_once(skeleton, self.prune_length)
        return skeleton

    def _prune_once(self, skeleton: np.ndarray, min_length: int) -> np.ndarray:
        """Single pruning pass: remove branches shorter than min_length pixels."""
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbour_count = ndimage.convolve(
            skeleton.astype(np.uint8), kernel, mode='constant', cval=0
        )
        endpoints = (skeleton > 0) & (neighbour_count == 2)

        labeled, _ = ndimage.label(skeleton)

        pruned = skeleton.copy()
        for coord in np.argwhere(endpoints):
            label_id = labeled[coord[0], coord[1]]
            if label_id == 0:
                continue
            if (labeled == label_id).sum() < min_length:
                pruned[labeled == label_id] = 0

        return pruned

    # ------------------------------------------------------------------
    # Evaluation 
    # ------------------------------------------------------------------

    def evaluate(self, image: np.ndarray,
                 gt_skeleton: np.ndarray,
                 gt_vessel_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate predicted skeleton against ground truth.

        Metrics:
            - precision, recall, f1  (pixel-level on skeletons)
            - clDice               (centerline Dice, requires gt_vessel_mask)
        """
        pred_skeleton, _ = self.extract_centerline(image)
        return self._compute_metrics(pred_skeleton, gt_skeleton, gt_vessel_mask)

    def _compute_metrics(self,
                         pred: np.ndarray,
                         gt: np.ndarray,
                         gt_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        pred_bin = pred > 0
        gt_bin   = gt > 0

        tp = np.logical_and(pred_bin, gt_bin).sum()
        fp = np.logical_and(pred_bin, ~gt_bin).sum()
        fn = np.logical_and(~pred_bin, gt_bin).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        metrics = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
        }

        if gt_mask is not None:
            gt_mask_bin = gt_mask > 0
            tprec  = np.logical_and(pred_bin, gt_mask_bin).sum() / (pred_bin.sum() + 1e-8)
            tsens  = np.logical_and(gt_bin,   gt_mask_bin).sum() / (gt_bin.sum()   + 1e-8)
            cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
            metrics["clDice"] = float(cldice)

        return metrics