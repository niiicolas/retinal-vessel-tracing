"""
Classical Frangi vesselness baseline for comparison.
"""
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import Dict, Optional
from data.fundus_preprocessor import FundusPreprocessor

class FrangiBaseline:
    """
    Classical vessel segmentation using Frangi filter, with preprocessing.
    
    Pipeline:
    1. FundusPreprocessor
    2. Frangi vesselness filter
    3. Thresholding
    4. Morphological operations
    5. Skeletonization
    6. Graph pruning
    """
    
    def __init__(self,
                 sigma_min: float = 1.0,
                 sigma_max: float = 3.0,
                 num_scales: int = 5,
                 threshold: float = 0.02,
                 min_size: int = 50,
                 prune_length: int = 10):
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.threshold = threshold
        self.min_size = min_size
        self.prune_length = prune_length
        
        self.preprocessor = FundusPreprocessor( )

    # ------------------------------------------------------------------
    # Centerline extraction
    # ------------------------------------------------------------------

    def extract_centerline(self, image: np.ndarray,
                           return_vesselness: bool = False,
                           external_fov_mask: Optional[np.ndarray] = None):
        """
        Extract vessel centerline from image using preprocessor.
        If external_fov_mask is provided (e.g. DRIVE dataset mask), it is used.
        Otherwise the preprocessor generates a FOV mask automatically.
        """
        preprocessed, _, _, _, mask = self.preprocessor.preprocess(
            image,
            external_mask=external_fov_mask,
            return_intermediate=True
        )

        # Frangi vesselness
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(preprocessed.astype(np.float64), sigmas=sigmas, black_ridges=True)

        # Normalize
        vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)

        # Apply FOV mask
        vesselness *= (mask > 0)

        # Threshold + morphological ops
        binary = vesselness > self.threshold
        binary = morphology.binary_closing(binary, morphology.disk(1))
        binary = ndimage.binary_fill_holes(binary)
        binary = remove_small_objects(binary.astype(bool), min_size=self.min_size)

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

        labeled, num_features = ndimage.label(skeleton)

        pruned = skeleton.copy()
        endpoint_coords = np.argwhere(endpoints)

        for coord in endpoint_coords:
            label_id = labeled[coord[0], coord[1]]
            if label_id == 0:
                continue
            branch_pixels = np.argwhere(labeled == label_id)
            if len(branch_pixels) < min_length:
                pruned[labeled == label_id] = 0

        return pruned

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, image: np.ndarray,
                 gt_skeleton: np.ndarray,
                 gt_vessel_mask: Optional[np.ndarray] = None,
                 external_fov_mask: Optional[np.ndarray] = None) -> Dict[str, float]:  # ← added
        """
        Evaluate predicted skeleton against ground truth.
        If external_fov_mask is provided, uses external FOV mask.
        Otherwise falls back to self-generated FOV mask.
        """
        pred_skeleton, _ = self.extract_centerline(
            image,
            external_fov_mask=external_fov_mask  # ← passed through
        )
        return self._compute_metrics(pred_skeleton, gt_skeleton, gt_vessel_mask)

    def _compute_metrics(self,
                         pred: np.ndarray,
                         gt: np.ndarray,
                         gt_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Computes metrics with a 2-pixel tolerance.
        """
        pred_bin = pred > 0
        gt_bin   = gt > 0

        tolerance = 2
        struct = ndimage.generate_binary_structure(2, 2)

        # Precision: did prediction land near GT?
        gt_dilated   = ndimage.binary_dilation(gt_bin, structure=struct, iterations=tolerance)
        tp_precision = np.logical_and(pred_bin, gt_dilated).sum()
        precision    = tp_precision / (np.sum(pred_bin) + 1e-8)

        # Recall: was GT captured by prediction?
        pred_dilated = ndimage.binary_dilation(pred_bin, structure=struct, iterations=tolerance)
        tp_recall    = np.logical_and(gt_bin, pred_dilated).sum()
        recall       = tp_recall / (np.sum(gt_bin) + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
        }

        # clDice
        if gt_mask is not None:
            gt_mask_bin = gt_mask > 0

            tprec = np.logical_and(pred_bin, gt_mask_bin).sum() / (np.sum(pred_bin) + 1e-8)
            tsens = np.logical_and(gt_bin, pred_dilated).sum() / (np.sum(gt_bin) + 1e-8)

            cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
            metrics["clDice"] = float(cldice)

        return metrics