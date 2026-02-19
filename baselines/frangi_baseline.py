"""
Classical Frangi vesselness baseline for comparison.
"""
import numpy as np
from scipy import ndimage
from skimage import filters, morphology
from skimage.morphology import skeletonize, remove_small_objects
from typing import Tuple, Dict, Optional
from data.fundus_preprocessor import FundusPreprocessor

class FrangiBaseline:
    """
    Classical vessel segmentation using Frangi filter, with preprocessing.
    
    Pipeline:
    1. FundusPreprocessor (green channel + CLAHE + FOV mask)
    2. Frangi vesselness filter
    3. Thresholding
    4. Morphological operations
    5. Skeletonization
    6. Graph pruning
    """
    
    def __init__(self,
                 sigma_min: float = 1.0,
                 sigma_max: float = 3.0,    # Reduced max sigma for thin vessels
                 num_scales: int = 5,
                 threshold: float = 0.02,   # Lowered threshold
                 min_size: int = 50,
                 prune_length: int = 10):
        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.threshold = threshold
        self.min_size = min_size
        self.prune_length = prune_length
        
        # Initialize the preprocessor
        self.preprocessor = FundusPreprocessor()

    def extract_centerline(self, image: np.ndarray,
                           return_vesselness: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract vessel centerline from image using preprocessor.
        """
        # --- Preprocessing ---
        # preprocessed, green, gamma, clahe, mask
        _, _, _, clahe, mask = self.preprocessor.preprocess(image, 
                                                            return_intermediate=True)

        # --- Frangi vesselness ---
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(clahe.astype(np.float64) / 255.0, sigmas=sigmas, black_ridges=True)

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
        # Find endpoints: pixels with exactly 1 neighbour in 8-connectivity
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbour_count = ndimage.convolve(
            skeleton.astype(np.uint8), kernel, mode='constant', cval=0
        )
        # An endpoint is a skeleton pixel with only 1 neighbour (itself + 1 other)
        endpoints = (skeleton > 0) & (neighbour_count == 2)

        # Label connected components
        labeled, num_features = ndimage.label(skeleton)

        # For each endpoint, trace its branch and remove if too short
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
                 gt_vessel_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate predicted skeleton against ground truth.
        """
        pred_skeleton, _ = self.extract_centerline(image)
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

        # --- 1. Tolerance (Dilation) ---
        # We make lines thicker so a 1-2 pixel shift counts as a match
        tolerance = 2
        struct = ndimage.generate_binary_structure(2, 2)
        
        # Dilate Ground Truth to check Precision (Did Pred hit near GT?)
        gt_dilated = ndimage.binary_dilation(gt_bin, structure=struct, iterations=tolerance)
        tp_precision = np.logical_and(pred_bin, gt_dilated).sum()
        precision = tp_precision / (np.sum(pred_bin) + 1e-8)

        # Dilate Prediction to check Recall (Was GT found by Pred?)
        pred_dilated = ndimage.binary_dilation(pred_bin, structure=struct, iterations=tolerance)
        tp_recall = np.logical_and(gt_bin, pred_dilated).sum()
        recall = tp_recall / (np.sum(gt_bin) + 1e-8)

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
        }

        # --- 2. clDice (Fixed) ---
        if gt_mask is not None:
            gt_mask_bin = gt_mask > 0
            
            # Precision: Fraction of Predicted Skeleton inside GT Mask (Thick)
            tprec = np.logical_and(pred_bin, gt_mask_bin).sum() / (np.sum(pred_bin) + 1e-8)
            
            # Sensitivity: Fraction of GT Skeleton captured by Predicted Skeleton (Dilated)
            tsens = np.logical_and(gt_bin, pred_dilated).sum() / (np.sum(gt_bin) + 1e-8)
            
            cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
            metrics["clDice"] = float(cldice)

        return metrics