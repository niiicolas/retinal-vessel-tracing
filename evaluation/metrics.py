# evaluation/metrics.py
"""
Evaluation metrics for retinal vessel centerline extraction.
Includes F1 at multiple tolerances, clDice, Betti-0 error, and HD95.
"""

import numpy as np
from scipy import ndimage
from skimage import measure
from typing import Dict, List, Optional, Tuple


class CenterlineMetrics:
    """
    Compute evaluation metrics for predicted vs ground-truth skeletons.
    """

    def __init__(self, tolerance_levels: List[int] = [1, 2, 3]):
        self.tolerance_levels = tolerance_levels
        self.struct = ndimage.generate_binary_structure(2, 2)

    def compute_all_metrics(self,
                            pred_skeleton: np.ndarray,
                            gt_skeleton: np.ndarray,
                            gt_vessel_mask: Optional[np.ndarray] = None
                            ) -> Dict[str, float]:
        """
        Compute all metrics for a single prediction.

        Returns:
            Dict[str, float]: precision@Nx, recall@Nx, f1@Nx, clDice, betti_0_error, hd95
        """
        metrics = {}
        
        # 1. F1 Scores at different tolerances
        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(pred_skeleton, gt_skeleton, tau)
            metrics[f'precision@{tau}px'] = precision
            metrics[f'recall@{tau}px'] = recall
            metrics[f'f1@{tau}px'] = f1

        # 2. clDice (only if vessel mask is provided)
        if gt_vessel_mask is not None:
            metrics['clDice'] = self.cl_dice(pred_skeleton, gt_skeleton, gt_vessel_mask)

        # 3. Topology Metrics 
        metrics['betti_0_error'] = self.betti_0_error(pred_skeleton, gt_skeleton)
        metrics['hd95'] = self.hd95(pred_skeleton, gt_skeleton)

        return metrics

    def centerline_f1(self,
                      pred: np.ndarray,
                      gt: np.ndarray,
                      tolerance: int = 2) -> Tuple[float, float, float]:
        """
        Compute centerline F1 with distance tolerance.
        """
        pred_bin = pred > 0
        gt_bin = gt > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 1.0, 1.0, 1.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0, 0.0, 0.0

        # Distance transforms
        gt_dist = ndimage.distance_transform_edt(1 - gt_bin)
        pred_dist = ndimage.distance_transform_edt(1 - pred_bin)

        # Precision
        pred_points = np.argwhere(pred_bin)
        tp_precision = sum(gt_dist[y, x] <= tolerance for y, x in pred_points)
        precision = tp_precision / len(pred_points)

        # Recall
        gt_points = np.argwhere(gt_bin)
        tp_recall = sum(pred_dist[y, x] <= tolerance for y, x in gt_points)
        recall = tp_recall / len(gt_points)

        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def cl_dice(self, pred_skeleton: np.ndarray, gt_skeleton: np.ndarray, gt_vessel_mask: np.ndarray) -> float:
        """
        Compute clDice coefficient (Topology-Aware).
        Reference: Shit et al. "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"
        """
        pred_bin = pred_skeleton > 0
        gt_bin = gt_skeleton > 0
        mask_bin = gt_vessel_mask > 0

        # Topology-Precision: Predicted skeleton inside GT vessel mask
        tprec = np.logical_and(pred_bin, mask_bin).sum() / (pred_bin.sum() + 1e-8) if pred_bin.sum() > 0 else 0.0

        # Topology-Sensitivity: GT skeleton covered by dilated predicted skeleton
        pred_dilated = ndimage.binary_dilation(pred_bin, iterations=2)
        tsens = np.logical_and(gt_bin, pred_dilated).sum() / (gt_bin.sum() + 1e-8) if gt_bin.sum() > 0 else 1.0

        return 2 * tprec * tsens / (tprec + tsens + 1e-8) if (tprec + tsens) > 0 else 0.0

    def betti_0_error(self, pred: np.ndarray, gt: np.ndarray) -> int:
        """
        Calculates absolute error in 0th Betti Number (Connected Components).
        """
        _, pred_b0 = measure.label(pred > 0, return_num=True, connectivity=2)
        _, gt_b0 = measure.label(gt > 0, return_num=True, connectivity=2)
        return abs(int(pred_b0) - int(gt_b0))

    def hd95(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        95th percentile Hausdorff Distance. 
        Measures the spatial distance between the sets of pixels.
        """
        p_bin, g_bin = pred > 0, gt > 0
        
        # Handle cases with no prediction or no GT
        if p_bin.sum() == 0 and g_bin.sum() == 0:
            return 0.0
        if p_bin.sum() == 0 or g_bin.sum() == 0:
            return float(np.sqrt(pred.shape[0]**2 + pred.shape[1]**2)) # Image diagonal penalty

        p_dist_map = ndimage.distance_transform_edt(1 - p_bin)
        g_dist_map = ndimage.distance_transform_edt(1 - g_bin)

        # Distances from pred pixels to nearest GT pixel and vice-versa
        d_pred_to_gt = g_dist_map[p_bin]
        d_gt_to_pred = p_dist_map[g_bin]

        hd95_p_g = np.percentile(d_pred_to_gt, 95)
        hd95_g_p = np.percentile(d_gt_to_pred, 95)

        return float(max(hd95_p_g, hd95_g_p))