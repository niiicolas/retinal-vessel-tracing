# evaluation/metrics.py
"""
Evaluation metrics for retinal vessel centerline extraction.
Includes F1 at multiple tolerances and clDice.
"""

import numpy as np
from scipy import ndimage
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
            Dict[str, float]: precision@Nx, recall@Nx, f1@Nx, clDice (optional)
        """
        metrics = {}
        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(pred_skeleton, gt_skeleton, tau)
            metrics[f'precision@{tau}px'] = precision
            metrics[f'recall@{tau}px'] = recall
            metrics[f'f1@{tau}px'] = f1

        if gt_vessel_mask is not None:
            metrics['clDice'] = self.cl_dice(pred_skeleton, gt_skeleton, gt_vessel_mask)

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

        f1 = 2 * precision * recall / (precision + recall + 1e-8) if precision + recall > 0 else 0.0

        return precision, recall, f1

    def cl_dice(self,
                pred_skeleton: np.ndarray,
                gt_skeleton: np.ndarray,
                gt_vessel_mask: np.ndarray) -> float:
        """
        Compute clDice coefficient.
        """
        pred_bin = pred_skeleton > 0
        gt_bin = gt_skeleton > 0
        mask_bin = gt_vessel_mask > 0

        tprec = np.logical_and(pred_bin, mask_bin).sum() / (pred_bin.sum() + 1e-8) if pred_bin.sum() > 0 else 0.0

        from scipy.ndimage import binary_dilation
        pred_dilated = binary_dilation(pred_bin, iterations=2)
        tsens = np.logical_and(gt_bin, pred_dilated).sum() / (gt_bin.sum() + 1e-8) if gt_bin.sum() > 0 else 1.0

        # Calculation of harmonic mean for clDice
        return 2 * tprec * tsens / (tprec + tsens + 1e-8) if (tprec + tsens) > 0 else 0.0