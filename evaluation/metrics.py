# evaluation/metrics.py
"""
Evaluation metrics for retinal vessel centerline extraction.

Includes:
- Centerline F1 at multiple tolerances (distance-based)
- clDice (mask-based, original formulation)
- Betti-0 error (connected component difference)
- HD95 (95th percentile Hausdorff distance)
"""

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize
from typing import Dict, List, Optional, Tuple


class CenterlineMetrics:
    """
    Compute evaluation metrics for predicted vs ground-truth skeletons
    and vessel masks.
    """

    def __init__(self, tolerance_levels: List[int] = [1, 2, 3]):
        self.tolerance_levels = tolerance_levels

    # ============================================================
    # MAIN ENTRY
    # ============================================================

    def compute_all_metrics(
        self,
        pred_skeleton: np.ndarray,
        gt_skeleton: np.ndarray,
        pred_vessel_mask: Optional[np.ndarray] = None,
        gt_vessel_mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single prediction.
        """

        metrics = {}

        # --------------------------------------------------------
        # 1. Centerline F1 Scores (skeleton-based)
        # --------------------------------------------------------
        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(
                pred_skeleton,
                gt_skeleton,
                tau
            )
            metrics[f'precision@{tau}px'] = precision
            metrics[f'recall@{tau}px'] = recall
            metrics[f'f1@{tau}px'] = f1

        # --------------------------------------------------------
        # 2. clDice (mask-based, if masks provided)
        # --------------------------------------------------------
        if pred_vessel_mask is not None and gt_vessel_mask is not None:
            metrics['clDice'] = self.cl_dice(
                pred_vessel_mask,
                gt_vessel_mask
            )

        # --------------------------------------------------------
        # 3. Topology Metrics
        # --------------------------------------------------------
        metrics['betti_0_error'] = self.betti_0_error(
            pred_skeleton,
            gt_skeleton
        )

        metrics['hd95'] = self.hd95(
            pred_skeleton,
            gt_skeleton
        )

        return metrics

    # ============================================================
    # CENTERLINE F1 (Distance-Tolerant, Vectorized)
    # ============================================================

    def centerline_f1(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        tolerance: int = 2
    ) -> Tuple[float, float, float]:
        """
        Compute centerline F1 with Euclidean distance tolerance.
        """

        pred_bin = pred > 0
        gt_bin = gt > 0

        # Edge cases
        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 1.0, 1.0, 1.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0, 0.0, 0.0

        # Distance transforms
        gt_dist = ndimage.distance_transform_edt(~gt_bin)
        pred_dist = ndimage.distance_transform_edt(~pred_bin)

        # Vectorized precision
        tp_precision = int((gt_dist[pred_bin] <= tolerance).sum())
        precision = tp_precision / pred_bin.sum()

        # Vectorized recall
        tp_recall = int((pred_dist[gt_bin] <= tolerance).sum())
        recall = tp_recall / gt_bin.sum()

        # F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    # ============================================================
    # PROPER clDice (Mask-Based)
    # ============================================================

    def cl_dice(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> float:
        """
        Compute original clDice metric.

        Tprec = |S(P) ∩ G| / |S(P)|
        Tsens = |S(G) ∩ P| / |S(G)|
        clDice = 2 Tprec Tsens / (Tprec + Tsens)

        P = predicted vessel mask
        G = ground-truth vessel mask
        S(.) = skeletonization
        """

        pred_bin = pred_mask > 0
        gt_bin = gt_mask > 0

        # Edge cases
        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 1.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0

        # Skeletonization
        skel_pred = skeletonize(pred_bin)
        skel_gt = skeletonize(gt_bin)

        if skel_pred.sum() == 0 or skel_gt.sum() == 0:
            return 0.0

        # Topology precision
        tprec = np.logical_and(skel_pred, gt_bin).sum() / skel_pred.sum()

        # Topology sensitivity
        tsens = np.logical_and(skel_gt, pred_bin).sum() / skel_gt.sum()

        if tprec + tsens == 0:
            return 0.0

        return 2 * tprec * tsens / (tprec + tsens)

    # ============================================================
    # BETTI-0 ERROR
    # ============================================================

    def betti_0_error(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ) -> int:
        """
        Absolute difference in number of connected components.
        """

        _, pred_b0 = measure.label(
            pred > 0,
            return_num=True,
            connectivity=2
        )

        _, gt_b0 = measure.label(
            gt > 0,
            return_num=True,
            connectivity=2
        )

        return abs(int(pred_b0) - int(gt_b0))

    # ============================================================
    # HD95 (Symmetric)
    # ============================================================

    def hd95(
        self,
        pred: np.ndarray,
        gt: np.ndarray
    ) -> float:
        """
        95th percentile symmetric Hausdorff distance.
        """

        p_bin = pred > 0
        g_bin = gt > 0

        if p_bin.sum() == 0 and g_bin.sum() == 0:
            return 0.0

        if p_bin.sum() == 0 or g_bin.sum() == 0:
            # Image diagonal penalty
            return float(
                np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)
            )

        p_dist_map = ndimage.distance_transform_edt(~p_bin)
        g_dist_map = ndimage.distance_transform_edt(~g_bin)

        d_pred_to_gt = g_dist_map[p_bin]
        d_gt_to_pred = p_dist_map[g_bin]

        hd95_p_g = np.percentile(d_pred_to_gt, 95)
        hd95_g_p = np.percentile(d_gt_to_pred, 95)

        return float(max(hd95_p_g, hd95_g_p))