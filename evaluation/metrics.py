# evaluation/metrics.py
"""Evaluation metrics for retinal vessel centerline extraction.

Includes:
- Centerline F1 at multiple tolerances (distance-based)
- clDice (mask-based, hard skeletonization — for evaluation only)
- cl_dice_from_probs (threshold prob map → binary → clDice)
- Betti-0 error (connected component difference)
- HD95 (95th percentile Hausdorff distance)
- IoU (Intersection over Union for binary vessel masks)

Note on clDice:
    Training uses a differentiable soft-skeleton approximation (see CenterlineLoss).
    Evaluation uses skimage.skeletonize on the thresholded probability map, which
    is the correct hard metric to report in results. The two are intentionally
    different: the soft version exists only to make gradients flow during training.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from skimage import measure
from skimage.morphology import skeletonize


def compute_betti0(binary_mask: np.ndarray, connectivity: int = 2) -> int:
    """Count connected components (Betti-0) of a binary mask.

    Cheap enough for periodic use during training.
    connectivity=2 → 8-connected (matches your existing betti_0_error).
    """
    if binary_mask.sum() == 0:
        return 0
    _, n_components = measure.label(
        binary_mask > 0,
        return_num=True,
        connectivity=connectivity,
    )
    return int(n_components)


def compute_gt_graph_metrics(
    pred_mask: np.ndarray,
    gt_centerline: np.ndarray,
    gt_graph=None,
    tolerance: float = 2.0,
    edge_coverage_threshold: float = 0.80,
) -> Dict[str, float]:
    """Graph-aware metrics that mirror the env-side ``covered_centerline`` view.

    Returns:
      betti_0_covered     — connected-component count of the covered GT centerline
                             (GT centerline pixels within ``tolerance`` of any pred
                             pixel).  Lower is better; matches Diag-3-style topology
                             feedback at the GT-skeleton level rather than at the
                             prediction level.
      gt_edge_cov80_frac  — fraction of GT graph edges (junction-to-junction or
                             junction-to-endpoint) where >= ``edge_coverage_threshold``
                             of the path pixels are covered.  Higher is better.

    If ``gt_graph`` is None, builds it inline via CenterlineExtractor.
    """
    pred_bool = pred_mask > 0
    gt_bool = gt_centerline > 0

    if not pred_bool.any() or not gt_bool.any():
        return {
            'betti_0_covered': 0,
            'gt_edge_cov80_frac': 0.0,
        }

    dist_to_pred = ndimage.distance_transform_edt(~pred_bool)
    covered_centerline = gt_bool & (dist_to_pred <= tolerance)

    if covered_centerline.any():
        _, n_components = measure.label(
            covered_centerline,
            return_num=True,
            connectivity=2,
        )
    else:
        n_components = 0

    if gt_graph is None:
        from data.centerline_extraction import (
            CenterlineExtractor,
        )

        gt_graph = CenterlineExtractor().skeleton_to_graph(gt_bool.astype(np.uint8))

    n_edges = gt_graph.number_of_edges()
    if n_edges == 0:
        edge_cov_frac = 0.0
    else:
        n_covered_edges = 0
        for _, _, data in gt_graph.edges(data=True):
            path = data.get('path', [])
            if not path:
                continue
            n_covered = sum(1 for (y, x) in path if covered_centerline[y, x])
            if n_covered / len(path) >= edge_coverage_threshold:
                n_covered_edges += 1
        edge_cov_frac = n_covered_edges / n_edges

    return {
        'betti_0_covered': int(n_components),
        'gt_edge_cov80_frac': float(edge_cov_frac),
    }


class CenterlineMetrics:
    """Compute evaluation metrics for predicted vs ground-truth skeletons
    and vessel masks.
    """

    def __init__(
        self,
        tolerance_levels: List[int] = [1, 2, 3],
    ):
        self.tolerance_levels = tolerance_levels

    # ============================================================
    # MAIN ENTRY
    # ============================================================

    def compute_all_metrics(
        self,
        pred_skeleton: np.ndarray,
        gt_skeleton: np.ndarray,
        pred_vessel_mask: Optional[np.ndarray] = None,
        gt_vessel_mask: Optional[np.ndarray] = None,
        pred_prob: Optional[np.ndarray] = None,  # raw model prob map (H, W) float [0,1]
        fov_mask: Optional[np.ndarray] = None,  # FOV mask — metrics computed inside ROI only
    ) -> Dict[str, float]:
        """Compute all metrics for a single prediction.

        Args:
            pred_skeleton    : (H, W) uint8  — predicted binary centerline
            gt_skeleton      : (H, W) uint8  — ground-truth binary centerline
            pred_vessel_mask : (H, W) uint8  — predicted binary vessel mask (optional)
            gt_vessel_mask   : (H, W) uint8  — GT binary vessel mask (optional)
            pred_prob        : (H, W) float  — raw sigmoid output; if provided,
                               clDice is computed via thresholding rather than
                               from pred_vessel_mask (preferred)
            fov_mask         : (H, W) uint8/bool — if provided, all metrics are
                               restricted to the FOV region (excludes black padding)

        """
        metrics = {}

        # --------------------------------------------------------
        # Apply FOV mask — zero out everything outside the retina
        # --------------------------------------------------------
        if fov_mask is not None:
            fov = fov_mask > 0
            pred_skeleton = pred_skeleton * fov
            gt_skeleton = gt_skeleton * fov
            if pred_vessel_mask is not None:
                pred_vessel_mask = pred_vessel_mask * fov
            if gt_vessel_mask is not None:
                gt_vessel_mask = gt_vessel_mask * fov
            if pred_prob is not None:
                pred_prob = pred_prob * fov

        # --------------------------------------------------------
        # 1. Centerline F1 Scores (skeleton-based)
        # --------------------------------------------------------
        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(
                pred_skeleton,
                gt_skeleton,
                tau,
            )
            metrics[f'precision@{tau}px'] = precision
            metrics[f'recall@{tau}px'] = recall
            metrics[f'f1@{tau}px'] = f1

        # --------------------------------------------------------
        # 2. clDice (mask-based)
        #    Prefer pred_prob (thresholded) over pred_vessel_mask
        #    because using the full vessel mask — not the skeleton —
        #    is what clDice was designed for.
        # --------------------------------------------------------
        if gt_vessel_mask is not None:
            if pred_prob is not None:
                metrics['clDice'] = self.cl_dice_from_probs(pred_prob, gt_vessel_mask)
            elif pred_vessel_mask is not None:
                metrics['clDice'] = self.cl_dice(
                    pred_vessel_mask,
                    gt_vessel_mask,
                )

        # --------------------------------------------------------
        # 3. IoU (mask-based)
        # --------------------------------------------------------
        if pred_vessel_mask is not None and gt_vessel_mask is not None:
            metrics['iou'] = self.iou(pred_vessel_mask, gt_vessel_mask)

        # --------------------------------------------------------
        # 4. Topology Metrics
        # --------------------------------------------------------
        metrics['betti_0_error'] = self.betti_0_error(pred_skeleton, gt_skeleton)
        metrics['hd95'] = self.hd95(pred_skeleton, gt_skeleton)

        # --------------------------------------------------------
        # 5. Width-stratified recall (diagnostic)
        # --------------------------------------------------------
        if gt_vessel_mask is not None:
            metrics.update(
                self.recall_by_width(
                    pred_skeleton,
                    gt_skeleton,
                    gt_vessel_mask,
                    tolerance=2,
                )
            )

        return metrics

    # ============================================================
    # CENTERLINE F1 (Distance-Tolerant, Vectorized)
    # ============================================================

    def centerline_f1(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        tolerance: int = 2,
    ) -> Tuple[float, float, float]:
        """Compute centerline F1 with Euclidean distance tolerance.

        A predicted pixel is a true positive if it lies within
        `tolerance` pixels of any GT centerline pixel, and vice versa.
        """
        pred_bin = pred > 0
        gt_bin = gt > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 0.0, 0.0, 0.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0, 0.0, 0.0

        gt_dist = ndimage.distance_transform_edt(~gt_bin)
        pred_dist = ndimage.distance_transform_edt(~pred_bin)

        tp_precision = int((gt_dist[pred_bin] <= tolerance).sum())
        tp_recall = int((pred_dist[gt_bin] <= tolerance).sum())

        precision = tp_precision / float(pred_bin.sum())
        recall = tp_recall / float(gt_bin.sum())

        if precision + recall == 0:
            return 0.0, 0.0, 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    # ============================================================
    # WIDTH-STRATIFIED RECALL (diagnostic)
    # ============================================================

    def recall_by_width(
        self,
        pred: np.ndarray,
        gt_skeleton: np.ndarray,
        gt_vessel_mask: np.ndarray,
        tolerance: int = 2,
        thin_max: float = 3.0,
        med_max: float = 6.0,
    ) -> Dict[str, float]:
        """Recall@tolerance stratified by local vessel width.

        Width at each GT centerline pixel is 2 × inward distance transform of
        the vessel mask (i.e. local vessel diameter in px).  Bins:
            thin   : width <= thin_max
            medium : thin_max < width <= med_max
            thick  : width >  med_max

        Recall in each bin = fraction of GT centerline pixels in that bin
        that lie within `tolerance` px of any predicted skeleton pixel.
        Comparable to ``recall@{tolerance}px`` from ``centerline_f1``.

        Returns counts as well so per-image rows can be aggregated.
        """
        out: Dict[str, float] = {
            f'recall@{tolerance}px_thin': 0.0,
            f'recall@{tolerance}px_med': 0.0,
            f'recall@{tolerance}px_thick': 0.0,
            'n_centerline_thin': 0.0,
            'n_centerline_med': 0.0,
            'n_centerline_thick': 0.0,
        }

        gt_bin = gt_skeleton > 0
        if not gt_bin.any() or not (gt_vessel_mask > 0).any():
            return out

        inward = ndimage.distance_transform_edt(gt_vessel_mask > 0).astype(np.float32)
        width = 2.0 * inward  # local diameter in px

        if (pred > 0).sum() == 0:
            pred_dist = np.full(
                pred.shape,
                np.inf,
                dtype=np.float32,
            )
        else:
            pred_dist = ndimage.distance_transform_edt(~(pred > 0))

        gt_idx = np.argwhere(gt_bin)
        widths = width[gt_idx[:, 0], gt_idx[:, 1]]
        within = pred_dist[gt_idx[:, 0], gt_idx[:, 1]] <= tolerance

        bins = {
            'thin': widths <= thin_max,
            'med': (widths > thin_max) & (widths <= med_max),
            'thick': widths > med_max,
        }
        for name, mask in bins.items():
            n = int(mask.sum())
            out[f'n_centerline_{name}'] = float(n)
            if n > 0:
                out[f'recall@{tolerance}px_{name}'] = float(within[mask].sum()) / n
        return out

    # ============================================================
    # clDice (Hard, Mask-Based — evaluation only)
    # ============================================================

    def cl_dice(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Hard clDice from binary vessel masks.

        Tprec = |S(P) ∩ G| / |S(P)|
        Tsens = |S(G) ∩ P| / |S(G)|
        clDice = 2 * Tprec * Tsens / (Tprec + Tsens)

        P = predicted vessel mask
        G = ground-truth vessel mask
        S(.) = skimage.skeletonize  (hard, non-differentiable)

        Use this for evaluation. For training, see CenterlineLoss
        which uses a differentiable soft-skeleton approximation.
        """
        pred_bin = pred_mask > 0
        gt_bin = gt_mask > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 0.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0

        skel_pred = skeletonize(pred_bin)
        skel_gt = skeletonize(gt_bin)

        if skel_pred.sum() == 0 or skel_gt.sum() == 0:
            return 0.0

        tprec = np.logical_and(skel_pred, gt_bin).sum() / float(skel_pred.sum())
        tsens = np.logical_and(skel_gt, pred_bin).sum() / float(skel_gt.sum())

        if tprec + tsens == 0:
            return 0.0

        return float(2 * tprec * tsens / (tprec + tsens))

    def cl_dice_from_probs(
        self,
        pred_prob: np.ndarray,
        gt_vessel_mask: np.ndarray,
        prob_threshold: float = 0.5,
    ) -> float:
        """Hard clDice computed from a raw probability map.

        Thresholds pred_prob → binary vessel mask → skeletonize → clDice.
        This is the correct evaluation path: the full thresholded vessel
        mask (not the post-processed skeleton) is what clDice operates on.

        Use this for evaluation, NOT for training.
        """
        pred_bin = (pred_prob >= prob_threshold).astype(np.uint8)
        return self.cl_dice(pred_bin, gt_vessel_mask)

    # ============================================================
    # IoU (Mask-Based)
    # ============================================================

    def iou(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """Intersection over Union for binary vessel masks.

        IoU = |P ∩ G| / |P ∪ G|

        Returns 1.0 if both masks are empty (perfect agreement),
        0.0 if exactly one is empty.
        """
        pred_bin = pred_mask > 0
        gt_bin = gt_mask > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 0.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()

        return float(intersection / union)

    # ============================================================
    # BETTI-0 ERROR
    # ============================================================

    def betti_0_error(self, pred: np.ndarray, gt: np.ndarray) -> float:
        pred_b0 = compute_betti0(pred)
        gt_b0 = compute_betti0(gt)
        return float(abs(pred_b0 - gt_b0))

    # def betti_0_error(
    #     self,
    #     pred: np.ndarray,
    #     gt: np.ndarray,
    # ) -> float:
    #     """Absolute difference in number of connected components (8-connectivity).
    #     Lower is better; 0 means topology matches GT exactly.
    #     """
    #     _, pred_b0 = measure.label(pred > 0, return_num=True, connectivity=2)
    #     _, gt_b0 = measure.label(gt > 0, return_num=True, connectivity=2)

    #     return float(abs(int(pred_b0) - int(gt_b0)))

    # ============================================================
    # HD95 (Symmetric)
    # ============================================================

    def hd95(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """95th percentile symmetric Hausdorff distance (pixels).

        When one input is empty, returns the image diagonal as a
        worst-case penalty (conventional choice — document in thesis).
        """
        p_bin = pred > 0
        g_bin = gt > 0

        if p_bin.sum() == 0 and g_bin.sum() == 0:
            return 0.0

        if p_bin.sum() == 0 or g_bin.sum() == 0:
            # Worst-case penalty: image diagonal
            return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

        p_dist_map = ndimage.distance_transform_edt(~p_bin)
        g_dist_map = ndimage.distance_transform_edt(~g_bin)

        hd95_p_g = float(np.percentile(g_dist_map[p_bin], 95))
        hd95_g_p = float(np.percentile(p_dist_map[g_bin], 95))

        return max(hd95_p_g, hd95_g_p)
