"""Evaluation metrics for retinal vessel centerline extraction.

Distance-tolerant centerline F1, hard clDice, IoU, Betti-0 error, HD95, and graph-aware
coverage. clDice here uses hard skeletonization (report metric), distinct from the
differentiable soft-skeleton used in training.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize


def compute_betti0(binary_mask: np.ndarray, connectivity: int = 2) -> int:
    """Count connected components (Betti-0) of a binary mask; ``connectivity=2`` is 8-connected."""
    if binary_mask.sum() == 0:
        return 0
    _, n_components = measure.label(binary_mask > 0, return_num=True, connectivity=connectivity)
    return int(n_components)


def compute_gt_graph_metrics(
    pred_mask: np.ndarray, gt_centerline: np.ndarray, gt_graph=None, tolerance: float = 2.0, edge_coverage_threshold: float = 0.80
) -> Dict[str, float]:
    """Graph-aware metrics over the covered GT centerline (env ``covered_centerline`` view).

    A GT centerline pixel is "covered" if within ``tolerance`` of any pred pixel.
    ``gt_graph`` is built inline via CenterlineExtractor when None.

    Returns:
        ``betti_0_covered`` (component count of the covered centerline, lower better) and
        ``gt_edge_cov80_frac`` (fraction of GT graph edges with >= ``edge_coverage_threshold``
        of their path pixels covered, higher better).
    """
    pred_bool = pred_mask > 0
    gt_bool = gt_centerline > 0

    if not pred_bool.any() or not gt_bool.any():
        return {'betti_0_covered': 0, 'gt_edge_cov80_frac': 0.0}

    dist_to_pred = ndimage.distance_transform_edt(~pred_bool)
    covered_centerline = gt_bool & (dist_to_pred <= tolerance)

    if covered_centerline.any():
        _, n_components = measure.label(covered_centerline, return_num=True, connectivity=2)
    else:
        n_components = 0

    if gt_graph is None:
        from data.centerline_extraction import CenterlineExtractor

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
    """Computes evaluation metrics for predicted vs ground-truth skeletons and vessel masks."""

    def __init__(self, tolerance_levels: List[int] = [1, 2, 3]):
        """Configure the pixel tolerances at which centerline F1 is reported."""
        self.tolerance_levels = tolerance_levels

    def compute_all_metrics(
        self,
        pred_skeleton: np.ndarray,
        gt_skeleton: np.ndarray,
        pred_vessel_mask: Optional[np.ndarray] = None,
        gt_vessel_mask: Optional[np.ndarray] = None,
        pred_prob: Optional[np.ndarray] = None,
        fov_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute the full metric set for one prediction.

        Args:
            pred_skeleton: (H, W) predicted binary centerline.
            gt_skeleton: (H, W) ground-truth binary centerline.
            pred_vessel_mask / gt_vessel_mask: (H, W) binary vessel masks (optional).
            pred_prob: (H, W) raw sigmoid map; when given, clDice is computed from it
                (thresholded) in preference to pred_vessel_mask.
            fov_mask: (H, W) — when given, restricts all metrics to the FOV region.

        Returns:
            Dict of metric name → value (F1/precision/recall per tolerance, clDice, iou,
            betti_0_error, hd95, width-stratified recall).
        """
        metrics = {}

        # Restrict every input to the FOV so black padding can't score.
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

        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(pred_skeleton, gt_skeleton, tau)
            metrics[f'precision@{tau}px'] = precision
            metrics[f'recall@{tau}px'] = recall
            metrics[f'f1@{tau}px'] = f1

        # clDice prefers the thresholded prob map (full vessel mask) over the skeleton.
        if gt_vessel_mask is not None:
            if pred_prob is not None:
                metrics['clDice'] = self.cl_dice_from_probs(pred_prob, gt_vessel_mask)
            elif pred_vessel_mask is not None:
                metrics['clDice'] = self.cl_dice(pred_vessel_mask, gt_vessel_mask)

        if pred_vessel_mask is not None and gt_vessel_mask is not None:
            metrics['iou'] = self.iou(pred_vessel_mask, gt_vessel_mask)

        metrics['betti_0_error'] = self.betti_0_error(pred_skeleton, gt_skeleton)
        metrics['hd95'] = self.hd95(pred_skeleton, gt_skeleton)

        if gt_vessel_mask is not None:
            metrics.update(self.recall_by_width(pred_skeleton, gt_skeleton, gt_vessel_mask, tolerance=2))

        return metrics

    def centerline_f1(self, pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> Tuple[float, float, float]:
        """Compute distance-tolerant centerline F1, returning ``(precision, recall, f1)``.

        A predicted pixel is a true positive if within ``tolerance`` px of any GT pixel,
        and symmetrically for recall.
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

    def recall_by_width(
        self, pred: np.ndarray, gt_skeleton: np.ndarray, gt_vessel_mask: np.ndarray, tolerance: int = 2, thin_max: float = 3.0, med_max: float = 6.0
    ) -> Dict[str, float]:
        """Recall@tolerance stratified into thin/medium/thick GT vessels by local diameter.

        Local width = 2 × inward distance transform of the vessel mask; bins split at
        ``thin_max`` and ``med_max`` px.

        Returns:
            Per-bin recall plus the GT centerline pixel counts per bin (for aggregation).
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
            pred_dist = np.full(pred.shape, np.inf, dtype=np.float32)
        else:
            pred_dist = ndimage.distance_transform_edt(~(pred > 0))

        gt_idx = np.argwhere(gt_bin)
        widths = width[gt_idx[:, 0], gt_idx[:, 1]]
        within = pred_dist[gt_idx[:, 0], gt_idx[:, 1]] <= tolerance

        bins = {'thin': widths <= thin_max, 'med': (widths > thin_max) & (widths <= med_max), 'thick': widths > med_max}
        for name, mask in bins.items():
            n = int(mask.sum())
            out[f'n_centerline_{name}'] = float(n)
            if n > 0:
                out[f'recall@{tolerance}px_{name}'] = float(within[mask].sum()) / n
        return out

    def cl_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Hard clDice between two binary vessel masks via skimage skeletonization.

        clDice = 2·Tprec·Tsens / (Tprec + Tsens) with Tprec = |S(P)∩G|/|S(P)|,
        Tsens = |S(G)∩P|/|S(G)|. Returns 0.0 if either mask (or skeleton) is empty.
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

    def cl_dice_from_probs(self, pred_prob: np.ndarray, gt_vessel_mask: np.ndarray, prob_threshold: float = 0.5) -> float:
        """Hard clDice from a probability map: threshold at ``prob_threshold`` then delegate to cl_dice."""
        pred_bin = (pred_prob >= prob_threshold).astype(np.uint8)
        return self.cl_dice(pred_bin, gt_vessel_mask)

    def iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Intersection-over-union of two binary masks; returns 0.0 if either is empty."""
        pred_bin = pred_mask > 0
        gt_bin = gt_mask > 0

        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            return 0.0
        if pred_bin.sum() == 0 or gt_bin.sum() == 0:
            return 0.0

        intersection = np.logical_and(pred_bin, gt_bin).sum()
        union = np.logical_or(pred_bin, gt_bin).sum()

        return float(intersection / union)

    def betti_0_error(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Absolute difference in connected-component count (8-connected); 0 means topology matches GT."""
        pred_b0 = compute_betti0(pred)
        gt_b0 = compute_betti0(gt)
        return float(abs(pred_b0 - gt_b0))

    def hd95(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """95th-percentile symmetric Hausdorff distance in pixels.

        Returns 0.0 when both inputs are empty, and the image diagonal (worst-case penalty)
        when exactly one is empty.
        """
        p_bin = pred > 0
        g_bin = gt > 0

        if p_bin.sum() == 0 and g_bin.sum() == 0:
            return 0.0

        if p_bin.sum() == 0 or g_bin.sum() == 0:
            return float(np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2))

        p_dist_map = ndimage.distance_transform_edt(~p_bin)
        g_dist_map = ndimage.distance_transform_edt(~g_bin)

        hd95_p_g = float(np.percentile(g_dist_map[p_bin], 95))
        hd95_g_p = float(np.percentile(p_dist_map[g_bin], 95))

        return max(hd95_p_g, hd95_g_p)
