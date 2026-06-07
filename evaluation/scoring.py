"""Single shared scorer for ALL models (RL, frangi, greedy, unet).

Every model must be scored through ``score_prediction`` so the numbers are
comparable: same post-processing, same metric set, same GT, same code path.
This is a verbatim extraction of the scoring block that ``run_rl_tracing.py``
used (the recorded v12 numbers), so routing the RL eval through here is a no-op
and the baselines inherit the *identical* treatment.

Contract: the caller supplies a predicted 1-px skeleton (``traced``) and the
GT arrays for ONE image, all at the SAME resolution. Returns the full e2e
metric dict (config.METRIC_COLS minus image_id).

NOTE on the dilation radius: ``distance_transform`` here is the GT
distance-FROM-centerline clipped at ``tolerance`` (data.centerline_extraction.
compute_distance_transform), so it is 0 on the centerline → the per-image
"vessel radius" below evaluates to 1 for every image. That is faithful to the
recorded RL eval; baselines must pass the SAME ``distance_transform`` (compute
it from the GT centerline with the same function) to get identical treatment.
"""

import csv
import os
from typing import Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize

from evaluation.metrics import (
    CenterlineMetrics,
    compute_gt_graph_metrics,
)

# Metric columns this scorer fills (matches config.METRIC_COLS).
SCORER_METRIC_KEYS = (
    'iou',
    'clDice',
    'betti_0_error_raw',
    'betti_0_error_postproc',
    'betti_0_covered',
    'gt_edge_cov80_frac',
    'hd95',
    'f1@1px',
    'precision@1px',
    'recall@1px',
    'f1@2px',
    'precision@2px',
    'recall@2px',
    'f1@3px',
    'precision@3px',
    'recall@3px',
)


def postprocess_skeleton(traced: np.ndarray, dilation_radius: int = 5) -> np.ndarray:
    """Bridge nearby segments: binary-dilate then re-skeletonize to 1 px."""
    binary = (traced > 0).astype(np.uint8)
    r = dilation_radius
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(binary, se, iterations=1)
    return skeletonize(dilated > 0).astype(np.uint8)


def score_prediction(
    traced: np.ndarray,
    *,
    centerline: np.ndarray,
    vessel_mask: np.ndarray,
    fov_mask: Optional[np.ndarray],
    distance_transform: Optional[np.ndarray],
    tolerance: float,
    dilation_radius: int = 5,
    metrics_calc: Optional[CenterlineMetrics] = None,
) -> dict:
    """Score one predicted skeleton against GT. See module docstring.

    Parameters
    ----------
    traced              : (H, W) predicted 1-px skeleton (>0 = vessel).
    centerline          : (H, W) GT centerline (>0 = vessel).
    vessel_mask         : (H, W) GT filled vessel mask.
    fov_mask            : (H, W) field-of-view mask, or None.
    distance_transform  : (H, W) GT distance-from-centerline clipped at
                          ``tolerance`` (sizes the eval dilation; pass the
                          same thing for every model).
    tolerance           : px tolerance for the graph-coverage metric.
    """
    if metrics_calc is None:
        metrics_calc = CenterlineMetrics(tolerance_levels=[1, 2, 3])

    raw_skel = (traced > 0).astype(np.uint8)
    gt_skel = (centerline > 0).astype(np.uint8)
    fov_bool = (fov_mask > 0) if fov_mask is not None else None
    gt_vessel = (vessel_mask > 0).astype(np.float32)

    # Post-processed skeleton (bridge → reskeletonize), FOV-masked.
    pred_skel = postprocess_skeleton(traced, dilation_radius)
    if fov_bool is not None:
        pred_skel = pred_skel * fov_bool

    # Per-image vessel radius from the GT DT (median on the centerline) → dilate
    # the predicted skeleton to a vessel-shaped mask for IoU / clDice.
    if distance_transform is not None:
        radii = distance_transform[gt_skel > 0]
        vessel_radius = int(round(float(np.median(radii)))) if radii.size else 2
    else:
        vessel_radius = 2
    vessel_radius = max(vessel_radius, 1)
    se_vessel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * vessel_radius + 1,
            2 * vessel_radius + 1,
        ),
    )
    pred_vessel_mask = cv2.dilate(
        pred_skel.astype(np.uint8),
        se_vessel,
        iterations=1,
    )
    if fov_bool is not None:
        pred_vessel_mask = pred_vessel_mask * fov_bool.astype(np.uint8)

    metrics = metrics_calc.compute_all_metrics(
        pred_skeleton=pred_skel,
        gt_skeleton=gt_skel,
        pred_vessel_mask=pred_vessel_mask,
        gt_vessel_mask=gt_vessel,
        fov_mask=fov_mask,
    )
    metrics['betti_0_error_postproc'] = metrics.pop('betti_0_error')
    metrics['betti_0_error_raw'] = metrics_calc.betti_0_error(raw_skel, gt_skel)

    graph_m = compute_gt_graph_metrics(
        pred_mask=raw_skel,
        gt_centerline=gt_skel,
        tolerance=tolerance,
        edge_coverage_threshold=0.80,
    )
    metrics['betti_0_covered'] = graph_m['betti_0_covered']
    metrics['gt_edge_cov80_frac'] = graph_m['gt_edge_cov80_frac']
    return metrics


def write_eval_csvs(output_dir: str, rows: list) -> None:
    """Write per-image + summary CSVs in the RL eval format (so any model scored
    via ``score_prediction`` is comparable and picked up by
    scripts/collect_ablation_metrics.py).

    ``rows`` is a list of metric dicts, each with ``image_id`` + SCORER_METRIC_KEYS.
    Writes <output_dir>/metrics_e2e.csv and <output_dir>/metrics_summary_e2e.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    cols = ['image_id'] + list(SCORER_METRIC_KEYS)
    with open(
        os.path.join(output_dir, 'metrics_e2e.csv'),
        'w',
        newline='',
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=cols,
            extrasaction='ignore',
        )
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    with open(
        os.path.join(output_dir, 'metrics_summary_e2e.csv'),
        'w',
        newline='',
    ) as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Mean +/- Std'])
        for k in SCORER_METRIC_KEYS:
            vals = [r[k] for r in rows if k in r]
            if vals:
                w.writerow(
                    [
                        k,
                        f'{np.mean(vals):.4f} +/- {np.std(vals):.4f}',
                    ]
                )
