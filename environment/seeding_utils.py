# environment/seeding_utils.py
"""Reusable seed generation utilities for retinal vessel tracing.

Provides:
  - fov_ring_seeds   : evenly-spaced peripheral seeds just inside the FOV boundary
  - merge_seeds      : combine detector seeds + ring seeds with slot reservation

These are the ONE external mechanism kept alongside the learned seed detector.
Ring seeds solve a problem no confidence-based detector can: guaranteed
peripheral coverage at the FOV boundary where heatmap response is always low.

Designed to be imported by any inference or evaluation script:

    from environment.seeding_utils import fov_ring_seeds, merge_seeds
"""

from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_N_RING_SEEDS = 0  # angular samples around FOV ring (every 15°)
DEFAULT_RING_INSET_PX = 40  # erosion depth — keeps seeds away from hard edges
DEFAULT_RING_DEDUP_PX = 35  # skip ring seed if detector seed within this distance
DEFAULT_OBS_HALF = 32  # half-width of observation patch (OBS_SIZE // 2)


def fov_ring_seeds(
    fov_mask: np.ndarray,
    n_seeds: int = DEFAULT_N_RING_SEEDS,
    inset_px: int = DEFAULT_RING_INSET_PX,
    obs_half: int = DEFAULT_OBS_HALF,
) -> List[Tuple[int, int]]:
    """Generate evenly-spaced seed points just inside the FOV boundary.

    Strategy:
        1. Erode the FOV mask by inset_px to produce an inner ring band.
        2. Sample n_seeds points at equal angles around the FOV centroid,
           snapping each direction to the nearest pixel in the ring band.
        3. Clamp all points to the observation safe-zone.

    Args:
        fov_mask : (H, W) uint8 binary FOV mask (1 = inside retina)
        n_seeds  : number of angular samples; 24 → one seed every 15°
        inset_px : erosion radius in pixels
        obs_half : half observation patch size for boundary clamping

    Returns:
        List of (y, x) integer seed coordinates, deduplicated.
    """
    h, w = fov_mask.shape

    # ---- Build inner ring band ----
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * inset_px + 1, 2 * inset_px + 1),
    )
    eroded = cv2.erode(
        fov_mask.astype(np.uint8),
        se,
        iterations=1,
    )
    ring = (fov_mask > 0) & (eroded == 0)

    if not ring.any():
        # Ring band empty — FOV covers entire image (e.g. FIVES) or is too small.
        # Fall back to evenly-spaced points just inside the FOV.
        fov_pts = np.argwhere(fov_mask > 0)
        if len(fov_pts) == 0:
            return []
        safe_y0, safe_y1 = (
            obs_half + 2,
            h - obs_half - 3,
        )
        safe_x0, safe_x1 = (
            obs_half + 2,
            w - obs_half - 3,
        )

        cy, cx = fov_pts.mean(axis=0)
        angles = np.linspace(0, 2 * np.pi, n_seeds, endpoint=False)
        radius = min(h, w) // 2 - inset_px
        if radius < 10:
            radius = min(h, w) // 3
        seeds = []
        for a in angles:
            y = int(
                np.clip(
                    cy + radius * np.sin(a),
                    safe_y0,
                    safe_y1,
                )
            )
            x = int(
                np.clip(
                    cx + radius * np.cos(a),
                    safe_x0,
                    safe_x1,
                )
            )
            if fov_mask[y, x] > 0:
                seeds.append((y, x))
        return list(dict.fromkeys(seeds))

    # ---- FOV centroid as angular reference ----
    fov_pts = np.argwhere(fov_mask > 0)
    cy, cx = fov_pts.mean(axis=0)
    ring_pts = np.argwhere(ring)  # (N, 2)

    # ---- Safe zone ----
    safe_y0, safe_y1 = (
        obs_half + 2,
        h - obs_half - 3,
    )
    safe_x0, safe_x1 = (
        obs_half + 2,
        w - obs_half - 3,
    )

    # Vectorized seed scoring
    rel = ring_pts - np.array([[cy, cx]])
    angles = np.linspace(0, 2 * np.pi, n_seeds, endpoint=False)
    directions = np.stack([np.sin(angles), np.cos(angles)], axis=1)
    scores = directions @ rel.T
    best_pts = ring_pts[np.argmax(scores, axis=1)]

    best_pts[:, 0] = np.clip(best_pts[:, 0], safe_y0, safe_y1)
    best_pts[:, 1] = np.clip(best_pts[:, 1], safe_x0, safe_x1)

    seeds = list(dict.fromkeys(map(tuple, best_pts)))
    return seeds


def merge_seeds(
    detector_seeds: List[Tuple[int, int, float]],
    fov_mask: np.ndarray,
    max_traces: int,
    n_ring_seeds: int = DEFAULT_N_RING_SEEDS,
    inset_px: int = DEFAULT_RING_INSET_PX,
    dedup_px: int = DEFAULT_RING_DEDUP_PX,
    obs_half: int = DEFAULT_OBS_HALF,
) -> Tuple[List[Tuple[int, int]], int]:
    """Merge detector seeds and FOV ring seeds with slot reservation.

    Slot reservation:
        n_ring_seeds slots are always reserved for ring seeds.  The detector
        uses the remaining (max_traces - n_ring_seeds) slots, taking the
        highest-confidence predictions first (assumed pre-sorted descending).

    Deduplication:
        A ring seed is skipped if any detector seed lies within dedup_px
        pixels (Manhattan distance).

    Args:
        detector_seeds : list of (y, x, confidence) from seed detector,
                         sorted by confidence descending
        fov_mask       : (H, W) uint8 binary FOV mask
        max_traces     : total seed budget (e.g. 80)
        n_ring_seeds   : slots reserved for peripheral ring seeds
        inset_px       : FOV erosion depth passed to fov_ring_seeds
        dedup_px       : Manhattan-distance dedup radius
        obs_half       : half observation size for boundary clamping

    Returns:
        merged  : list of (y, x) seed coordinates, detector first then ring
        n_added : number of ring seeds actually added
    """
    detector_slots = max_traces - n_ring_seeds
    detector_pts = [(y, x) for y, x, _ in detector_seeds[:detector_slots]]

    ring_pts = fov_ring_seeds(
        fov_mask,
        n_seeds=n_ring_seeds,
        inset_px=inset_px,
        obs_half=obs_half,
    )

    # Vectorized Manhattan distance deduplication
    if detector_pts and ring_pts:
        det_arr = np.array(detector_pts)
        ring_arr = np.array(ring_pts)
        dists = np.abs(ring_arr[:, None, :] - det_arr[None, :, :]).sum(axis=2)
        keep = dists.min(axis=1) >= dedup_px
        added_rings = [tuple(ring_arr[i]) for i in np.where(keep)[0]]
    else:
        added_rings = list(ring_pts)

    # If detector produced very few seeds, force ALL ring seeds in
    if len(detector_pts) < 5:
        added_rings = list(ring_pts)

    # Ordering matters: FrontierTracer consumes this list as a LIFO stack
    # (`frontier.pop()` takes the LAST element first).  detector_seeds is
    # confidence-descending, so we place the ring seeds first and the
    # detector seeds reversed (worst→best) at the end.  Under LIFO pop this
    # yields: best detector seed first, ..., worst detector seed, then ring
    # seeds last.
    #
    # Why this is essential: the tracer early-stops after
    # `max_low_gain_traces` consecutive low-gain traces.  With the previous
    # order (detector + rings), `pop()` drained ALL peripheral ring seeds
    # first; on images where the periphery has little vessel, 5 low-gain
    # ring traces tripped the early-stop before a single detector seed was
    # ever tried — the "only FOV ring seeds" failure.  Ring seeds are a
    # coverage backstop, so they belong last, not first.
    merged = added_rings + detector_pts[::-1]
    n_added = len(added_rings)
    return merged, n_added
