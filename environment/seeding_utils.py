"""Seed-generation utilities: FOV-ring peripheral seeds and detector/ring seed merging.

Ring seeds guarantee peripheral coverage at the FOV boundary, where any confidence-based
detector's heatmap response is low.
"""

from typing import List, Tuple

import cv2
import numpy as np

DEFAULT_N_RING_SEEDS = 0  # angular samples around the FOV ring
DEFAULT_RING_INSET_PX = 40  # erosion depth, keeping seeds off hard edges
DEFAULT_RING_DEDUP_PX = 35  # drop a ring seed if a detector seed is this close
DEFAULT_OBS_HALF = 32  # half observation-patch width (OBS_SIZE // 2)


def fov_ring_seeds(
    fov_mask: np.ndarray, n_seeds: int = DEFAULT_N_RING_SEEDS, inset_px: int = DEFAULT_RING_INSET_PX, obs_half: int = DEFAULT_OBS_HALF
) -> List[Tuple[int, int]]:
    """Generate evenly-spaced seed points just inside the FOV boundary.

    Erodes the FOV by ``inset_px`` to form a ring band, samples ``n_seeds`` equal-angle
    directions around the FOV centroid snapping each to the nearest band pixel, and clamps
    to the observation safe-zone.

    Args:
        fov_mask: (H, W) uint8 binary FOV mask (1 = inside retina).
        n_seeds: number of angular samples (24 → one every 15°).
        inset_px: erosion radius in pixels.
        obs_half: half observation-patch size for boundary clamping.

    Returns:
        Deduplicated list of (y, x) integer seed coordinates.
    """
    h, w = fov_mask.shape

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset_px + 1, 2 * inset_px + 1))
    eroded = cv2.erode(fov_mask.astype(np.uint8), se, iterations=1)
    ring = (fov_mask > 0) & (eroded == 0)

    if not ring.any():
        # Empty ring (FOV spans the whole image, e.g. FIVES, or is tiny): fall back to a
        # circle of points inside the FOV around its centroid.
        fov_pts = np.argwhere(fov_mask > 0)
        if len(fov_pts) == 0:
            return []
        safe_y0, safe_y1 = (obs_half + 2, h - obs_half - 3)
        safe_x0, safe_x1 = (obs_half + 2, w - obs_half - 3)

        cy, cx = fov_pts.mean(axis=0)
        angles = np.linspace(0, 2 * np.pi, n_seeds, endpoint=False)
        radius = min(h, w) // 2 - inset_px
        if radius < 10:
            radius = min(h, w) // 3
        seeds = []
        for a in angles:
            y = int(np.clip(cy + radius * np.sin(a), safe_y0, safe_y1))
            x = int(np.clip(cx + radius * np.cos(a), safe_x0, safe_x1))
            if fov_mask[y, x] > 0:
                seeds.append((y, x))
        return list(dict.fromkeys(seeds))

    fov_pts = np.argwhere(fov_mask > 0)
    cy, cx = fov_pts.mean(axis=0)
    ring_pts = np.argwhere(ring)

    safe_y0, safe_y1 = (obs_half + 2, h - obs_half - 3)
    safe_x0, safe_x1 = (obs_half + 2, w - obs_half - 3)

    # For each angle, pick the ring pixel whose direction-from-centroid best matches it.
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
    """Merge detector seeds and FOV ring seeds under a fixed total budget with slot reservation.

    ``n_ring_seeds`` slots are reserved for ring seeds; the detector takes the remaining
    ``max_traces - n_ring_seeds`` highest-confidence slots (input assumed sorted descending).
    A ring seed is dropped if any detector seed lies within ``dedup_px`` (Manhattan).

    Args:
        detector_seeds: (y, x, confidence) list, sorted by confidence descending.
        fov_mask: (H, W) uint8 binary FOV mask.
        max_traces: total seed budget.
        n_ring_seeds: slots reserved for ring seeds.
        inset_px: FOV erosion depth passed to fov_ring_seeds.
        dedup_px: Manhattan dedup radius.
        obs_half: half observation size for boundary clamping.

    Returns:
        ``(merged, n_added)`` — seed (y, x) list (ring seeds first) and ring-seed count added.
    """
    detector_slots = max_traces - n_ring_seeds
    detector_pts = [(y, x) for y, x, _ in detector_seeds[:detector_slots]]

    ring_pts = fov_ring_seeds(fov_mask, n_seeds=n_ring_seeds, inset_px=inset_px, obs_half=obs_half)

    if detector_pts and ring_pts:
        det_arr = np.array(detector_pts)
        ring_arr = np.array(ring_pts)
        dists = np.abs(ring_arr[:, None, :] - det_arr[None, :, :]).sum(axis=2)
        keep = dists.min(axis=1) >= dedup_px
        added_rings = [tuple(ring_arr[i]) for i in np.where(keep)[0]]
    else:
        added_rings = list(ring_pts)

    # Too few detector seeds: keep all ring seeds rather than dedup them away.
    if len(detector_pts) < 5:
        added_rings = list(ring_pts)

    merged = added_rings + detector_pts[::-1]
    n_added = len(added_rings)
    return merged, n_added
