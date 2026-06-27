"""Greedy steepest-ascent tracer baseline over a Frangi vesselness map, with skan pruning."""

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skan import Skeleton as SkanSkeleton
from skan import summarize
from skimage import filters
from skimage.morphology import remove_small_objects, skeletonize

from data.fundus_preprocessor import eroded_fov_mask


class GreedyTracer:
    """Traces vessels by steepest ascent from local-maxima seeds on a soft vesselness map.

    Returns both the binary skeleton and the per-trace trajectories for visualization.
    Post-processing (small-object removal) lives in GreedyTracerBaseline.
    """

    def __init__(self, seed_thresh: float = 0.15, step_thresh: float = 0.08, min_length: int = 10, thin_output: bool = True):
        """Store seed/step vesselness thresholds, the minimum trace length, and the thinning flag."""
        self.seed_thresh = seed_thresh
        self.step_thresh = step_thresh
        self.min_length = min_length
        self.thin_output = thin_output

        self._offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def _local_maxima(self, prob: np.ndarray) -> np.ndarray:
        """Return a boolean mask of pixels >= all 8 neighbours."""
        padded = np.pad(prob, 1, mode='constant', constant_values=0)
        lm = np.ones_like(prob, dtype=bool)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                shifted = padded[1 + dy : 1 + dy + prob.shape[0], 1 + dx : 1 + dx + prob.shape[1]]
                lm &= prob >= shifted
        return lm

    def _trace_from(self, prob: np.ndarray, visited: np.ndarray, start_r: int, start_c: int) -> List[Tuple[int, int]]:
        """Walk steepest-ascent from a seed (stopping below ``step_thresh``); returns the ordered (r, c) path."""
        H, W = prob.shape
        path = [(start_r, start_c)]
        visited[start_r, start_c] = True
        r, c = start_r, start_c

        while True:
            best_val = self.step_thresh
            best_rc = None
            for dr, dc in self._offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                    if prob[nr, nc] > best_val:
                        best_val = prob[nr, nc]
                        best_rc = (nr, nc)
            if best_rc is None:
                break
            r, c = best_rc
            visited[r, c] = True
            path.append((r, c))

        return path

    def trace(self, prob_map: np.ndarray, fov_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """Trace all seeds strongest-first and return the pruned skeleton plus trajectories.

        Args:
            prob_map: (H, W) float32 vesselness/probability map.
            fov_mask: (H, W) uint8 — restricts tracing to the FOV.

        Returns:
            ``(skeleton uint8 {0,255}, traces)`` where traces is a list of ordered (r, c) paths
            in visit order (trace 0 = strongest).
        """
        prob = prob_map.copy().astype(np.float32)
        if fov_mask is not None:
            prob[fov_mask == 0] = 0.0

        H, W = prob.shape
        skeleton = np.zeros((H, W), dtype=np.uint8)
        visited = np.zeros((H, W), dtype=bool)

        # Seeds: above seed_thresh and a strict local maximum.
        candidates = (prob >= self.seed_thresh) & self._local_maxima(prob)
        seed_coords = np.argwhere(candidates)

        if len(seed_coords) == 0:
            return skeleton, []

        # Trace strongest ridges first.
        seed_probs = prob[seed_coords[:, 0], seed_coords[:, 1]]
        order = np.argsort(-seed_probs)
        seed_coords = seed_coords[order]

        traces = []

        for sr, sc in seed_coords:
            if visited[sr, sc]:
                continue
            path = self._trace_from(prob, visited, sr, sc)
            if len(path) >= self.min_length:
                for r, c in path:
                    skeleton[r, c] = 255
                traces.append(path)

        if self.thin_output and skeleton.any():
            skeleton_bool = skeletonize(skeleton > 0)

            # Prune short tip-to-junction (type 1) branches, then re-skeletonize for clean junctions.
            try:
                skel = SkanSkeleton(skeleton_bool)
                stats = summarize(skel, separator='_')

                short_tips = stats[(stats['branch-type'] == 1) & (stats['branch-distance'] < self.min_length)]

                pruned = skeleton_bool.copy()
                for edge_idx in short_tips.index:
                    coords = skel.path_coordinates(edge_idx)
                    for r, c in coords.astype(int):
                        pruned[r, c] = False

                skeleton_bool = skeletonize(pruned)
            except Exception:
                pass  # graph too small to summarize — keep the unpruned skeleton

            skeleton = (skeleton_bool * 255).astype(np.uint8)

        return skeleton, traces


class GreedyTracerBaseline:
    """End-to-end greedy baseline: preprocessed image → Frangi vesselness → greedy trace → skeleton."""

    def __init__(
        self,
        sigma_min: float = 0.5,
        sigma_max: float = 3.0,
        num_scales: int = 5,
        gauss_sigma: float = 1.0,
        seed_thresh: float = 0.15,
        step_thresh: float = 0.08,
        min_length: int = 10,
        thin_output: bool = True,
        min_obj_size: int = 0,
    ):
        """Store Frangi/smoothing parameters and build the underlying GreedyTracer.

        ``min_obj_size`` removes isolated skeleton blobs after tracing (0 disables).
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.gauss_sigma = gauss_sigma
        self.min_obj_size = min_obj_size

        self.tracer = GreedyTracer(seed_thresh=seed_thresh, step_thresh=step_thresh, min_length=min_length, thin_output=thin_output)

    def _compute_vesselness(self, preprocessed: np.ndarray, safe_mask: np.ndarray) -> np.ndarray:
        """Multi-scale Frangi → normalize → Gaussian smooth → gate by the (already-eroded) FOV mask.

        Smoothing suppresses noisy background maxima; ``safe_mask`` is the FOV-radius-scaled
        eroded mask (see ``eroded_fov_mask``) that drops the Frangi edge-halo at the boundary.
        """
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.num_scales)
        vesselness = filters.frangi(preprocessed.astype(np.float64), sigmas=sigmas, black_ridges=True)

        vmin, vmax = (vesselness.min(), vesselness.max())
        vesselness = (vesselness - vmin) / (vmax - vmin + 1e-8)

        if self.gauss_sigma > 0:
            vesselness = gaussian_filter(vesselness, sigma=self.gauss_sigma)

        vesselness *= (safe_mask > 0).astype(np.float32)

        return vesselness.astype(np.float32)

    def extract_centerline(
        self, preprocessed: np.ndarray, fov_mask: Optional[np.ndarray] = None, return_vesselness: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List]:
        """Run the full pipeline on one image and return the traced skeleton.

        Args:
            preprocessed: (H, W) float32 CLAHE-enhanced grayscale in [0, 1].
            fov_mask: (H, W) uint8 FOV mask {0, 255}; full image if None.
            return_vesselness: when False the returned vesselness map is None.

        Returns:
            ``(skeleton, vesselness or None, traces)``.
        """
        mask = fov_mask if fov_mask is not None else np.ones(preprocessed.shape[:2], dtype=np.uint8) * 255
        # FOV-radius-scaled erosion (shared with the RL agent) gates both the vesselness and the trace.
        safe_mask = eroded_fov_mask(mask)
        vesselness = self._compute_vesselness(preprocessed, safe_mask)
        skeleton, traces = self.tracer.trace(vesselness, fov_mask=safe_mask)

        if skeleton.any() and self.min_obj_size > 0:
            skeleton_bool = remove_small_objects(skeleton > 0, min_size=self.min_obj_size)
            skeleton = (skeleton_bool * 255).astype(np.uint8)

        if return_vesselness:
            return skeleton, vesselness, traces
        return skeleton, None, traces
