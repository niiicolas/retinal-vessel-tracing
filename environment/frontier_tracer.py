# environment/frontier_tracer.py
"""Branch Coverage Manager for Retinal Vessel Tracing.
Implements the Frontier-Based Coverage (Algorithm 2) to trace the full
connected vascular tree.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import cv2
import numpy as np
import torch
from scipy.ndimage import convolve
from tqdm import tqdm


def _stamp_polyline(
    mask: np.ndarray,
    coords: np.ndarray,
    value: float = 1.0,
    max_gap: Optional[float] = None,
) -> None:
    """Stamp a CONNECTED polyline into ``mask`` (lines between consecutive
    points), so step_size>1 traces are continuous instead of dotted.

    With step_size=2 the agent's path points are 2px apart; stamping them as
    isolated pixels left a dotted skeleton that shattered into thousands of
    8-disconnected components (observed betti_0_error_raw≈4447). Densifying
    each segment to its Chebyshev length+1 samples yields 8-connected pixels.

    ``max_gap``: if two consecutive points are farther apart than this
    (Chebyshev), the connecting line is NOT drawn — only the vertices are
    stamped. Guards against painting a straight stroke across a jump (e.g.
    two neighbours that snapped to different ridges at a junction).
    """
    coords = np.asarray(coords)
    if len(coords) == 0:
        return
    iy = coords[:, 0].astype(np.intp)
    ix = coords[:, 1].astype(np.intp)
    mask[iy, ix] = value  # always stamp the vertices
    for j in range(len(coords) - 1):
        y0, x0 = (
            float(coords[j, 0]),
            float(coords[j, 1]),
        )
        y1, x1 = (
            float(coords[j + 1, 0]),
            float(coords[j + 1, 1]),
        )
        n = int(max(abs(y1 - y0), abs(x1 - x0)))  # Chebyshev steps
        if n <= 1:
            continue  # already 8-connected; vertices suffice
        if max_gap is not None and n > max_gap:
            continue  # jump — don't paint across the gap
        ys = np.round(np.linspace(y0, y1, n + 1)).astype(np.intp)
        xs = np.round(np.linspace(x0, x1, n + 1)).astype(np.intp)
        mask[ys, xs] = value


class FrontierTracer:
    """Single source of truth for Frontier-Based Coverage (Algorithm 2)."""

    def __init__(
        self,
        env,
        policy_model,
        device,
        obs_size: int = 65,
        snap_to_centerline=None,
        snap_radius=None,
        vessel_gate=None,
    ):
        self.env = env
        self.model = policy_model
        self.device = device
        self.obs_size = obs_size
        self.half = obs_size // 2

        # Inference-time centerline snap (no retrain): pull traced points onto
        # the predicted-centerline ridge to fix the ~2-3px off-centre drift.
        # vessel_gate: DROP traced points farther than snap_radius from any
        # predicted-centerline ridge pixel (and break the polyline there), so
        # the policy's straight-line strokes across non-vessel background are
        # not stamped as false positives. Read defaults from config; explicit
        # args override for A/B tests.
        from config import MODEL_CONFIG

        _inf = MODEL_CONFIG.get('inference', {})
        self.snap_to_centerline = (
            _inf.get('snap_to_centerline', True) if snap_to_centerline is None else bool(snap_to_centerline)
        )
        self.vessel_gate = _inf.get('vessel_gate', True) if vessel_gate is None else bool(vessel_gate)
        self.snap_radius = float(_inf.get('snap_radius_px', 3.5) if snap_radius is None else snap_radius)
        # GT-ablation certification: feed the env garbage GT to prove the
        # prediction does not depend on ground truth (see _setup_env).
        self._corrupt_gt = bool(_inf.get('corrupt_gt', False))
        # Max connector length when stamping (breaks straight jumps); set per
        # image in _setup_env once env.step_size is known.
        self._stamp_max_gap = None
        # Nearest-ridge index map, precomputed per image in _setup_env.
        self._snap_iy = self._snap_ix = self._snap_dist = None

        # Preallocated inference buffer — filled in-place each step, never reallocated
        n_channels = env.observation_space.shape[0]
        self._obs_buf = torch.zeros(
            1,
            n_channels,
            obs_size,
            obs_size,
            dtype=torch.float32,
            device=device,
        )

    def _extract_endpoint_seeds(
        self,
        on_vessel_mask: np.ndarray,
        existing_frontier: List[Tuple[int, int]],
        min_distance: int = 8,
    ) -> List[Tuple[int, int]]:
        """Find dangling tips of the on-vessel skeleton.

        IMPORTANT: accepts `on_vessel_mask` — the on-vessel-only pixels from
        sub_paths — NOT the full combined_mask which includes bridge segments.
        Using bridge tips as seeds would extend false connections further.

        Returns endpoints sorted farthest-from-covered first, suppressing
        near-duplicates from the existing frontier.
        """
        if not on_vessel_mask.any():
            return []

        h, w = on_vessel_mask.shape
        margin = self.half + 5
        skel = (on_vessel_mask > 0).astype(np.uint8)

        kernel = np.array(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
            dtype=np.uint8,
        )
        neighbour_count = convolve(skel, kernel, mode='constant', cval=0)
        endpoint_mask = (skel > 0) & (neighbour_count == 1)
        endpoints = np.argwhere(endpoint_mask)

        if len(endpoints) == 0:
            return []

        frontier_set = set((int(s[0]), int(s[1])) for s in existing_frontier)

        # Distance transform from covered pixels for re-ranking
        covered_bin = (on_vessel_mask > 0).astype(np.uint8)
        dist_from_covered = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)

        new_seeds = []
        for ep in endpoints:
            ey, ex = int(ep[0]), int(ep[1])
            if not (margin <= ey < h - margin and margin <= ex < w - margin):
                continue
            if (ey, ex) in frontier_set:
                continue
            too_close = any(abs(ey - fy) + abs(ex - fx) < min_distance for fy, fx in frontier_set)
            if not too_close:
                new_seeds.append((ey, ex))

        new_seeds.sort(
            key=lambda s: dist_from_covered[s[0], s[1]],
            reverse=True,
        )
        return new_seeds

    def _execute_single_trace(
        self,
        start_pos: Tuple[int, int],
        combined_mask: np.ndarray,
    ) -> Tuple[
        List[Tuple[int, int]],
        List[Tuple[int, int]],
    ]:
        """Executes a single continuous trace until the agent stops or terminates."""
        # Expose accumulated coverage from all previous traces so the agent can
        # observe what has already been traced via the prior_coverage channel.
        self.env.prior_coverage = combined_mask
        obs, _ = self.env.reset(start_position=start_pos)
        path = [start_pos]
        done = False
        alternate_branches = []

        self.model.eval()
        with torch.no_grad():
            while not done:
                self._obs_buf.copy_(torch.from_numpy(obs))
                logits, _, _ = self.model(self._obs_buf)
                action = logits.argmax(dim=-1).item()

                (
                    obs,
                    _,
                    terminated,
                    truncated,
                    _,
                ) = self.env.step(action)
                done = terminated or truncated

                y, x = self.env.position
                path.append((y, x))
                combined_mask[y, x] = 1.0

        return path, alternate_branches

    def trace_from_seeds(
        self,
        sample: Dict[str, Any],
        initial_seeds: List[Tuple[int, int]],
        min_coverage_gain: float = 0.0005,
        max_low_gain_traces: int = 5,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """End-to-end inference: stack-based frontier with dynamic endpoint growth.

        Improvements:
        - Fix B (skip covered starts): skips seeds whose start pixel is covered.
        - Fix A (endpoint growth): after each trace, extracts endpoints from the
          ON-VESSEL skeleton only (sub_paths, not the full path including bridges).
          This prevents bridge tips from seeding further false connections.
        - Early stopping: halts after consecutive low-gain traces.
        """
        self._setup_env(sample)
        h, w = sample['image'].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        all_paths = []
        # Leak-free coverage normaliser for the early-stop gain threshold:
        # predicted centerline pixel count, not GT (≈ same magnitude, but keeps
        # ground truth out of the stopping decision).
        _pred_cl = self.env.pred_centerline
        cov_norm = (
            float(max(np.asarray(_pred_cl).sum(), 1))
            if _pred_cl is not None
            else float(max(sample['centerline'].sum(), 1))
        )
        low_gain_streak = 0
        traces_run = 0

        frontier = list(initial_seeds)

        # Scale early-stop patience to the seed budget.  `max_low_gain_traces`
        # is a fixed constant (5); on a small-FOV image the detector
        # legitimately returns proportionally fewer seeds, and a few hard
        # (e.g. dark, low-contrast) regions can fail several traces in a row
        # while dozens of valid, well-placed seeds elsewhere are still
        # unused.  A fixed patience of 5 then abandons the whole image at
        # ~19% coverage (observed: 536_N).  Growing patience with the number
        # of seeds we were handed lets the tracer actually work through its
        # budget; it stays bounded because `frontier` only shrinks except
        # for the (self-limiting) gap reseeder, so this cannot run away.
        patience = max(
            max_low_gain_traces,
            int(round(0.4 * len(initial_seeds))),
        )

        # Livelock guards. The gap reseeder resets the low-gain streak, so if
        # traces stop adding coverage (e.g. the vessel gate drops their points
        # on a hard image) the streak can never reach `patience` and the loop
        # spins forever (observed: im0324 ran >5h). Bound the reseed rounds AND
        # require coverage to grow between reseeds, plus an absolute cap on
        # total traces as a backstop.
        max_gap_reseeds = 5
        gap_reseeds_done = 0
        cov_at_last_reseed = -1.0
        hard_trace_cap = max(200, 10 * len(initial_seeds))

        pbar = tqdm(
            total=len(frontier),
            desc='Tracing Seeds',
            unit='seed',
            leave=False,
        )

        while frontier:
            if traces_run >= hard_trace_cap:
                tqdm.write(f'    Hard trace cap ({hard_trace_cap}) reached — stopping')
                break
            start_pos = frontier.pop()
            pbar.update(1)

            # Skip seeds whose start pixel is already covered
            sy, sx = (
                int(start_pos[0]),
                int(start_pos[1]),
            )
            if combined_mask[sy, sx] > 0:
                continue

            covered_before = combined_mask.sum()
            # Trace on a COPY so the real combined_mask is only updated by the
            # snapped, on-vessel sub-paths below. The raw per-step stamps inside
            # _execute_single_trace are ~2-3px off-centre and would otherwise
            # persist as false positives even after snapping.
            path, alternate_branches = self._execute_single_trace(
                start_pos,
                combined_mask.copy(),
            )
            traces_run += 1

            # On-vessel filtering is LEAK-FREE: _snap_and_gate splits the path
            # into segments along the PREDICTED centerline ridge (dropping
            # points with no predicted vessel nearby) and snaps them onto it.
            # This replaces the old _split_trace_at_bridges, which used the
            # ground-truth distance transform to decide on-vessel — an
            # inference-time GT leak. Bridges never entered combined_mask (we
            # traced on a copy), so nothing to clear.
            on_vessel_mask = np.zeros((h, w), dtype=np.float32)
            for seg in self._snap_and_gate(np.array(path, dtype=np.intp)):
                if len(seg) < 3:
                    continue  # drop tiny fragments (matches old >=3-point rule)
                all_paths.append([tuple(int(v) for v in p) for p in seg])
                _stamp_polyline(
                    combined_mask,
                    seg,
                    max_gap=self._stamp_max_gap,
                )
                _stamp_polyline(
                    on_vessel_mask,
                    seg,
                    max_gap=self._stamp_max_gap,
                )

            # Coverage gain tracking for early stopping
            gain = (combined_mask.sum() - covered_before) / cov_norm

            if gain < min_coverage_gain:
                low_gain_streak += 1
            else:
                low_gain_streak = 0

            if low_gain_streak >= patience:
                tqdm.write(f'    Early stop: {patience} consecutive low-gain traces (gain < {min_coverage_gain:.4f})')
                break

            # Fix A: extract endpoints from the ON-VESSEL mask only.
            # Using the full path (including bridges) would seed the agent
            # at bridge tips, causing further false connections downstream.
            new_endpoints = self._extract_endpoint_seeds(on_vessel_mask, frontier)
            if new_endpoints:
                frontier.extend(new_endpoints)
                pbar.total += len(new_endpoints)

            # Gap reseeder: when the frontier is nearly empty and no new
            # endpoints were found, inject seeds at uncovered vessel regions
            # that are far (>40px) from any covered pixel.  This prevents
            # starvation when disconnected subtrees were never reached by any
            # trace endpoint.
            # Only reseed while we have rounds left AND coverage actually grew
            # since the last reseed. Without the coverage check, a reseed round
            # whose traces add nothing (e.g. gated out) keeps re-firing and
            # resetting the streak → livelock (im0324). When coverage stalls we
            # stop reseeding and let the low-gain early-stop fire.
            cur_cov = float(combined_mask.sum())
            if (
                not new_endpoints
                and len(frontier) < 3
                and gap_reseeds_done < max_gap_reseeds
                and cur_cov > cov_at_last_reseed
            ):
                gap_seeds = self._gap_reseeder(combined_mask, sample)
                if gap_seeds:
                    frontier.extend(gap_seeds)
                    pbar.total += len(gap_seeds)
                    # A fresh batch of seeds in far, uncovered regions deserves
                    # a fresh chance, so reset the low-gain streak (observed on
                    # 536_N: 40 seeds injected at 0.19 coverage were else
                    # early-stopped before being tried). Bounded by the round
                    # cap + coverage-grew gate above so it cannot loop forever.
                    low_gain_streak = 0
                    gap_reseeds_done += 1
                    cov_at_last_reseed = cur_cov
                    tqdm.write(
                        f'    Gap reseeder: injected {len(gap_seeds)} seeds '
                        f'(coverage={cur_cov / cov_norm:.3f}, '
                        f'round {gap_reseeds_done}/{max_gap_reseeds})'
                    )

            for branch_pos in alternate_branches:
                if (
                    combined_mask[
                        branch_pos[0],
                        branch_pos[1],
                    ]
                    == 0
                ):
                    frontier.append(branch_pos)
                    pbar.total += 1

        pbar.close()
        tqdm.write(f'    Frontier tracer: {traces_run} traces, coverage={combined_mask.sum() / cov_norm:.3f}')

        return combined_mask, all_paths

    def trace_with_gt_gaps(
        self,
        sample: Dict[str, Any],
        max_traces: int = 50,
        min_coverage_gain: float = 0.005,
    ) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """Evaluation method: Iteratively forces the agent into ground-truth gaps."""
        self._setup_env(sample)
        h, w = sample['image'].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        all_paths = []
        gt_total = float(max(sample['centerline'].sum(), 1))

        # Added tqdm to the GT evaluation loop
        for trace_idx in tqdm(
            range(max_traces),
            desc='GT Gap Tracing',
            unit='trace',
        ):
            start_pos = self._pick_frontier_seed_from_gt(
                sample['centerline'],
                combined_mask,
            )

            if start_pos is None:
                tqdm.write(f'    Full coverage after {trace_idx} traces.')
                break

            covered_before = combined_mask.sum()
            path, _ = self._execute_single_trace(start_pos, combined_mask.copy())

            # Split trace at off-vessel bridges
            dt = self.env.distance_transform
            tol = self.env.tolerance
            sub_paths = self._split_trace_at_bridges(path, dt, tol)

            # Stamp on-vessel segments only, snapped to the predicted centerline
            # ridge and connected into continuous lines. Bridges never entered
            # combined_mask (traced on a copy), so no off-vessel clearing needed.
            for sp in sub_paths:
                all_paths.append(sp)
                for seg in self._snap_and_gate(np.array(sp, dtype=np.intp)):
                    _stamp_polyline(
                        combined_mask,
                        seg,
                        max_gap=self._stamp_max_gap,
                    )

            gain = (combined_mask.sum() - covered_before) / gt_total
            coverage_pct = combined_mask.sum() / gt_total

            tqdm.write(
                f'    Trace {trace_idx + 1:3d} from {start_pos} -> {len(path)} steps  gain={gain:.3f}  coverage={coverage_pct:.3f}'
            )

            if trace_idx >= 3 and gain < min_coverage_gain:
                tqdm.write(f'    Early stop: gain {gain:.4f} < {min_coverage_gain}')
                break

        return combined_mask, all_paths

    def _gap_reseeder(
        self,
        combined_mask: np.ndarray,
        sample: Dict[str, Any],
        n_gap_seeds: int = 40,
        gap_threshold: int = 25,
    ) -> List[Tuple[int, int]]:
        """Find vessel pixels far from covered regions → new frontier seeds.

        Prevents starvation when the frontier runs dry while large uncovered
        vessel segments remain (e.g. disconnected subtrees or peripheral
        branches that no trace endpoint reached).

        LEAK-FREE: uses the PREDICTED centerline (UNet) as the vessel proxy,
        not GT — otherwise the reseeder would use ground truth to discover the
        regions the tracer missed (an inference-time leak). gap_threshold
        controls the minimum distance from any covered pixel before a vessel
        point is considered a gap.  Seeds are spaced at least gap_threshold//2
        apart.
        """
        from scipy.ndimage import (
            distance_transform_edt,
        )

        centerline = self.env.pred_centerline
        if centerline is None:
            return []
        centerline = np.asarray(centerline)
        h, w = centerline.shape
        margin = self.half + 5

        vessel_uncovered = (centerline > 0) & (combined_mask == 0)
        if not vessel_uncovered.any():
            return []

        if combined_mask.any():
            dist_from_covered = distance_transform_edt(combined_mask == 0).astype(np.float32)
        else:
            dist_from_covered = np.full(
                (h, w),
                float(max(h, w)),
                dtype=np.float32,
            )

        gap_mask = vessel_uncovered & (dist_from_covered > gap_threshold)
        gap_pts = np.argwhere(gap_mask)
        if len(gap_pts) == 0:
            return []

        valid = [(int(y), int(x)) for y, x in gap_pts if margin <= y < h - margin and margin <= x < w - margin]
        if not valid:
            return []
        valid.sort(
            key=lambda yx: dist_from_covered[yx[0], yx[1]],
            reverse=True,
        )

        min_half = max(4, gap_threshold // 4)
        selected: List[Tuple[int, int]] = []
        occupied = np.zeros((h, w), dtype=bool)
        for y, x in valid:
            if len(selected) >= n_gap_seeds:
                break
            if occupied[
                max(0, y - min_half) : y + min_half + 1,
                max(0, x - min_half) : x + min_half + 1,
            ].any():
                continue
            selected.append((y, x))
            occupied[
                max(0, y - min_half) : y + min_half + 1,
                max(0, x - min_half) : x + min_half + 1,
            ] = True

        return selected

    def _setup_env(self, sample: Dict[str, Any]):
        # GT-ablation: pass GARBAGE ground truth to the env (zeroed centerline,
        # large-constant distance transform) to certify leak-freedom. The
        # predicted inputs below and the GT used for metric scoring (which lives
        # on `sample` and is read directly by run_rl_tracing, not via the env)
        # are untouched. If metrics are unchanged vs the normal run, the
        # prediction provably does not depend on GT.
        gt_centerline = sample['centerline']
        gt_dt = sample['distance_transform']
        if self._corrupt_gt:
            gt_centerline = np.zeros_like(gt_centerline)
            gt_dt = np.full_like(gt_dt, 1.0e6)
        self.env.set_data(
            image=sample['image'],
            centerline=gt_centerline,
            distance_transform=gt_dt,
            fov_mask=sample['fov_mask'],
            vessel_orientation=sample.get('vessel_orientation'),
            unet_prior=sample.get('unet_prior'),
            pred_centerline=sample.get('pred_centerline'),
            pred_distance_transform=sample.get('pred_distance_transform'),
            pred_dt_gradient=sample.get('pred_dt_gradient'),
        )
        # Precompute the nearest predicted-centerline-ridge index map for
        # inference snapping AND vessel gating (set_data populates
        # env.pred_centerline — a binary skeleton — even when the sample didn't
        # supply one). Max connector length scales with snap radius + stride.
        self._stamp_max_gap = 2.0 * self.snap_radius + float(getattr(self.env, 'step_size', 1))
        self._snap_iy = self._snap_ix = self._snap_dist = None
        if (self.snap_to_centerline or self.vessel_gate) and self.env.pred_centerline is not None:
            ridge = np.asarray(self.env.pred_centerline) > 0
            if ridge.any():
                from scipy.ndimage import (
                    distance_transform_edt,
                )

                dist, (iy, ix) = distance_transform_edt(
                    ~ridge,
                    return_indices=True,
                )
                (
                    self._snap_dist,
                    self._snap_iy,
                    self._snap_ix,
                ) = dist, iy, ix

    def _snap_and_gate(self, coords: np.ndarray) -> List[np.ndarray]:
        """Return the trace as a list of CONTIGUOUS sub-segments after optional
        snapping and vessel gating against the predicted-centerline ridge.

        - snap: points within ``snap_radius`` of a ridge pixel are moved onto
          the nearest one (fixes the ~2-3px off-centre drift).
        - gate: points farther than ``snap_radius`` from any ridge pixel are
          DROPPED — they cross background the perception model sees no vessel
          in — and the segment is broken there, so no straight stroke is
          painted across the gap (the false-positive web).

        When the ridge map is unavailable, returns the input unchanged.
        """
        if self._snap_dist is None or len(coords) == 0:
            return [coords]
        ys, xs = coords[:, 0], coords[:, 1]
        near = self._snap_dist[ys, xs] <= self.snap_radius
        snapped = coords.copy()
        if self.snap_to_centerline and near.any():
            snapped[near, 0] = self._snap_iy[ys[near], xs[near]]
            snapped[near, 1] = self._snap_ix[ys[near], xs[near]]
        # If not gating, keep every point as a single segment (snapped or not).
        if not self.vessel_gate:
            return [snapped]
        # Gate: split into runs of consecutive kept (near-ridge) points.
        segments: List[np.ndarray] = []
        start = None
        for i, keep in enumerate(near):
            if keep and start is None:
                start = i
            elif not keep and start is not None:
                segments.append(snapped[start:i])
                start = None
        if start is not None:
            segments.append(snapped[start:])
        return segments

    def _pick_frontier_seed_from_gt(
        self,
        gt_centerline: np.ndarray,
        covered: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        uncovered = (gt_centerline > 0) & (covered == 0)
        if not uncovered.any():
            return None

        uncovered_pts = np.argwhere(uncovered)
        h, w = gt_centerline.shape

        covered_bin = (covered > 0).astype(np.uint8)
        if covered_bin.any():
            dist = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
            scores = dist[
                uncovered_pts[:, 0],
                uncovered_pts[:, 1],
            ]
            best = uncovered_pts[np.argmax(scores)]
        else:
            centre = np.array([h // 2, w // 2])
            dists = np.linalg.norm(uncovered_pts - centre, axis=1)
            best = uncovered_pts[np.argmin(dists)]

        y = int(
            np.clip(
                best[0],
                self.half,
                h - self.half - 1,
            )
        )
        x = int(
            np.clip(
                best[1],
                self.half,
                w - self.half - 1,
            )
        )
        return (y, x)

    def _split_trace_at_bridges(
        self,
        path: List[Tuple[int, int]],
        distance_transform: np.ndarray,
        tolerance: float,
    ) -> List[List[Tuple[int, int]]]:
        """Split a single trace into sub-traces wherever it went off-vessel.

        Removes bridge segments that cross non-vessel regions.
        Only keeps sub-traces with >= 3 on-vessel points.
        """
        if len(path) < 3:
            return [path]

        coords = np.array(path, dtype=np.intp)
        on_vessel = distance_transform[coords[:, 0], coords[:, 1]] <= tolerance

        # Find boundaries where on_vessel changes
        changes = np.diff(on_vessel.astype(np.int8))
        split_indices = np.where(changes != 0)[0] + 1
        chunks = np.split(np.arange(len(path)), split_indices)

        segments = []
        for chunk in chunks:
            if len(chunk) >= 3 and on_vessel[chunk[0]]:
                segments.append([tuple(coords[i]) for i in chunk])

        return segments if segments else [path]
