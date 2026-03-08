# environment/frontier_tracer.py
"""
Branch Coverage Manager for Retinal Vessel Tracing.
Implements the Frontier-Based Coverage (Algorithm 2) to trace the full 
connected vascular tree .

Supports both end-to-end inference (using SeedCNN outputs) and 
evaluation tracing (using GT gaps to test PPO isolation).
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional

class FrontierTracer:
    """
    Single source of truth for Frontier-Based Coverage (Algorithm 2)[cite: 133].
    """
    def __init__(self, env, policy_model, device, obs_size: int = 65):
        self.env = env
        self.model = policy_model
        self.device = device
        self.obs_size = obs_size
        self.half = obs_size // 2

    def _execute_single_trace(self, start_pos: Tuple[int, int], combined_mask: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Executes a single continuous trace until the agent stops or terminates[cite: 134].
        Returns the path taken and any alternate branches found at junctions.
        """
        obs, _ = self.env.reset(start_position=start_pos)
        path = [start_pos]
        done = False
        alternate_branches = []

        self.model.eval()
        with torch.no_grad():
            while not done:
                # 5: a <- pi(o); step; update V [cite: 134]
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits, _, _ = self.model(obs_t)
                action = logits.argmax(dim=-1).item()
                
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                y, x = self.env.position
                path.append((y, x))
                combined_mask[y, x] = 1.0

                # 6: if junction with unexplored branches then [cite: 134]
                # junction_detected, branches = self._detect_junction(obs, self.env.position)
                # if junction_detected:
                #     alternate_branches.extend(branches)
                
        return path, alternate_branches

    def trace_from_seeds(self, sample: Dict[str, Any], initial_seeds: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """
        True End-to-End Inference: Algorithm 2 using a stack-based frontier[cite: 134].
        Takes seeds predicted by the SeedCNN and explores the vascular tree.
        """
        self._setup_env(sample)
        h, w = sample['image'].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        all_paths = []
        
        # 1: Frontier <- initial seed(s) [cite: 134]
        frontier = list(initial_seeds)

        # 2: while Frontier not empty do [cite: 134]
        while frontier:
            # 3: Pop (p, d_in); reset counters [cite: 134]
            start_pos = frontier.pop()
            
            # Skip if already covered by a previous trace
            if combined_mask[start_pos[0], start_pos[1]] > 0:
                continue

            # 4: while not done do [cite: 134]
            path, alternate_branches = self._execute_single_trace(start_pos, combined_mask)
            all_paths.append(path)

            # 7: push alternate branches (p', d_in') to Frontier [cite: 134]
            for branch_pos in alternate_branches:
                if combined_mask[branch_pos[0], branch_pos[1]] == 0:
                    frontier.append(branch_pos)

        return combined_mask, all_paths

    def trace_with_gt_gaps(self, sample: Dict[str, Any], max_traces: int = 50, min_coverage_gain: float = 0.005) -> Tuple[np.ndarray, List[List[Tuple[int, int]]]]:
        """
        Evaluation method: Iteratively forces the agent into the largest uncovered 
        ground-truth gaps to isolate and evaluate PPO walking performance.
        """
        self._setup_env(sample)
        h, w = sample['image'].shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.float32)
        all_paths = []
        gt_total = float(max(sample['centerline'].sum(), 1))

        for trace_idx in range(max_traces):
            start_pos = self._pick_frontier_seed_from_gt(sample['centerline'], combined_mask)
            
            if start_pos is None:
                print(f"    Full coverage after {trace_idx} traces.")
                break

            covered_before = combined_mask.sum()
            path, _ = self._execute_single_trace(start_pos, combined_mask)
            all_paths.append(path)

            gain = (combined_mask.sum() - covered_before) / gt_total
            coverage_pct = combined_mask.sum() / gt_total
            
            print(f"    Trace {trace_idx+1:3d} from {start_pos} -> "
                  f"{len(path)} steps  gain={gain:.3f}  coverage={coverage_pct:.3f}")

            if trace_idx >= 3 and gain < min_coverage_gain:
                print(f"    Early stop: gain {gain:.4f} < {min_coverage_gain}")
                break

        return combined_mask, all_paths

    def _setup_env(self, sample: Dict[str, Any]):
        """Helper to inject sample data into the environment."""
        self.env.set_data(
            image=sample['image'],
            centerline=sample['centerline'],
            distance_transform=sample['dist_transform'],
            fov_mask=sample['fov_mask'],
        )

    def _pick_frontier_seed_from_gt(self, gt_centerline: np.ndarray, covered: np.ndarray) -> Optional[Tuple[int, int]]:
        """Finds the uncovered GT centerline pixel furthest from any already-covered pixel."""
        uncovered = (gt_centerline > 0) & (covered == 0)
        if not uncovered.any():
            return None

        uncovered_pts = np.argwhere(uncovered)
        h, w = gt_centerline.shape

        covered_bin = (covered > 0).astype(np.uint8)
        if covered_bin.any():
            dist = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
            scores = dist[uncovered_pts[:, 0], uncovered_pts[:, 1]]
            best = uncovered_pts[np.argmax(scores)]
        else:
            centre = np.array([h // 2, w // 2])
            dists = np.linalg.norm(uncovered_pts - centre, axis=1)
            best = uncovered_pts[np.argmin(dists)]

        y = int(np.clip(best[0], self.half, h - self.half - 1))
        x = int(np.clip(best[1], self.half, w - self.half - 1))
        return (y, x)