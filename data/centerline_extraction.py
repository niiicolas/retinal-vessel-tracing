# data/centerline_extraction.py
"""Extract centerlines from binary vessel masks."""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import (
    medial_axis,
    remove_small_objects,
    skeletonize,
)


def compute_vessel_width(
    vessel_mask: np.ndarray,
) -> float:
    """Median vessel width in pixels (= 2 × median medial-axis radius).

    Returns 0.0 for an empty mask. Used to scale skeleton-pruning, env
    tolerance, and junction-map dilation so the pipeline behaves
    consistently across datasets that span 2–5× in resized vessel width
    (DRHAGIS/HRF ~2 px vs CHASEDB1 ~4.5 px after the dataloader's
    aspect-preserving resize-to-512).
    """
    binary = vessel_mask > 0.5
    if not binary.any():
        return 0.0
    binary = remove_small_objects(binary, min_size=50)
    if not binary.any():
        return 0.0
    skel, dist = medial_axis(binary, return_distance=True)
    sk_dist = dist[skel]
    if sk_dist.size == 0:
        return 0.0
    return float(np.median(sk_dist) * 2.0)


class CenterlineExtractor:
    """Extract and process vessel centerlines from binary masks.

    ``min_branch_length`` is either:
      * a positive int — absolute pixel threshold (legacy behaviour); or
      * ``None``       — auto-scale per image: prune any terminal branch
                         shorter than ``min_branch_widths`` × median
                         vessel width, with ``_MIN_BRANCH_LENGTH_FLOOR``
                         as a hard lower bound.
    """

    def __init__(
        self,
        min_branch_length: Optional[int] = None,
        prune_iterations: int = 5,
        min_branch_widths: float = 2.5,
    ):
        self.min_branch_length = min_branch_length
        self.prune_iterations = prune_iterations
        self.min_branch_widths = float(min_branch_widths)

    def extract_centerline(self, vessel_mask: np.ndarray) -> np.ndarray:
        """Extract centerline from binary vessel mask using skeletonization."""
        # Ensure binary mask
        binary = vessel_mask > 0.5

        # Remove small disconnected components
        binary = remove_small_objects(binary, min_size=50)

        # Skeletonize
        skeleton = skeletonize(binary)

        # Determine pruning threshold (absolute px vs width-scaled).
        if self.min_branch_length is None:
            width = compute_vessel_width(vessel_mask)
            mbl = max(
                3,
                int(round(self.min_branch_widths * width)),
            )
        else:
            mbl = int(self.min_branch_length)

        # Prune spurious branches
        skeleton = self._prune_skeleton(skeleton, min_branch_length=mbl)

        return skeleton.astype(np.float32)

    def _prune_skeleton(
        self,
        skeleton: np.ndarray,
        min_branch_length: Optional[int] = None,
    ) -> np.ndarray:
        """Remove spurious short branches from skeleton.

        ``min_branch_length`` overrides ``self.min_branch_length``. Required
        when the constructor was passed ``min_branch_length=None`` (auto
        mode) — extract_centerline computes the per-image value and passes
        it in here.
        """
        mbl = (
            min_branch_length
            if min_branch_length is not None
            else (self.min_branch_length if self.min_branch_length is not None else 10)
        )
        for _ in range(self.prune_iterations):
            endpoints = self._find_endpoints(skeleton)
            for y, x in endpoints:
                branch_length = self._trace_branch_length(skeleton, y, x)
                if branch_length < mbl:
                    skeleton = self._remove_branch(
                        skeleton,
                        y,
                        x,
                        max_steps=mbl + 5,
                    )
        return skeleton

    def _get_neighbor_counts(self, skeleton: np.ndarray) -> np.ndarray:
        """Convolve once; callers threshold the result themselves."""
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        return ndimage.convolve(
            skeleton.astype(np.int32),
            kernel,
            mode='constant',
        )

    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        nc = self._get_neighbor_counts(skeleton)
        return [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc == 1))]

    def _find_junctions(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        nc = self._get_neighbor_counts(skeleton)
        return [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc > 2))]

    def _trace_branch_length(
        self,
        skeleton: np.ndarray,
        start_y: int,
        start_x: int,
        max_steps: int = 100,
    ) -> int:
        """Trace a branch from endpoint until junction or end."""
        visited = np.zeros_like(skeleton, dtype=bool)
        y, x = start_y, start_x
        length = 0

        for _ in range(max_steps):
            visited[y, x] = True
            length += 1
            # Find unvisited neighbors on skeleton
            neighbors = self._get_skeleton_neighbors(skeleton, y, x, visited)

            if len(neighbors) == 0:
                break  # Dead end
            elif len(neighbors) > 1:
                break  # Junction reached
            else:
                y, x = neighbors[0]

        return length

    def _get_skeleton_neighbors(
        self,
        skeleton: np.ndarray,
        y: int,
        x: int,
        visited: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Get unvisited skeleton neighbors of a pixel."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < skeleton.shape[0]
                    and 0 <= nx < skeleton.shape[1]
                    and skeleton[ny, nx] > 0
                    and not visited[ny, nx]
                ):
                    neighbors.append((ny, nx))
        return neighbors

    def _remove_branch(
        self,
        skeleton: np.ndarray,
        start_y: int,
        start_x: int,
        max_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Remove a branch starting from an endpoint.

        ``max_steps`` upper-bounds how far we walk; defaults to
        ``min_branch_length + 5`` when the attribute is an int, or 15 when
        it is ``None`` (auto mode — callers should pass in an explicit
        value matching the per-image threshold).
        """
        if max_steps is None:
            if isinstance(self.min_branch_length, int):
                max_steps = self.min_branch_length + 5
            else:
                max_steps = 15
        result = skeleton.copy()
        visited = np.zeros_like(skeleton, dtype=bool)
        y, x = start_y, start_x

        for _ in range(max_steps):
            visited[y, x] = True
            result[y, x] = 0
            neighbors = self._get_skeleton_neighbors(skeleton, y, x, visited)

            if len(neighbors) != 1:
                break
            y, x = neighbors[0]

        return result

    def skeleton_to_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Convert skeleton image to graph representation."""
        G = nx.Graph()

        nc = self._get_neighbor_counts(skeleton)
        endpoints = [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc == 1))]
        junctions = [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc > 2))]
        special_points = set(endpoints + junctions)

        # Add nodes
        for idx, point in enumerate(special_points):
            G.add_node(
                idx,
                pos=point,
                type='endpoint' if point in endpoints else 'junction',
            )

        # Create mapping from coordinates to node indices
        point_to_node = {point: idx for idx, point in enumerate(special_points)}

        # Trace edges between special points
        visited_edges = set()

        for start_point in special_points:
            neighbors = self._get_skeleton_neighbors(
                skeleton,
                start_point[0],
                start_point[1],
                np.zeros_like(skeleton, dtype=bool),
            )

            for neighbor in neighbors:
                edge_path = self._trace_edge(
                    skeleton,
                    start_point,
                    neighbor,
                    special_points,
                )

                if edge_path and edge_path[-1] in special_points:
                    end_point = edge_path[-1]
                    edge_key = tuple(
                        sorted(
                            [
                                start_point,
                                end_point,
                            ]
                        )
                    )

                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        G.add_edge(
                            point_to_node[start_point],
                            point_to_node[end_point],
                            path=edge_path,
                            length=len(edge_path),
                        )

        return G

    def _trace_edge(
        self,
        skeleton: np.ndarray,
        start: Tuple[int, int],
        first_step: Tuple[int, int],
        special_points: set,
        max_steps: int = 5000,
    ) -> List[Tuple[int, int]]:
        path = [start, first_step]
        visited = {start, first_step}
        current = first_step

        for _ in range(max_steps):
            if current in special_points and current != start:
                return path

            neighbors = [
                (y + dy, x + dx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if not (dy == 0 and dx == 0)
                for y, x in (current,)
                if 0 <= y + dy < skeleton.shape[0]
                and 0 <= x + dx < skeleton.shape[1]
                and skeleton[y + dy, x + dx] > 0
                and (y + dy, x + dx) not in visited
            ]

            if not neighbors:
                return path

            current = neighbors[0]
            path.append(current)
            visited.add(current)

        return path

    def compute_distance_transform(
        self,
        centerline: np.ndarray,
        tolerance: float = 2.0,
    ) -> np.ndarray:
        """Compute distance transform from centerline, clipped at tolerance."""
        if centerline.max() == 0:
            return np.ones_like(centerline) * tolerance

        distance = ndimage.distance_transform_edt(1 - centerline)
        return np.clip(distance, 0, tolerance)

    def generate_expert_traces(
        self,
        skeleton: np.ndarray,
        graph: Optional[nx.Graph] = None,
    ) -> List[List[Tuple[int, int]]]:
        """Generate expert traces by traversing the skeleton graph."""
        if graph is None:
            graph = self.skeleton_to_graph(skeleton)

        if len(graph.nodes) == 0:
            return []

        traces = []
        visited_edges = set()

        # Start from endpoints (degree 1 nodes)
        endpoints = [n for n in graph.nodes if graph.degree(n) == 1]

        if not endpoints:
            # No endpoints, start from any node
            endpoints = [list(graph.nodes)[0]]

        for start_node in endpoints:
            # DFS traversal from this endpoint
            stack = [(start_node, None)]

            while stack:
                current_node, prev_edge = stack.pop()

                for neighbor in graph.neighbors(current_node):
                    edge_key = tuple(
                        sorted(
                            [
                                current_node,
                                neighbor,
                            ]
                        )
                    )

                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        edge_data = graph.get_edge_data(
                            current_node,
                            neighbor,
                        )
                        path = edge_data.get('path', [])

                        if path:
                            traces.append(path)

                        stack.append((neighbor, edge_key))

        return traces


def compute_centerline_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float = 2.0,
) -> Dict[str, float]:
    """Standalone tolerance-aware centerline F1, precision, and recall.
    Used by the training loop and evaluation suite.
    """
    extractor = CenterlineExtractor()
    dist_to_gt = extractor.compute_distance_transform(gt, tolerance=1e9)
    dist_to_pred = extractor.compute_distance_transform(pred, tolerance=1e9)

    pred_px = pred > 0
    gt_px = gt > 0

    precision = float((dist_to_gt[pred_px] <= tolerance).sum()) / max(pred_px.sum(), 1)
    recall = float((dist_to_pred[gt_px] <= tolerance).sum()) / max(gt_px.sum(), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }
