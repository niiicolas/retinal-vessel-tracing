"""Vessel centerline extraction: skeletonization, pruning, graph/trace conversion, and F1 scoring."""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.morphology import (
    medial_axis,
    remove_small_objects,
    skeletonize,
)


def compute_vessel_width(vessel_mask: np.ndarray) -> float:
    """Return the median vessel width in pixels (2x the median medial-axis radius).

    Used to width-scale pruning, env tolerance, and junction dilation so the pipeline
    behaves consistently across datasets whose resized vessel widths span 2-5x.

    Args:
        vessel_mask: 2-D float or bool array; values > 0.5 are foreground.

    Returns:
        Median vessel diameter in pixels, or 0.0 if no foreground remains.
    """
    binary = vessel_mask > 0.5
    if not binary.any():
        return 0.0

    binary = remove_small_objects(binary, min_size=50)
    if not binary.any():
        return 0.0

    # dist gives the local half-width (largest inscribed-circle radius) per pixel.
    skel, dist = medial_axis(binary, return_distance=True)

    # Median over skeleton pixels only — robust to thick junctions.
    sk_dist = dist[skel]
    if sk_dist.size == 0:
        return 0.0

    return float(np.median(sk_dist) * 2.0)


class CenterlineExtractor:
    """Extracts pruned centerline skeletons from vessel masks and their graph/trace forms.

    With ``min_branch_length=None`` the prune threshold auto-scales per image to
    ``min_branch_widths`` x median vessel width; a positive int sets an absolute pixel
    threshold (legacy behavior).
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
        """Return a pruned float32 skeleton of the vessel mask.

        Args:
            vessel_mask: 2-D float or bool array; values > 0.5 are foreground.
        """
        binary = vessel_mask > 0.5
        binary = remove_small_objects(binary, min_size=50)
        skeleton = skeletonize(binary)

        # Auto-scale the prune threshold to vessel width unless a fixed value was set.
        if self.min_branch_length is None:
            width = compute_vessel_width(vessel_mask)
            mbl = max(3, int(round(self.min_branch_widths * width)))
        else:
            mbl = int(self.min_branch_length)

        skeleton = self._prune_skeleton(skeleton, min_branch_length=mbl)
        return skeleton.astype(np.float32)

    def _prune_skeleton(self, skeleton: np.ndarray, min_branch_length: Optional[int] = None) -> np.ndarray:
        """Iteratively erase terminal branches shorter than ``min_branch_length``.

        ``min_branch_length`` overrides the instance value and is required in auto mode,
        where extract_centerline computes and passes the per-image threshold.
        """
        mbl = min_branch_length if min_branch_length is not None else (self.min_branch_length if self.min_branch_length is not None else 10)
        for _ in range(self.prune_iterations):
            endpoints = self._find_endpoints(skeleton)
            for y, x in endpoints:
                if self._trace_branch_length(skeleton, y, x) < mbl:
                    skeleton = self._remove_branch(skeleton, y, x, max_steps=mbl + 5)
        return skeleton

    def _get_neighbor_counts(self, skeleton: np.ndarray) -> np.ndarray:
        """Return the 8-connectivity neighbor count for every pixel."""
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        return ndimage.convolve(skeleton.astype(np.int32), kernel, mode='constant')

    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Return skeleton pixels with exactly one neighbor (degree 1)."""
        nc = self._get_neighbor_counts(skeleton)
        return [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc == 1))]

    def _find_junctions(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Return skeleton pixels with more than two neighbors (degree > 2)."""
        nc = self._get_neighbor_counts(skeleton)
        return [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc > 2))]

    def _trace_branch_length(self, skeleton: np.ndarray, start_y: int, start_x: int, max_steps: int = 100) -> int:
        """Walk from an endpoint until a junction or dead end and return the step count."""
        visited = np.zeros_like(skeleton, dtype=bool)
        y, x = start_y, start_x
        length = 0

        for _ in range(max_steps):
            visited[y, x] = True
            length += 1
            neighbors = self._get_skeleton_neighbors(skeleton, y, x, visited)

            if len(neighbors) == 0:
                break
            elif len(neighbors) > 1:
                break  # Branch terminates at a junction.
            else:
                y, x = neighbors[0]

        return length

    def _get_skeleton_neighbors(self, skeleton: np.ndarray, y: int, x: int, visited: np.ndarray) -> List[Tuple[int, int]]:
        """Return unvisited 8-connected skeleton neighbors of (y, x)."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] > 0 and not visited[ny, nx]:
                    neighbors.append((ny, nx))
        return neighbors

    def _remove_branch(self, skeleton: np.ndarray, start_y: int, start_x: int, max_steps: Optional[int] = None) -> np.ndarray:
        """Return a copy with the terminal branch at (start_y, start_x) zeroed out.

        Walks until a junction, dead end, or ``max_steps``. ``max_steps`` defaults to
        ``min_branch_length + 5`` (int mode) or 15 (auto mode); callers should pass an
        explicit value in auto mode.
        """
        if max_steps is None:
            max_steps = self.min_branch_length + 5 if isinstance(self.min_branch_length, int) else 15

        result = skeleton.copy()
        visited = np.zeros_like(skeleton, dtype=bool)
        y, x = start_y, start_x

        for _ in range(max_steps):
            visited[y, x] = True
            result[y, x] = 0
            neighbors = self._get_skeleton_neighbors(skeleton, y, x, visited)

            # Stop erasing at a junction or dead end so the rest of the tree survives.
            if len(neighbors) != 1:
                break
            y, x = neighbors[0]

        return result

    def skeleton_to_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Convert a skeleton image to a NetworkX graph.

        Nodes are endpoints and junctions; each edge carries the connecting pixel
        ``path`` and its ``length``.
        """
        G = nx.Graph()

        nc = self._get_neighbor_counts(skeleton)
        endpoints = [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc == 1))]
        junctions = [(int(y), int(x)) for y, x in np.argwhere((skeleton > 0) & (nc > 2))]
        special_points = set(endpoints + junctions)

        for idx, point in enumerate(special_points):
            G.add_node(idx, pos=point, type='endpoint' if point in endpoints else 'junction')

        point_to_node = {point: idx for idx, point in enumerate(special_points)}

        visited_edges = set()
        for start_point in special_points:
            neighbors = self._get_skeleton_neighbors(
                skeleton,
                start_point[0],
                start_point[1],
                np.zeros_like(skeleton, dtype=bool),
            )
            for neighbor in neighbors:
                edge_path = self._trace_edge(skeleton, start_point, neighbor, special_points)

                if edge_path and edge_path[-1] in special_points:
                    end_point = edge_path[-1]
                    edge_key = tuple(sorted([start_point, end_point]))

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
        """Trace pixels from ``start`` through ``first_step`` to the next endpoint/junction."""
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
        """Return the per-pixel Euclidean distance to the centerline, clipped at ``tolerance``.

        Yields an all-``tolerance`` array for a blank centerline.
        """
        if centerline.max() == 0:
            return np.ones_like(centerline) * tolerance
        distance = ndimage.distance_transform_edt(1 - centerline)
        return np.clip(distance, 0, tolerance)

    def generate_expert_traces(self, skeleton: np.ndarray, graph: Optional[nx.Graph] = None) -> List[List[Tuple[int, int]]]:
        """Return per-edge pixel paths from a DFS traversal of the skeleton graph.

        Traversal starts from degree-1 endpoints, falling back to an arbitrary node for
        loop-only graphs. Each trace is a list of (y, x) coordinates along one edge.
        """
        if graph is None:
            graph = self.skeleton_to_graph(skeleton)
        if len(graph.nodes) == 0:
            return []

        traces = []
        visited_edges = set()

        endpoints = [n for n in graph.nodes if graph.degree(n) == 1]
        if not endpoints:
            endpoints = [list(graph.nodes)[0]]  # Loop-only graph: seed from any node.

        for start_node in endpoints:
            stack = [(start_node, None)]
            while stack:
                current_node, _ = stack.pop()
                for neighbor in graph.neighbors(current_node):
                    edge_key = tuple(sorted([current_node, neighbor]))
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)
                        path = graph.get_edge_data(current_node, neighbor).get('path', [])
                        if path:
                            traces.append(path)
                        stack.append((neighbor, edge_key))

        return traces


def compute_centerline_f1(pred: np.ndarray, gt: np.ndarray, tolerance: float = 2.0) -> Dict[str, float]:
    """Compute tolerance-aware centerline F1, precision, and recall.

    A predicted pixel is a true positive if within ``tolerance`` pixels of the GT
    centerline (symmetrically for recall).

    Args:
        pred: Binary predicted centerline (values > 0 are foreground).
        gt: Binary ground-truth centerline.
        tolerance: Max distance (px) for a pixel to count as a match.

    Returns:
        Dict with keys ``f1``, ``precision``, ``recall``.
    """
    extractor = CenterlineExtractor()
    # Effectively unbounded DT so the tolerance threshold can be applied post-hoc.
    dist_to_gt = extractor.compute_distance_transform(gt, tolerance=1e9)
    dist_to_pred = extractor.compute_distance_transform(pred, tolerance=1e9)

    pred_px = pred > 0
    gt_px = gt > 0

    precision = float((dist_to_gt[pred_px] <= tolerance).sum()) / max(pred_px.sum(), 1)
    recall = float((dist_to_pred[gt_px] <= tolerance).sum()) / max(gt_px.sum(), 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'f1': f1, 'precision': precision, 'recall': recall}
