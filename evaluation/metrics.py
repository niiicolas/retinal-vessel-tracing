# evaluation/metrics.py
"""
Evaluation metrics for vessel centerline tracing.
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Dict, List, Tuple, Optional
import networkx as nx


class CenterlineMetrics:
    """
    Metrics for evaluating centerline extraction quality.
    """
    
    def __init__(self, tolerance_levels: List[int] = [1, 2, 3]):
        self.tolerance_levels = tolerance_levels
        
    def compute_all_metrics(self,
                            pred_skeleton: np.ndarray,
                            gt_skeleton: np.ndarray,
                            gt_vessel_mask: Optional[np.ndarray] = None
                            ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            pred_skeleton: Predicted skeleton (binary)
            gt_skeleton: Ground truth skeleton (binary)
            gt_vessel_mask: Optional ground truth vessel mask
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # F1 at different tolerances
        for tau in self.tolerance_levels:
            precision, recall, f1 = self.centerline_f1(pred_skeleton, gt_skeleton, tau)
            metrics[f'precision@{tau}px'] = precision
            metrics[f'recall@{tau}px'] = recall
            metrics[f'f1@{tau}px'] = f1
        
        # clDice
        if gt_vessel_mask is not None:
            metrics['cldice'] = self.cl_dice(pred_skeleton, gt_skeleton, gt_vessel_mask)
        
        # Topology metrics
        topo_metrics = self.topology_metrics(pred_skeleton, gt_skeleton)
        metrics.update(topo_metrics)
        
        return metrics
    
    def centerline_f1(self,
                      pred: np.ndarray,
                      gt: np.ndarray,
                      tolerance: int = 2) -> Tuple[float, float, float]:
        """
        Compute centerline F1 with distance tolerance.
        
        Args:
            pred: Predicted skeleton
            gt: Ground truth skeleton
            tolerance: Distance tolerance in pixels
            
        Returns:
            precision, recall, f1
        """
        if pred.sum() == 0 or gt.sum() == 0:
            if pred.sum() == 0 and gt.sum() == 0:
                return 1.0, 1.0, 1.0
            return 0.0, 0.0, 0.0
        
        # Distance transforms
        gt_dist = ndimage.distance_transform_edt(1 - gt)
        pred_dist = ndimage.distance_transform_edt(1 - pred)
        
        # Precision: predicted pixels close to GT
        pred_points = np.argwhere(pred > 0)
        tp_precision = sum(gt_dist[y, x] <= tolerance for y, x in pred_points)
        precision = tp_precision / len(pred_points)
        
        # Recall: GT pixels close to prediction
        gt_points = np.argwhere(gt > 0)
        tp_recall = sum(pred_dist[y, x] <= tolerance for y, x in gt_points)
        recall = tp_recall / len(gt_points)
        
        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return precision, recall, f1
    
    def cl_dice(self,
                pred_skeleton: np.ndarray,
                gt_skeleton: np.ndarray,
                gt_vessel_mask: np.ndarray) -> float:
        """
        Compute centerline Dice coefficient.
        
        Args:
            pred_skeleton: Predicted skeleton
            gt_skeleton: Ground truth skeleton
            gt_vessel_mask: Ground truth vessel mask
            
        Returns:
            clDice value
        """
        # Ensure binary
        pred = pred_skeleton > 0
        gt = gt_skeleton > 0
        mask = gt_vessel_mask > 0
        
        # Topology precision: fraction of pred skeleton inside vessel mask
        if pred.sum() > 0:
            tprec = np.logical_and(pred, mask).sum() / pred.sum()
        else:
            tprec = 0.0
        
        # Topology sensitivity: fraction of GT skeleton covered by pred
        if gt.sum() > 0:
            # Dilate prediction slightly
            from scipy.ndimage import binary_dilation
            pred_dilated = binary_dilation(pred, iterations=2)
            tsens = np.logical_and(gt, pred_dilated).sum() / gt.sum()
        else:
            tsens = 1.0
        
        # clDice
        if tprec + tsens > 0:
            cldice = 2 * tprec * tsens / (tprec + tsens)
        else:
            cldice = 0.0
        
        return cldice
    
    def topology_metrics(self,
                         pred: np.ndarray,
                         gt: np.ndarray) -> Dict[str, float]:
        """
        Compute topology-related metrics.
        
        Args:
            pred: Predicted skeleton
            gt: Ground truth skeleton
            
        Returns:
            Dictionary of topology metrics
        """
        from data.centerline_extraction import CenterlineExtractor
        
        extractor = CenterlineExtractor()
        
        # Connected components
        pred_labels, pred_cc = ndimage.label(pred > 0)
        gt_labels, gt_cc = ndimage.label(gt > 0)
        
        # Endpoints and junctions
        pred_endpoints = len(extractor._find_endpoints(pred > 0))
        gt_endpoints = len(extractor._find_endpoints(gt > 0))
        
        pred_junctions = len(extractor._find_junctions(pred > 0))
        gt_junctions = len(extractor._find_junctions(gt > 0))
        
        # Betti numbers (simplified: B0 = connected components, B1 = holes/cycles)
        pred_b0 = pred_cc
        gt_b0 = gt_cc
        
        # Estimate B1 using Euler characteristic: χ = V - E + F = B0 - B1
        # For skeleton: V ≈ endpoints + junctions, E ≈ branches
        pred_b1 = max(0, pred_junctions - pred_endpoints + pred_cc)
        gt_b1 = max(0, gt_junctions - gt_endpoints + gt_cc)
        
        return {
            'connected_components_pred': pred_cc,
            'connected_components_gt': gt_cc,
            'cc_ratio': pred_cc / max(gt_cc, 1),
            'endpoints_pred': pred_endpoints,
            'endpoints_gt': gt_endpoints,
            'junctions_pred': pred_junctions,
            'junctions_gt': gt_junctions,
            'betti_0_error': abs(pred_b0 - gt_b0),
            'betti_1_error': abs(pred_b1 - gt_b1)
        }
    
    def path_efficiency(self,
                        trajectory: List[Tuple[int, int]],
                        gt_skeleton: np.ndarray,
                        tolerance: int = 2) -> Dict[str, float]:
        """
        Compute path efficiency metrics.
        
        Args:
            trajectory: List of (y, x) positions
            gt_skeleton: Ground truth skeleton
            tolerance: Distance tolerance
            
        Returns:
            Dictionary of efficiency metrics
        """
        if len(trajectory) == 0:
            return {'steps_per_covered_pixel': float('inf'), 'path_length': 0}
        
        gt_dist = ndimage.distance_transform_edt(1 - gt_skeleton)
        
        # Count covered GT pixels
        covered = np.zeros_like(gt_skeleton)
        for y, x in trajectory:
            if 0 <= y < covered.shape[0] and 0 <= x < covered.shape[1]:
                # Mark nearby GT pixels as covered
                y_min = max(0, y - tolerance)
                y_max = min(covered.shape[0], y + tolerance + 1)
                x_min = max(0, x - tolerance)
                x_max = min(covered.shape[1], x + tolerance + 1)
                
                for py in range(y_min, y_max):
                    for px in range(x_min, x_max):
                        if gt_skeleton[py, px] > 0:
                            dist = np.sqrt((py - y) ** 2 + (px - x) ** 2)
                            if dist <= tolerance:
                                covered[py, px] = 1
        
        covered_pixels = covered.sum()
        total_steps = len(trajectory)
        
        return {
            'steps_per_covered_pixel': total_steps / max(covered_pixels, 1),
            'path_length': total_steps,
            'covered_pixels': int(covered_pixels),
            'coverage_ratio': covered_pixels / max(gt_skeleton.sum(), 1)
        }


class EvaluationRunner:
    """Run evaluation on a dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        tolerance_levels = config.get('evaluation', {}).get('tolerance_levels', [1, 2, 3])
        self.metrics = CenterlineMetrics(tolerance_levels)
        
    def evaluate_model(self,
                       policy,
                       env,
                       dataset,
                       num_samples: Optional[int] = None,
                       device=None) -> Dict[str, float]:
        """
        Evaluate policy on dataset.
        
        Args:
            policy: Trained policy network
            env: Evaluation environment
            dataset: Evaluation dataset
            num_samples: Number of samples to evaluate
            device: Torch device
            
        Returns:
            Aggregated metrics
        """
        import torch
        from tqdm import tqdm
        
        if num_samples is None:
            num_samples = len(dataset)
        
        all_metrics = []
        
        policy.eval()
        
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
            sample = dataset[i]
            
            # Set up environment
            env.set_data(
                image=sample['image'].permute(1, 2, 0).numpy(),
                centerline=sample['centerline'].squeeze().numpy(),
                distance_transform=sample['distance_transform'].squeeze().numpy(),
                fov_mask=sample['fov_mask'].squeeze().numpy()
            )
            
            # Run episode
            obs, _ = env.reset()
            done = False
            trajectory = [tuple(env.position)]
            
            with torch.no_grad():
                hidden = None
                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                    if len(obs_tensor.shape) == 3:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    
                    action, _, _, _, hidden = policy.get_action_and_value(obs_tensor, hidden)
                    action = action.item()
                    
                    obs, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    trajectory.append(tuple(env.position))
            
            # Get predicted skeleton from visited mask
            pred_skeleton = env.visited_mask > 0
            gt_skeleton = sample['centerline'].squeeze().numpy() > 0
            gt_vessel_mask = sample['vessel_mask'].squeeze().numpy() if 'vessel_mask' in sample else None
            
            # Compute metrics
            sample_metrics = self.metrics.compute_all_metrics(
                pred_skeleton, gt_skeleton, gt_vessel_mask
            )
            
            # Add path efficiency
            path_metrics = self.metrics.path_efficiency(trajectory, gt_skeleton)
            sample_metrics.update(path_metrics)
            
            all_metrics.append(sample_metrics)
        
        # Aggregate
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
