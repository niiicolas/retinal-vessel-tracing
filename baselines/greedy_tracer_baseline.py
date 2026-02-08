# baselines/greedy_tracer_baseline.py
"""
Greedy tracing baseline that follows maximum vesselness.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, Optional, List

class GreedyTracer:
    DIRECTIONS = np.array([
        [-1, 0], [-1, 1], [0, 1], [1, 1],
        [1, 0], [1, -1], [0, -1], [-1, -1]
    ])
    
    def __init__(self, step_size: int = 1, max_steps: int = 1000,
                 min_vesselness: float = 0.05, momentum: float = 0.3):
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_vesselness = min_vesselness
        self.momentum = momentum
        self.vessel_cmap = LinearSegmentedColormap.from_list(
            'vessel', ['black', 'blue', 'cyan', 'yellow', 'red']
        )
        
    def trace(self, vesselness: np.ndarray, start_point: Tuple[int, int],
              mask: Optional[np.ndarray] = None, 
              visualize: bool = False) -> np.ndarray:
        """
        Trace a vessel from a starting point.
        
        Args:
            vesselness: Vesselness map
            start_point: (y, x) starting coordinates
            mask: Optional binary mask
            visualize: If True, show trajectory evolution
            
        Returns:
            Array of (y, x) coordinates along the trajectory
        """
        h, w = vesselness.shape
        visited = np.zeros((h, w), dtype=bool)
        trajectory = [start_point]
        current = np.array(start_point)
        prev_direction = None
        
        for step in range(self.max_steps):
            y, x = current
            visited[y, x] = True
            best_score = -1
            best_direction = None
            best_next = None
            
            for i, direction in enumerate(self.DIRECTIONS):
                next_pos = current + direction * self.step_size
                ny, nx = next_pos
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if mask is not None and mask[ny, nx] == 0:
                    continue
                if visited[ny, nx]:
                    continue
                
                score = vesselness[ny, nx]
                if prev_direction is not None:
                    prev_vec = self.DIRECTIONS[prev_direction]
                    similarity = np.dot(direction, prev_vec) / (
                        np.linalg.norm(direction) * np.linalg.norm(prev_vec) + 1e-8
                    )
                    score += self.momentum * max(0, similarity)
                
                if score > best_score:
                    best_score = score
                    best_direction = i
                    best_next = next_pos
            
            if best_next is None or vesselness[best_next[0], best_next[1]] < self.min_vesselness:
                break
            
            current = best_next
            prev_direction = best_direction
            trajectory.append(tuple(current))
        
        trajectory_array = np.array(trajectory)
        
        if visualize:
            self.visualize_trace(vesselness, trajectory_array, start_point)
        
        return trajectory_array
    
    def trace_from_seeds(self, vesselness: np.ndarray, seeds: List[Tuple[int, int]],
                         mask: Optional[np.ndarray] = None,
                         visualize: bool = False) -> np.ndarray:
        """
        Trace from multiple seed points.
        
        Args:
            vesselness: Vesselness map
            seeds: List of (y, x) seed coordinates
            mask: Optional binary mask
            visualize: If True, show all trajectories
            
        Returns:
            Binary skeleton map
        """
        h, w = vesselness.shape
        skeleton = np.zeros((h, w), dtype=np.float32)
        trajectories = []
        
        for seed in seeds:
            trajectory = self.trace(vesselness, seed, mask, visualize=False)
            trajectories.append(trajectory)
            for y, x in trajectory:
                if 0 <= y < h and 0 <= x < w:
                    skeleton[y, x] = 1.0
        
        if visualize:
            self.visualize_multiple_traces(vesselness, trajectories, seeds)
        
        return skeleton
    
    def visualize_trace(self, vesselness: np.ndarray, 
                       trajectory: np.ndarray,
                       start_point: Tuple[int, int],
                       save_path: Optional[str] = None):
        """Visualize a single trajectory."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Trajectory on vesselness
        ax1.imshow(vesselness, cmap=self.vessel_cmap, alpha=0.7)
        if len(trajectory) > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectory)))
            ax1.scatter(trajectory[:, 1], trajectory[:, 0], 
                       c=colors, s=30, marker='o', edgecolors='white', linewidths=0.5)
            ax1.plot(trajectory[:, 1], trajectory[:, 0], 
                    'w-', linewidth=2, alpha=0.6)
        ax1.scatter(start_point[1], start_point[0], 
                   c='lime', s=250, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax1.set_title(f'Trajectory ({len(trajectory)} steps)', fontsize=14)
        ax1.axis('off')
        
        # Right: Vesselness along path
        if len(trajectory) > 0:
            vessel_values = [vesselness[y, x] for y, x in trajectory]
            ax2.plot(vessel_values, linewidth=2, color='darkblue', label='Vesselness')
            ax2.axhline(y=np.mean(vessel_values), color='red', 
                       linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(vessel_values):.3f}')
            ax2.axhline(y=self.min_vesselness, color='orange', 
                       linestyle=':', linewidth=1.5, label=f'Threshold: {self.min_vesselness}')
            ax2.fill_between(range(len(vessel_values)), vessel_values, 
                            alpha=0.3, color='lightblue')
            ax2.set_xlabel('Step', fontsize=12)
            ax2.set_ylabel('Vesselness', fontsize=12)
            ax2.set_title('Vesselness Along Path', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_multiple_traces(self, vesselness: np.ndarray,
                                  trajectories: List[np.ndarray],
                                  seeds: List[Tuple[int, int]],
                                  save_path: Optional[str] = None):
        """Visualize multiple trajectories."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        ax.imshow(vesselness, cmap=self.vessel_cmap, alpha=0.6)
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(trajectories), 10)))
        
        for i, (traj, seed) in enumerate(zip(trajectories, seeds)):
            if len(traj) > 0:
                color = colors[i % 10]
                ax.plot(traj[:, 1], traj[:, 0], 
                       color=color, linewidth=2.5, alpha=0.8,
                       label=f'Trace {i+1} ({len(traj)} pts)')
                ax.scatter(seed[1], seed[0], 
                          c=[color], s=180, marker='*', 
                          edgecolors='white', linewidths=2, zorder=10)
        
        ax.set_title(f'{len(trajectories)} Trajectories', fontsize=16)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def compare_with_ground_truth(self, vesselness: np.ndarray,
                                 skeleton: np.ndarray,
                                 ground_truth: np.ndarray,
                                 save_path: Optional[str] = None):
        """Compare tracing result with ground truth."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Vesselness
        axes[0].imshow(vesselness, cmap=self.vessel_cmap)
        axes[0].set_title('Vesselness Map', fontsize=14)
        axes[0].axis('off')
        
        # Greedy result
        axes[1].imshow(skeleton, cmap='gray')
        axes[1].set_title('Greedy Tracing Result', fontsize=14)
        axes[1].axis('off')
        
        # Overlay comparison
        overlay = np.zeros((*vesselness.shape, 3))
        overlay[skeleton > 0] = [0, 1, 0]      # Green: detected
        overlay[ground_truth > 0] = [1, 0, 0]  # Red: ground truth
        overlap = (skeleton > 0) & (ground_truth > 0)
        overlay[overlap] = [1, 1, 0]           # Yellow: correct
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Green=Detected, Red=GT, Yellow=Match)', fontsize=14)
        axes[2].axis('off')
        
        # Calculate metrics
        tp = np.sum(overlap)
        fp = np.sum((skeleton > 0) & (ground_truth == 0))
        fn = np.sum((skeleton == 0) & (ground_truth > 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        fig.suptitle(f'Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
                    fontsize=16, y=1.02)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return {'precision': precision, 'recall': recall, 'f1': f1}