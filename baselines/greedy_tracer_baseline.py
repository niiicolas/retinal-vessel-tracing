# baselines/greedy_tracer_baseline.py
"""
Orientation-following vessel tracer using Hessian-based orientation extraction.
Traces vessels by following their local orientation on vesselness maps.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fundus_preprocessor import FundusPreprocessor


class OrientationTracer:
    """
    Vessel tracer that follows local vessel orientation computed from 
    the Hessian matrix eigenvalues/eigenvectors.
    """
    
    def __init__(self, 
                 step_size: float = 1.0,
                 max_steps: int = 1000,
                 min_vesselness: float = 0.05,
                 orientation_weight: float = 0.7,
                 vesselness_weight: float = 0.3,
                 angle_tolerance: float = 45.0,
                 smoothing_sigma: float = 1.0):
        self.step_size = step_size
        self.max_steps = max_steps
        self.min_vesselness = min_vesselness
        self.orientation_weight = orientation_weight
        self.vesselness_weight = vesselness_weight
        self.angle_tolerance = np.radians(angle_tolerance)
        self.smoothing_sigma = smoothing_sigma
        self.preprocessor = FundusPreprocessor()
        
        self.vessel_cmap = LinearSegmentedColormap.from_list(
            'vessel', ['black', 'blue', 'cyan', 'yellow', 'red']
        )
    
    def compute_hessian_2d(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D Hessian matrix components.
        
        Returns:
            Hxx, Hxy, Hyy: Second derivatives
        """
        smoothed = gaussian_filter(image, self.smoothing_sigma)
        
        Ix = np.gradient(smoothed, axis=1)
        Iy = np.gradient(smoothed, axis=0)
        
        Hxx = np.gradient(Ix, axis=1)
        Hxy = np.gradient(Ix, axis=0)
        Hyy = np.gradient(Iy, axis=0)
        
        return Hxx, Hxy, Hyy
    
    def compute_orientation_field(self, vesselness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vessel orientation field from vesselness map.
        
        Returns:
            orientation: Angle in radians (-pi to pi)
            coherence: Orientation coherence/confidence (0-1)
        """
        Hxx, Hxy, Hyy = self.compute_hessian_2d(vesselness)
        
        h, w = vesselness.shape
        orientation = np.zeros((h, w))
        coherence = np.zeros((h, w))
        
        for i in range(h):
            for j in range(w):
                H = np.array([[Hxx[i, j], Hxy[i, j]],
                             [Hxy[i, j], Hyy[i, j]]])
                
                eigenvalues, eigenvectors = np.linalg.eigh(H)
                
                idx = np.argsort(np.abs(eigenvalues))
                lambda1, lambda2 = eigenvalues[idx]
                
                vessel_direction = eigenvectors[:, idx[1]]
                
                orientation[i, j] = np.arctan2(vessel_direction[1], vessel_direction[0])
                
                if np.abs(lambda2) > 1e-6:
                    coherence[i, j] = 1 - np.abs(lambda1) / (np.abs(lambda2) + 1e-6)
                else:
                    coherence[i, j] = 0
        
        return orientation, coherence
    
    def get_direction_vector(self, angle: float) -> np.ndarray:
        """Convert angle to unit direction vector."""
        return np.array([np.cos(angle), np.sin(angle)])
    
    def angle_difference(self, angle1: float, angle2: float) -> float:
        """Compute minimum angular difference between two angles."""
        diff = np.abs(angle1 - angle2)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        return min(diff, np.pi - diff)
    
    def _preprocess_if_rgb(self, image: np.ndarray) -> np.ndarray:
        """Preprocess RGB fundus image if needed, otherwise pass through."""
        if image.ndim == 3:
            _, _, clahe, mask = self.preprocessor.preprocess(image, return_intermediate=True)
            vesselness = clahe.astype(np.float32) / 255.0
            vesselness *= (mask > 0)
            return vesselness
        return image

    def trace(self, 
              vesselness: np.ndarray,
              start_point: Tuple[int, int],
              mask: Optional[np.ndarray] = None,
              visualize: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Trace a vessel from a starting point following local orientation.
        
        Args:
            vesselness: Vesselness map or RGB fundus image
            start_point: (y, x) starting coordinates
            mask: Optional binary mask
            visualize: If True, show trajectory evolution
            
        Returns:
            trajectory: Array of (y, x) coordinates
            info: Dictionary with tracing information
        """
        vesselness = self._preprocess_if_rgb(vesselness)

        orientation, coherence = self.compute_orientation_field(vesselness)
        
        h, w = vesselness.shape
        visited = np.zeros((h, w), dtype=bool)
        trajectory = [start_point]
        current = np.array(start_point, dtype=float)
        
        y0, x0 = int(current[0]), int(current[1])
        current_angle = orientation[y0, x0]
        
        vesselness_values = []
        coherence_values = []
        angles = []
        
        for step in range(self.max_steps):
            y, x = int(current[0]), int(current[1])
            
            if y < 0 or y >= h or x < 0 or x >= w:
                break
            if mask is not None and mask[y, x] == 0:
                break
            if vesselness[y, x] < self.min_vesselness:
                break
            
            visited[y, x] = True
            
            vesselness_values.append(vesselness[y, x])
            coherence_values.append(coherence[y, x])
            angles.append(current_angle)
            
            local_angle = orientation[y, x]
            local_coherence = coherence[y, x]
            
            if local_coherence > 0.3:
                angle_diff = self.angle_difference(current_angle, local_angle)
                
                if angle_diff > np.pi / 2:
                    local_angle += np.pi
                    if local_angle > np.pi:
                        local_angle -= 2 * np.pi
                
                current_angle = 0.3 * current_angle + 0.7 * local_angle
            
            direction = self.get_direction_vector(current_angle)
            next_pos = current + direction * self.step_size
            next_y, next_x = int(next_pos[0]), int(next_pos[1])
            
            if 0 <= next_y < h and 0 <= next_x < w:
                if visited[next_y, next_x]:
                    break
            
            current = next_pos
            trajectory.append((int(current[0]), int(current[1])))
        
        trajectory_array = np.array(trajectory)
        
        info = {
            'vesselness_values': np.array(vesselness_values),
            'coherence_values': np.array(coherence_values),
            'angles': np.array(angles),
            'orientation_field': orientation,
            'coherence_field': coherence
        }
        
        if visualize:
            self.visualize_trace(vesselness, trajectory_array, start_point, info)
        
        return trajectory_array, info
    
    def trace_bidirectional(self,
                           vesselness: np.ndarray,
                           start_point: Tuple[int, int],
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Trace in both directions from a seed point."""
        vesselness = self._preprocess_if_rgb(vesselness)

        traj_forward, _ = self.trace(vesselness, start_point, mask, visualize=False)
        
        orientation, coherence = self.compute_orientation_field(vesselness)
        orientation_flipped = orientation + np.pi
        
        self._original_orientation = orientation.copy()
        
        traj_backward, _ = self.trace(vesselness, start_point, mask, visualize=False)
        
        trajectory = np.vstack([traj_backward[::-1][:-1], traj_forward])
        
        return trajectory
    
    def trace_from_seeds(self,
                        vesselness: np.ndarray,
                        seeds: List[Tuple[int, int]],
                        mask: Optional[np.ndarray] = None,
                        bidirectional: bool = True,
                        visualize: bool = False) -> np.ndarray:
        """
        Trace from multiple seed points.
        
        Args:
            vesselness: Vesselness map or RGB fundus image
            seeds: List of (y, x) seed coordinates
            mask: Optional binary mask
            bidirectional: If True, trace in both directions
            visualize: If True, show all trajectories
            
        Returns:
            Binary skeleton map
        """
        vesselness = self._preprocess_if_rgb(vesselness)

        h, w = vesselness.shape
        skeleton = np.zeros((h, w), dtype=np.float32)
        trajectories = []
        
        for seed in seeds:
            if bidirectional:
                trajectory = self.trace_bidirectional(vesselness, seed, mask)
            else:
                trajectory, _ = self.trace(vesselness, seed, mask, visualize=False)
            
            trajectories.append(trajectory)
            
            for y, x in trajectory:
                if 0 <= y < h and 0 <= x < w:
                    skeleton[y, x] = 1.0
        
        if visualize:
            self.visualize_multiple_traces(vesselness, trajectories, seeds)
        
        return skeleton
    
    def visualize_orientation_field(self,
                                   vesselness: np.ndarray,
                                   save_path: Optional[str] = None):
        """Visualize the computed orientation field."""
        orientation, coherence = self.compute_orientation_field(vesselness)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im0 = axes[0].imshow(vesselness, cmap=self.vessel_cmap)
        axes[0].set_title('Vesselness Map', fontsize=14)
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        
        orientation_normalized = (orientation + np.pi) / (2 * np.pi)
        im1 = axes[1].imshow(orientation_normalized, cmap='hsv')
        axes[1].set_title('Orientation Field', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, label='Angle')
        
        im2 = axes[2].imshow(coherence, cmap='viridis')
        axes[2].set_title('Orientation Coherence', fontsize=14)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        step = max(vesselness.shape) // 20
        Y, X = np.mgrid[0:vesselness.shape[0]:step, 0:vesselness.shape[1]:step]
        U = np.cos(orientation[0::step, 0::step])
        V = np.sin(orientation[0::step, 0::step])
        C = coherence[0::step, 0::step]
        
        axes[0].quiver(X, Y, U, V, C, cmap='hot', alpha=0.6, 
                      scale=20, width=0.003, headwidth=3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_trace(self,
                       vesselness: np.ndarray,
                       trajectory: np.ndarray,
                       start_point: Tuple[int, int],
                       info: dict,
                       save_path: Optional[str] = None):
        """Visualize a single trajectory with orientation information."""
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(vesselness, cmap=self.vessel_cmap, alpha=0.7)
        
        orientation = info['orientation_field']
        coherence = info['coherence_field']
        step = max(vesselness.shape) // 25
        Y, X = np.mgrid[0:vesselness.shape[0]:step, 0:vesselness.shape[1]:step]
        U = np.cos(orientation[0::step, 0::step])
        V = np.sin(orientation[0::step, 0::step])
        C = coherence[0::step, 0::step]
        ax1.quiver(X, Y, U, V, C, cmap='hot', alpha=0.4, 
                  scale=20, width=0.003, headwidth=3)
        
        if len(trajectory) > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectory)))
            ax1.scatter(trajectory[:, 1], trajectory[:, 0],
                       c=colors, s=40, marker='o', edgecolors='white', linewidths=0.5)
            ax1.plot(trajectory[:, 1], trajectory[:, 0],
                    'w-', linewidth=2.5, alpha=0.7)
        ax1.scatter(start_point[1], start_point[0],
                   c='lime', s=300, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax1.set_title(f'Trajectory on Orientation Field ({len(trajectory)} steps)', fontsize=12)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if len(info['vesselness_values']) > 0:
            steps = range(len(info['vesselness_values']))
            ax2.plot(steps, info['vesselness_values'], 
                    linewidth=2, color='darkblue', label='Vesselness')
            ax2.axhline(y=np.mean(info['vesselness_values']), color='red',
                       linestyle='--', linewidth=1.5, 
                       label=f'Mean: {np.mean(info["vesselness_values"]):.3f}')
            ax2.axhline(y=self.min_vesselness, color='orange',
                       linestyle=':', linewidth=1.5, 
                       label=f'Threshold: {self.min_vesselness}')
            ax2.fill_between(steps, info['vesselness_values'],
                            alpha=0.3, color='lightblue')
            ax2.set_xlabel('Step', fontsize=11)
            ax2.set_ylabel('Vesselness', fontsize=11)
            ax2.set_title('Vesselness Along Path', fontsize=12)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        if len(info['coherence_values']) > 0:
            steps = range(len(info['coherence_values']))
            ax3.plot(steps, info['coherence_values'],
                    linewidth=2, color='darkgreen', label='Coherence')
            ax3.axhline(y=np.mean(info['coherence_values']), color='red',
                       linestyle='--', linewidth=1.5,
                       label=f'Mean: {np.mean(info["coherence_values"]):.3f}')
            ax3.fill_between(steps, info['coherence_values'],
                            alpha=0.3, color='lightgreen')
            ax3.set_xlabel('Step', fontsize=11)
            ax3.set_ylabel('Coherence', fontsize=11)
            ax3.set_title('Orientation Coherence Along Path', fontsize=12)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, :])
        orientation_normalized = (orientation + np.pi) / (2 * np.pi)
        im = ax4.imshow(orientation_normalized, cmap='hsv', alpha=0.6)
        ax4.imshow(vesselness, cmap='gray', alpha=0.4)
        
        if len(trajectory) > 0:
            ax4.plot(trajectory[:, 1], trajectory[:, 0],
                    'w-', linewidth=3, alpha=0.9)
            ax4.scatter(trajectory[:, 1], trajectory[:, 0],
                       c='yellow', s=30, marker='o', edgecolors='black', linewidths=0.5)
        ax4.scatter(start_point[1], start_point[0],
                   c='lime', s=300, marker='*', edgecolors='black', linewidths=2, zorder=10)
        ax4.set_title('Trajectory on Orientation Map', fontsize=12)
        ax4.axis('off')
        plt.colorbar(im, ax=ax4, fraction=0.046, label='Orientation (radians)')
        
        plt.suptitle('Orientation-Following Vessel Tracing', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_multiple_traces(self,
                                  vesselness: np.ndarray,
                                  trajectories: List[np.ndarray],
                                  seeds: List[Tuple[int, int]],
                                  save_path: Optional[str] = None):
        """Visualize multiple trajectories with orientation field."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(vesselness, cmap=self.vessel_cmap, alpha=0.7)
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(trajectories), 10)))
        
        for i, (traj, seed) in enumerate(zip(trajectories, seeds)):
            if len(traj) > 0:
                color = colors[i % 10]
                ax1.plot(traj[:, 1], traj[:, 0],
                        color=color, linewidth=2.5, alpha=0.8,
                        label=f'Trace {i+1} ({len(traj)} pts)')
                ax1.scatter(seed[1], seed[0],
                           c=[color], s=200, marker='*',
                           edgecolors='white', linewidths=2, zorder=10)
        
        ax1.set_title(f'{len(trajectories)} Trajectories on Vesselness', fontsize=14)
        ax1.legend(loc='upper right', fontsize=9, ncol=2)
        ax1.axis('off')
        
        orientation, coherence = self.compute_orientation_field(vesselness)
        orientation_normalized = (orientation + np.pi) / (2 * np.pi)
        ax2.imshow(orientation_normalized, cmap='hsv', alpha=0.6)
        ax2.imshow(vesselness, cmap='gray', alpha=0.3)
        
        for i, (traj, seed) in enumerate(zip(trajectories, seeds)):
            if len(traj) > 0:
                color = colors[i % 10]
                ax2.plot(traj[:, 1], traj[:, 0],
                        color=color, linewidth=2.5, alpha=0.8)
                ax2.scatter(seed[1], seed[0],
                           c=[color], s=200, marker='*',
                           edgecolors='white', linewidths=2, zorder=10)
        
        ax2.set_title(f'{len(trajectories)} Trajectories on Orientation Field', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()