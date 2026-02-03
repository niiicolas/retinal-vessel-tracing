#!/usr/bin/env python
# scripts/demo.py
"""
Interactive demo for vessel tracing.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import RetinalImagePreprocessor
from data.centerline_extraction import CenterlineExtractor
from environment.vessel_env import VesselTracingEnv
from models.policy_network import ActorCriticNetwork
from models.seed_detector import SeedDetector
from evaluation.visualization import TracingVisualizer


class VesselTracingDemo:
    """Interactive vessel tracing demo."""
    
    def __init__(self, policy_path: str, config_path: str, 
                 seed_detector_path: str = None, device: str = 'cuda'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.policy = ActorCriticNetwork(self.config).to(self.device)
        checkpoint = torch.load(policy_path, map_location=self.device)
        if 'policy_state_dict' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            self.policy.load_state_dict(checkpoint)
        self.policy.eval()
        
        self.seed_detector = None
        if seed_detector_path:
            self.seed_detector = SeedDetector(self.config).to(self.device)
            self.seed_detector.load_state_dict(
                torch.load(seed_detector_path, map_location=self.device)
            )
            self.seed_detector.eval()
        
        # Initialize components
        self.preprocessor = RetinalImagePreprocessor(self.config)
        self.centerline_extractor = CenterlineExtractor()
        self.env = VesselTracingEnv(self.config)
        self.visualizer = TracingVisualizer()
        
    def process_image(self, image_path: str, output_path: str = None,
                      max_traces: int = 20):
        """
        Process a single image and trace vessels.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save output
            max_traces: Maximum number of traces to run
            
        Returns:
            result: Dictionary with skeleton and metrics
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocessor.normalize_image(image)
        
        h, w = image.shape[:2]
        
        # Compute vesselness for better tracing
        vesselness = self.preprocessor.compute_vesselness(image)
        
        # Create FOV mask
        fov_mask = self.preprocessor.create_fov_mask(image)
        
        # Create dummy centerline and distance transform for env
        # (In real use, these would come from predictions or not be needed)
        dummy_centerline = np.zeros((h, w), dtype=np.float32)
        dummy_dt = np.ones((h, w), dtype=np.float32) * self.config['environment']['tolerance']
        
        self.env.set_data(
            image=image,
            centerline=dummy_centerline,
            distance_transform=dummy_dt,
            vesselness=vesselness,
            fov_mask=fov_mask
        )
        
        # Get seeds
        seeds = []
        if self.seed_detector:
            image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            detected_seeds, heatmap = self.seed_detector.detect_seeds(image_tensor, return_heatmap=True)
            seeds = detected_seeds[0]
        
        if not seeds:
            # Use grid of starting points
            step = 100
            for y in range(50, h - 50, step):
                for x in range(50, w - 50, step):
                    if fov_mask[y, x] > 0 and vesselness[y, x] > 0.1:
                        seeds.append((y, x, vesselness[y, x]))
        
        # Trace from seeds
        all_trajectories = []
        predicted_skeleton = np.zeros((h, w), dtype=np.float32)
        
        for seed_idx, (y, x, conf) in enumerate(seeds[:max_traces]):
            obs, _ = self.env.reset(start_position=(y, x))
            
            trajectory = [tuple(self.env.position)]
            hidden = None
            done = False
            
            with torch.no_grad():
                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    obs_tensor = obs_tensor.unsqueeze(0)
                    
                    action, _, _, _, hidden = self.policy.get_action_and_value(obs_tensor, hidden)
                    action = action.item()
                    
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    trajectory.append(tuple(self.env.position))
            
            all_trajectories.append(trajectory)
            
            # Update skeleton
            for py, px in trajectory:
                if 0 <= py < h and 0 <= px < w:
                    predicted_skeleton[py, px] = 1.0
        
        # Create output visualization
        vis_image = (image * 255).astype(np.uint8).copy()
        
        # Draw skeleton in red
        vis_image[predicted_skeleton > 0] = [255, 0, 0]
        
        # Draw trajectories with different colors
        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255], [0, 255, 255]
        ]
        for i, trajectory in enumerate(all_trajectories):
            color = colors[i % len(colors)]
            for py, px in trajectory:
                cv2.circle(vis_image, (px, py), 1, color, -1)
        
        # Save output
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"Output saved to: {output_path}")
        
        result = {
            'skeleton': predicted_skeleton,
            'trajectories': all_trajectories,
            'num_traces': len(all_trajectories),
            'skeleton_pixels': int(predicted_skeleton.sum()),
            'visualization': vis_image
        }
        
        return result
    
    def run_interactive(self, image_path: str):
        """Run interactive demo with mouse-based seed selection."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.preprocessor.normalize_image(image)
        
        h, w = image.shape[:2]
        
        # Prepare environment
        vesselness = self.preprocessor.compute_vesselness(image)
        fov_mask = self.preprocessor.create_fov_mask(image)
        dummy_centerline = np.zeros((h, w), dtype=np.float32)
        dummy_dt = np.ones((h, w), dtype=np.float32) * 2.0
        
        self.env.set_data(
            image=image,
            centerline=dummy_centerline,
            distance_transform=dummy_dt,
            vesselness=vesselness,
            fov_mask=fov_mask
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Display image
        axes[0].imshow(image)
        axes[0].set_title('Click to place seed points')
        axes[0].axis('off')
        
        # Result display
        result_display = axes[1].imshow(image)
        axes[1].set_title('Traced vessels')
        axes[1].axis('off')
        
        # State
        seeds = []
        trajectories = []
        skeleton = np.zeros((h, w), dtype=np.float32)
        
        def onclick(event):
            if event.inaxes != axes[0]:
                return
            
            x, y = int(event.xdata), int(event.ydata)
            
            if 0 <= x < w and 0 <= y < h:
                seeds.append((y, x))
                axes[0].plot(x, y, 'go', markersize=10)
                fig.canvas.draw()
                
                # Trace from this seed
                trace_from_seed(y, x)
        
        def trace_from_seed(y, x):
            nonlocal skeleton
            
            obs, _ = self.env.reset(start_position=(y, x))
            trajectory = [tuple(self.env.position)]
            hidden = None
            done = False
            
            with torch.no_grad():
                while not done:
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    obs_tensor = obs_tensor.unsqueeze(0)
                    
                    action, _, _, _, hidden = self.policy.get_action_and_value(obs_tensor, hidden)
                    action = action.item()
                    
                    obs, _, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    trajectory.append(tuple(self.env.position))
            
            trajectories.append(trajectory)
            
            # Update skeleton
            for py, px in trajectory:
                if 0 <= py < h and 0 <= px < w:
                    skeleton[py, px] = 1.0
            
            # Update display
            vis = image.copy()
            vis[skeleton > 0] = [1, 0, 0]
            result_display.set_data(vis)
            axes[1].set_title(f'Traced vessels ({len(trajectories)} traces)')
            fig.canvas.draw()
        
        def clear_traces(event):
            nonlocal seeds, trajectories, skeleton
            seeds = []
            trajectories = []
            skeleton = np.zeros((h, w), dtype=np.float32)
            
            axes[0].clear()
            axes[0].imshow(image)
            axes[0].set_title('Click to place seed points')
            axes[0].axis('off')
            
            result_display.set_data(image)
            axes[1].set_title('Traced vessels')
            fig.canvas.draw()
        
        # Add clear button
        ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
        btn_clear = Button(ax_button, 'Clear')
        btn_clear.on_clicked(clear_traces)
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Vessel tracing demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to policy checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed-detector', type=str, default=None,
                        help='Path to seed detector checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    demo = VesselTracingDemo(
        policy_path=args.checkpoint,
        config_path=args.config,
        seed_detector_path=args.seed_detector,
        device=args.device
    )
    
    if args.interactive:
        demo.run_interactive(args.image)
    else:
        result = demo.process_image(
            image_path=args.image,
            output_path=args.output or 'output.png'
        )
        
        print(f"\nResults:")
        print(f"  Number of traces: {result['num_traces']}")
        print(f"  Skeleton pixels: {result['skeleton_pixels']}")


if __name__ == '__main__':
    main()
