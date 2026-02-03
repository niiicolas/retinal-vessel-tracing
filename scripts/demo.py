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
                    
                    obs, _, 
