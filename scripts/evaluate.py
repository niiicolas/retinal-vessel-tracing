#!/usr/bin/env python
# scripts/evaluate.py
"""
Evaluation script for trained vessel tracing model.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import RetinalVesselDataset
from environment.vessel_env import VesselTracingEnv
from models.policy_network import ActorCriticNetwork
from models.seed_detector import SeedDetector
from evaluation.metrics import EvaluationRunner, CenterlineMetrics
from evaluation.visualization import TracingVisualizer
from tqdm import tqdm


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load trained model from checkpoint."""
    policy = ActorCriticNetwork(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'policy_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['policy_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    return policy


def trace_image(policy, env, seed_detector, image_data: dict, 
                device: torch.device, config: dict):
    """
    Trace vessels in a single image.
    
    Returns:
        trajectory: List of (y, x) positions
        actions: List of actions taken
        info: Final episode info
    """
    # Set up environment
    env.set_data(
        image=image_data['image'].permute(1, 2, 0).numpy(),
        centerline=image_data['centerline'].squeeze().numpy(),
        distance_transform=image_data['distance_transform'].squeeze().numpy(),
        fov_mask=image_data['fov_mask'].squeeze().numpy()
    )
    
    # Get seeds
    if seed_detector is not None:
        image_tensor = image_data['image'].unsqueeze(0).to(device)
        seeds, _ = seed_detector.detect_seeds(image_tensor)
        seeds = seeds[0]  # First (only) image
    else:
        # Use random start on centerline
        seeds = None
    
    all_trajectories = []
    all_actions = []
    
    # Trace from each seed (or just reset for random start)
    frontier = []
    
    if seeds:
        for y, x, conf in seeds[:config.get('seed_detector', {}).get('top_k_seeds', 10)]:
            frontier.append((y, x))
    
    visited_global = np.zeros_like(image_data['centerline'].squeeze().numpy())
    
    # Main tracing loop with frontier
    while frontier or not all_trajectories:
        if frontier:
            start_pos = frontier.pop()
            obs, _ = env.reset(start_position=start_pos)
        else:
            obs, _ = env.reset()
        
        trajectory = [tuple(env.position)]
        actions = []
        done = False
        hidden = None
        
        with torch.no_grad():
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                if len(obs_tensor.shape) == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                action, _, _, _, hidden = policy.get_action_and_value(obs_tensor, hidden)
                action = action.item()
                
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                trajectory.append(tuple(env.position))
                actions.append(action)
                
                # Check for junction - add unexplored branches to frontier
                # (Simplified - in full implementation would detect junctions)
        
        all_trajectories.extend(trajectory)
        all_actions.extend(actions)
        
        # Update global visited mask
        for y, x in trajectory:
            if 0 <= y < visited_global.shape[0] and 0 <= x < visited_global.shape[1]:
                visited_global[y, x] = 1
        
        # Break after first trajectory if no seeds
        if not seeds:
            break
    
    return all_trajectories, all_actions, info, visited_global


def evaluate_dataset(policy, seed_detector, dataset, config: dict, 
                     device: torch.device, output_dir: Path, 
                     num_samples: int = None, visualize: bool = True):
    """Evaluate on entire dataset."""
    
    env = VesselTracingEnv(config)
    metrics_calculator = CenterlineMetrics(
        tolerance_levels=config.get('evaluation', {}).get('tolerance_levels', [1, 2, 3])
    )
    visualizer = TracingVisualizer()
    
    if num_samples is None:
        num_samples = len(dataset)
    
    all_metrics = []
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    for i in tqdm(range(min(num_samples, len(dataset))), desc="Evaluating"):
        sample = dataset[i]
        
        # Trace
        trajectory, actions, info, pred_skeleton = trace_image(
            policy, env, seed_detector, sample, device, config
        )
        
        # Compute metrics
        gt_skeleton = sample['centerline'].squeeze().numpy()
        gt_vessel_mask = sample['vessel_mask'].squeeze().numpy() if 'vessel_mask' in sample else None
        
        sample_metrics = metrics_calculator.compute_all_metrics(
            pred_skeleton > 0, gt_skeleton > 0, gt_vessel_mask
        )
        
        # Add path efficiency
        path_metrics = metrics_calculator.path_efficiency(trajectory, gt_skeleton > 0)
        sample_metrics.update(path_metrics)
        sample_metrics['sample_id'] = sample.get('id', str(i))
        
        all_metrics.append(sample_metrics)
        
        # Visualize (first few samples)
        if visualize and i < 10:
            image = sample['image'].permute(1, 2, 0).numpy()
            fig = visualizer.visualize_episode(
                image=image,
                gt_centerline=gt_skeleton,
                trajectory=trajectory,
                actions=actions,
                title=f"Sample {sample_metrics['sample_id']} - F1@2px: {sample_metrics.get('f1@2px', 0):.3f}"
            )
            fig.savefig(vis_dir / f'sample_{i:03d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    # Aggregate metrics
    aggregated = {}
    for key in all_metrics[0].keys():
        if key == 'sample_id':
            continue
        values = [m[key] for m in all_metrics]
        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
        aggregated[f'{key}_min'] = float(np.min(values))
        aggregated[f'{key}_max'] = float(np.max(values))
    
    return aggregated, all_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate vessel tracing model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--dataset-name', type=str, default='DRIVE',
                        help='Dataset name (DRIVE, STARE, etc.)')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split (train/val/test)')
    parser.add_argument('--output', type=str, default='eval_results',
                        help='Output directory')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed-detector', type=str, default=None,
                        help='Path to seed detector checkpoint')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    policy = load_model(args.checkpoint, config, device)
    
    # Load seed detector if provided
    seed_detector = None
    if args.seed_detector:
        print(f"Loading seed detector from {args.seed_detector}")
        seed_detector = SeedDetector(config).to(device)
        seed_detector.load_state_dict(torch.load(args.seed_detector, map_location=device))
        seed_detector.eval()
    
    # Load dataset
    print(f"Loading {args.dataset_name} dataset from {args.data}")
    dataset = RetinalVesselDataset(
        root_dir=args.data,
        dataset_name=args.dataset_name,
        split=args.split,
        config=config,
        transform=False
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    aggregated_metrics, all_metrics = evaluate_dataset(
        policy=policy,
        seed_detector=seed_detector,
        dataset=dataset,
        config=config,
        device=device,
        output_dir=output_dir,
        num_samples=args.num_samples,
        visualize=not args.no_visualize
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nDataset: {args.dataset_name} ({args.split})")
    print(f"Samples evaluated: {len(all_metrics)}")
    print()
    
    # Group metrics by category
    print("Centerline Metrics:")
    print("-" * 40)
    for tau in config.get('evaluation', {}).get('tolerance_levels', [1, 2, 3]):
        p = aggregated_metrics.get(f'precision@{tau}px_mean', 0)
        r = aggregated_metrics.get(f'recall@{tau}px_mean', 0)
        f1 = aggregated_metrics.get(f'f1@{tau}px_mean', 0)
        print(f"  @{tau}px: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    
    if 'cldice_mean' in aggregated_metrics:
        print(f"  clDice: {aggregated_metrics['cldice_mean']:.4f}")
    
    print("\nTopology 
