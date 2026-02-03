#!/usr/bin/env python
# scripts/train.py
"""
Main training script for vessel tracing RL agent.
"""

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
import sys
import os
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import RetinalVesselDataset, TileDataset
from data.preprocessing import RetinalImagePreprocessor
from environment.vessel_env import VesselTracingEnv
from models.policy_network import ActorCriticNetwork
from models.seed_detector import SeedDetector, SeedDetectorTrainer
from training.ppo import PPOTrainer
from training.imitation import ExpertTraceGenerator, ExpertTraceDataset, ImitationLearner
from training.curriculum import CurriculumManager
from evaluation.metrics import EvaluationRunner
from evaluation.visualization import TracingVisualizer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_seed_detector(config: dict, device: torch.device, output_dir: Path):
    """Train the seed detection network."""
    print("\n" + "="*50)
    print("Training Seed Detector")
    print("="*50 + "\n")
    
    # Load dataset
    dataset_config = config['data']['datasets'][0]
    dataset = RetinalVesselDataset(
        root_dir=dataset_config['path'],
        dataset_name=dataset_config['name'],
        split='train',
        config=config,
        transform=True
    )
    
    val_dataset = RetinalVesselDataset(
        root_dir=dataset_config['path'],
        dataset_name=dataset_config['name'],
        split='val',
        config=config,
        transform=False
    )
    
    # Create model
    seed_detector = SeedDetector(config).to(device)
    trainer = SeedDetectorTrainer(seed_detector, config, device)
    
    # Training loop
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4
    )
    
    best_loss = float('inf')
    num_epochs = config.get('seed_detector', {}).get('epochs', 30)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch in train_loader:
            images = batch['image'].to(device)
            centerlines = batch['centerline'].to(device)
            masks = batch['fov_mask'].to(device)
            
            loss = trainer.train_step(images, centerlines, masks)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}: loss = {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(seed_detector.state_dict(), output_dir / 'seed_detector_best.pth')
            print(f"  -> New best model saved!")
    
    print(f"\nSeed detector training complete. Best loss: {best_loss:.4f}")
    return seed_detector


def generate_expert_traces(config: dict, dataset, device: torch.device):
    """Generate expert traces for imitation learning."""
    print("\n" + "="*50)
    print("Generating Expert Traces")
    print("="*50 + "\n")
    
    generator = ExpertTraceGenerator(config)
    
    all_traces = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        image = sample['image'].permute(1, 2, 0).numpy()
        centerline = sample['centerline'].squeeze().numpy()
        
        traces = generator.generate_traces(
            image=image,
            centerline=centerline,
            max_traces=50
        )
        
        all_traces.extend(traces)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataset)} images, {len(all_traces)} traces")
    
    print(f"\nTotal traces generated: {len(all_traces)}")
    return all_traces


def train_with_imitation(config: dict, policy: ActorCriticNetwork, 
                         traces: list, device: torch.device):
    """Warm-start policy with imitation learning."""
    print("\n" + "="*50)
    print("Imitation Learning Warm-Start")
    print("="*50 + "\n")
    
    # Split traces
    n_train = int(0.9 * len(traces))
    train_traces = traces[:n_train]
    val_traces = traces[n_train:]
    
    train_dataset = ExpertTraceDataset(train_traces, config)
    val_dataset = ExpertTraceDataset(val_traces, config) if val_traces else None
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    learner = ImitationLearner(policy, config, device)
    history = learner.train(train_dataset, val_dataset)
    
    return history


def train_ppo(config: dict, policy: ActorCriticNetwork, 
              dataset, device: torch.device, output_dir: Path):
    """Train policy with PPO."""
    print("\n" + "="*50)
    print("PPO Training")
    print("="*50 + "\n")
    
    # Create environment
    env = VesselTracingEnv(config)
    
    # Initialize with first sample
    sample = dataset[0]
    env.set_data(
        image=sample['image'].permute(1, 2, 0).numpy(),
        centerline=sample['centerline'].squeeze().numpy(),
        distance_transform=sample['distance_transform'].squeeze().numpy(),
        fov_mask=sample['fov_mask'].squeeze().numpy()
    )
    
    # Create trainer
    trainer = PPOTrainer(policy, config, device)
    
    # Curriculum manager
    curriculum = CurriculumManager(config)
    
    # Callback for checkpointing
    best_coverage = 0
    
    def checkpoint_callback(update, metrics, policy):
        nonlocal best_coverage
        
        if update % 100 == 0 and len(trainer.coverage_ratios) > 0:
            avg_coverage = np.mean(trainer.coverage_ratios)
            
            if avg_coverage > best_coverage:
                best_coverage = avg_coverage
                trainer.save_checkpoint(
                    str(output_dir / 'policy_best.pth'),
                    {'coverage': best_coverage, 'update': update}
                )
                print(f"\nNew best coverage: {best_coverage:.2%}")
            
            # Update curriculum
            curriculum.step(success=avg_coverage > 0.5)
            
            # Sample new training data
            sample_idx = np.random.randint(len(dataset))
            sample = dataset[sample_idx]
            env.set_data(
                image=sample['image'].permute(1, 2, 0).numpy(),
                centerline=sample['centerline'].squeeze().numpy(),
                distance_transform=sample['distance_transform'].squeeze().numpy(),
                fov_mask=sample['fov_mask'].squeeze().numpy()
            )
    
    # Train
    history = trainer.train(env, callback=checkpoint_callback)
    
    # Save final checkpoint
    trainer.save_checkpoint(str(output_dir / 'policy_final.pth'))
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train vessel tracing RL agent')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-seed-detector', action='store_true',
                        help='Skip seed detector training')
    parser.add_argument('--skip-imitation', action='store_true',
                        help='Skip imitation learning')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable W&B logging')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Initialize W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.get('logging', {}).get('project_name', 'vessel-tracing'),
            config=config,
            name=f'run_{timestamp}'
        )
        config['logging']['use_wandb'] = True
    
    # Load dataset
    print("\n" + "="*50)
    print("Loading Dataset")
    print("="*50 + "\n")
    
    dataset_config = config['data']['datasets'][0]
    train_dataset = RetinalVesselDataset(
        root_dir=dataset_config['path'],
        dataset_name=dataset_config['name'],
        split='train',
        config=config,
        transform=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Use tile dataset for training
    tile_dataset = TileDataset(
        train_dataset,
        tile_size=config['data']['tile_size'],
        tiles_per_image=10
    )
    
    print(f"Total tiles: {len(tile_dataset)}")
    
    # Step 1: Train seed detector (optional)
    seed_detector = None
    if not args.skip_seed_detector:
        seed_detector = train_seed_detector(config, device, output_dir)
    
    # Step 2: Create policy network
    print("\n" + "="*50)
    print("Creating Policy Network")
    print("="*50 + "\n")
    
    policy = ActorCriticNetwork(config).to(device)
    
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
    
    # Step 3: Imitation learning warm-start (optional)
    if not args.skip_imitation and not args.resume:
        traces = generate_expert_traces(config, train_dataset, device)
        if traces:
            imitation_history = train_with_imitation(config, policy, traces, device)
            
            # Save imitation checkpoint
            torch.save(policy.state_dict(), output_dir / 'policy_after_imitation.pth')
            
            # Visualize imitation learning
            visualizer = TracingVisualizer()
            fig = visualizer.plot_training_history(imitation_history)
            fig.savefig(output_dir / 'imitation_history.png', dpi=150)
    
    # Step 4: PPO training
    ppo_history = train_ppo(config, policy, tile_dataset, device, output_dir)
    
    # Save training history visualization
    visualizer = TracingVisualizer()
    fig = visualizer.plot_training_history(ppo_history, str(output_dir / 'ppo_history.png'))
    
    # Step 5: Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50 + "\n")
    
    val_dataset = RetinalVesselDataset(
        root_dir=dataset_config['path'],
        dataset_name=dataset_config['name'],
        split='val',
        config=config,
        transform=False
    )
    
    evaluator = EvaluationRunner(config)
    env = VesselTracingEnv(config)
    
    metrics = evaluator.evaluate_model(policy, env, val_dataset, num_samples=10, device=device)
    
    print("\nEvaluation Results:")
    print("-" * 40)
    for key, value in sorted(metrics.items()):
        if 'mean' in key:
            print(f"{key}: {value:.4f}")
    
    # Save metrics
    import json
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    
    # Close W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
