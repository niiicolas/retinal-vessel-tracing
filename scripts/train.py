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
    print("\n=== Training Seed Detector ===\n")
    
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
    
    for epoch in range(config.get('seed_detector', {}).get('epochs', 30)):
        epoch_losses = []
        
        for batch in train_loader:
            images = batch['image'].to(device)
            centerlines = batch['centerline'].to(device)
            masks = batch['fov_mask'].to(device)
            
            loss = trainer.train_step(images, centerlines, masks)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(seed_detector.state_dict(), output_dir / 'seed_detector_best.pth')
    
    return seed_detector


def generate_expert_traces(config: dict, dataset, device: torch.device):
    """Generate expert traces for imitation learning."""
    print("\n=== Generating Expert Traces ===\n")
    
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
    
    print(f"Total traces generated: {len(all_traces)}")
    return all_traces


def train_with_imitation(config: dict, policy: ActorCriticNetwork, 
                         traces: list, device: torch.device):
    """Warm-start policy with imitation learning."""
    print("\n=== Imitation Learning Warm-Start ===\n")
    
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
    print("\n=== PPO Training ===\n")
    
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
    parser.add_argument('--output', type
