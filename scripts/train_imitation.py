"""Imitation-learning (behavior cloning) entry point — run before train_ppo.py.

Loads data via the unified dataloader, generates expert traces, and behavior-clones the
policy; core logic lives in training/imitation.py. Supports feedforward and LSTM modes.
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import DEVICE
from config import IMITATION_WEIGHTS_PATH as SAVE_PATH
from config import IMITATION_LOG_PATH
from config import MODEL_CONFIG as CONFIG
from config import OBS_SIZE, TOLERANCE

_imi = CONFIG['training']['imitation']
LEARNING_RATE = _imi['lr']
BATCH_SIZE = _imi['batch_size']
LSTM_BATCH_SIZE = _imi['lstm_batch_size']
NUM_EPOCHS = _imi['num_epochs']
USE_AUGMENT = _imi['use_augment']
from data.centerline_extraction import CenterlineExtractor
from data.dataloader import get_data
from models.policy_network import ActorCriticNetwork
from training.imitation import ImitationDataset, ImitationTrainer, augment_sample, generate_expert_metadata, generate_expert_sequence_metadata

USE_LSTM = CONFIG['policy']['use_lstm']

process = psutil.Process()


def _extract_single_sample(args):
    """Build one imitation sample (image, GT maps, expert traces) for parallel extraction.

    Expert traces are walked on the GT skeleton (the supervision signal); predicted priors
    are carried alongside so the observation pipeline gets non-leaking geometry without
    re-running the UNet.
    """
    raw_sample, extractor_kwargs = args
    extractor = CenterlineExtractor(**extractor_kwargs)
    centerline = raw_sample['centerline'].squeeze(0).numpy()
    out = {
        'image': raw_sample['image'].permute(1, 2, 0).numpy(),
        'centerline': centerline,
        'distance_transform': raw_sample['distance_transform'].squeeze(0).numpy(),
        'fov_mask': raw_sample['fov_mask'].squeeze(0).numpy(),
        'expert_traces': extractor.generate_expert_traces(centerline),
    }
    if 'pred_centerline' in raw_sample:
        out['pred_centerline'] = raw_sample['pred_centerline'].squeeze(0).numpy()
    if 'pred_distance_transform' in raw_sample:
        out['pred_distance_transform'] = raw_sample['pred_distance_transform'].squeeze(0).numpy()
    if 'pred_dt_gradient' in raw_sample:
        out['pred_dt_gradient'] = raw_sample['pred_dt_gradient'].numpy()
    if 'vessel_orientation' in raw_sample:
        out['vessel_orientation'] = raw_sample['vessel_orientation'].numpy()
    return out


def load_training_samples():
    """Load combined training samples and generate their expert traces in parallel."""
    ds, _ = get_data('rl_agent', 'train', tolerance=TOLERANCE)

    extractor_kwargs = {'min_branch_length': 10, 'prune_iterations': 5}
    raw_samples = [ds[i] for i in range(len(ds))]

    n_workers = min(4, len(raw_samples))
    print(f'Extracting expert traces from {len(raw_samples)} images ({n_workers} workers)...')

    with Pool(n_workers) as pool:
        samples = pool.map(_extract_single_sample, [(s, extractor_kwargs) for s in raw_samples])

    print(f'Loaded {len(samples)} training samples. Memory: {process.memory_info().rss / 1e9:.1f} GB\n')
    return samples


def main():
    """Generate expert-trace metadata (and LSTM sequences), split it, and behavior-clone the policy.

    Writes imitation-pretrained weights to IMITATION_WEIGHTS_PATH for PPO to warm-start from.
    """
    print(f'Device: {DEVICE}')
    print(f'LSTM:   {"ON" if USE_LSTM else "OFF"}')
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print('\nLoading combined training samples...')
    all_samples = load_training_samples()

    all_samples = all_samples  # kept mutable so augmented samples can be appended in place
    all_metadata = []
    all_sequences = []

    # Walk the skeleton at the env's per-action stride so imitation→PPO action semantics match.
    STEP_SIZE = int(CONFIG['environment'].get('step_size', 1))
    # Invert the env's tangent rotation at expert-gen time so action labels match what the env
    # executes; a separate flag from environment.tangent_relative_actions for mismatch experiments.
    TANGENT_AWARE = bool(CONFIG.get('training', {}).get('imitation', {}).get('tangent_aware', True))

    for sample_idx, sample in enumerate(all_samples):
        meta = generate_expert_metadata(sample, sample_idx, OBS_SIZE, STEP_SIZE, tangent_aware=TANGENT_AWARE)
        all_metadata.extend(meta)

        if USE_LSTM:
            seqs = generate_expert_sequence_metadata(sample, sample_idx, OBS_SIZE, STEP_SIZE, tangent_aware=TANGENT_AWARE)
            all_sequences.extend(seqs)

        n_info = f'{len(meta)} steps'
        if USE_LSTM:
            n_info += f', {len(seqs)} sequences'
        print(f'  [{sample_idx}] -> {n_info}')

        if USE_AUGMENT:
            for aug in augment_sample(sample, TOLERANCE):
                all_samples.append(aug)
                aug_idx = len(all_samples) - 1
                aug_meta = generate_expert_metadata(aug, aug_idx, OBS_SIZE, STEP_SIZE, tangent_aware=TANGENT_AWARE)
                all_metadata.extend(aug_meta)
                if USE_LSTM:
                    aug_seqs = generate_expert_sequence_metadata(aug, aug_idx, OBS_SIZE, STEP_SIZE, tangent_aware=TANGENT_AWARE)
                    all_sequences.extend(aug_seqs)

    print(f'\nTotal samples (incl. augmented): {len(all_samples)}')
    print(f'Total step metadata: {len(all_metadata)}')

    train_sequences = None
    val_sequences = None

    if USE_LSTM:
        print(f'Total sequences: {len(all_sequences)}  (avg length {np.mean([s["length"] for s in all_sequences]):.1f})')

        indices = np.random.permutation(len(all_sequences))
        split = int(len(all_sequences) * 0.9)
        train_sequences = [all_sequences[i] for i in indices[:split]]
        val_sequences = [all_sequences[i] for i in indices[split:]]
        print(f'Train: {len(train_sequences)} seqs  |  Val: {len(val_sequences)} seqs')

    if not all_metadata:
        print('ERROR: No metadata generated. Check data paths.')
        return

    # Deterministic-by-seed 90/10 split of feedforward step metadata.
    indices = np.random.permutation(len(all_metadata))
    split = int(len(all_metadata) * 0.9)
    train_meta = [all_metadata[i] for i in indices[:split]]
    val_meta = [all_metadata[i] for i in indices[split:]]

    print(f'FF split: {len(train_meta)} train  |  {len(val_meta)} val')

    train_ds = ImitationDataset(all_samples, train_meta, CONFIG)
    val_ds = ImitationDataset(all_samples, val_meta, CONFIG)

    # Reuse the FF dataset's precomputed maps for the LSTM path instead of recomputing them.
    stacked_sources = train_ds.stacked_sources if USE_LSTM else None
    vesselness_maps = train_ds.vesselness_maps if USE_LSTM else None
    unet_priors = train_ds.unet_priors if USE_LSTM else None

    model = ActorCriticNetwork(CONFIG).to(DEVICE)

    trainer = ImitationTrainer(model, DEVICE, CONFIG, lr=LEARNING_RATE, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, lstm_batch_size=LSTM_BATCH_SIZE)
    trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        save_path=SAVE_PATH,
        config=CONFIG,
        log_path=IMITATION_LOG_PATH,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        samples=all_samples if USE_LSTM else None,
        stacked_sources=stacked_sources,
        vesselness_maps=vesselness_maps,
        unet_priors=unet_priors,
    )


if __name__ == '__main__':
    main()
