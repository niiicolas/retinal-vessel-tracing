"""scripts/train_seed_detector.py
=================================
Entry script for the SOTA seed detector.

Loads the combined TRAIN_DATASETS = (FIVES, STARE, CHASEDB1, HRF, LES-AV)
via the unified dataloader, instantiates the Attention U-Net multi-task
SeedDetector, and trains it.

Run via SLURM (see run.sh):
    python -u -m scripts.train_seed_detector

The ``-u`` flag forces unbuffered stdout; without it, a job redirected by
SLURM to a file shows no progress in the log until the process exits, so
a job that hits the wall-clock limit looks silent in retrospect.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force line-buffered stdout/stderr even when SLURM redirects to a file —
# defence in depth in case run.sh forgot to pass ``python -u``.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from config import (
    DEVICE,
    SEED_WEIGHTS_PATH as SAVE_PATH,
    TOLERANCE,
    SEED_CONFIG,
)
from data.dataloader import get_data
from models.seed_detector import (
    SeedDetector,
    frangi_vesselness,
)
from training.seed_detector_trainer import (
    SeedDetectorTrainer,
)


def load_samples(split: str, use_frangi: bool = True) -> List[Dict[str, Any]]:
    """Load combined-dataset samples in numpy form for the trainer.

    If ``use_frangi`` is True, also precompute the multi-scale Frangi
    vesselness map per sample ONCE (≈360 ms / 512×512 image). Without this
    cache the trainer recomputes Frangi every ``__getitem__`` call — at
    500+ samples × 200 epochs that alone exceeds the SLURM wall-clock
    limit. Geometric augmentations are applied to the cached Frangi
    alongside the image, so caching does not break the augmentation
    pipeline.
    """
    print(
        f'[load_samples] split={split}  use_frangi={use_frangi}',
        flush=True,
    )
    ds, _ = get_data(
        'rl_agent',
        split,
        tolerance=TOLERANCE,
        max_samples_per_dataset=None,
        # Chicken-and-egg: this trainer PRODUCES seed_detector.pt, so its
        # data pipeline must not depend on the predicted-prior bundle
        # (which would require seed_detector.pt to already exist).
        require_predicted_priors=False,
    )
    n = len(ds)
    samples: List[Dict[str, Any]] = []
    t_total = time.time()
    for i in range(n):
        t0 = time.time()
        s = ds[i]
        sid = s['id']
        img_np = s['image'].permute(1, 2, 0).numpy()  # (H, W, 3)
        cl_np = s['centerline'].squeeze(0).numpy()
        vessel_np = s['vessel_mask'].squeeze(0).numpy()
        fov_np = s['fov_mask'].squeeze(0).numpy()

        entry: Dict[str, Any] = {
            'id': sid,
            'image': img_np,
            'centerline': cl_np,
            'vessel_mask': vessel_np,
            'fov_mask': fov_np,
        }
        if use_frangi:
            entry['frangi'] = frangi_vesselness(img_np[..., 1])  # (H, W)

        samples.append(entry)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            elapsed = time.time() - t_total
            print(
                f'  [{split}] {i + 1}/{n}  last_id={sid}  image={img_np.shape}  elapsed={elapsed:.1f}s',
                flush=True,
            )

    print(
        f'[load_samples] {split}: {len(samples)} samples (precomputed frangi={"yes" if use_frangi else "no"}) in {time.time() - t_total:.1f}s\n',
        flush=True,
    )
    return samples


def main():
    print(f'Device: {DEVICE}', flush=True)

    sd_cfg = SEED_CONFIG.get('seed_detector', {})
    tr_cfg = SEED_CONFIG.get('training', {})
    use_frangi = bool(sd_cfg.get('use_frangi_input', True))

    train_samples = load_samples('train', use_frangi=use_frangi)
    val_samples = load_samples('val', use_frangi=use_frangi)
    if not train_samples:
        print(
            'ERROR: No training samples loaded.',
            flush=True,
        )
        return

    model = SeedDetector(sd_cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {n_params:,}', flush=True)

    trainer = SeedDetectorTrainer(
        model,
        DEVICE,
        lr=float(tr_cfg.get('lr', 1e-3)),
        batch_size=int(tr_cfg.get('batch_size', 4)),
        num_epochs=int(tr_cfg.get('num_epochs', 100)),
        warmup_epochs=int(tr_cfg.get('warmup_epochs', 5)),
        sigma=float(tr_cfg.get('sigma', 1.5)),
        use_frangi_input=use_frangi,
        num_workers=int(tr_cfg.get('num_workers', 2)),
    )

    trainer.train(
        train_samples=train_samples,
        val_samples=val_samples,
        save_path=SAVE_PATH,
        config={
            'version': 'v4_attn_unet_multitask',
            'seed_detector': sd_cfg,
            'training': tr_cfg,
        },
    )


if __name__ == '__main__':
    main()
