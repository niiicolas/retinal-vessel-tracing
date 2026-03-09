# scripts/train_imitation.py
"""
Imitation Learning — Step 1 of 2 before PPO.
Run this BEFORE train_ppo.py.

All logic lives in rl_training/imitation.py.
This script handles: paths, config, data loading, and wiring.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from rl_models.policy_network import ActorCriticNetwork
from data.centerline_extraction import CenterlineExtractor
from data.fundus_preprocessor import FundusPreprocessor
from rl_training.imitation import (
    ImitationTrainer,
    augment_sample,
    generate_expert_pairs,
)

_preprocessor = FundusPreprocessor()

# ==========================================
# CONFIG
# ==========================================
DRIVE_ROOT  = r"C:\ZHAW\BA\data\DRIVE\training"
SAVE_PATH   = r"C:\ZHAW\BA\weights\imitation_policy.pt"

IMAGES_DIR  = os.path.join(DRIVE_ROOT, "images")
VESSELS_DIR = os.path.join(DRIVE_ROOT, "1st_manual")
MASKS_DIR   = os.path.join(DRIVE_ROOT, "mask")

LEARNING_RATE = 3e-4
BATCH_SIZE    = 128
NUM_EPOCHS    = 30
TOLERANCE     = 2.0
OBS_SIZE      = 65
USE_AUGMENT   = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'policy': {
        'hidden_dim':   128,
        'lstm_hidden':  128,
        'use_lstm':     False,
        'dropout':      0.0,
        'encoder_type': 'cnn',
    },
    'environment': {
        'observation_size': OBS_SIZE,
        'tolerance':        TOLERANCE,
        'use_vesselness':   False,
    },
    'training': {'ppo': {'gamma': 0.99}},
}

# Train : 21-35 (15 images)
# Val   : 36-37 (monitored during PPO)
# Test  : 38-40 (never seen during training)
TRAIN_IDS = [str(i).zfill(2) for i in range(21, 36)]


# ==========================================
# DATA LOADING  (DRIVE)
# ==========================================

def load_sample(img_id: str) -> dict:
    img_path    = os.path.join(IMAGES_DIR,  f"{img_id}_training.tif")
    vessel_path = os.path.join(VESSELS_DIR, f"{img_id}_manual1.gif")
    mask_path   = os.path.join(MASKS_DIR,   f"{img_id}_training_mask.gif")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    vessel     = np.array(Image.open(vessel_path).convert('L'))
    vessel_bin = (vessel > 128).astype(np.uint8)

    fov_raw = np.array(Image.open(mask_path).convert('L'))
    fov_bin = (fov_raw > 128).astype(np.uint8)

    enhanced_green = _preprocessor.preprocess(
        image=(img_rgb * 255).astype(np.uint8),
        external_mask=fov_raw,
    )
    img_rgb[:, :, 1] = enhanced_green

    extractor     = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    centerline    = extractor.extract_centerline(vessel_bin)
    dist_tf       = extractor.compute_distance_transform(centerline, tolerance=TOLERANCE)
    expert_traces = extractor.generate_expert_traces(centerline)

    print(f"  [{img_id}] centerline px: {int(centerline.sum())}  "
          f"traces: {len(expert_traces)}")

    return {
        'image':          img_rgb,
        'centerline':     centerline,
        'dist_transform': dist_tf,
        'fov_mask':       fov_bin,
        'expert_traces':  expert_traces,
    }


# ==========================================
# MAIN
# ==========================================

def main():
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f"\nLoading DRIVE training samples {TRAIN_IDS[0]}-{TRAIN_IDS[-1]}...")
    all_pairs = []

    for img_id in TRAIN_IDS:
        img_path = os.path.join(IMAGES_DIR, f"{img_id}_training.tif")
        if not os.path.exists(img_path):
            print(f"  [{img_id}] not found, skipping")
            continue

        sample = load_sample(img_id)
        pairs  = generate_expert_pairs(sample, CONFIG, OBS_SIZE)
        all_pairs.extend(pairs)
        print(f"  [{img_id}] -> {len(pairs)} pairs")

        if USE_AUGMENT:
            for aug in augment_sample(sample, TOLERANCE):
                all_pairs.extend(generate_expert_pairs(aug, CONFIG, OBS_SIZE))

    print(f"\nTotal (obs, action) pairs: {len(all_pairs)}")
    if not all_pairs:
        print("ERROR: No pairs generated. Check DRIVE paths.")
        return

    split       = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    model   = ActorCriticNetwork(CONFIG).to(DEVICE)
    trainer = ImitationTrainer(
        model, DEVICE,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
    )
    trainer.train(train_pairs, val_pairs, SAVE_PATH, CONFIG)


if __name__ == "__main__":
    main()