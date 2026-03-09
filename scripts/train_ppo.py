# scripts/train_ppo.py
"""
PPO Training — Step 2 of 2. Run AFTER train_imitation.py.

All PPO logic lives in rl_training/ppo.py.
This script handles: paths, config, and DRIVE-specific data loading.
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_models.policy_network import ActorCriticNetwork
from data.centerline_extraction import CenterlineExtractor
from data.fundus_preprocessor import FundusPreprocessor
from rl_training.ppo import PPOTrainer

_preprocessor = FundusPreprocessor()

# ==========================================
# CONFIG
# ==========================================
DRIVE_ROOT        = r"C:\ZHAW\BA\data\DRIVE\training"
IMITATION_WEIGHTS = r"C:\ZHAW\BA\weights\imitation_policy.pt"
SAVE_PATH         = r"C:\ZHAW\BA\weights\ppo_policy.pt"
LOG_PATH          = r"C:\ZHAW\BA\weights\ppo_log.txt"

IMAGES_DIR  = os.path.join(DRIVE_ROOT, "images")
VESSELS_DIR = os.path.join(DRIVE_ROOT, "1st_manual")
MASKS_DIR   = os.path.join(DRIVE_ROOT, "mask")

TOLERANCE      = 2.0
OBS_SIZE       = 65
MAX_STEPS      = 2000
USE_VESSELNESS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'policy': {
        'hidden_dim':   128,
        'lstm_hidden':  128,
        'use_lstm':     False,
        'dropout':      0.05,
        'encoder_type': 'cnn',
    },
    'environment': {
        'observation_size':      OBS_SIZE,
        'tolerance':             TOLERANCE,
        'use_vesselness':        USE_VESSELNESS,
        'max_steps_per_episode': MAX_STEPS,
        'max_off_track_streak':  8,
        'step_size':             1,
    },
    'reward': {
        'alpha_near':            0.1,
        'beta_coverage':         1.0,
        'gamma_off':            -0.5,
        'lambda_revisit':       -1.0,
        'step_cost':            -0.01,
        'direction_bonus':       0.05,
        'terminal_f1_weight':    5.0,
        'use_potential_shaping': False,
    },
    'training': {'ppo': {'gamma': 0.99}},
}

# Train : 21-35  |  Val : 36-37  |  Test : 38-40 (never touched here)
TRAIN_IDS = list(range(21, 36))
VAL_IDS   = list(range(36, 38))


# ==========================================
# DATA LOADING  (DRIVE-specific)
# ==========================================

def load_sample(img_id: str) -> Optional[Dict]:
    img_path    = os.path.join(IMAGES_DIR,  f"{img_id}_training.tif")
    vessel_path = os.path.join(VESSELS_DIR, f"{img_id}_manual1.gif")
    mask_path   = os.path.join(MASKS_DIR,   f"{img_id}_training_mask.gif")

    if not os.path.exists(img_path):
        return None

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

    extractor  = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    centerline = extractor.extract_centerline(vessel_bin)
    dist_tf    = extractor.compute_distance_transform(centerline, tolerance=TOLERANCE)

    return {
        'id':             img_id,
        'image':          img_rgb,
        'centerline':     centerline,
        'dist_transform': dist_tf,
        'fov_mask':       fov_bin,
    }


def load_samples(ids: List[int], label: str) -> List[Dict]:
    print(f"Loading {label} samples {ids[0]}-{ids[-1]}...")
    samples = []
    for i in ids:
        s = load_sample(str(i))
        if s:
            samples.append(s)
            print(f"  [{s['id']}] centerline px: {int(s['centerline'].sum())}")
    print(f"Loaded {len(samples)} {label} samples.\n")
    return samples


# ==========================================
# MAIN CONFIG
# ==========================================

def main():
    print(f"Device: {DEVICE}")

    train_samples = load_samples(TRAIN_IDS, "train")
    val_samples   = load_samples(VAL_IDS,   "val")

    if not train_samples:
        print("ERROR: No training samples loaded.")
        return

    model   = ActorCriticNetwork(CONFIG).to(DEVICE)
    trainer = PPOTrainer(
        model, CONFIG, DEVICE,
        lr              = 1e-4,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_eps        = 0.1,
        entropy_coef    = 0.05,
        value_coef      = 0.5,
        max_grad_norm   = 1.0,
        ppo_epochs      = 4,
        mini_batch_size = 256,
        steps_per_iter  = 4096,
        num_iterations  = 1000,
        eval_every      = 25,
        save_every      = 50,
        tolerance       = TOLERANCE,
    )

    trainer.train(
        train_samples  = train_samples,
        val_samples    = val_samples,
        save_path      = SAVE_PATH,
        log_path       = LOG_PATH,
        imitation_path = IMITATION_WEIGHTS,
    )


if __name__ == "__main__":
    main()