# scripts/train_seed_detector.py
"""
Seed Detector Training — Step 0 (optional but recommended for end-to-end inference).

Trains a UNet-based heatmap predictor to detect vessel endpoints and junctions.
Architecture: same UNet backbone as centerline_unet_baseline.py (DSConv blocks,
skip connections, ~0.5M params) with in_channels=3 for RGB input.

Its output replaces the GT-dependent _pick_frontier_seed() at inference time,
making the pipeline fully end-to-end.

Run BEFORE or independently of train_imitation.py / train_ppo.py.

All logic lives in training/seed_detector_trainer.py.
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

from rl_models.seed_detector import SeedDetector
from data.centerline_extraction import CenterlineExtractor
from data.fundus_preprocessor import FundusPreprocessor
from rl_training.seed_detector_trainer import SeedDetectorTrainer

_preprocessor = FundusPreprocessor()

# ==========================================
# CONFIG
# ==========================================
DRIVE_ROOT  = r"C:\ZHAW\BA\data\DRIVE\training"
SAVE_PATH   = r"C:\ZHAW\BA\retinal-vessel-tracing\weights\seed_detector.pt"

IMAGES_DIR  = os.path.join(DRIVE_ROOT, "images")
VESSELS_DIR = os.path.join(DRIVE_ROOT, "1st_manual")
MASKS_DIR   = os.path.join(DRIVE_ROOT, "mask")

TOLERANCE   = 2.0
SIGMA       = 3.0     # Gaussian blob size around each endpoint/junction
NUM_EPOCHS  = 30
BATCH_SIZE  = 4       # full 565×584 images — keep small to fit VRAM
LR          = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train : 21-35  |  Val : 36-37  |  Test : 38-40 (never touched here)
TRAIN_IDS = list(range(21, 36))
VAL_IDS   = list(range(36, 38))

CONFIG = {
    'seed_detector': {
        'base_ch':              16,    # UNet channel width — 16 → ~0.5M params
        'nms_radius':           10,    # min distance between peaks after NMS
        'confidence_threshold': 0.3,   # min heatmap value to count as a seed
        'top_k_seeds':          50,    # max seeds returned per image
    }
}

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

    # Same preprocessing as PPO training — consistent green channel enhancement
    enhanced_green = _preprocessor.preprocess(
        image=(img_rgb * 255).astype(np.uint8),
        external_mask=fov_raw,
    )
    img_rgb[:, :, 1] = enhanced_green

    extractor  = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    centerline = extractor.extract_centerline(vessel_bin)

    return {
        'id':         img_id,
        'image':      img_rgb,           # (H, W, 3) float32
        'centerline': centerline,        # (H, W)    float32 binary
        'fov_mask':   fov_bin,           # (H, W)    uint8   binary
    }


def load_samples(ids: List[int], label: str) -> List[Dict]:
    print(f"Loading {label} samples {ids[0]}–{ids[-1]}...")
    samples = []
    for i in ids:
        s = load_sample(str(i))
        if s:
            samples.append(s)
            extractor = CenterlineExtractor()
            n_ep = len(extractor._find_endpoints(s['centerline']))
            n_jn = len(extractor._find_junctions(s['centerline']))
            print(f"  [{s['id']}]  endpoints={n_ep}  junctions={n_jn}  "
                  f"seeds={n_ep + n_jn}")
    print(f"Loaded {len(samples)} {label} samples.\n")
    return samples


# ==========================================
# MAIN
# ==========================================

def main():
    print(f"Device: {DEVICE}")

    train_samples = load_samples(TRAIN_IDS, "train")
    val_samples   = load_samples(VAL_IDS,   "val")

    if not train_samples:
        print("ERROR: No training samples loaded. Check DRIVE paths.")
        return

    model   = SeedDetector(CONFIG).to(DEVICE)
    trainer = SeedDetectorTrainer(
        model, DEVICE,
        lr         = LR,
        batch_size = BATCH_SIZE,
        num_epochs = NUM_EPOCHS,
        sigma      = SIGMA,
    )

    trainer.train(
        train_samples = train_samples,
        val_samples   = val_samples,
        save_path     = SAVE_PATH,
        config        = CONFIG,
    )


if __name__ == "__main__":
    main()