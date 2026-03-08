# scripts/train_imitation.py
"""
Imitation Learning Warm-Start (Behavior Cloning)
Appendix A3 from proposal — Step 1 of 2 before PPO.

Loads DRIVE training images directly, generates expert traces
from GT centerlines, and trains the policy network via supervised
cross-entropy loss on (observation, expert_action) pairs.

Run this BEFORE train_ppo.py.
"""

import os
import sys
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.policy_network import ActorCriticNetwork
from environment.observation import ObservationBuilder
from data.centerline_extraction import CenterlineExtractor
from data.fundus_preprocessor import FundusPreprocessor

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
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'policy': {
        'hidden_dim':  128,
        'lstm_hidden': 128,
        'use_lstm':    False,
        'dropout':     0.0,    # match PPO config — was 0.05
        'encoder_type': 'cnn',
    },
    'environment': {
        'observation_size': OBS_SIZE,
        'tolerance':        TOLERANCE,
        'use_vesselness':   False,
    },
    'training': {
        'ppo': {'gamma': 0.99}
    }
}


# ==========================================
# STEP 1 — LOAD A DRIVE SAMPLE
# ==========================================

def load_sample(img_id: str) -> Dict[str, np.ndarray]:
    img_path    = os.path.join(IMAGES_DIR, f"{img_id}_training.tif")
    vessel_path = os.path.join(VESSELS_DIR, f"{img_id}_manual1.gif")
    mask_path   = os.path.join(MASKS_DIR,   f"{img_id}_training_mask.gif")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    vessel     = np.array(Image.open(vessel_path).convert('L'))
    vessel_bin = (vessel > 128).astype(np.uint8)

    fov_raw = np.array(Image.open(mask_path).convert('L'))
    fov_bin = (fov_raw > 128).astype(np.uint8)

    # Enhance green channel with full preprocessing pipeline:
    # green extraction → gamma → median blur → FOV mask → CLAHE → ROI normalise.
    # R and B channels are kept as raw normalised values.
    # Preprocessor receives full RGB (uint8) + raw FOV mask and handles
    # green extraction internally.
    enhanced_green = _preprocessor.preprocess(
        image         = (img_rgb * 255).astype(np.uint8),
        external_mask = fov_raw,
    )
    img_rgb[:, :, 1] = enhanced_green  # replace green channel only

    extractor      = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    centerline     = extractor.extract_centerline(vessel_bin)
    dist_transform = extractor.compute_distance_transform(centerline, tolerance=TOLERANCE)
    expert_traces  = extractor.generate_expert_traces(centerline)

    print(f"  [{img_id}] centerline px: {int(centerline.sum())}  "
          f"traces: {len(expert_traces)}")

    return {
        'image':          img_rgb,
        'centerline':     centerline,
        'dist_transform': dist_transform,
        'fov_mask':       fov_bin,
        'expert_traces':  expert_traces,
    }


# ==========================================
# STEP 2 — AUGMENTATION
# ==========================================

# Action remapping tables for geometric transforms.
# Actions: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7, STOP=8
_FLIP_H_REMAP  = {0:0, 1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1, 8:8}  # E<->W, NE<->NW, SE<->SW
_FLIP_V_REMAP  = {0:4, 1:3, 2:2, 3:1, 4:0, 5:7, 6:6, 7:5, 8:8}  # N<->S, NE<->SE, NW<->SW
_ROT90_REMAP   = {0:2, 1:3, 2:4, 3:5, 4:6, 5:7, 6:0, 7:1, 8:8}  # each action rotates 90° CW
_ROT180_REMAP  = {0:4, 1:5, 2:6, 3:7, 4:0, 5:1, 6:2, 7:3, 8:8}
_ROT270_REMAP  = {0:6, 1:7, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:8}


def _remap_traces(traces, action_remap, transform_fn):
    """Apply a coordinate transform + action remap to expert traces."""
    new_traces = []
    for trace in traces:
        new_traces.append([transform_fn(y, x) for y, x in trace])
    return new_traces


def augment_sample(sample: Dict) -> List[Dict]:
    """
    Generate augmented copies of a sample.
    Returns a list of new sample dicts (original not included).

    Augmentations:
      - Horizontal flip
      - Vertical flip
      - Rotation 90 / 180 / 270
      - Brightness/contrast jitter (image only, no label change)
    """
    img   = sample['image']          # (H, W, 3) float32 [0,1]
    cl    = sample['centerline']     # (H, W) uint8
    dt    = sample['dist_transform'] # (H, W) float32
    fov   = sample['fov_mask']       # (H, W) uint8
    traces = sample['expert_traces']
    h, w  = img.shape[:2]

    augmented = []

    def make_sample(new_img, new_cl, new_dt, new_fov, new_traces):
        # Recompute dist_transform for augmented centerline to keep it consistent
        extractor = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
        new_dt    = extractor.compute_distance_transform(new_cl, tolerance=TOLERANCE)
        return {
            'image':          new_img,
            'centerline':     new_cl,
            'dist_transform': new_dt,
            'fov_mask':       new_fov,
            'expert_traces':  new_traces,
        }

    # --- Horizontal flip (left <-> right) ---
    img_hf  = img[:, ::-1, :].copy()
    cl_hf   = cl[:, ::-1].copy()
    fov_hf  = fov[:, ::-1].copy()
    tr_hf   = _remap_traces(traces, _FLIP_H_REMAP,
                             lambda y, x: (y, w - 1 - x))
    augmented.append(make_sample(img_hf, cl_hf, None, fov_hf, tr_hf))

    # --- Vertical flip (top <-> bottom) ---
    img_vf  = img[::-1, :, :].copy()
    cl_vf   = cl[::-1, :].copy()
    fov_vf  = fov[::-1, :].copy()
    tr_vf   = _remap_traces(traces, _FLIP_V_REMAP,
                             lambda y, x: (h - 1 - y, x))
    augmented.append(make_sample(img_vf, cl_vf, None, fov_vf, tr_vf))

    # --- Rotation 90° CW ---
    # np.rot90 with k=3 rotates CW; coordinate: (y,x) -> (x, h-1-y)
    img_r90  = np.rot90(img,  k=3).copy()
    cl_r90   = np.rot90(cl,   k=3).copy()
    fov_r90  = np.rot90(fov,  k=3).copy()
    tr_r90   = _remap_traces(traces, _ROT90_REMAP,
                              lambda y, x: (x, h - 1 - y))
    augmented.append(make_sample(img_r90, cl_r90, None, fov_r90, tr_r90))

    # --- Rotation 180° ---
    img_r180  = np.rot90(img,  k=2).copy()
    cl_r180   = np.rot90(cl,   k=2).copy()
    fov_r180  = np.rot90(fov,  k=2).copy()
    tr_r180   = _remap_traces(traces, _ROT180_REMAP,
                               lambda y, x: (h - 1 - y, w - 1 - x))
    augmented.append(make_sample(img_r180, cl_r180, None, fov_r180, tr_r180))

    # --- Rotation 270° CW (= 90° CCW) ---
    # coordinate: (y,x) -> (w-1-x, y)
    img_r270  = np.rot90(img,  k=1).copy()
    cl_r270   = np.rot90(cl,   k=1).copy()
    fov_r270  = np.rot90(fov,  k=1).copy()
    tr_r270   = _remap_traces(traces, _ROT270_REMAP,
                               lambda y, x: (w - 1 - x, y))
    augmented.append(make_sample(img_r270, cl_r270, None, fov_r270, tr_r270))

    # --- Brightness/contrast jitter (image only, no label change) ---
    for brightness, contrast in [(0.8, 1.0), (1.2, 1.0), (1.0, 0.8), (1.0, 1.2)]:
        img_jit = np.clip(img * contrast + (brightness - 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        augmented.append({
            'image':          img_jit,
            'centerline':     cl,
            'dist_transform': dt,   # unchanged — geometry not modified
            'fov_mask':       fov,
            'expert_traces':  traces,
        })

    return augmented


# ==========================================
# STEP 3 — CONVERT TRACES TO (obs, action) PAIRS
# ==========================================

def direction_to_action(dy: int, dx: int) -> int:
    """
    Convert (dy, dx) step to discrete action index (0–7).
    N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7, STOP=8
    """
    direction_map = {
        (-1,  0): 0,
        (-1,  1): 1,
        ( 0,  1): 2,
        ( 1,  1): 3,
        ( 1,  0): 4,
        ( 1, -1): 5,
        ( 0, -1): 6,
        (-1, -1): 7,
    }
    return direction_map.get((dy, dx), 8)


def generate_expert_pairs(sample: Dict) -> List[Tuple[np.ndarray, int]]:
    """
    Walk each expert trace and build (observation, action) pairs.
    """
    obs_builder = ObservationBuilder(CONFIG)
    image       = sample['image']
    dist_tf     = sample['dist_transform']   # needed for new obs channel
    h, w        = image.shape[:2]
    half        = OBS_SIZE // 2

    pairs        = []
    visited_mask = np.zeros((h, w), dtype=np.float32)

    for trace in sample['expert_traces']:
        if len(trace) < 2:
            continue

        for i in range(len(trace) - 1):
            y, x   = trace[i]
            ny, nx = trace[i + 1]

            if y < half or y >= h - half or x < half or x >= w - half:
                continue

            dy     = int(ny) - int(y)
            dx     = int(nx) - int(x)
            action = direction_to_action(dy, dx)

            if action == 8:
                continue

            prev_dir = direction_to_action(
                int(y) - int(trace[i-1][0]) if i > 0 else 0,
                int(x) - int(trace[i-1][1]) if i > 0 else 0,
            ) if i > 0 else None

            # Pass distance_transform so observation has 6 channels
            obs = obs_builder.build(
                image              = image,
                visited_mask       = visited_mask,
                vesselness         = None,
                position           = np.array([y, x]),
                prev_direction     = prev_dir,
                distance_transform = dist_tf,   # was missing — caused 5-channel obs
            )

            pairs.append((obs, action))
            visited_mask[y, x] = 1.0

    return pairs


# ==========================================
# STEP 3 — PYTORCH DATASET
# ==========================================

class ImitationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[np.ndarray, int]]):
        self.obs     = [p[0] for p in pairs]
        self.actions = [p[1] for p in pairs]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs    = torch.from_numpy(self.obs[idx]).float()
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return obs, action


# ==========================================
# STEP 4 — TRAINING LOOP
# ==========================================

def train_imitation():
    print(f"Device: {DEVICE}")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Clean data split — must match train_ppo.py exactly:
    #   Train : 21-35 (15 images) — imitation + PPO training
    #   Val   : 36-37 (2 images)  — PPO validation during training
    #   Test  : 38-40 (3 images)  — final evaluation only, never seen during training
    print("\nLoading DRIVE training samples (IDs 21-35)...")
    all_ids = [str(i).zfill(2) for i in range(21, 36)]

    all_pairs = []
    for img_id in all_ids:
        img_path = os.path.join(IMAGES_DIR, f"{img_id}_training.tif")
        if not os.path.exists(img_path):
            continue
        sample = load_sample(img_id)

        pairs = generate_expert_pairs(sample)
        all_pairs.extend(pairs)
        print(f"  [{img_id}] → {len(pairs)} (obs, action) pairs")

    print(f"\nTotal pairs: {len(all_pairs)}")

    if len(all_pairs) == 0:
        print("ERROR: No expert pairs generated. Check your DRIVE paths.")
        return

    split       = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split]
    val_pairs   = all_pairs[split:]

    train_ds = ImitationDataset(train_pairs)
    val_ds   = ImitationDataset(val_pairs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model     = ActorCriticNetwork(CONFIG).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for obs_batch, action_batch in train_loader:
            obs_batch    = obs_batch.to(DEVICE)
            action_batch = action_batch.to(DEVICE)

            logits, _, _ = model(obs_batch)
            loss = criterion(logits, action_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item() * len(action_batch)
            train_correct += (logits.argmax(-1) == action_batch).sum().item()
            train_total   += len(action_batch)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for obs_batch, action_batch in val_loader:
                obs_batch    = obs_batch.to(DEVICE)
                action_batch = action_batch.to(DEVICE)

                logits, _, _ = model(obs_batch)
                loss = criterion(logits, action_batch)

                val_loss    += loss.item() * len(action_batch)
                val_correct += (logits.argmax(-1) == action_batch).sum().item()
                val_total   += len(action_batch)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':            val_loss,
                'val_acc':             val_acc,
                'config':              CONFIG,
            }, SAVE_PATH)
            print(f"  ✓ Saved best model → {SAVE_PATH}")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train_imitation()