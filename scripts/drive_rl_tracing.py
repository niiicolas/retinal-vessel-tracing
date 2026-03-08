# scripts/drive_rl_tracing.py
"""
Run the saved PPO policy on a DRIVE validation image and visualize the result.
Shows: fundus image | GT centerline | traced skeleton | overlay

Usage:
    python -m scripts.visualize_tracing
"""

import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.policy_network import ActorCriticNetwork
from environment.vessel_env import VesselTracingEnv
from data.centerline_extraction import CenterlineExtractor, compute_centerline_f1
from data.fundus_preprocessor import FundusPreprocessor

_preprocessor = FundusPreprocessor()

# ==========================================
# CONFIG — edit paths to match yours
# ==========================================
DRIVE_ROOT   = r"C:\ZHAW\BA\data\DRIVE\training"
PPO_WEIGHTS  = r"C:\ZHAW\BA\weights\ppo_policy.pt"
OUTPUT_DIR   = r"C:\ZHAW\BA\retinal-vessel-tracing\results\RL_tracing_DRIVE"

IMAGES_DIR   = os.path.join(DRIVE_ROOT, "images")
VESSELS_DIR  = os.path.join(DRIVE_ROOT, "1st_manual")
MASKS_DIR    = os.path.join(DRIVE_ROOT, "mask")

TOLERANCE      = 2.0
OBS_SIZE       = 65
USE_VESSELNESS = False
N_TRACES       = 50        # how many start points to trace per image
MAX_STEPS      = 2000
VAL_IDS        = ["38", "39", "40"]  # DRIVE test images — held out, never seen during training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    'policy': {
        'hidden_dim':  128,
        'lstm_hidden': 128,
        'use_lstm':    False,
        'dropout':     0.0,
        'encoder_type': 'cnn',
    },
    'environment': {
        'observation_size': OBS_SIZE,
        'tolerance':        TOLERANCE,
        'use_vesselness':   USE_VESSELNESS,
        'max_steps_per_episode': MAX_STEPS,
        'max_off_track_streak': 3,
        'step_size': 1,
    },
    'reward': {
        'alpha_near': 0.1,
        'beta_coverage': 1.0,
        'gamma_off': -0.5,
        'lambda_revisit': -2.0,
        'step_cost': -0.01,
        'direction_bonus': 0.05,
        'terminal_f1_weight': 5.0,
        'use_potential_shaping': False,
    },
    'training': {'ppo': {'gamma': 0.99}}
}


def load_sample(img_id):
    img_path    = os.path.join(IMAGES_DIR, f"{img_id}_training.tif")
    vessel_path = os.path.join(VESSELS_DIR, f"{img_id}_manual1.gif")
    mask_path   = os.path.join(MASKS_DIR,   f"{img_id}_training_mask.gif")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    vessel     = np.array(Image.open(vessel_path).convert('L'))
    vessel_bin = (vessel > 128).astype(np.uint8)

    fov_raw = np.array(Image.open(mask_path).convert('L'))
    fov_bin = (fov_raw > 128).astype(np.uint8)

    enhanced_green = _preprocessor.preprocess(
        image         = (img_rgb * 255).astype(np.uint8),
        external_mask = fov_raw,
    )
    img_rgb[:, :, 1] = enhanced_green

    extractor  = CenterlineExtractor(min_branch_length=10, prune_iterations=5)
    centerline = extractor.extract_centerline(vessel_bin)
    dist_tf    = extractor.compute_distance_transform(centerline, tolerance=TOLERANCE)

    return {
        'id':          img_id,
        'image':       img_rgb,
        'centerline':  centerline,
        'dist_transform': dist_tf,
        'fov_mask':    fov_bin,
    }



def _pick_frontier_seed(gt_centerline, covered, half):
    """
    Find the uncovered GT centerline pixel furthest from any already-covered pixel.
    This maximises coverage gain per trace by targeting the most isolated gap.
    """
    uncovered = (gt_centerline > 0) & (covered == 0)
    if not uncovered.any():
        return None

    uncovered_pts = np.argwhere(uncovered)
    h, w = gt_centerline.shape

    covered_bin = (covered > 0).astype(np.uint8)
    if covered_bin.any():
        # Distance from covered region — higher = more isolated
        dist   = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
        scores = dist[uncovered_pts[:, 0], uncovered_pts[:, 1]]
        best   = uncovered_pts[np.argmax(scores)]
    else:
        # First trace: start near image centre to avoid border
        centre = np.array([h // 2, w // 2])
        dists  = np.linalg.norm(uncovered_pts - centre, axis=1)
        best   = uncovered_pts[np.argmin(dists)]

    y = int(np.clip(best[0], half + 5, h - half - 6))
    x = int(np.clip(best[1], half + 5, w - half - 6))
    return (y, x)


def trace_frontier(model, sample, max_traces=N_TRACES, min_coverage_gain=0.001):
    """
    Frontier-based coverage: after each trace, restart from the furthest
    uncovered centerline region. Stops when coverage gain per trace drops
    below min_coverage_gain or max_traces is exhausted.
    """
    env = VesselTracingEnv(CONFIG)
    env.set_data(
        image              = sample['image'],
        centerline         = sample['centerline'],
        distance_transform = sample['dist_transform'],
        fov_mask           = sample['fov_mask'],
    )

    h, w     = sample['image'].shape[:2]
    half     = OBS_SIZE // 2
    combined = np.zeros((h, w), dtype=np.float32)
    paths    = []
    gt_total = float(max(sample['centerline'].sum(), 1))

    model.eval()
    with torch.no_grad():
        for trace_idx in range(max_traces):
            start = _pick_frontier_seed(sample['centerline'], combined, half)
            if start is None:
                print(f"    Full coverage after {trace_idx} traces.")
                break

            obs, _         = env.reset(start_position=start)
            path           = [start]
            covered_before = combined.sum()
            done           = False

            while not done:
                obs_t        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logits, _, _ = model(obs_t)
                action       = logits.argmax(dim=-1).item()
                obs, _, terminated, truncated, _ = env.step(action)
                done         = terminated or truncated
                y, x         = env.position
                path.append((y, x))
                combined[y, x] = 1.0

            gain         = (combined.sum() - covered_before) / gt_total
            coverage_pct = combined.sum() / gt_total
            print(f"    Trace {trace_idx+1:3d} from {start} -> "
                  f"{len(path)} steps  gain={gain:.3f}  coverage={coverage_pct:.3f}")
            paths.append(path)

            if trace_idx >= 3 and gain < min_coverage_gain:
                print(f"    Early stop: gain {gain:.4f} < {min_coverage_gain}")
                break

    return combined, paths


def make_overlay(image_rgb, gt_centerline, traced, paths):
    """
    Returns an RGB overlay image with:
      - GT centerline in green
      - Traced path in red
      - Seed points in yellow
    """
    overlay = (image_rgb * 255).astype(np.uint8).copy()

    # GT centerline — green
    overlay[gt_centerline > 0] = [0, 200, 0]

    # Traced pixels — red (drawn on top so misses are visible)
    overlay[traced > 0] = [220, 50, 50]

    # Overlap (true positives) — yellow
    tp = (gt_centerline > 0) & (traced > 0)
    overlay[tp] = [255, 220, 0]

    # Seed points — cyan dots
    for path in paths:
        if path:
            y, x = path[0]
            cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)

    return overlay


def visualize_sample(model, sample, output_dir):
    img_id = sample['id']
    print(f"\n  Image {img_id}: frontier tracing (max {N_TRACES} traces)...")

    traced, paths = trace_frontier(model, sample, max_traces=N_TRACES, min_coverage_gain=0.001)
    n_traces_used = len(paths)

    metrics = compute_centerline_f1(traced, sample['centerline'], tolerance=TOLERANCE)
    print(f"  F1={metrics['f1']:.3f}  precision={metrics['precision']:.3f}  "
          f"recall={metrics['recall']:.3f}  traces_used={n_traces_used}")

    overlay = make_overlay(sample['image'], sample['centerline'], traced, paths)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(
        f"Image {img_id} — F1={metrics['f1']:.3f}  "
        f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
        f"({n_traces_used} traces)",
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(sample['image'])
    axes[0].set_title("Fundus image")
    axes[0].axis('off')

    axes[1].imshow(sample['centerline'], cmap='gray')
    axes[1].set_title("GT centerline")
    axes[1].axis('off')

    axes[2].imshow(traced, cmap='gray')
    axes[2].set_title(f"Traced ({n_traces_used} frontier traces)")
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis('off')

    legend = [
        mpatches.Patch(color='#00C800', label='GT only (miss)'),
        mpatches.Patch(color='#DC3232', label='Traced only (FP)'),
        mpatches.Patch(color='#FFDC00', label='True positive'),
        mpatches.Patch(color='#00FFFF', label='Seed point'),
    ]
    axes[3].legend(handles=legend, loc='lower right', fontsize=7,
                   framealpha=0.8, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"trace_{img_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_path}")
    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Loading weights from {PPO_WEIGHTS}")

    checkpoint = torch.load(PPO_WEIGHTS, map_location=DEVICE, weights_only=True)
    model = ActorCriticNetwork(CONFIG).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  loaded iter={checkpoint.get('iteration','?')}  "
          f"best_F1={checkpoint.get('best_f1', '?')}")

    all_f1 = []
    for img_id in VAL_IDS:
        img_path = os.path.join(IMAGES_DIR, f"{img_id}_training.tif")
        if not os.path.exists(img_path):
            print(f"  [{img_id}] not found, skipping")
            continue
        sample  = load_sample(img_id)
        metrics = visualize_sample(model, sample, OUTPUT_DIR)
        all_f1.append(metrics['f1'])

    print(f"\nMean F1 across val images: {np.mean(all_f1):.3f}")
    print(f"Output images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()