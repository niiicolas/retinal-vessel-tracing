# scripts/drive_rl_tracing.py
"""
End-to-end inference: SeedDetector → FrontierTracer → F1 evaluation.
"""

import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_models.policy_network import ActorCriticNetwork
from rl_models.seed_detector import SeedDetector
from rl_environment.vessel_env import VesselTracingEnv
from rl_environment.frontier_tracer import FrontierTracer
from data.centerline_extraction import CenterlineExtractor, compute_centerline_f1
from data.fundus_preprocessor import FundusPreprocessor

_preprocessor = FundusPreprocessor()

# ==========================================
# MODE — switch between gt and e2e
# ==========================================
MODE = 'e2e'   # 'gt' | 'e2e' 

# ==========================================
# PATHS
# ==========================================
DRIVE_ROOT      = r"C:\ZHAW\BA\data\DRIVE\training"
PPO_WEIGHTS     = r"C:\ZHAW\BA\retinal-vessel-tracing\weights\ppo_policy.pt"
SEED_WEIGHTS    = r"C:\ZHAW\BA\retinal-vessel-tracing\weights\seed_detector.pt"
OUTPUT_DIR      = r"C:\ZHAW\BA\retinal-vessel-tracing\results\RL_tracing_seeddetector_DRIVE"

IMAGES_DIR      = os.path.join(DRIVE_ROOT, "images")
VESSELS_DIR     = os.path.join(DRIVE_ROOT, "1st_manual")
MASKS_DIR       = os.path.join(DRIVE_ROOT, "mask")

TOLERANCE       = 2.0
OBS_SIZE        = 65
MAX_STEPS       = 2000
MAX_TRACES      = 80
MIN_COV_GAIN    = 0.001
TEST_IDS        = ["38", "39", "40"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PPO_CONFIG = {
    'policy': {
        'hidden_dim':    128,
        'lstm_hidden':   128,
        'use_lstm':      False,
        'dropout':       0.0,
        'encoder_type':  'cnn',
    },
    'environment': {
        'observation_size':      OBS_SIZE,
        'tolerance':             TOLERANCE,
        'use_vesselness':        False,
        'max_steps_per_episode': 600,
        'max_off_track_streak':  5,
        'step_size':             2,
    },
    'reward': {
        'alpha_near':            0.1,
        'beta_coverage':         1.0,
        'gamma_off':            -0.5,
        'lambda_revisit':       -2.0,
        'step_cost':            -0.01,
        'direction_bonus':       0.05,
        'terminal_f1_weight':    5.0,
        'use_potential_shaping': False,
    },
    'training': {'ppo': {'gamma': 0.99}},
}

SEED_CONFIG = {
    'seed_detector': {
        'base_ch':              16,    
        'nms_radius':           15,
        'confidence_threshold': 0.3,    
        'top_k_seeds':          MAX_TRACES,
    }
}


# ==========================================
# DATA LOADING
# ==========================================

def load_sample(img_id: str) -> dict:
    img_path    = os.path.join(IMAGES_DIR,  f"{img_id}_training.tif")
    vessel_path = os.path.join(VESSELS_DIR, f"{img_id}_manual1.gif")
    mask_path   = os.path.join(MASKS_DIR,   f"{img_id}_training_mask.gif")

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # Save a copy of the original RGB image for the first visualization panel
    orig_img_rgb = img_rgb.copy()

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
        'image_orig':     orig_img_rgb,  # <-- Added original image to dict
        'image':          img_rgb,
        'centerline':     centerline,
        'dist_transform': dist_tf,
        'fov_mask':       fov_bin,
    }


# ==========================================
# GT-BASED SEED PICKING  (MODE='gt')
# ==========================================

def _pick_frontier_seed_gt(gt_centerline, covered, half):
    uncovered = (gt_centerline > 0) & (covered == 0)
    if not uncovered.any():
        return None

    uncovered_pts = np.argwhere(uncovered)
    h, w = gt_centerline.shape
    covered_bin = (covered > 0).astype(np.uint8)

    if covered_bin.any():
        dist   = cv2.distanceTransform(1 - covered_bin, cv2.DIST_L2, 5)
        scores = dist[uncovered_pts[:, 0], uncovered_pts[:, 1]]
        best   = uncovered_pts[np.argmax(scores)]
    else:
        centre = np.array([h // 2, w // 2])
        dists  = np.linalg.norm(uncovered_pts - centre, axis=1)
        best   = uncovered_pts[np.argmin(dists)]

    y = int(np.clip(best[0], half + 5, h - half - 6))
    x = int(np.clip(best[1], half + 5, w - half - 6))
    return (y, x)


# ==========================================
# GT MODE TRACING
# ==========================================

def trace_gt_mode(ppo_model, sample):
    env = VesselTracingEnv(PPO_CONFIG)
    env.set_data(
        image=sample['image'],
        centerline=sample['centerline'],
        distance_transform=sample['dist_transform'],
        fov_mask=sample['fov_mask'],
    )

    h, w     = sample['image'].shape[:2]
    half     = OBS_SIZE // 2
    combined = np.zeros((h, w), dtype=np.float32)
    paths    = []
    gt_total = float(max(sample['centerline'].sum(), 1))

    ppo_model.eval()
    with torch.no_grad():
        for trace_idx in tqdm(range(MAX_TRACES), desc=f"Img {sample['id']} Tracing", unit="trace", leave=False):
            start = _pick_frontier_seed_gt(sample['centerline'], combined, half)
            if start is None:
                tqdm.write(f"    Full GT coverage after {trace_idx} traces.")
                break

            obs, _         = env.reset(start_position=start)
            path           = [start]
            covered_before = combined.sum()
            done           = False

            while not done:
                obs_t        = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                logits, _, _ = ppo_model(obs_t)
                action       = logits.argmax(dim=-1).item()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                y, x = env.position
                path.append((y, x))
                combined[y, x] = 1.0

            gain         = (combined.sum() - covered_before) / gt_total
            coverage_pct = combined.sum() / gt_total
            
            tqdm.write(f"    Trace {trace_idx+1:3d} gain={gain:.3f} coverage={coverage_pct:.3f}")
            paths.append(path)

            if trace_idx >= 3 and gain < MIN_COV_GAIN:
                tqdm.write(f"    Early stop: gain {gain:.4f} < {MIN_COV_GAIN}")
                break

    return combined, paths


# ==========================================
# E2E MODE TRACING
# ==========================================

def trace_e2e_mode(ppo_model, seed_model, sample):
    img_t = torch.from_numpy(
        sample['image'].transpose(2, 0, 1)
    ).unsqueeze(0).float().to(DEVICE)

    fov_t = torch.from_numpy(sample['fov_mask']).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    batch_seeds, heatmap = seed_model.detect_seeds(
        img_t, 
        obs_half=OBS_SIZE // 2, 
        return_heatmap=True,
        fov_mask=fov_t
    )
    seeds = batch_seeds[0] 
    tqdm.write(f"    Seed detector: {len(seeds)} seeds predicted")

    if not seeds:
        tqdm.write("    WARNING: No seeds found, falling back to image centre")
        h, w = sample['image'].shape[:2]
        seeds = [(h // 2, w // 2, 0.5)]

    env = VesselTracingEnv(PPO_CONFIG)
    tracer = FrontierTracer(env, ppo_model, DEVICE, obs_size=OBS_SIZE)
    initial_seeds = [(y, x) for y, x, _ in seeds]

    combined, paths = tracer.trace_from_seeds(sample, initial_seeds)

    return combined, paths, heatmap[0, 0].cpu().numpy()


# ==========================================
# VISUALISATION
# ==========================================

def make_overlay(image_orig, gt_centerline, traced, paths):
    """Creates a darkened grayscale background with colored traces over it."""
    # Convert to grayscale and darken
    gray = cv2.cvtColor((image_orig * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dark_gray = (gray * 0.4).astype(np.uint8)
    overlay = cv2.cvtColor(dark_gray, cv2.COLOR_GRAY2RGB)
    
    # Overlay colors
    overlay[gt_centerline > 0]                        = [0,   200,  0]   # GT (missed) — green
    overlay[traced > 0]                               = [220,  50, 50]   # Traced (FP) — red
    overlay[(gt_centerline > 0) & (traced > 0)]       = [255, 220,  0]   # TP — yellow
    for path in paths:
        if path:
            y, x = path[0]
            cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)            # Seeds — cyan
    return overlay


def visualize_sample(ppo_model, seed_model, sample, output_dir):
    img_id = sample['id']
    tqdm.write(f"\nProcessing Image {img_id} [Mode: {MODE}]")

    if MODE == 'gt':
        traced, paths = trace_gt_mode(ppo_model, sample)
    else:
        # We don't need to plot the heatmap anymore, so we can ignore it
        traced, paths, _ = trace_e2e_mode(ppo_model, seed_model, sample)

    metrics       = compute_centerline_f1(traced, sample['centerline'], tolerance=TOLERANCE)
    n_traces_used = len(paths)
    tqdm.write(f"  Result: F1={metrics['f1']:.3f} | Prec={metrics['precision']:.3f} | Rec={metrics['recall']:.3f}")

    # Pass the original image to make the darkened background
    overlay = make_overlay(sample['image_orig'], sample['centerline'], traced, paths)

    # 5 Columns to match your target layout exactly
    n_cols = 5
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # Target Title Format
    fig.suptitle(f"Image {img_id} — F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  ({n_traces_used} traces)", 
                 fontsize=14, fontweight='bold')

    # Subplots with proper labels
    axes[0].imshow(sample['image_orig']); axes[0].set_title("(a) Original RGB Fundus", fontsize=10); axes[0].axis('off')
    axes[1].imshow(sample['image']);      axes[1].set_title("(b) Preprocessed (Agent Input)", fontsize=10); axes[1].axis('off')
    axes[2].imshow(sample['centerline'], cmap='gray'); axes[2].set_title("(c) GT Centerline", fontsize=10); axes[2].axis('off')
    axes[3].imshow(traced, cmap='gray');  axes[3].set_title(f"(d) Agent Traced ({n_traces_used} paths)", fontsize=10); axes[3].axis('off')
    axes[4].imshow(overlay);              axes[4].set_title("(e) Darkened Overlay", fontsize=10); axes[4].axis('off')

    # Reordered Legend matching your target image
    legend = [
        mpatches.Patch(color='#00C800', label='GT only (miss)'),
        mpatches.Patch(color='#FFDC00', label='True positive'),
        mpatches.Patch(color='#DC3232', label='Traced only (FP)'),
        mpatches.Patch(color='#00FFFF', label='Seed Point'),
    ]
    axes[4].legend(handles=legend, loc='lower right', fontsize=8,
                   framealpha=0.8, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"trace_{img_id}_{MODE}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    tqdm.write(f"  Saved → {out_path}")
    return metrics


# ==========================================
# MAIN
# ==========================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}  |  Mode: {MODE}")

    # Load PPO model
    ppo_ckpt  = torch.load(PPO_WEIGHTS, map_location=DEVICE, weights_only=True)
    ppo_model = ActorCriticNetwork(PPO_CONFIG).to(DEVICE)
    ppo_model.load_state_dict(ppo_ckpt['model_state_dict'])
    ppo_model.eval()

    # Load seed detector
    seed_model = None
    if MODE == 'e2e':
        seed_ckpt  = torch.load(SEED_WEIGHTS, map_location=DEVICE, weights_only=True)
        seed_model = SeedDetector(SEED_CONFIG).to(DEVICE)
        seed_model.load_state_dict(seed_ckpt['model_state_dict'])
        seed_model.eval()

    # Master progress bar for all test images
    for img_id in tqdm(TEST_IDS, desc="Total Benchmark", unit="img"):
        img_path = os.path.join(IMAGES_DIR, f"{img_id}_training.tif")
        if not os.path.exists(img_path):
            continue
        sample  = load_sample(img_id)
        metrics = visualize_sample(ppo_model, seed_model, sample, OUTPUT_DIR)

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()