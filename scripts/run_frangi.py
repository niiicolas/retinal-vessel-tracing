"""Frangi vesselness baseline evaluation driver, dataset-agnostic.

Runs the Frangi centerline extractor on the val/test splits at the RL agent's settings and
scores it through the shared scorer for direct comparability.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TOLERANCE
from data.centerline_extraction import CenterlineExtractor
from data.dataloader import OUTPUT_DIR as _OUTPUT_BASE
from data.dataloader import TEST_DATASETS, get_data, get_test_data
from evaluation.scoring import score_prediction, write_eval_csvs
from models.frangi import FrangiBaseline

# Match the RL agent's eval (run_rl_tracing): same resolution, tolerance, FOV, shared scorer.
RESIZE = (512, 512)
_extractor = CenterlineExtractor()

# Per-dataset Frangi parameters; DEFAULT_FRANGI_PARAMS is used for the combined val set.
FRANGI_PARAMS = {
    'DRIVE': dict(
        sigma_min=1.0,
        sigma_max=8.0,
        num_scales=10,
        threshold=0.005,
        gauss_sigma=1.2,
        min_size=75,
    ),
    'DRHAGIS': dict(
        sigma_min=0.5,
        sigma_max=4.0,
        num_scales=10,
        threshold=0.004,
        gauss_sigma=1.0,
        min_size=100,
    ),
}

DEFAULT_FRANGI_PARAMS = dict(
    sigma_min=1.0,
    sigma_max=3.0,
    num_scales=5,
    threshold=0.05,
    gauss_sigma=1.0,
    min_size=50,
)


def evaluate(split):
    """Run Frangi on one split and write panels, a mosaic, and metric CSVs; returns the per-image metrics.

    Args:
        split: ``"val"`` for the combined val set, otherwise a test dataset name.
    """
    # Output mirrors the RL layout so collect_ablation_metrics.py finds it.
    output_dir = str(_OUTPUT_BASE / 'frangi' / 'RL_tracing_e2e' / split)
    panels_dir = os.path.join(output_dir, 'panels')
    os.makedirs(panels_dir, exist_ok=True)

    if split == 'val':
        dataset, _ = get_data('frangi', 'val', batch_size=1, resize=RESIZE, tolerance=TOLERANCE)
        params = DEFAULT_FRANGI_PARAMS
    else:
        dataset, _ = get_test_data(split, 'frangi', batch_size=1, resize=RESIZE, tolerance=TOLERANCE)
        params = FRANGI_PARAMS.get(split, DEFAULT_FRANGI_PARAMS)
    model = FrangiBaseline(**params)

    print(f'[{split}]  {len(dataset)} images\n')

    all_metrics = []
    mosaic_data = []

    for i in tqdm(range(len(dataset)), desc=f'Frangi — {split}'):
        sample = dataset[i]
        image_id = sample['id']

        pred_skeleton, vesselness, _ = model.extract_centerline(sample['preprocessed'], return_vesselness=True, fov_mask=sample['fov_mask'])

        # Same GT (dataloader centerline) and shared scorer as the RL agent → comparable to v12.
        gt_skeleton = (sample['centerline'] > 0).astype(np.uint8)
        dt = _extractor.compute_distance_transform(gt_skeleton.astype(np.float32), TOLERANCE)
        raw_metrics = score_prediction(
            pred_skeleton,
            centerline=gt_skeleton,
            vessel_mask=sample['vessel_mask'],
            fov_mask=sample['fov_mask'],
            distance_transform=dt,
            tolerance=TOLERANCE,
        )
        metrics_entry = {'image_id': image_id}
        metrics_entry.update(raw_metrics)
        all_metrics.append(metrics_entry)

        mosaic_data.append({'image_id': image_id, 'gt_skeleton': gt_skeleton, 'pred_skeleton': pred_skeleton, 'metrics': metrics_entry})

        fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor='white')
        axes[0].imshow(sample['image'])
        axes[0].set_title(f'Original Image (ID: {image_id})', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(vesselness, cmap='gray')
        axes[1].set_title('Frangi Vesselness', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        combined_skel = np.hstack((gt_skeleton.astype(np.uint8) * 255, pred_skeleton.astype(np.uint8) * 255))
        axes[2].imshow(combined_skel, cmap='gray')
        axes[2].set_title('1px Skeletons\n(Left: GT | Right: Pred)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        h, w = pred_skeleton.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :, 1] = gt_skeleton.astype(np.uint8) * 255
        overlay[:, :, 0] = pred_skeleton.astype(np.uint8) * 255
        axes[3].imshow(overlay)
        axes[3].set_title(
            f'Overlay Analysis\n'
            f'F1@2px: {raw_metrics.get("f1@2px", 0):.3f} | '
            f'clDice: {raw_metrics.get("clDice", 0):.3f} | '
            f'IoU: {raw_metrics.get("iou", 0):.3f}',
            fontsize=14,
            fontweight='bold',
            color='darkblue',
        )
        axes[3].axis('off')
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='GT'),
            Patch(facecolor='red', edgecolor='black', label='Pred'),
            Patch(facecolor='yellow', edgecolor='black', label='Match'),
        ]
        axes[3].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(panels_dir, f'{image_id}_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if mosaic_data:
        n = len(mosaic_data)
        n_cols = 4
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = np.array(axes).flatten()
        for i, data in enumerate(mosaic_data):
            h, w = data['pred_skeleton'].shape
            ov = np.zeros((h, w, 3), dtype=np.uint8)
            ov[:, :, 1] = data['gt_skeleton'].astype(np.uint8) * 255
            ov[:, :, 0] = data['pred_skeleton'].astype(np.uint8) * 255
            axes[i].imshow(ov)
            axes[i].set_title(
                f'[{data["image_id"]}]\n'
                f'clDice: {data["metrics"].get("clDice", 0):.3f} | '
                f'IoU: {data["metrics"].get("iou", 0):.3f}\n'
                f'F1@2px: {data["metrics"].get("f1@2px", 0):.3f}',
                fontsize=9,
                fontweight='bold',
            )
            axes[i].axis('off')
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mosaic_overview.png'), dpi=200, bbox_inches='tight')
        plt.close()

    write_eval_csvs(output_dir, all_metrics)
    f1 = np.mean([m['f1@2px'] for m in all_metrics]) if all_metrics else float('nan')
    print(f'\n[frangi/{split}] {len(all_metrics)} imgs  f1@2px={f1:.4f}  → {output_dir}')
    return all_metrics


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='store_true', help='Evaluate on the val set')
    ap.add_argument('--test', action='store_true', help='Evaluate on the test datasets')
    args = ap.parse_args()
    if not args.eval and not args.test:  # no flag → run both
        args.eval = args.test = True
    if args.eval:
        evaluate('val')
    if args.test:
        for name in TEST_DATASETS:
            evaluate(name)
