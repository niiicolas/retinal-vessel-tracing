"""Seed detector trainer — multi-task with heavy augmentation.

Pairs with `models.seed_detector.SeedDetector` (Attention U-Net + 5 heads
+ aux deep-supervision heads).

Key differences from the v2/v3 trainers (ignored on purpose):
  * Five-task target stack (centerline tube, vessel mask, radius map,
    orientation map, FOV mask) built on the *augmented* image so geometric
    transforms stay consistent.
  * AdamW + linear warmup + cosine decay over the full schedule.
  * Augmentations: 90° rotations, flips, mild affine, colour jitter, CLAHE-
    free, Gaussian noise, gamma — matches the cross-dataset domain shift
    between FIVES/STARE/CHASEDB1/HRF/LES-AV and the test sets DRIVE/DRHAGIS.
  * Optional Frangi auxiliary channel precomputed on the augmented green
    channel (the model expects it when ``use_frangi_input`` is on).
  * Best-checkpoint selection by validation centerline F1 (peak-recall) so
    we save the model that's actually best at *seed placement*, not just at
    minimising the multi-task loss.
"""

from __future__ import annotations

import math
import os
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.seed_detector import (
    SeedDetector,
    build_centerline_tube,
    build_radius_map,
    build_orientation_map,
    frangi_vesselness,
    seed_detector_loss,
)


# ===========================================================================
# Augmented multi-task dataset
# ===========================================================================


class SeedDataset(Dataset):
    """On-the-fly augmented dataset with multi-task target generation.

    All geometric augmentations are applied BEFORE building targets, so the
    centerline tube, radius map and orientation map are all in the same
    augmented frame as the image. Photometric augmentations touch the image
    only.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        sigma: float = 1.5,
        resize: Tuple[int, int] = (512, 512),
        augment: bool = False,
        use_frangi_input: bool = True,
    ):
        self.samples = samples
        self.sigma = sigma
        self.resize = resize
        self.augment = augment
        self.use_frangi_input = use_frangi_input

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def _augment(
        self,
        img: np.ndarray,
        vessel: np.ndarray,
        fov: np.ndarray,
        frangi: Optional[np.ndarray] = None,
    ):
        """Geometric + photometric augmentation kept consistent across
        image / vessel / fov (and the cached Frangi channel, if given).

        Photometric transforms are applied to the image only — Frangi is a
        purely geometric vessel response so it should NOT be re-photometried,
        otherwise the cached value would drift away from "Frangi on the
        clean image". Geometric augs (rotation / flip / affine) are applied
        identically to the Frangi map so it stays spatially registered with
        image/vessel/fov.
        """
        # 90° rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k, axes=(0, 1)).copy()
            vessel = np.rot90(vessel, k, axes=(0, 1)).copy()
            fov = np.rot90(fov, k, axes=(0, 1)).copy()
            if frangi is not None:
                frangi = np.rot90(frangi, k, axes=(0, 1)).copy()
        # Flips
        if np.random.rand() < 0.5:
            img, vessel, fov = (np.fliplr(a).copy() for a in (img, vessel, fov))
            if frangi is not None:
                frangi = np.fliplr(frangi).copy()
        if np.random.rand() < 0.5:
            img, vessel, fov = (np.flipud(a).copy() for a in (img, vessel, fov))
            if frangi is not None:
                frangi = np.flipud(frangi).copy()
        # Small in-plane rotation (±15°) + minor scale
        if np.random.rand() < 0.5:
            h, w = img.shape[:2]
            angle = np.random.uniform(-15.0, 15.0)
            scale = np.random.uniform(0.92, 1.08)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            img = cv2.warpAffine(
                img,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            vessel = cv2.warpAffine(
                vessel,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            fov = cv2.warpAffine(
                fov,
                M,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            if frangi is not None:
                frangi = cv2.warpAffine(
                    frangi,
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        # Photometric — image only (NEVER Frangi)
        if np.random.rand() < 0.8:
            img = np.clip(
                img + np.random.uniform(-0.12, 0.12),
                0.0,
                1.0,
            )
        if np.random.rand() < 0.8:
            f = np.random.uniform(0.7, 1.3)
            mu = img.mean()
            img = np.clip((img - mu) * f + mu, 0.0, 1.0)
        if np.random.rand() < 0.5:
            for c in range(3):
                img[:, :, c] = np.clip(
                    img[:, :, c] + np.random.uniform(-0.08, 0.08),
                    0,
                    1,
                )
        if np.random.rand() < 0.5:
            gamma = np.random.uniform(0.7, 1.4)
            img = np.clip(img**gamma, 0.0, 1.0)
        if np.random.rand() < 0.3:
            noise = np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.01, 0.04)
            img = np.clip(img + noise, 0.0, 1.0)
        return img, vessel, fov, frangi

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = s['image'].copy()
        vessel = s['vessel_mask'].copy()
        fov = s['fov_mask'].copy()
        # Cached Frangi (precomputed in load_samples). If missing and the
        # caller asked for the Frangi input channel, compute it lazily —
        # but warn-once because that's the slow path that caused the 12 h
        # SLURM timeout.
        frangi = s.get('frangi')
        if self.use_frangi_input and frangi is None:
            if not getattr(
                SeedDataset,
                '_warned_no_cache',
                False,
            ):
                print(
                    'WARN: SeedDataset received samples without a '
                    "precomputed 'frangi' key — falling back to per-getitem "
                    'Frangi (slow). Precompute it in load_samples().',
                    flush=True,
                )
                SeedDataset._warned_no_cache = True
            frangi = frangi_vesselness(img[..., 1])
        elif frangi is not None:
            frangi = frangi.copy()

        if self.augment:
            img, vessel, fov, frangi = self._augment(img, vessel, fov, frangi)

        if self.resize is not None:
            th, tw = self.resize
            img = cv2.resize(
                img,
                (tw, th),
                interpolation=cv2.INTER_LINEAR,
            )
            vessel = cv2.resize(
                vessel.astype(np.float32),
                (tw, th),
                interpolation=cv2.INTER_NEAREST,
            )
            fov = cv2.resize(
                fov.astype(np.float32),
                (tw, th),
                interpolation=cv2.INTER_NEAREST,
            )
            if frangi is not None:
                frangi = cv2.resize(
                    frangi,
                    (tw, th),
                    interpolation=cv2.INTER_LINEAR,
                )

        from skimage.morphology import skeletonize

        vessel_bin = vessel > 0.5
        skel = skeletonize(vessel_bin).astype(np.float32)

        cl_tube = build_centerline_tube(skel, sigma=self.sigma)
        rad_map = build_radius_map(vessel_bin)
        orient = build_orientation_map(skel)  # (2, H, W)

        if self.use_frangi_input:
            img4 = np.concatenate([img, frangi[..., None]], axis=-1)
        else:
            img4 = img

        img_t = torch.from_numpy(img4.transpose(2, 0, 1)).float()
        cl_t = torch.from_numpy(cl_tube).unsqueeze(0).float()
        ve_t = torch.from_numpy(vessel_bin.astype(np.float32)).unsqueeze(0).float()
        rad_t = torch.from_numpy(rad_map).unsqueeze(0).float()
        or_t = torch.from_numpy(orient).float()  # (2, H, W) already
        fov_t = torch.from_numpy(fov.astype(np.float32)).unsqueeze(0).float()

        return (
            img_t,
            cl_t,
            ve_t,
            rad_t,
            or_t,
            fov_t,
        )


# ===========================================================================
# Trainer
# ===========================================================================


class SeedDetectorTrainer:
    """Multi-task seed-detector trainer with cosine schedule + deep super.

    Best-checkpoint selection uses a validation centerline F1@2px against
    the GT skeleton (a proxy for downstream RL success) — not raw loss —
    so we save the checkpoint that best places seed candidates.
    """

    def __init__(
        self,
        model: SeedDetector,
        device: torch.device,
        lr: float = 1e-3,
        batch_size: int = 4,
        num_epochs: int = 100,
        warmup_epochs: int = 5,
        sigma: float = 1.5,
        use_frangi_input: bool = True,
        num_workers: int = 2,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.sigma = sigma
        self.use_frangi_input = use_frangi_input
        self.num_workers = num_workers
        self.max_grad_norm = max_grad_norm
        self._last_loss_components: Dict[str, float] = {}

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = None  # built in train() once dataset is known

    # ------------------------------------------------------------------
    def train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        save_path: str,
        config: Dict[str, Any],
    ):
        os.makedirs(
            os.path.dirname(save_path) or '.',
            exist_ok=True,
        )

        train_ds = SeedDataset(
            train_samples,
            sigma=self.sigma,
            augment=True,
            use_frangi_input=self.use_frangi_input,
        )
        val_ds = SeedDataset(
            val_samples,
            sigma=self.sigma,
            augment=False,
            use_frangi_input=self.use_frangi_input,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == 'cuda',
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == 'cuda',
        )

        total_steps = self.num_epochs * max(1, len(train_loader))
        warmup_steps = self.warmup_epochs * max(1, len(train_loader))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        print(
            f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}',
            flush=True,
        )
        print(
            f'Epochs: {self.num_epochs} (warmup {self.warmup_epochs})  '
            f'sigma={self.sigma}  frangi={self.use_frangi_input}  '
            f'num_workers={self.num_workers}',
            flush=True,
        )

        best_metric = -float('inf')
        train_start = time.time()
        for epoch in range(1, self.num_epochs + 1):
            t_epoch = time.time()
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            val_f1 = self._val_seed_f1(val_samples)

            lr_now = self.optimizer.param_groups[0]['lr']
            dt = time.time() - t_epoch
            eta_h = (self.num_epochs - epoch) * dt / 3600.0
            print(
                f'Epoch {epoch:3d}/{self.num_epochs}  '
                f'train={train_loss:.4f}  val={val_loss:.4f}  '
                f'seed_f1={val_f1:.3f}  lr={lr_now:.2e}  '
                f'dt={dt:.1f}s  ETA={eta_h:.2f}h',
                flush=True,
            )

            if val_f1 > best_metric:
                best_metric = val_f1
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'seed_f1': val_f1,
                        'config': config,
                    },
                    save_path,
                )
                print(
                    f'  ✓ Saved (seed_f1={val_f1:.3f})',
                    flush=True,
                )

        total_h = (time.time() - train_start) / 3600.0
        print(
            f'\nDone in {total_h:.2f}h. Best val seed_f1={best_metric:.3f}  →  {save_path}',
            flush=True,
        )

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        total = 0.0
        comp_sums: Dict[str, float] = {}
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for (
                img,
                cl,
                ve,
                ra,
                ori,
                fov,
            ) in loader:
                img = img.to(self.device)
                cl = cl.to(self.device)
                ve = ve.to(self.device)
                ra = ra.to(self.device)
                ori = ori.to(self.device)
                fov = fov.to(self.device)

                preds = self.model(img, return_aux=True)
                loss, comps = seed_detector_loss(
                    preds,
                    {
                        'centerline_tube': cl,
                        'vessel_mask': ve,
                        'radius_map': ra,
                        'orient_map': ori,
                    },
                    fov_mask=fov,
                )
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total += float(loss.item())
                for k, v in comps.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v

        n = max(1, len(loader))
        self._last_loss_components = {k: v / n for k, v in comp_sums.items()}
        self._last_loss_components['phase'] = 'train' if train else 'val'
        return total / n

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_seed_f1(
        self,
        val_samples: List[Dict[str, Any]],
        n_images: int = 8,
        tol_prec: int = 3,
        tol_recall: int = 20,
    ) -> float:
        """Real F1 for best-checkpoint selection.

        Precision (tight tol): fraction of predicted seeds whose pixel lies
        within ``tol_prec`` of the GT centerline. Penalises off-vessel and
        clustered-on-OD seeds.

        Recall (loose tol): fraction of GT-centerline pixels that have AT
        LEAST ONE predicted seed within ``tol_recall``. Penalises sparse /
        peripheral-vessel-starved seed sets — the failure mode the prior
        precision-only metric was blind to (it saturated at ≈0.97 across
        every reasonable checkpoint and gave no discrimination signal).

        F1 = 2·P·R / (P + R).
        """
        import cv2 as _cv2

        self.model.eval()
        tp_prec = fp_prec = 0
        gt_total = 0
        gt_covered = 0
        for s in val_samples[:n_images]:
            img_np = s['image']
            fov_np = s['fov_mask']
            cl_np = (s['centerline'] > 0).astype(np.uint8)
            img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
            fov_t = torch.from_numpy(fov_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            seeds, _, _ = self.model.detect_seeds(img_t, fov_mask=fov_t)
            seeds = seeds[0]

            gt_total += int(cl_np.sum())
            if not seeds:
                continue

            # Precision: seeds inside GT-skel dilated by tol_prec
            k_p = 2 * tol_prec + 1
            cl_dil_p = _cv2.dilate(
                cl_np,
                np.ones((k_p, k_p), np.uint8),
            )
            hits = sum(1 for y, x, _ in seeds if cl_dil_p[y, x] > 0)
            tp_prec += hits
            fp_prec += len(seeds) - hits

            # Recall: GT-skel pixels with a seed within tol_recall
            seed_mask = np.zeros_like(cl_np)
            for y, x, _ in seeds:
                if 0 <= y < cl_np.shape[0] and 0 <= x < cl_np.shape[1]:
                    seed_mask[y, x] = 1
            k_r = 2 * tol_recall + 1
            seed_dil = _cv2.dilate(
                seed_mask,
                np.ones((k_r, k_r), np.uint8),
            )
            gt_covered += int(((cl_np > 0) & (seed_dil > 0)).sum())

        prec = tp_prec / max(1, tp_prec + fp_prec)
        recall = gt_covered / max(1, gt_total)
        f1 = 2.0 * prec * recall / max(1e-6, prec + recall)
        return f1
