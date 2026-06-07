"""Unified dataloader for retinal vessel segmentation.

Datasets
--------
Training / validation (combined, balanced):
    FIVES, STARE, CHASE_DB1, HRF, LES_AV

External test (used in full, no split):
    DRIVE, DR_HAGIS

Supported targets
-----------------
unet           – (1,H,W) CLAHE-preprocessed grayscale + skeleton GT
frangi         – (H,W,3) raw RGB uint8 + binary annotations (numpy)
greedy_tracer  – (H,W,3) raw RGB uint8 + binary annotations (numpy)
rl_agent       – (3,H,W) float32 RGB + centerline, distance transform, …

Usage
-----
    from data.dataloader import get_data, get_test_data

    # Training & validation (balanced across 5 datasets)
    train_ds, train_loader = get_data("unet", "train", batch_size=4)
    val_ds,   val_loader   = get_data("unet", "val",   batch_size=1)

    # External test sets (entire dataset, no split)
    test_ds, test_loader = get_test_data("AV_WIDE",  "unet", batch_size=1)
    test_ds, test_loader = get_test_data("DR_HAGIS", "unet", batch_size=1)
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
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
from PIL import Image
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from .centerline_extraction import (
    CenterlineExtractor,
)
from .fundus_preprocessor import (
    FundusPreprocessor,
)
from environment.observation import (
    ObservationBuilder,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset root resolution
# ---------------------------------------------------------------------------
_DATA_BASE = Path('/cfs/earth/scratch/icls/shared/icls-retinal-vessel-tracing/retinal-vessel-tracing/data')

_PROJECT_ROOT = _DATA_BASE.parent

# When RVT_RUN_NAME is set, namespace all weights/results under a per-run
# subdirectory so concurrent ablation jobs don't clobber each other's
# checkpoints and CSV logs. When unset, behaviour is unchanged.
_RUN_NAME = os.environ.get('RVT_RUN_NAME', '').strip()
if _RUN_NAME:
    WEIGHTS_DIR = _PROJECT_ROOT / 'weights' / _RUN_NAME
    OUTPUT_DIR = _PROJECT_ROOT / 'results' / _RUN_NAME
else:
    WEIGHTS_DIR = _PROJECT_ROOT / 'weights'
    OUTPUT_DIR = _PROJECT_ROOT / 'results'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 5

# ---------------------------------------------------------------------------
# Centerline-probability predictor — lazy singleton.
#
# Source: the *multi-task* SeedDetector's centerline_prob head, loaded from
# ``weights/seed_detector.pt``. The RL pipeline used to pull this signal
# from a separate standalone CenterlineUNet (weights/centerline_unet.pt) for
# historical reasons — that checkpoint is the thesis comparison baseline
# (frangi vs greedy vs unet vs RL) and remains untouched. The RL prior
# now reuses the seed detector's already-trained centerline head so we
# don't maintain two centerline models.
# ---------------------------------------------------------------------------
_UNET_PREDICTOR = None
_UNET_LOAD_FAILED = False


def _get_unet_predictor():
    """Return a cached SeedDetector (eval mode) or None if checkpoint missing.

    Loaded once per process. The returned model is the *frozen* multi-task
    seed detector; the RL pipeline only consumes its centerline_prob head.
    """
    global _UNET_PREDICTOR, _UNET_LOAD_FAILED
    if _UNET_PREDICTOR is not None or _UNET_LOAD_FAILED:
        return _UNET_PREDICTOR
    try:
        import torch as _torch

        from models.seed_detector import (
            SeedDetector,
        )

        ckpt_path = WEIGHTS_DIR / 'seed_detector.pt'
        if not ckpt_path.exists():
            logger.warning(
                'Seed detector checkpoint missing at %s — UNet-derived '
                'observation priors are unavailable. Train it via '
                'scripts/train_seed_detector.py.',
                ckpt_path,
            )
            _UNET_LOAD_FAILED = True
            return None
        device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        ckpt = _torch.load(
            str(ckpt_path),
            map_location=device,
            weights_only=True,
        )
        model = SeedDetector().to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        _UNET_PREDICTOR = model
        logger.info(
            'Loaded seed detector (used as RL centerline prior) from %s (device=%s)',
            ckpt_path,
            device,
        )
        return _UNET_PREDICTOR
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            'Failed to load seed detector for UNet prior: %s',
            exc,
        )
        _UNET_LOAD_FAILED = True
        return None


def compute_unet_prior(
    image: np.ndarray,
) -> Optional[np.ndarray]:
    """Run the frozen seed-detector and return its centerline probability map.

    ``image`` is the (H, W, 3) RGB float32 array produced by ``_fmt_rl_agent``.
    The seed detector accepts 3-channel RGB and internally prepends a Frangi
    vesselness channel when its ``use_frangi_input`` flag is on (the default).
    Returns (H, W) float32 in [0, 1], or None if the checkpoint is unavailable.
    """
    predictor = _get_unet_predictor()
    if predictor is None:
        return None
    import torch as _torch

    if image.ndim == 2:
        # Single-channel input — fake an RGB by triplicating.
        rgb = np.stack([image, image, image], axis=-1)
    else:
        rgb = image
    img_t = (
        _torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    )
    device = next(predictor.parameters()).device
    img_t = img_t.to(device)
    with _torch.no_grad():
        # SeedDetector.forward returns (centerline, vessel, radius, orient, logvar)
        centerline_prob = predictor.forward(img_t, return_aux=False)[0]
    return centerline_prob.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def compute_predicted_priors(
    image: np.ndarray,
    tolerance: float,
    threshold: float = 0.5,
    centerline_extractor: Optional['CenterlineExtractor'] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Derive observation-channel geometry from the frozen centerline UNet.

    Pipeline: UNet probability → threshold → skeletonise → DT (clipped at
    ``tolerance``) → DT gradient. This is the only path producing the
    centerline/DT/DT-gradient that go into the RL observation; the GT
    skeleton is reserved for reward and coverage computation.

    Returns a dict with keys {centerline, distance_transform, dt_gradient,
    unet_prior} on success, or ``None`` if the centerline UNet checkpoint
    is unavailable. Callers MUST treat None as an error — falling back to
    GT here would re-introduce the leakage this pipeline exists to prevent.
    """
    prob = compute_unet_prior(image)
    if prob is None:
        return None
    pred_vessel = (prob > threshold).astype(np.float32)
    extractor = centerline_extractor or CenterlineExtractor()
    pred_cl = extractor.extract_centerline(pred_vessel)
    pred_dt = extractor.compute_distance_transform(pred_cl, tolerance)
    pred_dt_grad = ObservationBuilder.compute_dt_gradient(pred_dt)
    return {
        'centerline': pred_cl.astype(np.float32, copy=False),
        'distance_transform': pred_dt.astype(np.float32, copy=False),
        'dt_gradient': pred_dt_grad.astype(np.float32, copy=False),
        'unet_prior': prob.astype(np.float32, copy=False),
    }


def get_root(dataset_name: str) -> Path:
    """Return the root directory for a dataset."""
    canon = dataset_name.upper()
    if canon not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Known: {sorted(set(DATASET_REGISTRY))}")

    env_key = f'RETINAL_DATA_{canon}'
    if env_key in os.environ:
        root = Path(os.environ[env_key])
    else:
        root = _DATA_BASE / canon

    if not root.is_dir():
        raise FileNotFoundError(f'Dataset root for {canon} does not exist: {root}\n')
    return root


# ---------------------------------------------------------------------------
# Dataset layout descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetConfig:
    """File-system layout for one retinal fundus dataset."""

    image_dir: str
    vessel_dir: str
    image_glob: str
    vessel_suffix: str
    mask_dir: Optional[str] = None
    mask_suffix: Optional[str] = None
    stem_rule: Optional[str] = None
    train_subdir: Optional[str] = None

    def vessel_filename(self, image_stem: str) -> str:
        stem = self._transform_stem(image_stem)
        return f'{stem}{self.vessel_suffix}'

    def mask_filename(self, image_stem: str) -> str:
        if self.mask_suffix is None:
            raise ValueError('No mask_suffix configured.')
        return f'{image_stem}{self.mask_suffix}'

    def _transform_stem(self, stem: str) -> str:
        if self.stem_rule == 'drive':
            return stem.replace('_training', '_manual1').replace('_test', '_manual1')
        return stem


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    'DRIVE': DatasetConfig(
        image_dir='images',
        vessel_dir='1st_manual',
        image_glob='*.tif',
        vessel_suffix='.gif',
        mask_dir='mask',
        mask_suffix='_mask.gif',
        stem_rule='drive',
        train_subdir='training',
    ),
    'STARE': DatasetConfig(
        image_dir='.',
        vessel_dir='.',
        image_glob='*.ppm',
        vessel_suffix='.vk.ppm',
    ),
    'CHASEDB1': DatasetConfig(
        image_dir='.',
        vessel_dir='.',
        image_glob='*.jpg',
        vessel_suffix='_1stHO.png',
    ),
    'HRF': DatasetConfig(
        image_dir='images',
        vessel_dir='manual1',
        image_glob='*.[jJ][pP][gG]',
        vessel_suffix='.tif',
        mask_dir='mask',
        mask_suffix='_mask.tif',
    ),
    'LES-AV': DatasetConfig(
        image_dir='images',
        vessel_dir='vessel-segmentations',
        image_glob='*.png',
        vessel_suffix='.png',
        mask_dir='masks',
        mask_suffix='_mask.gif',
    ),
    'DRHAGIS': DatasetConfig(
        image_dir='Fundus_Images',
        vessel_dir='Manual_Segmentations',
        image_glob='*.jpg',
        vessel_suffix='_manual_orig.png',
        mask_dir='Mask_images',
        mask_suffix='_mask_orig.png',
    ),
    'FIVES': DatasetConfig(
        image_dir='Original',
        vessel_dir='Ground truth',
        image_glob='*.png',
        vessel_suffix='.png',
        train_subdir='train',
    ),
}


TRAIN_DATASETS = (
    'FIVES',
    'STARE',
    'CHASEDB1',
    'HRF',
    'LES-AV',
)
TEST_DATASETS = ('DRIVE', 'DRHAGIS')
VALID_TARGETS = (
    'unet',
    'frangi',
    'greedy_tracer',
    'rl_agent',
)


# ---------------------------------------------------------------------------
# Collate for numpy-dict targets (frangi / greedy_tracer)
# ---------------------------------------------------------------------------
def _list_collate(batch: list) -> list:
    return batch


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------
class RetinalFundusDataset(Dataset):
    """PyTorch Dataset for a single retinal fundus dataset.

    Parameters
    ----------
    root_dir      : top-level dataset directory (e.g. "data/DRIVE")
    dataset_name  : key in DATASET_REGISTRY
    target        : output format ("unet", "frangi", "greedy_tracer", "rl_agent")
    split         : "train", "val", or None (= return all samples)
    train_frac    : fraction of samples used for training (rest → val)
    resize        : (H, W) to resize all images/masks, or None
    tolerance     : distance-transform radius for rl_agent target
    cache_centerlines : persist skeletons to disk
    transform     : optional albumentations pipeline (unet target only)
    preprocessor  : shared FundusPreprocessor instance
    centerline_extractor : shared CenterlineExtractor instance

    """

    def __init__(
        self,
        root_dir: str,
        dataset_name: str,
        target: str = 'rl_agent',
        split: Optional[str] = None,
        train_frac: float = 0.8,
        resize: Optional[Tuple[int, int]] = None,
        tolerance: float = 2.0,
        cache_centerlines: bool = True,
        transform=None,
        fundus_preprocessor: Optional[FundusPreprocessor] = None,
        centerline_extractor: Optional[CenterlineExtractor] = None,
        max_samples: Optional[int] = None,
        use_unet_prior: bool = False,
        require_predicted_priors: bool = True,
    ):
        if target not in VALID_TARGETS:
            raise ValueError(f"target must be one of {VALID_TARGETS}, got '{target}'")

        canon = dataset_name.upper()
        if canon not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Known: {sorted(set(DATASET_REGISTRY))}")

        self.dataset_name = canon
        self.cfg = DATASET_REGISTRY[canon]
        self.target = target
        self.resize = resize
        self.tolerance = tolerance  # fallback only; per-sample value is width-scaled
        self.transform = transform
        self.fundus_preprocessor = fundus_preprocessor or FundusPreprocessor()
        self.cl_extractor = centerline_extractor or CenterlineExtractor()
        self.use_unet_prior = use_unet_prior
        # Predicted-prior derivation depends on the seed detector checkpoint.
        # The trainers that PRODUCE that checkpoint (scripts/train_seed_detector,
        # scripts/run_cnn for the standalone UNet baseline) can't depend on it
        # being present yet — they set this to False so _fmt_rl_agent skips
        # the prior bundle. RL-pipeline callers (PPO trainer, imitation,
        # inference) keep the default True so leakage stays removed.
        self.require_predicted_priors = require_predicted_priors

        # Resolve root directory
        self.root = self._resolve_root(Path(root_dir))

        # Discover and split samples
        self.samples = self._discover_samples()
        if split is not None:
            self.samples = self._apply_split(self.samples, split, train_frac)

        # Cap samples
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError(
                f'No samples found for {canon} in {self.root}. '
                f"Expected images in '{self.cfg.image_dir}/' matching "
                f"'{self.cfg.image_glob}' with vessels in '{self.cfg.vessel_dir}/'."
            )

        # Centerline cache — stores (skeleton, vessel_width_px) per sample
        self._cl_mem: Dict[str, Tuple[np.ndarray, float]] = {}
        self._cache_dir: Optional[Path] = None
        if cache_centerlines and self.resize is None:
            self._cache_dir = self.root / 'centerlines_cache'
            self._cache_dir.mkdir(exist_ok=True)

        self._vo_mem: Dict[str, np.ndarray] = {}
        self._up_mem: Dict[str, np.ndarray] = {}  # UNet prior cache
        # Predicted-prior bundle cache (centerline / DT / DT-grad / UNet prob,
        # all derived from the frozen UNet — the non-leaking replacement for
        # GT geometry in the RL observation channels).
        self._pp_mem: Dict[str, Dict[str, np.ndarray]] = {}

        logger.info(
            '%s  %d samples  target=%s  split=%s',
            canon,
            len(self.samples),
            target,
            split,
        )

    # -- Root resolution ---------------------------------------------------
    def _resolve_root(self, root_dir: Path) -> Path:
        """Find the directory that actually contains images."""
        if self.cfg.train_subdir is not None:
            subdir = root_dir / self.cfg.train_subdir
            if subdir.is_dir():
                return subdir
            if root_dir.name == self.cfg.train_subdir:
                return root_dir  # user passed the subdir directly
        return root_dir

    # -- Sample discovery --------------------------------------------------
    def _discover_samples(
        self,
    ) -> List[Dict[str, Any]]:
        image_dir = self.root / self.cfg.image_dir
        vessel_dir = self.root / self.cfg.vessel_dir
        mask_dir = (self.root / self.cfg.mask_dir) if self.cfg.mask_dir else None

        samples: List[Dict[str, Any]] = []
        for img_path in sorted(image_dir.glob(self.cfg.image_glob)):
            vessel_path = vessel_dir / self.cfg.vessel_filename(img_path.stem)
            if not vessel_path.exists():
                continue

            entry: Dict[str, Any] = {
                'id': img_path.stem,
                'image': img_path,
                'vessel': vessel_path,
            }
            if mask_dir is not None and self.cfg.mask_suffix is not None:
                mask_path = mask_dir / self.cfg.mask_filename(img_path.stem)
                if mask_path.exists():
                    entry['mask'] = mask_path

            samples.append(entry)
        return samples

    # -- Train/val split ---------------------------------------------------
    @staticmethod
    def _apply_split(
        samples: List[Dict],
        split: str,
        train_frac: float,
    ) -> List[Dict]:
        """Deterministic train/val split (sorted filenames → reproducible)."""
        n = len(samples)
        t = max(1, min(int(train_frac * n), n - 1))
        if split == 'train':
            return samples[:t]
        elif split == 'val':
            return samples[t:]
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    # -- I/O ---------------------------------------------------------------
    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert('RGB'))

    @staticmethod
    def _load_gray(path: Path) -> np.ndarray:
        return np.array(Image.open(path).convert('L'))

    def _load_vessel(self, path: Path) -> np.ndarray:
        return (self._load_gray(path) > 127).astype(np.float32)

    def _load_fov(self, path: Path) -> np.ndarray:
        return (self._load_gray(path) > 127).astype(np.uint8) * 255

    # -- Cache helpers -----------------------------------------------------
    @staticmethod
    def _cache_is_fresh(
        cache_path: Path,
        source_path: Optional[Path],
    ) -> bool:
        """True iff cache file is newer than the source it was derived from.

        Mask regenerations are silent: with no mtime guard the dataloader
        was happy to keep serving a stale skeleton forever. A cache file
        whose mtime is older than its source is treated as invalid and
        recomputed.
        """
        if not cache_path.exists():
            return False
        if source_path is None or not source_path.exists():
            return True  # no source to compare against — trust the cache
        return cache_path.stat().st_mtime >= source_path.stat().st_mtime

    # -- Centerline cache --------------------------------------------------
    # v2: cache stores (skeleton, vessel_width_px) together so per-image
    # width-aware tolerance / pruning thresholds survive reload. v1 .npy
    # files from the old code are ignored (skeleton-only, no width).
    _CL_CACHE_VERSION = 'v2'

    def _get_centerline(
        self,
        sid: str,
        vessel: np.ndarray,
        source_path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, float]:
        """Return (skeleton, vessel_width_px) for sample ``sid``.

        ``source_path`` is the on-disk vessel-mask file; used as the mtime
        reference for cache freshness. Pass ``None`` to skip the freshness
        check (e.g. when the mask was generated on the fly).
        """
        if sid in self._cl_mem:
            return self._cl_mem[sid]
        if self._cache_dir is not None:
            cache_file = self._cache_dir / f'{sid}_cl_{self._CL_CACHE_VERSION}.npz'
            if self._cache_is_fresh(cache_file, source_path):
                with np.load(cache_file) as data:
                    cl = data['skeleton']
                    w = float(data['width_px'])
                self._cl_mem[sid] = (cl, w)
                return cl, w
        cl = self.cl_extractor.extract_centerline(vessel)
        from .centerline_extraction import (
            compute_vessel_width,
        )

        w = compute_vessel_width(vessel)
        self._cl_mem[sid] = (cl, w)
        if self._cache_dir is not None:
            np.savez(
                self._cache_dir / f'{sid}_cl_{self._CL_CACHE_VERSION}.npz',
                skeleton=cl,
                width_px=np.float32(w),
            )
        return cl, w

    def _get_vessel_orientation(
        self,
        sid: str,
        image: np.ndarray,
        source_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Return cached vessel orientation, computing + persisting on first call."""
        if sid in self._vo_mem:
            return self._vo_mem[sid]
        if self._cache_dir is not None:
            cache_file = self._cache_dir / f'{sid}_vessel_orientation.npy'
            if self._cache_is_fresh(cache_file, source_path):
                vo = np.load(cache_file)
                self._vo_mem[sid] = vo
                return vo
        vo = ObservationBuilder.compute_vessel_orientation(image)
        self._vo_mem[sid] = vo
        if self._cache_dir is not None:
            np.save(
                self._cache_dir / f'{sid}_vessel_orientation.npy',
                vo,
            )
        return vo

    def _get_unet_prior(
        self,
        sid: str,
        image: np.ndarray,
        source_path: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """Return cached UNet prior, computing + persisting on first call."""
        if sid in self._up_mem:
            return self._up_mem[sid]
        if self._cache_dir is not None:
            cache_file = self._cache_dir / f'{sid}_unet_prior.npy'
            if self._cache_is_fresh(cache_file, source_path):
                up = np.load(cache_file)
                self._up_mem[sid] = up
                return up
        up = compute_unet_prior(image)
        if up is None:
            return None
        self._up_mem[sid] = up
        if self._cache_dir is not None:
            np.save(
                self._cache_dir / f'{sid}_unet_prior.npy',
                up,
            )
        return up

    _PP_CACHE_VERSION = 'v1'

    def _get_predicted_priors(
        self,
        sid: str,
        image: np.ndarray,
        source_path: Optional[Path] = None,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return cached predicted-prior bundle, computing + persisting on first call.

        Bundle keys: ``centerline``, ``distance_transform``, ``dt_gradient``,
        ``unet_prior``. Returns ``None`` if the UNet checkpoint is unavailable
        (callers in the RL pipeline must surface this as a hard error).
        """
        if sid in self._pp_mem:
            return self._pp_mem[sid]
        if self._cache_dir is not None:
            cache_file = self._cache_dir / f'{sid}_pred_priors_{self._PP_CACHE_VERSION}.npz'
            if self._cache_is_fresh(cache_file, source_path):
                with np.load(cache_file) as data:
                    bundle = {
                        'centerline': data['centerline'],
                        'distance_transform': data['distance_transform'],
                        'dt_gradient': data['dt_gradient'],
                        'unet_prior': data['unet_prior'],
                    }
                self._pp_mem[sid] = bundle
                return bundle
        bundle = compute_predicted_priors(
            image,
            self.tolerance,
            centerline_extractor=self.cl_extractor,
        )
        if bundle is None:
            return None
        self._pp_mem[sid] = bundle
        if self._cache_dir is not None:
            np.savez(
                self._cache_dir / f'{sid}_pred_priors_{self._PP_CACHE_VERSION}.npz',
                **bundle,
            )
        return bundle

    # -- FOV mask ----------------------------------------------------------

    def _get_fov(self, sample: Dict, rgb: np.ndarray) -> np.ndarray:
        if 'mask' in sample:
            return self._load_fov(sample['mask'])
        green = self.fundus_preprocessor.extract_green_channel(rgb)
        if green.dtype != np.uint8:
            green = np.clip(green * 255, 0, 255).astype(np.uint8)
        return self.fundus_preprocessor.create_fov_mask(green)

    # -- Resize ------------------------------------------------------------
    def _resize(
        self,
        *arrays: np.ndarray,
        interp: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, ...]:
        if self.resize is None:
            return arrays
        target_h, target_w = self.resize
        src_h, src_w = arrays[0].shape[:2]

        # Scale to fit within target while preserving aspect ratio
        scale = min(target_h / src_h, target_w / src_w)
        new_h, new_w = (
            int(src_h * scale),
            int(src_w * scale),
        )

        # Padding offsets (center the image)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2

        out = []
        for i, arr in enumerate(arrays):
            if interp is not None and i < len(interp):
                flag = interp[i]
            elif arr.ndim == 2:
                flag = cv2.INTER_NEAREST
            else:
                flag = cv2.INTER_LINEAR

            resized = cv2.resize(
                arr,
                (new_w, new_h),
                interpolation=flag,
            )

            # Create padded output (black/zero padding)
            if arr.ndim == 3:
                padded = np.zeros(
                    (
                        target_h,
                        target_w,
                        arr.shape[2],
                    ),
                    dtype=arr.dtype,
                )
            else:
                padded = np.zeros(
                    (target_h, target_w),
                    dtype=arr.dtype,
                )

            padded[
                pad_top : pad_top + new_h,
                pad_left : pad_left + new_w,
            ] = resized
            out.append(padded)

        return tuple(out)

    # -- __len__ / __getitem__ ---------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sid = sample['id']

        rgb = self._load_rgb(sample['image'])
        vessel = self._load_vessel(sample['vessel'])
        fov = self._get_fov(sample, rgb)

        if self.resize is not None:
            rgb, vessel, fov = self._resize(
                rgb,
                vessel,
                fov,
                interp=[
                    cv2.INTER_LINEAR,
                    cv2.INTER_NEAREST,
                    cv2.INTER_NEAREST,
                ],
            )
            vessel = (vessel > 0.5).astype(np.float32)

        return getattr(self, f'_fmt_{self.target}')(sid, rgb, vessel, fov, sample)

    # -- Target formatters -------------------------------------------------
    def _fmt_unet(
        self,
        sid: str,
        rgb: np.ndarray,
        vessel: np.ndarray,
        fov: np.ndarray,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        ext_mask = fov if fov.max() > 0 else None
        preprocessed = self.fundus_preprocessor.preprocess(rgb, external_mask=ext_mask)
        # Skeleton cache is only valid at native resolution; under resize
        # the source path's mtime no longer matches the in-memory mask.
        src = sample['vessel'] if self.resize is None else None
        cl, vessel_width_px = self._get_centerline(sid, vessel, source_path=src)
        fov_f = (fov > 0).astype(np.float32)

        if self.transform is not None:
            img_u8 = np.clip(preprocessed * 255, 0, 255).astype(np.uint8)
            assert img_u8.shape == cl.shape == fov_f.shape == vessel.shape, (
                f'Shape mismatch: img={img_u8.shape} cl={cl.shape} fov={fov_f.shape} vessel={vessel.shape}'
            )
            aug = self.transform(
                image=img_u8,
                mask=cl,
                fov=fov_f,
                thick_gt=vessel,
            )
            preprocessed = aug['image'].astype(np.float32) / 255.0
            cl = aug['mask']
            fov_f = aug['fov']
            vessel = aug['thick_gt']

        return {
            'id': sid,
            'image': torch.from_numpy(preprocessed).unsqueeze(0).float(),
            'centerline': torch.from_numpy(cl).unsqueeze(0).float(),
            'vessel_mask': torch.from_numpy(vessel).unsqueeze(0).float(),
            'fov_mask': torch.from_numpy(fov_f).unsqueeze(0).float(),
            'vessel_width_px': float(vessel_width_px),
        }

    def _fmt_frangi(self, sid, rgb, vessel, fov, sample):
        ext_mask = fov if fov.max() > 0 else None
        preprocessed = self.fundus_preprocessor.preprocess(rgb, external_mask=ext_mask)
        src = sample['vessel'] if self.resize is None else None
        cl, vessel_width_px = self._get_centerline(sid, vessel, source_path=src)
        return {
            'id': sid,
            'image': rgb,
            'preprocessed': preprocessed,
            'vessel_mask': (vessel * 255).astype(np.uint8),
            'centerline': (cl * 255).astype(np.uint8),
            'fov_mask': fov,
            'vessel_width_px': float(vessel_width_px),
        }

    def _fmt_greedy_tracer(
        self,
        sid: str,
        rgb: np.ndarray,
        vessel: np.ndarray,
        fov: np.ndarray,
        sample,
    ) -> Dict[str, Any]:
        return self._fmt_frangi(sid, rgb, vessel, fov, sample)

    def _fmt_rl_agent(
        self,
        sid: str,
        rgb: np.ndarray,
        vessel: np.ndarray,
        fov: np.ndarray,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        img_f = rgb.astype(np.float32) / 255.0
        img_orig = img_f.copy()

        ext_mask = fov if fov.max() > 0 else None
        enhanced_green = self.fundus_preprocessor.preprocess(rgb, external_mask=ext_mask)
        img_f[:, :, 1] = enhanced_green

        # Per-channel percentile normalization for R and B channels.
        # Green is already CLAHE-enhanced; R/B are raw [0,1] which lets the
        # optic disc's high raw intensity dominate. Clip to [2nd, 98th]
        # percentile within the FOV to suppress brightness outliers.
        fov_mask_bool = fov > 0
        for ch in [0, 2]:  # Red, Blue
            channel = img_f[:, :, ch]
            roi = channel[fov_mask_bool] if fov_mask_bool.any() else channel.ravel()
            if roi.size > 0:
                vmin, vmax = np.percentile(roi, [2.0, 98.0])
                channel = np.clip(
                    (channel - vmin) / (vmax - vmin + 1e-8),
                    0.0,
                    1.0,
                )
            img_f[:, :, ch] = channel

        src_vessel = sample['vessel'] if self.resize is None else None
        src_image = sample['image'] if self.resize is None else None
        cl, vessel_width_px = self._get_centerline(
            sid,
            vessel,
            source_path=src_vessel,
        )
        dt = self.cl_extractor.compute_distance_transform(cl, self.tolerance)
        fov_f = (fov > 0).astype(np.float32)

        # Precompute per-image arrays (vessel_orientation is disk-cached)
        vessel_orientation = self._get_vessel_orientation(sid, img_f, source_path=src_image)

        # Predicted priors (UNet → skeleton → DT → DT-grad) replace GT
        # centerline / DT in the RL observation channels. The GT versions
        # below stay around so the reward path (coverage, on-track checks,
        # terminal F-β) keeps its supervision signal.
        #
        # `require_predicted_priors` is False ONLY for the trainers that
        # produce the seed-detector / UNet checkpoint (chicken-and-egg —
        # they can't depend on the very file they're about to write). All
        # RL-pipeline callers (PPO, imitation, inference) leave it True so
        # leakage stays removed.
        if self.require_predicted_priors:
            pred = self._get_predicted_priors(sid, img_f, source_path=src_image)
            if pred is None:
                raise RuntimeError(
                    'Seed detector checkpoint is required for the RL '
                    'observation channels (centerline / DT / DT-grad / '
                    'junction are derived from its centerline-prob head — '
                    'GT-derived priors were intentionally removed to '
                    'eliminate ground-truth leakage). Train it via '
                    'scripts/train_seed_detector.py to produce '
                    f'{WEIGHTS_DIR / "seed_detector.pt"}, then retry. '
                    'NOTE: the environment.use_unet_prior flag only '
                    'controls whether the raw probability map is also '
                    'exposed as an extra observation channel — it does '
                    'NOT make the checkpoint optional, because the '
                    'predicted-prior pipeline is always on.'
                )
        else:
            pred = None

        out = {
            'id': sid,
            'image': torch.from_numpy(img_f).permute(2, 0, 1).float(),
            'image_orig': torch.from_numpy(img_orig).permute(2, 0, 1).float(),
            'vessel_mask': torch.from_numpy(vessel).unsqueeze(0).float(),
            # GT centerline / DT — for reward + coverage, never the observation.
            'centerline': torch.from_numpy(cl).unsqueeze(0).float(),
            'fov_mask': torch.from_numpy(fov_f).unsqueeze(0).float(),
            'distance_transform': torch.from_numpy(dt).unsqueeze(0).float(),
            'vessel_orientation': torch.from_numpy(vessel_orientation).float(),
            # vessel_width_px is emitted for diagnostics / future use; it
            # does NOT drive the env tolerance (which stays absolute, from
            # config). Keeping it harmless and free lets downstream tools
            # reason about per-image scale without a separate medial-axis pass.
            'vessel_width_px': float(vessel_width_px),
        }
        if pred is not None:
            # Predicted priors — fed to ObservationBuilder.prepare_stacked_sources.
            out['pred_centerline'] = torch.from_numpy(pred['centerline']).unsqueeze(0).float()
            out['pred_distance_transform'] = torch.from_numpy(pred['distance_transform']).unsqueeze(0).float()
            out['pred_dt_gradient'] = torch.from_numpy(pred['dt_gradient']).float()
        if self.use_unet_prior and pred is not None:
            out['unet_prior'] = torch.from_numpy(pred['unet_prior']).unsqueeze(0).float()
        return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_data(
    target: str = 'rl_agent',
    split: str = 'train',
    batch_size: int = 1,
    num_workers: int = 0,
    resize: Tuple[int, int] = (512, 512),
    train_frac: float = 0.8,
    balance: bool = True,
    max_samples_per_dataset: Optional[int] = MAX_SAMPLES,
    **kwargs,
) -> Tuple[ConcatDataset, DataLoader]:
    """Load the combined train/val set (DRIVE + STARE + CHASE_DB1 + HRF + LES_AV).

    For training with ``balance=True``, a WeightedRandomSampler ensures
    each dataset contributes equally per epoch despite different sizes.

    Parameters
    ----------
    target      : output format
    split       : "train" or "val"
    batch_size  : batch size
    num_workers : DataLoader workers
    resize      : (H, W) — required for cross-dataset batching
    train_frac  : fraction of each dataset used for training (rest → val)
    balance     : use inverse-frequency weighted sampling for training
    **kwargs    : forwarded to RetinalFundusDataset (transform, tolerance, …)

    Returns
    -------
    (ConcatDataset, DataLoader)

    """
    if split not in ('train', 'val'):
        raise ValueError(f"split must be 'train' or 'val', got '{split}'")

    shared_pre = kwargs.pop('fundus_preprocessor', None) or FundusPreprocessor()
    shared_ext = kwargs.pop('centerline_extractor', None) or CenterlineExtractor()

    sub_datasets: List[RetinalFundusDataset] = []
    for name in TRAIN_DATASETS:
        try:
            root = get_root(name)
        except (
            KeyError,
            FileNotFoundError,
        ) as exc:
            logger.warning('Skipping %s: %s', name, exc)
            continue
        try:
            ds = RetinalFundusDataset(
                str(root),
                name,
                target=target,
                split=split,
                train_frac=train_frac,
                resize=resize,
                fundus_preprocessor=shared_pre,
                centerline_extractor=shared_ext,
                max_samples=max_samples_per_dataset,
                **kwargs,
            )
            sub_datasets.append(ds)
        except FileNotFoundError as exc:
            logger.warning('Skipping %s: %s', name, exc)

    if not sub_datasets:
        raise FileNotFoundError(
            f'No datasets loaded. Check that at least one of {TRAIN_DATASETS} exists under the data root.'
        )

    combined = ConcatDataset(sub_datasets)
    parts = [f'{ds.dataset_name}={len(ds)}' for ds in sub_datasets]
    logger.info(
        'Combined %s: %s  total=%d',
        split,
        '  '.join(parts),
        len(combined),
    )

    # Balanced sampling for training
    sampler = None
    shuffle = False
    if split == 'train' and balance:
        weights: List[float] = []
        for ds in sub_datasets:
            w = 1.0 / len(ds)
            weights.extend([w] * len(ds))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(combined),
            replacement=True,
        )
    elif split == 'train':
        shuffle = True

    collate_fn = _list_collate if target in ('frangi', 'greedy_tracer') else None
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=target in ('unet', 'rl_agent'),
    )
    return combined, loader


def get_test_data(
    dataset_name: str,
    target: str = 'rl_agent',
    batch_size: int = 1,
    num_workers: int = 0,
    resize: Optional[Tuple[int, int]] = (
        512,
        512,
    ),
    max_samples: Optional[int] = MAX_SAMPLES,
    **kwargs,
) -> Tuple[RetinalFundusDataset, DataLoader]:
    """Load an external test dataset in full (no split).

    Parameters
    ----------
    dataset_name : "AV_WIDE" or "DR_HAGIS"
    target       : output format
    batch_size   : batch size
    num_workers  : DataLoader workers
    resize       : (H, W) or None
    **kwargs     : forwarded to RetinalFundusDataset

    Returns
    -------
    (RetinalFundusDataset, DataLoader)

    """
    root = get_root(dataset_name)
    ds = RetinalFundusDataset(
        str(root),
        dataset_name,
        target=target,
        split=None,
        resize=resize,
        max_samples=max_samples,
        **kwargs,
    )
    collate_fn = _list_collate if target in ('frangi', 'greedy_tracer') else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=target in ('unet', 'rl_agent'),
    )
    return ds, loader
