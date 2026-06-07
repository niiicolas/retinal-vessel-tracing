"""Imitation learning (behavior cloning) for the tracing policy."""

import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Discrete action indices: 0..7 are 8-neighbour moves (N, NE, E, SE, S, SW, W, NW), 8 is STOP.
STOP_ACTION = 8
N_ACTIONS = 9

DIRECTION_MAP = {
    (-1, 0): 0,
    (-1, 1): 1,
    (0, 1): 2,
    (1, 1): 3,
    (1, 0): 4,
    (1, -1): 5,
    (0, -1): 6,
    (-1, -1): 7,
}


def _remap_traces(traces, transform_fn):
    """Apply a coordinate transform to every (y, x) point of each trace (for geometric augmentation)."""
    return [[transform_fn(y, x) for y, x in trace] for trace in traces]


def augment_sample(sample: Dict, tolerance: float) -> List[Dict]:
    """Return 9 augmented copies (5 geometric + 4 brightness/contrast) of a sample; original excluded.

    Args:
        sample: dict with image, centerline, distance_transform, fov_mask, expert_traces.
        tolerance: centerline tolerance for recomputing the distance transform after geometry changes.

    Returns:
        List of augmented sample dicts in the input format.
    """
    from data.centerline_extraction import CenterlineExtractor

    img, cl, fov = (sample['image'], sample['centerline'], sample['fov_mask'])
    traces = sample['expert_traces']
    h, w = img.shape[:2]

    ext = CenterlineExtractor(min_branch_length=10, prune_iterations=5)

    def make(new_img, new_cl, new_fov, new_traces):
        new_dt = ext.compute_distance_transform(new_cl, tolerance=tolerance)
        return {'image': new_img, 'centerline': new_cl, 'distance_transform': new_dt, 'fov_mask': new_fov, 'expert_traces': new_traces}

    aug = []

    # Horizontal flip.
    aug.append(make(img[:, ::-1, :].copy(), cl[:, ::-1].copy(), fov[:, ::-1].copy(), _remap_traces(traces, lambda y, x: (y, w - 1 - x))))

    # Vertical flip.
    aug.append(make(img[::-1, :, :].copy(), cl[::-1, :].copy(), fov[::-1, :].copy(), _remap_traces(traces, lambda y, x: (h - 1 - y, x))))

    # Rotation 90° CW.
    aug.append(
        make(np.rot90(img, k=3).copy(), np.rot90(cl, k=3).copy(), np.rot90(fov, k=3).copy(), _remap_traces(traces, lambda y, x: (x, h - 1 - y)))
    )

    # Rotation 180°.
    aug.append(
        make(
            np.rot90(img, k=2).copy(), np.rot90(cl, k=2).copy(), np.rot90(fov, k=2).copy(), _remap_traces(traces, lambda y, x: (h - 1 - y, w - 1 - x))
        )
    )

    # Rotation 270° CW.
    aug.append(
        make(np.rot90(img, k=1).copy(), np.rot90(cl, k=1).copy(), np.rot90(fov, k=1).copy(), _remap_traces(traces, lambda y, x: (w - 1 - x, y)))
    )

    # Brightness/contrast jitter — geometry unchanged, so reuse the existing DT and traces.
    dt = sample['distance_transform']
    for brightness, contrast in [(0.8, 1.0), (1.2, 1.0), (1.0, 0.8), (1.0, 1.2)]:
        img_jit = np.clip(img * contrast + (brightness - 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        aug.append({'image': img_jit, 'centerline': cl, 'distance_transform': dt, 'fov_mask': fov, 'expert_traces': traces})

    return aug


def direction_to_action(dy: int, dx: int) -> int:
    """Map an (dy, dx) unit step to a movement-action index (0–7), returning -1 for off-grid steps.

    -1 must be skipped by callers; it is never STOP (index 8).
    """
    return DIRECTION_MAP.get((dy, dx), -1)


# Canonical 8-direction vectors (index-aligned with DIRECTION_MAP) for cosine matching.
_CANONICAL_DIRS = np.array(
    [
        [-1, 0],  # 0: N
        [-1, 1],  # 1: NE
        [0, 1],  # 2: E
        [1, 1],  # 3: SE
        [1, 0],  # 4: S
        [1, -1],  # 5: SW
        [0, -1],  # 6: W
        [-1, -1],  # 7: NW
    ],
    dtype=np.float64,
)
_CANONICAL_MAG = np.linalg.norm(_CANONICAL_DIRS, axis=1)


def world_to_canonical_action(world_dy: float, world_dx: float, ty: float, tx: float, prev_world_vec=None) -> int:
    """Invert the env's tangent rotation to find the action whose rotated direction matches a world step."""
    # Sign-align the tangent (mirrors _tangent_aligned_at).
    if prev_world_vec is not None:
        ref_y, ref_x = (float(prev_world_vec[0]), float(prev_world_vec[1]))
    else:
        ref_y, ref_x = (-1.0, 0.0)  # image-up default
    tmag = (ty * ty + tx * tx) ** 0.5
    if tmag < 1e-6:
        # Degenerate tangent → fall back to plain world-frame mapping.
        dy = int(np.sign(world_dy))
        dx = int(np.sign(world_dx))
        return direction_to_action(dy, dx)
    if ty * ref_y + tx * ref_x < 0.0:
        ty, tx = -ty, -tx
    # Inverse rotation: canonical = R^T @ world.
    can_dy = -ty * world_dy - tx * world_dx
    can_dx = tx * world_dy - ty * world_dx
    can_mag = (can_dy * can_dy + can_dx * can_dx) ** 0.5
    if can_mag < 1e-6:
        return -1
    cos = (_CANONICAL_DIRS[:, 0] * can_dy + _CANONICAL_DIRS[:, 1] * can_dx) / (_CANONICAL_MAG * can_mag)
    return int(np.argmax(cos))


def soft_correct_count(preds, targets, n_dirs: int = 8):
    """Count predictions correct within ±1 on the direction ring (movement actions only).

    Forgives adjacent-direction confusion (e.g. E vs NE) that is near-harmless when tracing a
    shallow-angle vessel; STOP gets no leniency (exact match only).

    Args:
        preds, targets: 1-D LongTensors of action indices.

    Returns:
        Count of soft-correct predictions (int).
    """
    exact = preds == targets
    both_move = (preds < n_dirs) & (targets < n_dirs)
    diff = (preds - targets).abs()
    circ = torch.minimum(diff, n_dirs - diff)  # circular ring distance
    adjacent = both_move & (circ <= 1)
    return int((exact | adjacent).sum().item())


def generate_expert_metadata(sample: Dict, sample_idx: int, obs_size: int, step_size: int = 1, tangent_aware: bool = False) -> List[Dict]:
    """Walk a sample's expert traces into lightweight per-step metadata for the FF dataset.

    Patches are cropped on the fly later, avoiding OOM. ``step_size`` pairs trace[i] with
    trace[i+step_size] and ±1-normalises the delta onto the 8-direction grid; ``tangent_aware``
    inverts the env's tangent rotation so labels match deployment.

    Returns:
        List of ``{sample_idx, trace_idx, step_idx, pos, action, prev_dir}`` dicts; ``trace_idx``/
        ``step_idx`` let the dataset reconstruct the visited mask for each step.
    """
    h, w = sample['image'].shape[:2]
    half = obs_size // 2
    steps_meta = []
    s = max(int(step_size), 1)

    # Tangent-aware path precomputes the orientation field once per sample.
    if tangent_aware:
        from environment.observation import ObservationBuilder

        vo = ObservationBuilder.compute_vessel_orientation(sample['image'])
    else:
        vo = None

    for trace_idx, trace in enumerate(sample.get('expert_traces', [])):
        if len(trace) < s + 1:
            continue
        last_valid = None  # (step_idx, pos, action) for the final STOP supervision
        prev_world_vec = None
        for i in range(0, len(trace) - s, s):
            y, x = trace[i]
            ny, nx = trace[i + s]

            if y < half or y >= h - half or x < half or x >= w - half:
                continue

            # ±1-normalise the stride-s delta so curved segments still land on the 8-dir grid.
            dy = int(np.sign(int(ny) - int(y)))
            dx = int(np.sign(int(nx) - int(x)))
            if tangent_aware and vo is not None:
                ty, tx = (float(vo[y, x, 0]), float(vo[y, x, 1]))
                action = world_to_canonical_action(dy, dx, ty, tx, prev_world_vec)
            else:
                action = direction_to_action(dy, dx)
            if action < 0:
                continue

            if i >= s:
                py, px = trace[i - s]
                pdy = int(np.sign(int(y) - int(py)))
                pdx = int(np.sign(int(x) - int(px)))
                if tangent_aware and vo is not None:
                    pty, ptx = (float(vo[py, px, 0]), float(vo[py, px, 1]))
                    raw_prev = world_to_canonical_action(pdy, pdx, pty, ptx, None)
                else:
                    raw_prev = direction_to_action(pdy, pdx)
            else:
                raw_prev = None
            prev_dir = raw_prev if (raw_prev is not None and raw_prev >= 0) else None

            steps_meta.append(
                {'sample_idx': sample_idx, 'trace_idx': trace_idx, 'step_idx': i, 'pos': (y, x), 'action': action, 'prev_dir': prev_dir}
            )
            last_valid = (i + s, (ny, nx), action)
            # Cache world-frame move direction for the next step's tangent sign-alignment.
            wmag = (dy * dy + dx * dx) ** 0.5
            if wmag > 0:
                prev_world_vec = (dy / wmag, dx / wmag)

        # Teach termination: append a STOP label at each trace endpoint.
        if last_valid is not None:
            (stop_step_idx, (sy, sx), last_action) = last_valid
            if half <= sy < h - half and half <= sx < w - half:
                steps_meta.append(
                    {
                        'sample_idx': sample_idx,
                        'trace_idx': trace_idx,
                        'step_idx': stop_step_idx,
                        'pos': (sy, sx),
                        'action': STOP_ACTION,
                        'prev_dir': last_action,
                    }
                )

    return steps_meta


def generate_expert_sequences(sample: Dict, config: dict, obs_size: int, step_size: int = 1, tangent_aware: bool = False) -> List[Dict[str, Any]]:
    """Walk expert traces into full per-episode observation/action sequences for LSTM training.

    Preserves temporal order and gives each trace its own visited mask (fresh episode).
    ``step_size``/``tangent_aware``: see ``generate_expert_metadata``.

    Returns:
        List of ``{observations, actions, length}`` dicts (observations are (C, H, W) arrays).
    """
    from environment.observation import ObservationBuilder

    obs_builder = ObservationBuilder(config)
    image = sample['image']
    # Stack the predicted-prior geometry once so build() consumes non-leaking channels (matches env.set_data).
    _build_stacked_sources([sample], obs_builder)
    h, w = image.shape[:2]
    half = obs_size // 2
    sequences = []
    s = max(int(step_size), 1)
    vo = ObservationBuilder.compute_vessel_orientation(image) if tangent_aware else None

    for trace in sample.get('expert_traces', []):
        if len(trace) < s + 1:
            continue

        visited_mask = np.zeros((h, w), dtype=np.float32)
        seq_obs: List[np.ndarray] = []
        seq_actions: List[int] = []
        last_valid = None  # ((ny, nx), action) for STOP supervision
        prev_world_vec = None

        for i in range(0, len(trace) - s, s):
            y, x = trace[i]
            ny, nx = trace[i + s]

            if y < half or y >= h - half or x < half or x >= w - half:
                continue

            dy = int(np.sign(int(ny) - int(y)))
            dx = int(np.sign(int(nx) - int(x)))
            if tangent_aware and vo is not None:
                ty, tx = (float(vo[y, x, 0]), float(vo[y, x, 1]))
                action = world_to_canonical_action(dy, dx, ty, tx, prev_world_vec)
            else:
                action = direction_to_action(dy, dx)
            if action < 0:
                continue

            if i >= s:
                py, px = trace[i - s]
                pdy = int(np.sign(int(y) - int(py)))
                pdx = int(np.sign(int(x) - int(px)))
                if tangent_aware and vo is not None:
                    pty, ptx = (float(vo[py, px, 0]), float(vo[py, px, 1]))
                    raw_prev = world_to_canonical_action(pdy, pdx, pty, ptx, None)
                else:
                    raw_prev = direction_to_action(pdy, pdx)
            else:
                raw_prev = None
            prev_dir = raw_prev if (raw_prev is not None and raw_prev >= 0) else None

            obs = obs_builder.build(image=image, visited_mask=visited_mask, vesselness=None, position=np.array([y, x]), prev_direction=prev_dir)
            seq_obs.append(obs)
            seq_actions.append(action)
            visited_mask[y, x] = 1.0
            last_valid = ((ny, nx), action)
            wmag = (dy * dy + dx * dx) ** 0.5
            if wmag > 0:
                prev_world_vec = (dy / wmag, dx / wmag)

        # STOP supervision at the trace endpoint.
        if last_valid is not None:
            (sy, sx), last_action = last_valid
            if half <= sy < h - half and half <= sx < w - half:
                stop_obs = obs_builder.build(
                    image=image,
                    visited_mask=visited_mask,
                    vesselness=None,
                    position=np.array([sy, sx]),
                    prev_direction=last_action,
                )
                seq_obs.append(stop_obs)
                seq_actions.append(STOP_ACTION)

        if len(seq_obs) >= 2:
            sequences.append({'observations': seq_obs, 'actions': seq_actions, 'length': len(seq_obs)})

    return sequences


def generate_expert_sequence_metadata(
    sample: Dict, sample_idx: int, obs_size: int, step_size: int = 1, tangent_aware: bool = False
) -> List[Dict[str, Any]]:
    """Walk expert traces into lightweight LSTM sequence metadata (positions/actions only).

    Same logic as ``generate_expert_sequences`` but stores ~100 B/step instead of full
    observation tensors; observations are built on the fly by ImitationSequenceDataset.

    Returns:
        List of ``{sample_idx, steps, length}`` dicts where each step is ``{pos, action, prev_dir}``.
    """
    traces = sample.get('expert_traces', [])
    h, w = sample['image'].shape[:2]
    half = obs_size // 2
    sequences = []
    s = max(int(step_size), 1)
    if tangent_aware:
        from environment.observation import ObservationBuilder

        vo = ObservationBuilder.compute_vessel_orientation(sample['image'])
    else:
        vo = None

    for trace in traces:
        if len(trace) < s + 1:
            continue
        steps: List[Dict[str, Any]] = []
        last_valid = None
        prev_world_vec = None
        for i in range(0, len(trace) - s, s):
            y, x = trace[i]
            ny, nx = trace[i + s]
            if y < half or y >= h - half or x < half or x >= w - half:
                continue
            dy = int(np.sign(int(ny) - int(y)))
            dx = int(np.sign(int(nx) - int(x)))
            if tangent_aware and vo is not None:
                ty, tx = (float(vo[y, x, 0]), float(vo[y, x, 1]))
                action = world_to_canonical_action(dy, dx, ty, tx, prev_world_vec)
            else:
                action = direction_to_action(dy, dx)
            if action < 0:
                continue
            if i >= s:
                py, px = trace[i - s]
                pdy = int(np.sign(int(y) - int(py)))
                pdx = int(np.sign(int(x) - int(px)))
                if tangent_aware and vo is not None:
                    pty, ptx = (float(vo[py, px, 0]), float(vo[py, px, 1]))
                    raw_prev = world_to_canonical_action(pdy, pdx, pty, ptx, None)
                else:
                    raw_prev = direction_to_action(pdy, pdx)
            else:
                raw_prev = None
            prev_dir = raw_prev if (raw_prev is not None and raw_prev >= 0) else None
            steps.append({'pos': (y, x), 'action': action, 'prev_dir': prev_dir})
            last_valid = ((ny, nx), action)
            wmag = (dy * dy + dx * dx) ** 0.5
            if wmag > 0:
                prev_world_vec = (dy / wmag, dx / wmag)
        # STOP supervision at the trace endpoint.
        if last_valid is not None:
            (sy, sx), last_action = last_valid
            if half <= sy < h - half and half <= sx < w - half:
                steps.append({'pos': (sy, sx), 'action': STOP_ACTION, 'prev_dir': last_action})
        if len(steps) >= 2:
            sequences.append({'sample_idx': sample_idx, 'steps': steps, 'length': len(steps)})
    return sequences


def _build_stacked_sources(samples, obs_builder, unet_priors=None):
    """Pre-stack each sample's static observation channels via the env's code path; returns the stacks list.

    Geometry comes from the predicted priors (UNet → skeleton → DT), never GT — same contract as
    VesselTracingEnv.set_data — and is derived on the fly when not already on the sample.
    """
    from environment.observation import ObservationBuilder
    from data.dataloader import compute_predicted_priors

    stacks: List[np.ndarray] = []
    for i, s in enumerate(samples):
        orient = s['vessel_orientation'] if 'vessel_orientation' in s else ObservationBuilder.compute_vessel_orientation(s['image'])
        if 'pred_centerline' in s and 'pred_distance_transform' in s and 'pred_dt_gradient' in s:
            pred_cl = s['pred_centerline']
            pred_dt = s['pred_distance_transform']
            pred_dt_grad = s['pred_dt_gradient']
        else:
            # Derive predicted priors lazily; errors clearly if the seed-detector checkpoint is missing.
            tol = float(obs_builder.tolerance)
            bundle = compute_predicted_priors(s['image'], tol)
            if bundle is None:
                raise RuntimeError(
                    'Predicted priors required but the seed-detector '
                    'checkpoint is missing. Train it via '
                    'scripts/train_seed_detector.py '
                    '(writes weights/seed_detector.pt) before running '
                    'imitation. Predicted priors are unconditionally on '
                    'post-P0 GT-leakage removal — the use_unet_prior flag '
                    'only toggles the raw probability obs channel.'
                )
            pred_cl = bundle['centerline']
            pred_dt = bundle['distance_transform']
            pred_dt_grad = bundle['dt_gradient']
        prior = unet_priors[i] if unet_priors is not None else None
        obs_builder.prepare_stacked_sources(
            distance_transform=pred_dt,
            dt_gradient=pred_dt_grad,
            centerline=pred_cl,
            vessel_orientation=orient,
            unet_prior=prior,
            vesselness=s.get('vesselness'),
        )
        # Grab the stack now before the next iteration overwrites obs_builder._stacked_sources.
        stacks.append(obs_builder._stacked_sources)
    return stacks


def _build_unet_priors(samples, enabled: bool):
    """Compute the centerline-UNet prior per sample when ``enabled``; returns a list (None entries when off/unavailable)."""
    if not enabled:
        return [None] * len(samples)
    from data.dataloader import compute_unet_prior

    print(f'Computing UNet centerline prior for {len(samples)} samples...')
    out: List[Optional[np.ndarray]] = []
    for s in samples:
        out.append(compute_unet_prior(s['image']))
    return out


def _build_vesselness_maps(samples, enabled: bool):
    """Compute the Frangi vesselness map per sample when ``enabled``; returns a list (None entries when off)."""
    if not enabled:
        return [None] * len(samples)
    from skimage.filters import frangi

    print(f'Computing Frangi vesselness for {len(samples)} samples...')
    maps: List[Optional[np.ndarray]] = []
    sigmas = np.linspace(1.0, 3.0, 5)
    for s in samples:
        img = s['image']
        gray = img[:, :, 1] if img.ndim == 3 else img
        v = frangi(gray.astype(np.float64), sigmas=sigmas, black_ridges=True)
        maps.append(v.astype(np.float32))
    return maps


class ImitationDataset(Dataset):
    """Feedforward BC dataset that stores only step metadata and crops observation patches on demand."""

    def __init__(self, samples: List[Dict], metadata: List[Dict], config: dict):
        """Hold samples + step metadata and precompute the per-sample static observation channels.

        Args:
            samples: full image dicts.
            metadata: step metadata from generate_expert_metadata().
            config: full CONFIG dict for ObservationBuilder.
        """
        self.samples = samples
        self.metadata = metadata

        from environment.observation import ObservationBuilder

        self.obs_builder = ObservationBuilder(config)

        # Precompute static channels via the env's path so the layout matches PPO observations.
        env_cfg = config.get('environment', {})
        self.use_vesselness = env_cfg.get('use_vesselness', False)
        self.use_unet_prior = env_cfg.get('use_unet_prior', False)
        print(f'Pre-computing static observation channels for {len(samples)} samples...')
        self.unet_priors = _build_unet_priors(samples, enabled=self.use_unet_prior)
        self.stacked_sources = _build_stacked_sources(samples, self.obs_builder, unet_priors=self.unet_priors)
        self.vesselness_maps = _build_vesselness_maps(samples, enabled=self.use_vesselness)
        print('  Done.')

        # Per-sample reusable visited-mask buffers; refilled per step in __getitem__. Safe under
        # DataLoader workers since fork gives each worker its own copy.
        self.visited_masks = [np.zeros(s['image'].shape[:2], dtype=np.float32) for s in samples]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Build the ``(obs, action)`` pair for expert step ``idx``, cropping the patch on the fly."""
        m = self.metadata[idx]
        sidx = m['sample_idx']
        s = self.samples[sidx]

        self.obs_builder._stacked_sources = self.stacked_sources[sidx]

        # Reconstruct the visited mask from the trace prefix so the obs matches deployment.
        visited = self.visited_masks[sidx]
        visited.fill(0.0)
        step_idx = m.get('step_idx', 0)
        if step_idx > 0:
            trace = s['expert_traces'][m['trace_idx']]
            prefix = np.asarray(trace[:step_idx], dtype=np.intp)
            visited[prefix[:, 0], prefix[:, 1]] = 1.0

        obs = self.obs_builder.build(
            image=s['image'],
            visited_mask=visited,
            vesselness=self.vesselness_maps[sidx] if self.use_vesselness else None,
            position=np.array(m['pos']),
            prev_direction=m['prev_dir'],
        )

        return (torch.from_numpy(obs).float(), torch.tensor(m['action'], dtype=torch.long))


class ImitationSequenceDataset(Dataset):
    """LSTM BC dataset of variable-length episodes; builds observations on the fly from metadata."""

    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        samples: List[Dict],
        config: dict,
        stacked_sources: Optional[List[np.ndarray]] = None,
        vesselness_maps: Optional[List[np.ndarray]] = None,
        unet_priors: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """Hold sequences + samples and reuse (or compute) the per-sample static channels and priors."""
        self.sequences = sequences
        self.samples = samples

        from environment.observation import ObservationBuilder

        self.obs_builder = ObservationBuilder(config)
        env_cfg = config.get('environment', {})
        self.use_vesselness = env_cfg.get('use_vesselness', False)
        self.use_unet_prior = env_cfg.get('use_unet_prior', False)

        if unet_priors is not None:
            self.unet_priors = unet_priors
        else:
            self.unet_priors = _build_unet_priors(samples, enabled=self.use_unet_prior)

        if stacked_sources is not None:
            self.stacked_sources = stacked_sources
        else:
            print(f'Pre-computing static observation channels for {len(samples)} samples (seq)...')
            self.stacked_sources = _build_stacked_sources(samples, self.obs_builder, unet_priors=self.unet_priors)
            print('  Done.')

        if vesselness_maps is not None:
            self.vesselness_maps = vesselness_maps
        else:
            self.vesselness_maps = _build_vesselness_maps(samples, enabled=self.use_vesselness)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Build the ``(observations, actions, length)`` tensors for one expert episode."""
        seq = self.sequences[idx]
        sample = self.samples[seq['sample_idx']]
        sidx = seq['sample_idx']
        self.obs_builder._stacked_sources = self.stacked_sources[sidx]

        h, w = sample['image'].shape[:2]
        visited_mask = np.zeros((h, w), dtype=np.float32)
        obs_list = []
        actions = []
        vmap = self.vesselness_maps[sidx] if self.use_vesselness else None

        for step in seq['steps']:
            obs = self.obs_builder.build(
                image=sample['image'],
                visited_mask=visited_mask,
                vesselness=vmap,
                position=np.array(step['pos']),
                prev_direction=step['prev_dir'],
            )
            obs_list.append(obs)
            actions.append(step['action'])
            visited_mask[step['pos'][0], step['pos'][1]] = 1.0

        return {
            'observations': torch.from_numpy(np.stack(obs_list)).float(),
            'actions': torch.tensor(actions, dtype=torch.long),
            'length': seq['length'],
        }


def sequence_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length episodes to the batch max for LSTM training.

    Returns:
        Dict with time-first ``observations`` (T_max, B, C, H, W), ``actions`` / ``mask`` /
        ``dones`` (T_max, B), and ``lengths``; ``dones`` marks each sequence's last valid step.
    """
    lengths = [item['length'] for item in batch]
    T_max = max(lengths)
    B = len(batch)
    C, H, W = batch[0]['observations'].shape[1:]

    obs_padded = torch.zeros(T_max, B, C, H, W)
    act_padded = torch.zeros(T_max, B, dtype=torch.long)
    mask = torch.zeros(T_max, B)
    dones = torch.zeros(T_max, B)

    for b, item in enumerate(batch):
        L = item['length']
        obs_padded[:L, b] = item['observations']  # already (T, C, H, W)
        act_padded[:L, b] = item['actions']
        mask[:L, b] = 1.0
        # Mark the last valid step so the LSTM resets its hidden state over the padding region.
        dones[L - 1, b] = 1.0

    return {'observations': obs_padded, 'actions': act_padded, 'mask': mask, 'dones': dones, 'lengths': lengths}


class ImitationTrainer:
    """Behavior-cloning trainer for the actor-critic policy, supporting feedforward and LSTM modes."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        lr: float = 3e-4,
        batch_size: int = 128,
        num_epochs: int = 30,
        lstm_batch_size: int = 16,
    ):
        """Configure the optimizer, LR scheduler, and mode (FF vs LSTM) from the model/config."""
        self.model = model
        self.config = config
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lstm_batch_size = lstm_batch_size
        self.use_lstm = getattr(model, 'use_lstm', False)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('training', {}).get('imitation', {}).get('lr_step_size', 10),
            gamma=config.get('training', {}).get('imitation', {}).get('lr_gamma', 0.5),
        )
        # Class weights are set in train() from the action mix; STOP is rare (~1/trace vs ~50
        # moves) and is otherwise never learned.
        self._class_weights: Optional[torch.Tensor] = None
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _action_class_weights(
        actions: List[int],
    ) -> torch.Tensor:
        """Return inverse-frequency action-class weights normalised to mean 1.0 over present classes.

        Mean-1 normalisation keeps the loss scale (so LR/grad-clip behave the same) while upweighting
        rare actions like STOP; absent classes get weight 0 so they don't inflate the normaliser.
        """
        counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=N_ACTIONS)
        smoothed = np.maximum(counts, 1).astype(np.float64)
        inv = 1.0 / smoothed
        inv[counts == 0] = 0.0
        present = (counts > 0).sum()
        if present == 0:
            return torch.ones(N_ACTIONS, dtype=torch.float32)
        weights = inv * present / inv.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def train(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        save_path: str,
        config: dict,
        log_path: Optional[str] = None,
        train_sequences: Optional[List[Dict[str, Any]]] = None,
        val_sequences: Optional[List[Dict[str, Any]]] = None,
        samples: Optional[List[Dict]] = None,
        stacked_sources: Optional[List[np.ndarray]] = None,
        vesselness_maps: Optional[List[np.ndarray]] = None,
        unet_priors: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Set action class weights and dispatch to the FF or LSTM training loop, saving best weights.

        LSTM mode additionally requires ``train_sequences``, ``val_sequences``, and ``samples``;
        ``stacked_sources`` / ``vesselness_maps`` / ``unet_priors`` are reused if supplied.
        """
        _log = log_path or save_path.replace('.pt', '_log.csv')
        if self.use_lstm:
            if train_sequences is None or val_sequences is None or samples is None:
                raise ValueError(
                    'LSTM mode requires train_sequences, val_sequences, and samples. '
                    'Use generate_expert_sequence_metadata() to create sequence metadata.'
                )
            train_actions = [step['action'] for seq in train_sequences for step in seq['steps']]
            self._class_weights = self._action_class_weights(train_actions).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self._class_weights)
            print(f'Action class weights (LSTM): {self._class_weights.cpu().tolist()}')
            self._train_lstm(train_sequences, val_sequences, save_path, config, _log, samples, stacked_sources, vesselness_maps, unet_priors)
        else:
            train_actions = [m['action'] for m in train_ds.metadata]
            self._class_weights = self._action_class_weights(train_actions).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self._class_weights)
            print(f'Action class weights (FF): {self._class_weights.cpu().tolist()}')
            self._train_ff(train_ds, val_ds, save_path, config, _log)

    def _train_ff(self, train_ds: Dataset, val_ds: Dataset, save_path: str, config: dict, log_path: str) -> None:
        """Run the feedforward BC loop: per-step cross-entropy, best-val-loss checkpointing, CSV logging."""
        n_workers = config.get('training', {}).get('imitation', {}).get('num_workers', 4)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=n_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)

        print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}')

        best_val_loss = float('inf')
        _csv_fields = ['epoch', 'train_loss', 'train_acc', 'train_soft_acc', 'train_grad_norm', 'val_loss', 'val_acc', 'val_soft_acc', 'lr']
        with open(log_path, 'w', newline='', encoding='utf-8') as _csv_file:
            _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
            _csv_writer.writeheader()

            for epoch in range(1, self.num_epochs + 1):
                (train_loss, train_acc, train_soft, train_gn) = self._run_epoch_ff(train_loader, train=True)
                val_loss, val_acc, val_soft, _ = self._run_epoch_ff(val_loader, train=False)
                current_lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()

                print(
                    f'Epoch {epoch:3d}/{self.num_epochs}  '
                    f'train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  '
                    f'val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  '
                    f'val_soft_acc={val_soft:.3f}'
                )
                _csv_writer.writerow(
                    {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_soft_acc': train_soft,
                        'train_grad_norm': train_gn,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_soft_acc': val_soft,
                        'lr': current_lr,
                    }
                )
                _csv_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'config': config,
                        },
                        save_path,
                    )
                    print(f'  ✓ Saved best model (val_loss={val_loss:.4f})')

        print(f'\nDone. Best val_loss={best_val_loss:.4f}  →  {save_path}')

        try:
            from training.plots import plot_imitation_log

            png = plot_imitation_log(log_path)
            if png:
                print(f'Plot: {png}')
        except Exception as e:
            print(f'[plot_imitation_log] skipped: {e}')

    def _run_epoch_ff(self, loader: DataLoader, train: bool) -> Tuple[float, float, float]:
        """Run one feedforward epoch (train or eval); returns (loss, acc, soft_acc, mean_grad_norm)."""
        self.model.train(mode=train)
        (total_loss, correct, soft_correct, total) = 0.0, 0, 0, 0
        total_gn, n_updates = 0.0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for obs_batch, action_batch in loader:
                obs_batch = obs_batch.to(self.device)
                action_batch = action_batch.to(self.device)

                logits, _, _ = self.model(obs_batch)
                loss = self.criterion(logits, action_batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('training', {}).get('imitation', {}).get('max_grad_norm', 1.0),
                    )
                    self.optimizer.step()
                    total_gn += grad_norm.item()
                    n_updates += 1

                preds = logits.argmax(-1)
                total_loss += loss.item() * len(action_batch)
                correct += (preds == action_batch).sum().item()
                soft_correct += soft_correct_count(preds, action_batch)
                total += len(action_batch)

        return (
            total_loss / max(total, 1),
            correct / max(total, 1),
            soft_correct / max(total, 1),
            total_gn / max(n_updates, 1),
        )

    def _train_lstm(
        self,
        train_sequences: List[Dict[str, Any]],
        val_sequences: List[Dict[str, Any]],
        save_path: str,
        config: dict,
        log_path: str,
        samples: Optional[List[Dict]] = None,
        stacked_sources: Optional[List[np.ndarray]] = None,
        vesselness_maps: Optional[List[np.ndarray]] = None,
        unet_priors: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Run the LSTM BC loop over padded episode batches: masked cross-entropy, best-val checkpointing, CSV logging."""
        # Compute static channels/priors/vesselness once so both datasets share them instead of
        # recomputing ~1 GB of stacks each.
        if stacked_sources is None or unet_priors is None or vesselness_maps is None:
            from environment.observation import ObservationBuilder

            shared_builder = ObservationBuilder(config)
            env_cfg = config.get('environment', {})
            if unet_priors is None:
                unet_priors = _build_unet_priors(samples, enabled=env_cfg.get('use_unet_prior', False))
            if stacked_sources is None:
                stacked_sources = _build_stacked_sources(samples, shared_builder, unet_priors=unet_priors)
            if vesselness_maps is None:
                vesselness_maps = _build_vesselness_maps(samples, enabled=env_cfg.get('use_vesselness', False))

        train_loader = DataLoader(
            ImitationSequenceDataset(
                train_sequences, samples, config, stacked_sources=stacked_sources, vesselness_maps=vesselness_maps, unet_priors=unet_priors
            ),
            batch_size=self.lstm_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=sequence_collate_fn,
        )
        val_loader = DataLoader(
            ImitationSequenceDataset(
                val_sequences, samples, config, stacked_sources=stacked_sources, vesselness_maps=vesselness_maps, unet_priors=unet_priors
            ),
            batch_size=self.lstm_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=sequence_collate_fn,
        )

        train_steps = sum(s['length'] for s in train_sequences)
        val_steps = sum(s['length'] for s in val_sequences)
        print(f'LSTM imitation: {len(train_sequences)} train seqs ({train_steps} steps)  |  {len(val_sequences)} val seqs ({val_steps} steps)')
        print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}')

        best_val_loss = float('inf')
        _csv_fields = ['epoch', 'train_loss', 'train_acc', 'train_soft_acc', 'train_grad_norm', 'val_loss', 'val_acc', 'val_soft_acc', 'lr']
        with open(log_path, 'w', newline='', encoding='utf-8') as _csv_file:
            _csv_writer = csv.DictWriter(_csv_file, fieldnames=_csv_fields)
            _csv_writer.writeheader()

            for epoch in range(1, self.num_epochs + 1):
                (train_loss, train_acc, train_soft, train_gn) = self._run_epoch_lstm(train_loader, train=True)
                val_loss, val_acc, val_soft, _ = self._run_epoch_lstm(val_loader, train=False)
                current_lr = self.scheduler.get_last_lr()[0]
                self.scheduler.step()

                print(
                    f'Epoch {epoch:3d}/{self.num_epochs}  '
                    f'train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  '
                    f'val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  '
                    f'val_soft_acc={val_soft:.3f}'
                )
                _csv_writer.writerow(
                    {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'train_soft_acc': train_soft,
                        'train_grad_norm': train_gn,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_soft_acc': val_soft,
                        'lr': current_lr,
                    }
                )
                _csv_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_acc': val_acc,
                            'config': config,
                        },
                        save_path,
                    )
                    print(f'  ✓ Saved best model (val_loss={val_loss:.4f})')

        print(f'\nDone. Best val_loss={best_val_loss:.4f}  →  {save_path}')

        try:
            from training.plots import plot_imitation_log

            png = plot_imitation_log(log_path)
            if png:
                print(f'Plot: {png}')
        except Exception as e:
            print(f'[plot_imitation_log] skipped: {e}')

    def _run_epoch_lstm(self, loader: DataLoader, train: bool) -> Tuple[float, float, float]:
        """Run one LSTM epoch over padded sequence batches; returns (loss, acc, soft_acc, mean_grad_norm)."""
        self.model.train(mode=train)
        (total_loss, correct, soft_correct, total) = 0.0, 0, 0, 0
        total_gn, n_updates = 0.0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                obs_seq = batch['observations'].to(self.device)  # (T, B, C, H, W)
                actions = batch['actions'].to(self.device)  # (T, B)
                mask = batch['mask'].to(self.device)  # (T, B)
                dones = batch['dones'].to(self.device)  # (T, B)

                T, B = obs_seq.shape[:2]

                init_state = self.model.init_hidden(batch_size=B, device=self.device)
                logits_seq, _ = self.model.forward_sequence(obs_seq, init_state, dones)

                logits_flat = logits_seq.reshape(T * B, -1)
                actions_flat = actions.reshape(T * B)
                mask_flat = mask.reshape(T * B)

                # Mask out padded steps from the loss.
                per_step_loss = nn.functional.cross_entropy(logits_flat, actions_flat, weight=self._class_weights, reduction='none')
                loss = (per_step_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.get('training', {}).get('imitation', {}).get('max_grad_norm', 1.0)
                    )
                    self.optimizer.step()
                    total_gn += grad_norm.item()
                    n_updates += 1

                preds = logits_seq.argmax(dim=-1)  # (T, B)
                valid_steps = mask.sum().item()
                correct += ((preds == actions) * mask).sum().item()
                mbool = mask.bool()
                soft_correct += soft_correct_count(preds[mbool], actions[mbool])
                total_loss += loss.item() * valid_steps
                total += valid_steps

        return (total_loss / max(total, 1), correct / max(total, 1), soft_correct / max(total, 1), total_gn / max(n_updates, 1))
