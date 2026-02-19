# data/dataset.py
"""
Dataset classes for loading retinal vessel data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
from PIL import Image
import json

from .preprocessing import RetinalImagePreprocessor, DataAugmentor
from .centerline_extraction import CenterlineExtractor


class RetinalVesselDataset(Dataset):
    """Dataset for retinal vessel images with centerline annotations."""
    
    DATASET_CONFIGS = {
        'DRIVE': {
            'image_dir': 'images',
            'mask_dir': 'mask',
            'vessel_dir': '1st_manual',
            'image_ext': '.tif',
            'mask_ext': '_mask.gif',
            'vessel_ext': '.gif'
        },
        'STARE': {
            'image_dir': 'images',
            'vessel_dir': 'labels-ah',
            'image_ext': '.ppm',
            'vessel_ext': '.ah.ppm'
        },
        'CHASE_DB1': {
            'image_dir': 'images',
            'vessel_dir': 'labels',
            'image_ext': '.jpg',
            'vessel_ext': '_1stHO.png'
        },
        'HRF': {
            'image_dir': 'images',
            'vessel_dir': 'manual1',
            'mask_dir': 'mask',
            'image_ext': '.jpg',
            'vessel_ext': '.tif',
            'mask_ext': '_mask.tif'
        }
    }
    
    def __init__(self, 
                 root_dir: str,
                 dataset_name: str,
                 split: str = 'train',
                 config: Optional[Dict[str, Any]] = None,
                 transform: bool = True,
                 precompute_centerlines: bool = True):
        """
        Initialize dataset.
        
        Args:
            root_dir: Path to dataset root
            dataset_name: One of DRIVE, STARE, CHASE_DB1, HRF
            split: 'train', 'val', or 'test'
            config: Configuration dictionary
            transform: Whether to apply data augmentation
            precompute_centerlines: Whether to precompute centerlines from masks
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.split = split
        self.config = config or {}
        self.transform = transform and split == 'train'
        
        # Get dataset-specific configuration
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        self.dataset_config = self.DATASET_CONFIGS[dataset_name]
        
        # Initialize processors
        self.preprocessor = RetinalImagePreprocessor(config) if config else None
        self.augmentor = DataAugmentor(config) if config and self.transform else None
        self.centerline_extractor = CenterlineExtractor()
        
        # Load file list
        self.samples = self._load_samples()
        
        # Precompute centerlines if needed
        self.precompute_centerlines = precompute_centerlines
        self.centerline_cache = {}
        if precompute_centerlines:
            self._precompute_all_centerlines()
    
    def _load_samples(self) -> List[Dict[str, Path]]:
        """Load list of samples with paths to images and annotations."""
        samples = []
        
        image_dir = self.root_dir / self.dataset_config['image_dir']
        vessel_dir = self.root_dir / self.dataset_config['vessel_dir']
        
        image_ext = self.dataset_config['image_ext']
        vessel_ext = self.dataset_config['vessel_ext']
        
        for image_path in sorted(image_dir.glob(f'*{image_ext}')):
            sample_id = image_path.stem
            
            # Find corresponding vessel annotation
            if self.dataset_name == 'DRIVE':
                vessel_name = sample_id.replace('_training', '_manual1').replace('_test', '_manual1')
            elif self.dataset_name == 'CHASE_DB1':
                vessel_name = sample_id + '_1stHO'
            else:
                vessel_name = sample_id
                
            vessel_path = vessel_dir / f'{vessel_name}{vessel_ext}'
            
            if vessel_path.exists():
                sample = {
                    'id': sample_id,
                    'image': image_path,
                    'vessel': vessel_path
                }
                
                # Add mask if available
                if 'mask_dir' in self.dataset_config:
                    mask_dir = self.root_dir / self.dataset_config['mask_dir']
                    mask_ext = self.dataset_config['mask_ext']
                    mask_path = mask_dir / f'{sample_id}{mask_ext}'
                    if mask_path.exists():
                        sample['mask'] = mask_path
                        
                samples.append(sample)
        
        # Split samples
        samples = self._apply_split(samples)
        
        return samples
    
    def _apply_split(self, samples: List[Dict]) -> List[Dict]:
        """Apply train/val/test split."""
        n = len(samples)
        if self.split == 'train':
            return samples[:int(0.7 * n)]
        elif self.split == 'val':
            return samples[int(0.7 * n):int(0.85 * n)]
        else:  # test
            return samples[int(0.85 * n):]
    
    def _precompute_all_centerlines(self):
        """Precompute centerlines for all samples."""
        centerline_cache_dir = self.root_dir / 'centerlines_cache'
        centerline_cache_dir.mkdir(exist_ok=True)
        
        for sample in self.samples:
            cache_path = centerline_cache_dir / f"{sample['id']}_centerline.npy"
            
            if cache_path.exists():
                self.centerline_cache[sample['id']] = np.load(cache_path)
            else:
                # Load vessel mask and extract centerline
                vessel_mask = self._load_vessel_mask(sample['vessel'])
                centerline = self.centerline_extractor.extract_centerline(vessel_mask)
                
                # Cache it
                np.save(cache_path, centerline)
                self.centerline_cache[sample['id']] = centerline
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image."""
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.preprocessor:
            image = self.preprocessor.normalize_image(image)
        else:
            image = image.astype(np.float32) / 255.0
            
        return image
    
    def _load_vessel_mask(self, path: Path) -> np.ndarray:
        """Load vessel mask."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        return mask
    
    def _load_fov_mask(self, path: Path) -> np.ndarray:
        """Load field-of-view mask."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image'])
        
        # Load vessel mask
        vessel_mask = self._load_vessel_mask(sample['vessel'])
        
        # Get centerline
        if sample['id'] in self.centerline_cache:
            centerline = self.centerline_cache[sample['id']]
        else:
            centerline = self.centerline_extractor.extract_centerline(vessel_mask)
        
        # Load FOV mask if available
        if 'mask' in sample:
            fov_mask = self._load_fov_mask(sample['mask'])
        else:
            fov_mask = self.preprocessor.create_fov_mask(image) if self.preprocessor else np.ones_like(vessel_mask)
        
        # Apply augmentation
        if self.augmentor:
            image, vessel_mask, centerline = self.augmentor(
                (image * 255).astype(np.uint8), 
                vessel_mask, 
                centerline
            )
            image = image.astype(np.float32) / 255.0
        
        # Compute distance transform
        distance_transform = self.centerline_extractor.compute_distance_transform(
            centerline, 
            tolerance=self.config.get('environment', {}).get('tolerance', 2.0)
        )
        
        # Convert to tensors
        return {
            'id': sample['id'],
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),
            'vessel_mask': torch.from_numpy(vessel_mask).unsqueeze(0).float(),
            'centerline': torch.from_numpy(centerline).unsqueeze(0).float(),
            'fov_mask': torch.from_numpy(fov_mask).unsqueeze(0).float(),
            'distance_transform': torch.from_numpy(distance_transform).unsqueeze(0).float()
        }


class TileDataset(Dataset):
    """Dataset that provides tiles from full images for training."""
    
    def __init__(self, base_dataset: RetinalVesselDataset, 
                 tile_size: int = 512,
                 tiles_per_image: int = 10):
        self.base_dataset = base_dataset
        self.tile_size = tile_size
        self.tiles_per_image = tiles_per_image
        
    def __len__(self) -> int:
        return len(self.base_dataset) * self.tiles_per_image
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_idx = idx // self.tiles_per_image
        sample = self.base_dataset[image_idx]
        
        # Random crop
        _, h, w = sample['image'].shape
        
        if h > self.tile_size and w > self.tile_size:
            # Try to find a tile with vessels
            for _ in range(10):
                y = np.random.randint(0, h - self.tile_size)
                x = np.random.randint(0, w - self.tile_size)
                
                centerline_crop = sample['centerline'][:, y:y+self.tile_size, x:x+self.tile_size]
                if centerline_crop.sum() > 10:  # Has some vessel content
                    break
        else:
            y, x = 0, 0
        
        # Crop all tensors
        cropped = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                if len(value.shape) == 3:
                    cropped[key] = value[:, y:y+self.tile_size, x:x+self.tile_size]
                else:
                    cropped[key] = value[y:y+self.tile_size, x:x+self.tile_size]
            else:
                cropped[key] = value
                
        cropped['tile_position'] = torch.tensor([y, x])
        
        return cropped
