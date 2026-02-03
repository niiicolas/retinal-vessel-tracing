# data/preprocessing.py
"""
Data preprocessing utilities for retinal vessel images.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, filters, exposure
from skimage.morphology import skeletonize, remove_small_objects
from typing import Tuple, Optional, Dict, Any
import albumentations as A
from pathlib import Path


class RetinalImagePreprocessor:
    """Preprocessing pipeline for retinal fundus images."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tile_size = config['data']['tile_size']
        self.overlap = config['data']['overlap']
        
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range with CLAHE enhancement."""
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        # Apply CLAHE to green channel (best vessel contrast)
        if len(image.shape) == 3:
            lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
            return enhanced
        return image
    
    def extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        """Extract and enhance green channel for vessel detection."""
        if len(image.shape) == 3:
            green = image[:, :, 1]
        else:
            green = image
        return exposure.equalize_adapthist(green)
    
    def create_fov_mask(self, image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """Create field-of-view mask from retinal image."""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
            
        mask = gray > threshold
        mask = ndimage.binary_fill_holes(mask)
        mask = morphology.binary_opening(mask, morphology.disk(10))
        return mask.astype(np.float32)
    
    def compute_vesselness(self, image: np.ndarray, 
                           sigma_range: Tuple[float, float] = (1.0, 3.0),
                           num_scales: int = 4) -> np.ndarray:
        """Compute Frangi vesselness filter response."""
        green = self.extract_green_channel(image)
        sigmas = np.linspace(sigma_range[0], sigma_range[1], num_scales)
        vesselness = filters.frangi(green, sigmas=sigmas, black_ridges=False)
        return vesselness
    
    def tile_image(self, image: np.ndarray, 
                   mask: Optional[np.ndarray] = None) -> list:
        """Split image into overlapping tiles."""
        h, w = image.shape[:2]
        tiles = []
        step = self.tile_size - self.overlap
        
        for y in range(0, h - self.tile_size + 1, step):
            for x in range(0, w - self.tile_size + 1, step):
                tile_img = image[y:y + self.tile_size, x:x + self.tile_size]
                tile_info = {
                    'image': tile_img,
                    'position': (y, x),
                    'original_size': (h, w)
                }
                if mask is not None:
                    tile_info['mask'] = mask[y:y + self.tile_size, x:x + self.tile_size]
                tiles.append(tile_info)
                
        return tiles
    
    def stitch_tiles(self, tiles: list, output_shape: Tuple[int, int]) -> np.ndarray:
        """Stitch tiles back together with overlap blending."""
        h, w = output_shape
        result = np.zeros((h, w), dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        
        # Create weight mask for blending
        blend_mask = self._create_blend_mask()
        
        for tile_info in tiles:
            y, x = tile_info['position']
            tile = tile_info['prediction']
            
            result[y:y + self.tile_size, x:x + self.tile_size] += tile * blend_mask
            weights[y:y + self.tile_size, x:x + self.tile_size] += blend_mask
            
        # Avoid division by zero
        weights = np.maximum(weights, 1e-8)
        return result / weights
    
    def _create_blend_mask(self) -> np.ndarray:
        """Create smooth blending mask for tile stitching."""
        ramp = np.linspace(0, 1, self.overlap)
        mask_1d = np.ones(self.tile_size)
        mask_1d[:self.overlap] = ramp
        mask_1d[-self.overlap:] = ramp[::-1]
        return np.outer(mask_1d, mask_1d)


class DataAugmentor:
    """Data augmentation for training."""
    
    def __init__(self, config: Dict[str, Any]):
        aug_config = config['data']['augmentation']
        
        self.transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(aug_config['brightness_range'][0] - 1, 
                                  aug_config['brightness_range'][1] - 1),
                contrast_limit=(aug_config['contrast_range'][0] - 1,
                               aug_config['contrast_range'][1] - 1),
                p=0.5
            ),
            A.Rotate(limit=aug_config['rotation_range'], p=0.5),
            A.ElasticTransform(
                alpha=aug_config['elastic_alpha'],
                sigma=aug_config['elastic_sigma'],
                p=0.3
            ),
            A.GaussNoise(var_limit=(0, aug_config['noise_std'] ** 2 * 255 ** 2), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.HueSaturationValue(p=0.3),
            A.RandomGamma(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ], additional_targets={'mask': 'mask', 'centerline': 'mask'})
        
    def __call__(self, image: np.ndarray, mask: np.ndarray, 
                 centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply augmentations to image and corresponding masks."""
        transformed = self.transform(
            image=image, 
            mask=mask, 
            centerline=centerline
        )
        return (
            transformed['image'],
            transformed['mask'],
            transformed['centerline']
        )
