import cv2
import numpy as np
from typing import Tuple, Optional

class FundusPreprocessor:
    """
    Preprocessing pipeline for blood vessel centerline extraction:
    1. Green channel extraction
    2. Gamma Correction (Brightness adjustment)
    3. CLAHE (Local contrast enhancement)
    4. FOV (Field of View) mask application
    """
    
    def __init__(self, clahe_clip_limit: float = 2.5, clahe_tile_size: int = 8, gamma: float = 0.8):
        """
        Initialize the preprocessor.
        
        Args:
            clahe_clip_limit: Clip limit for CLAHE (typically 2-3).
            clahe_tile_size: Grid size for CLAHE tiles.
            gamma: Power-law constant
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        
        # Define gamma so apply_gamma_correction can find it
        self.gamma = gamma 
        
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
    
    def extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        """Green channel typically has the best vessel contrast."""
        if len(image.shape) == 2:
            return image
        return image[:, :, 1]

    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Enhances image using power-law transformation."""
        # Look-up table for faster computation
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance local contrast."""
        return self.clahe.apply(image)
    
    def create_fov_mask(self, image: np.ndarray, block_size: int = 51, C: int = 10,
                        erosion_size: int = 5) -> np.ndarray:
        """Create Field of View mask to exclude background regions."""
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, block_size, -C 
        )
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest contour (the retina)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Erosion prevents Frangi artifacts at the sharp circular border
        if erosion_size > 0:
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size * 2 + 1, erosion_size * 2 + 1))
            mask = cv2.erode(mask, erosion_kernel, iterations=1)
        
        return mask
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(image, image, mask=mask)
    
    def preprocess(self, image: np.ndarray, return_intermediate: bool = False) -> Tuple:
        """
        Complete pipeline order:
        Green Channel -> Gamma Correction -> CLAHE -> Masking
        """
        # Step 1: Green channel
        green = self.extract_green_channel(image)

        # Step 2: Gamma correction
        gamma_corrected = self.apply_gamma_correction(green)
        
        # Step 3: Apply CLAHE to the gamma-corrected image
        clahe_enhanced = self.apply_clahe(gamma_corrected)
        
        # Step 4: Create mask
        mask = self.create_fov_mask(green)
        
        # Step 5: Final output
        preprocessed = self.apply_mask(clahe_enhanced, mask)
        
        if return_intermediate:
            # Added gamma_corrected to the return for your visualizations
            return preprocessed, green, gamma_corrected, clahe_enhanced, mask
        return preprocessed

    def preprocess_batch(self, images: list) -> list:
        return [self.preprocess(img) for img in images]