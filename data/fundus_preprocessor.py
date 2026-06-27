"""Fundus preprocessing: green-channel CLAHE enhancement with FOV masking."""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# FOV-radius-scaled rim erosion, shared with the RL seed detector so the baselines
# gate vesselness/probability exactly like the RL agent. See [[fov-scale-invariance]].
FOV_EROSION_FRAC = 0.04  # rim erosion ≈4% of FOV radius
FOV_EROSION_MIN = 4
FOV_EROSION_MAX = 17


def eroded_fov_mask(fov_mask: np.ndarray) -> np.ndarray:
    """Erode an FOV mask by an FOV-radius-scaled rim, matching the RL seed detector.

    Drops the bright Frangi edge-halo that hugs the FOV boundary (the step edge left
    by image masking) instead of letting it survive as a ring of false vessels.

    Args:
        fov_mask: (H, W) mask, any positive value = inside the retina.

    Returns:
        (H, W) uint8 {0, 1} eroded mask.
    """
    m = (fov_mask > 0).astype(np.uint8)
    radius = np.sqrt(max(int(m.sum()), 1) / np.pi)
    erode_px = int(np.clip(round(FOV_EROSION_FRAC * radius), FOV_EROSION_MIN, FOV_EROSION_MAX))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    return cv2.erode(m, se, iterations=1)


class FundusPreprocessor:
    """Turns a raw RGB fundus photo into a contrast-enhanced, FOV-masked [0, 1] grayscale image."""

    def __init__(self, clahe_clip_limit: float = 2.5, clahe_tile_size: int = 8, gamma: float = 0.8, median_kernel: int = 3):
        """Configure gamma, denoising, and CLAHE parameters used by preprocess()."""
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.gamma = gamma
        self.median_kernel = median_kernel
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(self.clahe_tile_size, self.clahe_tile_size))

    def extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        """Return the green channel (highest vessel contrast); pass through if grayscale."""
        if len(image.shape) == 2:
            return image
        return image[:, :, 1]

    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply LUT-based gamma correction; gamma < 1 brightens dark regions."""
        # Normalize floats/non-uint8 back to uint8 before the LUT lookup.
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
        return cv2.LUT(image, table)

    def apply_median_blur(self, image: np.ndarray) -> np.ndarray:
        """Suppress salt-and-pepper noise while preserving vessel edges."""
        if self.median_kernel > 0:
            return cv2.medianBlur(image, self.median_kernel)
        return image

    def _get_dynamic_kernel_size(self, image: np.ndarray, base_size: int = 5) -> int:
        """Scale a kernel size by image diagonal, normalized to DRIVE's ~812 px."""
        diag = np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2)
        scale = diag / 812.0
        return int(max(1, round(base_size * scale)))

    def create_fov_mask(self, image: np.ndarray, block_size: int = 51, C: int = 10, erosion_size: Optional[int] = None) -> np.ndarray:
        """Segment the circular field of view, returning an eroded {0, 255} uint8 mask.

        Adaptive thresholding is the primary path; Otsu and a centered-disc fallback
        guard against degenerate or empty results on challenging images.
        """
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        try:
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, -C)
        except cv2.error:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Fall back to Otsu when adaptive thresholding is near-empty or near-full.
        coverage = binary.sum() / (binary.size * 255)
        if coverage < 0.05 or coverage > 0.95:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_size = self._get_dynamic_kernel_size(image, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Keep only the largest contour as the FOV, else synthesize a disc.
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = int(min(h, w) * 0.45)
            cv2.circle(mask, center, radius, 255, -1)

        # Reject implausibly small masks (<10% area) in favor of the disc fallback.
        final_coverage = mask.sum() / (mask.size * 255)
        if final_coverage < 0.10:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = int(min(h, w) * 0.45)
            cv2.circle(mask, center, radius, 255, -1)

        e_size = erosion_size if erosion_size is not None else kernel_size
        if e_size > 0:
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e_size * 2 + 1, e_size * 2 + 1))
            mask = cv2.erode(mask, erosion_kernel, iterations=1)

        return mask

    def load_external_mask(self, mask: np.ndarray, erosion_size: Optional[int] = None) -> np.ndarray:
        """Binarize a supplied FOV mask and erode it to avoid the bright high-contrast rim."""
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = (mask > 128).astype(np.uint8) * 255

        e_size = erosion_size if erosion_size is not None else self._get_dynamic_kernel_size(mask, 5)
        if e_size > 0:
            erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e_size * 2 + 1, e_size * 2 + 1))
            mask = cv2.erode(mask, erosion_kernel, iterations=1)

        return mask

    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Zero out everything outside the FOV mask."""
        return cv2.bitwise_and(image, image, mask=mask)

    def preprocess(
        self, image: np.ndarray, external_mask: Optional[np.ndarray] = None, return_intermediate: bool = False
    ) -> Union[np.ndarray, Tuple]:
        """Run the full pipeline on one image, returning the [0, 1] enhanced grayscale result.

        With ``return_intermediate`` also returns the green, gamma, CLAHE, and mask stages
        for debugging/visualization.
        """
        green = self.extract_green_channel(image)
        gamma_corrected = self.apply_gamma_correction(green)
        denoised = self.apply_median_blur(gamma_corrected)

        if external_mask is not None:
            mask = self.load_external_mask(external_mask)
        else:
            mask = self.create_fov_mask(gamma_corrected)

        # Mask before CLAHE so the FOV border is not amplified into a false vessel.
        gamma_masked = self.apply_mask(denoised, mask)
        clahe_enhanced = self.clahe.apply(gamma_masked)

        # Normalize within the FOV using robust 1st/99th percentiles.
        roi_pixels = clahe_enhanced[mask > 0]
        if roi_pixels.size > 0:
            vmin, vmax = np.percentile(roi_pixels, [1.0, 99.0])
            preprocessed = np.clip((clahe_enhanced - vmin) / (vmax - vmin + 1e-8), 0, 1)
        else:
            preprocessed = clahe_enhanced.astype(np.float32) / 255.0

        preprocessed = preprocessed.astype(np.float32)

        if return_intermediate:
            return (preprocessed, green, gamma_corrected, clahe_enhanced, mask)

        return preprocessed

    def preprocess_batch(self, images: List[np.ndarray], masks: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """Preprocess a list of images, pairing each with its mask when provided."""
        results = []
        if masks is not None:
            for img, m in zip(images, masks):
                results.append(self.preprocess(img, external_mask=m))
        else:
            for img in images:
                results.append(self.preprocess(img))

        return results
