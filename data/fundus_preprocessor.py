import cv2
import numpy as np
from typing import Tuple, Optional


class FundusPreprocessor:
    """
    Preprocessing pipeline for blood vessel centerline extraction:
    1. Green channel extraction
    2. Gamma Correction
    3. CLAHE
    4. FOV mask (external or internally created)
    """

    def __init__(self, clahe_clip_limit: float = 2.5,
                 clahe_tile_size: int = 8,
                 gamma: float = 0.8):

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.gamma = gamma

        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )

    # --------------------------------------------------
    # CHANNEL EXTRACTION
    # --------------------------------------------------
    def extract_green_channel(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return image[:, :, 1]

    # --------------------------------------------------
    # GAMMA CORRECTION
    # --------------------------------------------------
    def apply_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        invGamma = 1.0 / self.gamma
        table = np.array([
            ((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")

        return cv2.LUT(image, table)

    # --------------------------------------------------
    # CLAHE
    # --------------------------------------------------
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        return self.clahe.apply(image)

    # --------------------------------------------------
    # INTERNAL FOV CREATION
    # --------------------------------------------------
    def create_fov_mask(self,
                        image: np.ndarray,
                        block_size: int = 51,
                        C: int = 10,
                        erosion_size: int = 5) -> np.ndarray:

        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size, -C
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [largest], -1, 255, -1)

        if erosion_size > 0:
            erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erosion_size * 2 + 1, erosion_size * 2 + 1)
            )
            mask = cv2.erode(mask, erosion_kernel, iterations=1)

        return mask

    # --------------------------------------------------
    # EXTERNAL FOV HANDLING
    # --------------------------------------------------
    def load_external_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Accepts already loaded mask and ensures correct format.
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = (mask > 0).astype(np.uint8) * 255
        return mask

    # --------------------------------------------------
    # APPLY MASK
    # --------------------------------------------------
    def apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.bitwise_and(image, image, mask=mask)

    # --------------------------------------------------
    # MAIN PIPELINE
    # --------------------------------------------------
    def preprocess(self,
                   image: np.ndarray,
                   external_mask: Optional[np.ndarray] = None,
                   return_intermediate: bool = False) -> Tuple:

        # 1️⃣ Green channel
        green = self.extract_green_channel(image)

        # 2️⃣ Gamma
        gamma_corrected = self.apply_gamma_correction(green)

        # 3️⃣ CLAHE
        clahe_enhanced = self.apply_clahe(gamma_corrected)

        # 4️⃣ Mask selection
        if external_mask is not None:
            mask = self.load_external_mask(external_mask)
        else:
            mask = self.create_fov_mask(green)

        # 5️⃣ Apply mask
        preprocessed = self.apply_mask(clahe_enhanced, mask)

        if return_intermediate:
            return (
                preprocessed,
                green,
                gamma_corrected,
                clahe_enhanced,
                mask
            )

        return preprocessed

    # --------------------------------------------------
    # BATCH
    # --------------------------------------------------
    def preprocess_batch(self,
                         images: list,
                         masks: Optional[list] = None) -> list:

        results = []

        if masks is not None:
            for img, m in zip(images, masks):
                results.append(self.preprocess(img, external_mask=m))
        else:
            for img in images:
                results.append(self.preprocess(img))

        return results