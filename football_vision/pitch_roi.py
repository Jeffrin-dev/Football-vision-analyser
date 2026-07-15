import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# HSV Tunable constants for green pitch grass.
# Note: lighting/grass-color variation, shadows, or different venues may require retuning these values.
GREEN_HUE_MIN = 30
GREEN_HUE_MAX = 95
GREEN_SAT_MIN = 30
GREEN_VAL_MIN = 30

def detect_pitch_mask(frame: np.ndarray) -> np.ndarray:
    """
    Detects the pitch area using color segmentation in the HSV color space.

    1. Converts the frame to HSV.
    2. Thresholds for green range based on tunable constants.
    3. Applies morphological operations to remove noise.
    4. Finds contours and keeps only the largest one (the pitch).
    5. Returns a binary mask (same H, W as frame, type np.uint8) where 255 represents the pitch.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold green
    lower_green = np.array([GREEN_HUE_MIN, GREEN_SAT_MIN, GREEN_VAL_MIN], dtype=np.uint8)
    upper_green = np.array([GREEN_HUE_MAX, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to clean up small noise specks and fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pitch_mask = np.zeros_like(mask)
    if contours:
        # Keep only the largest connected contour
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(pitch_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return pitch_mask
