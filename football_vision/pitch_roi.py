import logging
import cv2
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

def select_pitch_vertical_band(
    first_frame: np.ndarray,
    skip_roi_selection: bool = False
) -> Tuple[float, float]:
    """
    Selects the vertical band (PITCH_Y_MIN, PITCH_Y_MAX) of the playing field.

    ASSUMPTION: Camera only pans horizontally, does not tilt or zoom, so the vertical
    extent of the visible pitch stays roughly constant across the clip even as horizontal
    framing changes. This is a heuristic, not full field calibration — it will not work
    correctly if the camera tilts/zooms, and that limitation should be logged as a startup warning.
    """
    height, width = first_frame.shape[:2]
    default_min = 0.0
    default_max = float(height)

    # Log the startup heuristic warning
    logger.warning(
        "[PITCH ROI] WARNING: The pitch vertical-band filter assumes the camera ONLY pans "
        "horizontally, without tilting or zooming. If the camera tilts, zooms, or has dynamic "
        "vertical movement, this vertical band heuristic WILL NOT function correctly."
    )

    if skip_roi_selection:
        logger.info(
            f"[PITCH ROI] Skipped interactive selection. Using full frame height: "
            f"PITCH_Y_MIN={default_min:.2f}, PITCH_Y_MAX={default_max:.2f}"
        )
        return default_min, default_max

    # Setup interactive selection
    logger.info(
        "[PITCH ROI] Opening interactive selection window. "
        "Please click two points on the resized frame:\n"
        "  1. A point on the TOPMOST boundary of the playing area.\n"
        "  2. A point on the BOTTOMMOST boundary of the playing area.\n"
        "Press any key to finalize after making selections (or if you wish to exit/bypass)."
    )

    clicked_y = []
    display_frame = first_frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_y) < 2:
                clicked_y.append(float(y))
                # Draw a horizontal line at the selected y-coordinate
                color = (0, 255, 0) if len(clicked_y) == 1 else (0, 0, 255)
                label = "TOP Boundary" if len(clicked_y) == 1 else "BOTTOM Boundary"
                cv2.line(display_frame, (0, y), (width, y), color, 2)
                cv2.putText(
                    display_frame,
                    label,
                    (10, y - 10 if y - 10 > 15 else y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )
                cv2.imshow(window_name, display_frame)
                if len(clicked_y) == 1:
                    logger.info(f"[PITCH ROI] Click 1 (Top boundary) captured at y={y}")
                elif len(clicked_y) == 2:
                    logger.info(f"[PITCH ROI] Click 2 (Bottom boundary) captured at y={y}")

    window_name = "Pitch ROI Selection - Click Top then Bottom boundary"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Show initial frame
    cv2.imshow(window_name, display_frame)

    # Wait for user input
    while True:
        key = cv2.waitKey(100) & 0xFF
        # Break if any key is pressed or we have both coordinates and window is closed
        if key != 255:
            break
        # Also break if window is closed (OpenCV doesn't always support getWindowProperty easily, but we try)
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            pass

    cv2.destroyWindow(window_name)

    if len(clicked_y) == 2:
        y_min = min(clicked_y)
        y_max = max(clicked_y)
        logger.info(
            f"[PITCH ROI] Successfully selected pitch vertical band: "
            f"PITCH_Y_MIN={y_min:.2f}, PITCH_Y_MAX={y_max:.2f}"
        )
        return y_min, y_max
    elif len(clicked_y) == 1:
        logger.warning(
            f"[PITCH ROI] Incomplete selection (only 1 click registered). "
            f"Bypassing selection and using full frame height: "
            f"PITCH_Y_MIN={default_min:.2f}, PITCH_Y_MAX={default_max:.2f}"
        )
        return default_min, default_max
    else:
        logger.info(
            f"[PITCH ROI] No selections made. Using full frame height: "
            f"PITCH_Y_MIN={default_min:.2f}, PITCH_Y_MAX={default_max:.2f}"
        )
        return default_min, default_max
