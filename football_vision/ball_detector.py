import numpy as np
import supervision as sv
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class BallDetector:
    def __init__(
        self,
        max_interpolation_gap: int = 15,
        static_distance_threshold: float = 3.0,
        static_frames_threshold: int = 20
    ):
        """
        Initializes the Ball Detector.
        - max_interpolation_gap: maximum number of consecutive missing frames that can be interpolated.
        - static_distance_threshold: configurable distance threshold to identify static markings.
        - static_frames_threshold: configurable consecutive frame count threshold to identify static markings.
        """
        self.max_interpolation_gap = max_interpolation_gap
        self.static_distance_threshold = static_distance_threshold
        self.static_frames_threshold = static_frames_threshold
        # Track raw ball center positions (x, y) or None for each frame
        self.raw_positions: List[Optional[Tuple[float, float]]] = []
        # Track final positions after interpolation (x, y) or None
        self.interpolated_positions: List[Optional[Tuple[float, float]]] = []
        # Source tag for each frame: "detected", "interpolated", or "missing"
        self.sources: List[str] = []

    def process_frame_detections(self, detections: sv.Detections):
        """
        Processes detections for a single frame, filtered for "sports ball" (COCO class 32).
        Finds and records the single highest-confidence ball detection.
        If no ball is detected, records None.
        """
        if len(detections) == 0:
            self.raw_positions.append(None)
            return

        # Find the single highest-confidence ball detection
        highest_conf_idx = int(np.argmax(detections.confidence))
        bbox = detections.xyxy[highest_conf_idx]

        # Calculate the center (x, y) of the bounding box
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        self.raw_positions.append((float(center_x), float(center_y)))

    def filter_static_positions(self):
        """
        Filters raw positions to remove static pitch markings.
        If a ball candidate stays within self.static_distance_threshold of the same
        position for more than self.static_frames_threshold consecutive frames,
        it is discarded (set to None).
        """
        n = len(self.raw_positions)
        discarded_indices = set()

        # Identify all contiguous segments of non-None positions
        i = 0
        while i < n:
            if self.raw_positions[i] is not None:
                start_idx = i
                while i < n and self.raw_positions[i] is not None:
                    i += 1
                end_idx = i  # Exclusive

                segment_len = end_idx - start_idx
                # We need more than static_frames_threshold consecutive frames
                if segment_len > self.static_frames_threshold:
                    # Let's find static sub-segments
                    window_size = self.static_frames_threshold + 1
                    w_start = start_idx
                    while w_start <= end_idx - window_size:
                        ref_pos = self.raw_positions[w_start]
                        is_static_window = True
                        for k in range(w_start + 1, w_start + window_size):
                            pos = self.raw_positions[k]
                            dist = np.sqrt((pos[0] - ref_pos[0])**2 + (pos[1] - ref_pos[1])**2)
                            if dist > self.static_distance_threshold:
                                is_static_window = False
                                break

                        if is_static_window:
                            # Mark the entire window as static
                            for k in range(w_start, w_start + window_size):
                                discarded_indices.add(k)

                            # Extend the static segment
                            curr_idx = w_start + window_size
                            while curr_idx < end_idx:
                                pos = self.raw_positions[curr_idx]
                                dist = np.sqrt((pos[0] - ref_pos[0])**2 + (pos[1] - ref_pos[1])**2)
                                if dist <= self.static_distance_threshold:
                                    discarded_indices.add(curr_idx)
                                    curr_idx += 1
                                else:
                                    break
                            # Move w_start past this static segment
                            w_start = curr_idx
                        else:
                            w_start += 1
            else:
                i += 1

        if discarded_indices:
            logger.info(f"Filtering static candidates: Discarding {len(discarded_indices)} likely static pitch markings.")
            for idx in sorted(discarded_indices):
                pos = self.raw_positions[idx]
                logger.info(f"Discarded static candidate at frame {idx + 1}: pos={pos}")
                self.raw_positions[idx] = None

    def interpolate_gaps(self):
        """
        After processing all frames, fills short gaps of missing ball detections
        (up to self.max_interpolation_gap frames) via linear interpolation.
        Longer gaps stay marked as "missing" (and coordinate remains None).
        """
        self.filter_static_positions()
        n = len(self.raw_positions)
        self.interpolated_positions = [None] * n
        self.sources = ["missing"] * n

        # Populate with raw detections first
        for idx, pos in enumerate(self.raw_positions):
            if pos is not None:
                self.interpolated_positions[idx] = pos
                self.sources[idx] = "detected"

        # Interpolate gaps of None
        i = 0
        while i < n:
            if self.interpolated_positions[i] is None:
                start_gap = i
                while i < n and self.interpolated_positions[i] is None:
                    i += 1
                end_gap = i  # Index of next known position or n

                left_idx = start_gap - 1
                right_idx = end_gap

                gap_len = right_idx - left_idx - 1

                # We can interpolate only if we have known points on both sides,
                # and the gap size is <= max_interpolation_gap
                if left_idx >= 0 and right_idx < n and gap_len <= self.max_interpolation_gap:
                    # Linearly interpolate between left_idx and right_idx
                    x_start, y_start = self.interpolated_positions[left_idx]
                    x_end, y_end = self.interpolated_positions[right_idx]

                    for j in range(start_gap, end_gap):
                        t = (j - left_idx) / (right_idx - left_idx)
                        x_interp = x_start + t * (x_end - x_start)
                        y_interp = y_start + t * (y_end - y_start)
                        self.interpolated_positions[j] = (float(x_interp), float(y_interp))
                        self.sources[j] = "interpolated"
            else:
                i += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Aggregates and returns ball trajectory and detection statistics.
        Output format matching requirements:
        {
          "frames_detected": count,
          "frames_interpolated": count,
          "frames_missing": count,
          "trajectory": [ { "frame": frame_num, "x": x, "y": y, "source": "detected"|"interpolated" } ]
        }
        """
        frames_detected = sum(1 for s in self.sources if s == "detected")
        frames_interpolated = sum(1 for s in self.sources if s == "interpolated")
        frames_missing = sum(1 for s in self.sources if s == "missing")

        trajectory = []
        for idx, (pos, src) in enumerate(zip(self.interpolated_positions, self.sources)):
            if pos is not None:
                trajectory.append({
                    "frame": idx + 1,  # 1-based frame index
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "source": src
                })

        return {
            "frames_detected": frames_detected,
            "frames_interpolated": frames_interpolated,
            "frames_missing": frames_missing,
            "trajectory": trajectory
        }
