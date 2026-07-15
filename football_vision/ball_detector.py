import logging
import numpy as np
import supervision as sv
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Constant for ball detector player proximity filtering (in pixels)
# start at 80 pixels, at the 640px-wide working resolution — may need adjusting based on typical player spacing observed
MAX_PLAUSIBLE_PLAYER_DISTANCE = 80.0

class BallDetector:
    def __init__(
        self,
        max_interpolation_gap: int = 15,
        static_threshold_dist: float = 3.0,
        static_threshold_frames: int = 20
    ):
        """
        Initializes the Ball Detector.
        - max_interpolation_gap: maximum number of consecutive missing frames that can be interpolated.
        - static_threshold_dist: small pixel-distance threshold to consider a detection static.
        - static_threshold_frames: number of consecutive frames a candidate must stay static to be discarded.
        """
        self.max_interpolation_gap = max_interpolation_gap
        self.static_threshold_dist = static_threshold_dist
        self.static_threshold_frames = static_threshold_frames
        # Track raw ball center positions (x, y) or None for each frame
        self.raw_positions: List[Optional[Tuple[float, float]]] = []
        # Track raw ball bounding box widths or None for each frame
        self.raw_widths: List[Optional[float]] = []
        # Track final positions after interpolation (x, y) or None
        self.interpolated_positions: List[Optional[Tuple[float, float]]] = []
        # Track final widths after interpolation or None
        self.interpolated_widths: List[Optional[float]] = []
        # Source tag for each frame: "detected", "interpolated", or "missing"
        self.sources: List[str] = []

    def process_frame_detections(self, detections: sv.Detections, tracked_players: Optional[sv.Detections] = None):
        """
        Processes detections for a single frame, filtered for "sports ball" (COCO class 32).
        Uses proximity-to-player reranking to find and record the ball detection.
        If no ball is detected, records None.
        """
        if len(detections) == 0:
            self.raw_positions.append(None)
            self.raw_widths.append(None)
            frame_num = len(self.raw_positions)
            logger.info(f"[BALL DETECTOR] Frame {frame_num}: raw_candidates=0, decision=none")
            return

        # 1. Collect ALL "sports ball" class detections per frame, each with (x, y, confidence, bbox_width)
        candidates = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            x_min, y_min, x_max, y_max = bbox
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
            width = float(x_max - x_min)
            candidates.append({
                "x": center_x,
                "y": center_y,
                "confidence": conf,
                "width": width,
                "bbox": bbox
            })

        # 2. Extract player bounding box centers in that same frame
        player_centers = []
        if tracked_players is not None and tracked_players.xyxy is not None:
            for bbox in tracked_players.xyxy:
                x_min, y_min, x_max, y_max = bbox
                player_centers.append(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))

        # 3. For each candidate, compute distance to nearest player center
        candidates_with_dists = []
        for cand in candidates:
            cx, cy = cand["x"], cand["y"]
            min_dist = float('inf')
            for px, py in player_centers:
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < min_dist:
                    min_dist = dist
            candidates_with_dists.append((cand, min_dist))

        # 4. Filter candidates within MAX_PLAUSIBLE_PLAYER_DISTANCE
        passing_candidates = [item for item in candidates_with_dists if item[1] <= MAX_PLAUSIBLE_PLAYER_DISTANCE]

        chosen_candidate = None
        is_fallback = False
        nearest_distance = float('inf')

        if passing_candidates:
            # Pick highest confidence passing candidate
            best_item = max(passing_candidates, key=lambda x: x[0]["confidence"])
            chosen_candidate, nearest_distance = best_item
        else:
            # Zero candidates pass: fallback to highest-confidence raw candidate
            if candidates_with_dists:
                is_fallback = True
                best_item = max(candidates_with_dists, key=lambda x: x[0]["confidence"])
                chosen_candidate, nearest_distance = best_item

        frame_num = len(self.raw_positions) + 1

        if chosen_candidate is not None:
            self.raw_positions.append((chosen_candidate["x"], chosen_candidate["y"]))
            self.raw_widths.append(chosen_candidate["width"])

            # Log decision
            if is_fallback:
                logger.info(
                    f"[BALL DETECTOR] Frame {frame_num}: raw_candidates={len(candidates)}, "
                    f"nearest_player_distance={nearest_distance:.2f}, decision=fallback, "
                    f"tag=no_player_proximity_fallback, x={chosen_candidate['x']:.2f}, y={chosen_candidate['y']:.2f}"
                )
            else:
                logger.info(
                    f"[BALL DETECTOR] Frame {frame_num}: raw_candidates={len(candidates)}, "
                    f"nearest_player_distance={nearest_distance:.2f}, decision=normal, "
                    f"x={chosen_candidate['x']:.2f}, y={chosen_candidate['y']:.2f}"
                )
        else:
            self.raw_positions.append(None)
            self.raw_widths.append(None)
            logger.info(f"[BALL DETECTOR] Frame {frame_num}: raw_candidates=0, decision=none")

    def apply_motion_plausibility_filter(self):
        """
        Deactivated. Proximity-to-player is now the primary signal.
        """
        pass

    def interpolate_gaps(self):
        """
        After processing all frames, fills short gaps of missing ball detections
        (up to self.max_interpolation_gap frames) via linear interpolation.
        Longer gaps stay marked as "missing" (and coordinate remains None).
        """
        self.apply_motion_plausibility_filter()

        n = len(self.raw_positions)
        self.interpolated_positions = [None] * n
        self.interpolated_widths = [None] * n
        self.sources = ["missing"] * n

        # Populate with raw detections first
        for idx, pos in enumerate(self.raw_positions):
            if pos is not None:
                self.interpolated_positions[idx] = pos
                self.interpolated_widths[idx] = self.raw_widths[idx]
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
                    x_start, y_start = self.interpolated_positions[left_idx]
                    x_end, y_end = self.interpolated_positions[right_idx]

                    w_start = self.interpolated_widths[left_idx]
                    w_end = self.interpolated_widths[right_idx]

                    if w_start is None:
                        w_start = 8.0
                    if w_end is None:
                        w_end = 8.0

                    for j in range(start_gap, end_gap):
                        t = (j - left_idx) / (right_idx - left_idx)
                        x_interp = x_start + t * (x_end - x_start)
                        y_interp = y_start + t * (y_end - y_start)
                        w_interp = w_start + t * (w_end - w_start)
                        self.interpolated_positions[j] = (float(x_interp), float(y_interp))
                        self.interpolated_widths[j] = float(w_interp)
                        self.sources[j] = "interpolated"
            else:
                i += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Aggregates and returns ball trajectory and detection statistics.
        """
        frames_detected = sum(1 for s in self.sources if s == "detected")
        frames_interpolated = sum(1 for s in self.sources if s == "interpolated")
        frames_missing = sum(1 for s in self.sources if s == "missing")

        trajectory = []
        for idx, (pos, src) in enumerate(zip(self.interpolated_positions, self.sources)):
            if pos is not None:
                width = self.interpolated_widths[idx] if idx < len(self.interpolated_widths) else None
                trajectory.append({
                    "frame": idx + 1,  # 1-based frame index
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "width": float(width) if width is not None else 8.0,
                    "source": src
                })

        return {
            "frames_detected": frames_detected,
            "frames_interpolated": frames_interpolated,
            "frames_missing": frames_missing,
            "trajectory": trajectory
        }
