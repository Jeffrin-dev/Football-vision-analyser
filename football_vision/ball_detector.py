import logging
import numpy as np
import supervision as sv
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Constant for ball detector player proximity filtering (in pixels)
# start at 80 pixels, at the 640px-wide working resolution — may need adjusting based on typical player spacing observed
MAX_PLAUSIBLE_PLAYER_DISTANCE = 80.0

# Constant for ball trajectory continuity check (pixels/frame)
# Panning adds apparent motion on top of real ball speed so this needs to stay generous — better to under-reject than falsely reject a genuinely fast/long ball.
MAX_BALL_SPEED_PX_PER_FRAME = 40.0

# Number of initial confirmed detections to accept as-is to bootstrap without continuity check
BOOTSTRAP_CONFIRMED_COUNT = 3

# Constants for ball tracker staleness / anchor recovery
ANCHOR_STALENESS_LIMIT = 45
ANCHOR_JITTER_TOLERANCE = 5.0

class BallDetector:
    def __init__(
        self,
        max_interpolation_gap: int = 15,
        static_threshold_dist: float = 3.0,
        static_threshold_frames: int = 20,
        anchor_staleness_limit: int = ANCHOR_STALENESS_LIMIT
    ):
        """
        Initializes the Ball Detector.
        - max_interpolation_gap: maximum number of consecutive missing frames that can be interpolated.
        - static_threshold_dist: small pixel-distance threshold to consider a detection static.
        - static_threshold_frames: number of consecutive frames a candidate must stay static to be discarded.
        - anchor_staleness_limit: number of consecutive frames the anchor position stays static before being distrusted.
        """
        self.max_interpolation_gap = max_interpolation_gap
        self.static_threshold_dist = static_threshold_dist
        self.static_threshold_frames = static_threshold_frames
        self.anchor_staleness_limit = anchor_staleness_limit
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

        # Ball trajectory continuity state
        self.last_confirmed_pos: Optional[Tuple[float, float]] = None
        self.last_confirmed_frame: Optional[int] = None
        self.confirmed_count: int = 0
        self.stale_frame_count: int = 0

    def process_frame_detections(self, detections: sv.Detections, tracked_players: Optional[sv.Detections] = None):
        """
        Processes detections for a single frame, filtered for "sports ball" (COCO class 32).
        Uses proximity-to-player reranking to find and record the ball detection.
        If no ball is detected, records None.
        """
        frame_num = len(self.raw_positions) + 1
        prev_anchor = self.last_confirmed_pos
        is_distrusted = (prev_anchor is not None) and (self.stale_frame_count > self.anchor_staleness_limit)
        accepted_new = False

        if len(detections) == 0:
            self.raw_positions.append(None)
            self.raw_widths.append(None)

            # Update staleness state on early return
            if prev_anchor is not None:
                self.stale_frame_count += 1
                if self.stale_frame_count == self.anchor_staleness_limit + 1:
                    logger.warning(
                        f"[BALL DETECTOR] STALE ANCHOR DETECTED: Frame {frame_num} triggered a stale-anchor event. "
                        f"The anchor has been stuck for {self.stale_frame_count} frames at position "
                        f"({prev_anchor[0]:.2f}, {prev_anchor[1]:.2f}). Marking anchor as distrusted."
                    )

            logger.info(
                f"[BALL DETECTOR] Frame {frame_num}: raw_candidates=0, proximity_passing=0, "
                f"continuity_candidates_considered=0, chosen_speed=None, continuity_status=had-no-candidates, "
                f"decision=none"
            )
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

        if passing_candidates:
            # Check bootstrap status or distrusted status
            is_bootstrap = (self.last_confirmed_pos is None) or (self.confirmed_count < BOOTSTRAP_CONFIRMED_COUNT)
            use_fallback_selection = is_bootstrap or is_distrusted

            if use_fallback_selection:
                # Pick the highest confidence passing candidate without continuity checks
                best_item = max(passing_candidates, key=lambda x: x[0]["confidence"])
                chosen_candidate, nearest_distance = best_item

                # Update continuity states since this is a proximity-passing (confirmed) detection
                self.last_confirmed_pos = (chosen_candidate["x"], chosen_candidate["y"])
                self.last_confirmed_frame = frame_num
                self.confirmed_count += 1
                accepted_new = True

                self.raw_positions.append((chosen_candidate["x"], chosen_candidate["y"]))
                self.raw_widths.append(chosen_candidate["width"])

                decision_str = "distrusted_fallback" if is_distrusted else "bootstrap_normal"
                logger.info(
                    f"[BALL DETECTOR] Frame {frame_num}: raw_candidates={len(candidates)}, "
                    f"proximity_passing={len(passing_candidates)}, continuity_candidates_considered={len(passing_candidates)}, "
                    f"chosen_speed=None, continuity_status=passed, decision={decision_str}, "
                    f"x={chosen_candidate['x']:.2f}, y={chosen_candidate['y']:.2f}"
                )
            else:
                # Not bootstrap/distrusted: perform continuity check
                # Compute implied speed for each passing candidate
                frames_elapsed = frame_num - self.last_confirmed_frame
                for item in passing_candidates:
                    cand = item[0]
                    dx = cand["x"] - self.last_confirmed_pos[0]
                    dy = cand["y"] - self.last_confirmed_pos[1]
                    disp = np.sqrt(dx**2 + dy**2)
                    cand["implied_speed"] = disp / frames_elapsed

                valid_continuity_candidates = [
                    item for item in passing_candidates
                    if item[0]["implied_speed"] <= MAX_BALL_SPEED_PX_PER_FRAME
                ]

                if valid_continuity_candidates:
                    # Choose highest-confidence candidate among the valid ones
                    best_item = max(valid_continuity_candidates, key=lambda x: x[0]["confidence"])
                    chosen_candidate, nearest_distance = best_item

                    # Update continuity states with confirmed detection
                    self.last_confirmed_pos = (chosen_candidate["x"], chosen_candidate["y"])
                    self.last_confirmed_frame = frame_num
                    self.confirmed_count += 1
                    accepted_new = True

                    self.raw_positions.append((chosen_candidate["x"], chosen_candidate["y"]))
                    self.raw_widths.append(chosen_candidate["width"])

                    logger.info(
                        f"[BALL DETECTOR] Frame {frame_num}: raw_candidates={len(candidates)}, "
                        f"proximity_passing={len(passing_candidates)}, continuity_candidates_considered={len(passing_candidates)}, "
                        f"chosen_speed={chosen_candidate['implied_speed']:.2f}, continuity_status=passed, decision=normal, "
                        f"x={chosen_candidate['x']:.2f}, y={chosen_candidate['y']:.2f}"
                    )
                else:
                    # All passing candidates exceed the speed limit: do not force a pick
                    self.raw_positions.append(None)
                    self.raw_widths.append(None)

                    logger.info(
                        f"[BALL DETECTOR] Frame {frame_num}: raw_candidates={len(candidates)}, "
                        f"proximity_passing={len(passing_candidates)}, continuity_candidates_considered={len(passing_candidates)}, "
                        f"chosen_speed=None, continuity_status=failed, decision=none_no_plausible_continuation"
                    )
        else:
            # Zero candidates pass proximity: fallback to highest-confidence raw candidate
            # This does NOT update confirmed positions
            chosen_candidate = None
            nearest_distance = float('inf')

            if candidates_with_dists:
                best_item = max(candidates_with_dists, key=lambda x: x[0]["confidence"])
                chosen_candidate, nearest_distance = best_item

            if chosen_candidate is not None:
                self.raw_positions.append((chosen_candidate["x"], chosen_candidate["y"]))
                self.raw_widths.append(chosen_candidate["width"])

                logger.info(
                    f"[BALL DETECTOR] Frame {frame_num}: raw_candidates={len(candidates)}, "
                    f"proximity_passing=0, continuity_candidates_considered=0, chosen_speed=None, "
                    f"continuity_status=had-no-candidates, decision=fallback, tag=no_player_proximity_fallback, "
                    f"x={chosen_candidate['x']:.2f}, y={chosen_candidate['y']:.2f}"
                )
            else:
                self.raw_positions.append(None)
                self.raw_widths.append(None)

                logger.info(
                    f"[BALL DETECTOR] Frame {frame_num}: raw_candidates={len(candidates)}, "
                    f"proximity_passing=0, continuity_candidates_considered=0, chosen_speed=None, "
                    f"continuity_status=had-no-candidates, decision=none"
                )

        # Update staleness state
        if prev_anchor is not None:
            if is_distrusted:
                if accepted_new:
                    self.stale_frame_count = 0
                else:
                    self.stale_frame_count += 1
            else:
                if self.last_confirmed_pos is not None:
                    dx = self.last_confirmed_pos[0] - prev_anchor[0]
                    dy = self.last_confirmed_pos[1] - prev_anchor[1]
                    movement = np.sqrt(dx**2 + dy**2)
                    if movement < ANCHOR_JITTER_TOLERANCE:
                        self.stale_frame_count += 1
                    else:
                        self.stale_frame_count = 0
                else:
                    self.stale_frame_count += 1

            if self.stale_frame_count == self.anchor_staleness_limit + 1:
                logger.warning(
                    f"[BALL DETECTOR] STALE ANCHOR DETECTED: Frame {frame_num} triggered a stale-anchor event. "
                    f"The anchor has been stuck for {self.stale_frame_count} frames at position "
                    f"({prev_anchor[0]:.2f}, {prev_anchor[1]:.2f}). Marking anchor as distrusted."
                )

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
