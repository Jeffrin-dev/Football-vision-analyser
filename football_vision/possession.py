import numpy as np
from typing import Dict, List, Tuple, Optional

# Tunable constant for ball proximity (in pixels) for a 640px-wide frame.
# This default (e.g. 40 pixels) is a sensible distance representation for players close to the ball.
PROXIMITY_THRESHOLD = 40.0

class PossessionTracker:
    def __init__(self, proximity_threshold: float = PROXIMITY_THRESHOLD):
        """
        Initializes the possession tracker.
        - proximity_threshold: maximum distance in pixels between ball and player to consider possession.
        """
        self.proximity_threshold = proximity_threshold
        # Maps frame_idx (0-based) -> list of player bounding boxes and track_ids in that frame
        # List[Tuple[track_id, (x_min, y_min, x_max, y_max)]]
        self.frame_player_boxes: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}

        # Possession assigned per frame: None if missing/unknown, otherwise (track_id, player_center, distance) or None if contested/none
        # Let's record as: {frame_idx: assigned_track_id_or_none}
        self.frame_possession: Dict[int, Optional[int]] = {}

        # Aggregated stats per player: track_id -> possession_frame_count
        self.player_possession_counts: Dict[int, int] = {}
        # Aggregated stats per team: "A" -> count, "B" -> count, "contested" -> count
        self.team_possession_counts: Dict[str, int] = {
            "A": 0,
            "B": 0,
            "contested": 0
        }

    def record_players(self, frame_idx: int, tracked_detections):
        """
        Records the tracked players' bounding boxes for the given frame.
        """
        if frame_idx not in self.frame_player_boxes:
            self.frame_player_boxes[frame_idx] = []

        if tracked_detections.tracker_id is not None:
            for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                self.frame_player_boxes[frame_idx].append((int(track_id), tuple(bbox)))

    def compute_possession(self, frame_count: int, ball_positions: List[Optional[Tuple[float, float]]]):
        """
        Calculates proximity-based possession for each frame where the ball position is known (detected or interpolated).
        For each frame:
        - Calculate Euclidean distance from ball center to each player's position (center of bounding box).
        - Assign possession to the closest player within proximity_threshold.
        - If no player is within threshold, assign possession as "contested" (represented as None).
        - If the ball is missing in a frame, do not compute possession for that frame (do not assign or count).
        Finally, aggregates the total possession frames per player and per team into counts.
        """
        # Ensure we initialize/reset counts
        self.player_possession_counts = {}
        self.team_possession_counts = {
            "A": 0,
            "B": 0,
            "contested": 0
        }

        for frame_idx in range(frame_count):
            ball_pos = ball_positions[frame_idx] if frame_idx < len(ball_positions) else None

            if ball_pos is None:
                # Ball is missing/unknown, so do not compute possession or count it.
                self.frame_possession[frame_idx] = None
                continue

            ball_x, ball_y = ball_pos
            players = self.frame_player_boxes.get(frame_idx, [])

            closest_player_id = None
            min_dist = float('inf')

            for track_id, bbox in players:
                # Calculate player center
                x_min, y_min, x_max, y_max = bbox
                player_x = (x_min + x_max) / 2.0
                player_y = (y_min + y_max) / 2.0

                dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_player_id = track_id

            if min_dist <= self.proximity_threshold and closest_player_id is not None:
                self.frame_possession[frame_idx] = closest_player_id
            else:
                self.frame_possession[frame_idx] = None  # contested/none

    def aggregate_possession_stats(self, team_assignments: Dict[int, str]):
        """
        Aggregates total possession frames per player and per team based on computed frame possession.
        """
        for frame_idx, assigned_id in self.frame_possession.items():
            # If the ball was missing/unknown for this frame, we skipped it
            # We must distinguish between:
            # 1. Ball was missing: self.frame_possession[frame_idx] is None, but actually ball_pos is None.
            # 2. Ball was present but no player was close: assigned_id is None.
            # To handle this cleanly, we check if the frame is in our recorded player boxes and had a ball.
            # But wait, we can just check if ball_pos was known. Let's make sure we only aggregate if the frame was evaluated.
            pass

        # Since we set self.frame_possession[frame_idx] = None for both missing ball and contested,
        # let's be more precise. Let's rerun the aggregation cleanly inside compute_possession or pass a reference to ball_positions.
        pass

    def compute_and_aggregate_possession(self, frame_count: int, ball_positions: List[Optional[Tuple[float, float]]], team_assignments: Dict[int, str]):
        """
        Computes frame-by-frame possession and aggregates the stats for teams and players.
        """
        self.player_possession_counts = {}
        self.team_possession_counts = {
            "A": 0,
            "B": 0,
            "contested": 0
        }

        for frame_idx in range(frame_count):
            ball_pos = ball_positions[frame_idx] if frame_idx < len(ball_positions) else None

            if ball_pos is None:
                continue

            ball_x, ball_y = ball_pos
            players = self.frame_player_boxes.get(frame_idx, [])

            closest_player_id = None
            min_dist = float('inf')

            for track_id, bbox in players:
                # Calculate player center
                x_min, y_min, x_max, y_max = bbox
                player_x = (x_min + x_max) / 2.0
                player_y = (y_min + y_max) / 2.0

                dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_player_id = track_id

            if min_dist <= self.proximity_threshold and closest_player_id is not None:
                self.frame_possession[frame_idx] = closest_player_id
                # Update player count
                self.player_possession_counts[closest_player_id] = self.player_possession_counts.get(closest_player_id, 0) + 1
                # Update team count
                team = team_assignments.get(closest_player_id, "A")  # Default to "A" if unassigned
                self.team_possession_counts[team] = self.team_possession_counts.get(team, 0) + 1
            else:
                self.frame_possession[frame_idx] = None  # contested/none
                self.team_possession_counts["contested"] += 1
