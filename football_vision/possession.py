import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Tunable constant for ball proximity (in pixels) for a 640px-wide frame.
# This is moderately tightened (e.g. 30 pixels) so only closer proximity counts as possession.
# Note: This reduces false locks in ambiguous cases but does not fully solve possession accuracy —
# true ball-contact detection is out of scope for this phase.
PROXIMITY_THRESHOLD = 30.0
POSSESSION_MARGIN_RATIO = 0.15

class PossessionTracker:
    def __init__(
        self,
        proximity_threshold: float = PROXIMITY_THRESHOLD,
        possession_margin_ratio: float = POSSESSION_MARGIN_RATIO
    ):
        """
        Initializes the possession tracker.
        - proximity_threshold: maximum distance in pixels between ball and player to consider possession.
        - possession_margin_ratio: margin check percentage where closest player must be at least this much closer than the second closest.
        """
        self.proximity_threshold = proximity_threshold
        self.possession_margin_ratio = possession_margin_ratio

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
        Includes the margin requirement where the closest player must be closer than the second-closest by at least the margin.
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

            player_distances = []
            for track_id, bbox in players:
                x_min, y_min, x_max, y_max = bbox
                player_x = (x_min + x_max) / 2.0
                player_y = (y_min + y_max) / 2.0

                dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                player_distances.append((track_id, dist))

            # Sort by distance
            player_distances.sort(key=lambda x: x[1])

            is_possession_assigned = False
            assigned_player_id = None

            if len(player_distances) > 0:
                closest_id, dist_1 = player_distances[0]
                if dist_1 <= self.proximity_threshold:
                    if len(player_distances) > 1:
                        second_closest_id, dist_2 = player_distances[1]
                        if dist_1 <= dist_2 * (1.0 - self.possession_margin_ratio):
                            is_possession_assigned = True
                            assigned_player_id = closest_id
                            logger.debug(
                                f"[POSSESSION] Frame {frame_idx}: Clear winner. Player #{closest_id} (dist={dist_1:.2f}) "
                                f"vs Player #{second_closest_id} (dist={dist_2:.2f}), margin check passed."
                            )
                        else:
                            logger.info(
                                f"[POSSESSION] Frame {frame_idx}: Contested due to margin. Player #{closest_id} (dist={dist_1:.2f}) "
                                f"vs Player #{second_closest_id} (dist={dist_2:.2f}), margin check failed."
                            )
                    else:
                        is_possession_assigned = True
                        assigned_player_id = closest_id

            if is_possession_assigned and assigned_player_id is not None:
                self.frame_possession[frame_idx] = assigned_player_id
            else:
                self.frame_possession[frame_idx] = None  # contested/none

    def aggregate_possession_stats(self, team_assignments: Dict[int, str]):
        """
        Aggregates total possession frames per player and per team based on computed frame possession.
        """
        pass

    def compute_and_aggregate_possession(
        self,
        frame_count: int,
        ball_positions: List[Optional[Tuple[float, float]]],
        team_assignments: Dict[int, str]
    ):
        """
        Computes frame-by-frame possession and aggregates the stats for teams and players.
        Includes the margin requirement where the closest player must be closer than the second-closest by at least the margin.
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

            player_distances = []
            for track_id, bbox in players:
                # Calculate player center
                x_min, y_min, x_max, y_max = bbox
                player_x = (x_min + x_max) / 2.0
                player_y = (y_min + y_max) / 2.0

                dist = np.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                player_distances.append((track_id, dist))

            player_distances.sort(key=lambda x: x[1])

            is_possession_assigned = False
            assigned_player_id = None

            if len(player_distances) > 0:
                closest_id, dist_1 = player_distances[0]
                if dist_1 <= self.proximity_threshold:
                    if len(player_distances) > 1:
                        second_closest_id, dist_2 = player_distances[1]
                        if dist_1 <= dist_2 * (1.0 - self.possession_margin_ratio):
                            is_possession_assigned = True
                            assigned_player_id = closest_id
                            logger.debug(
                                f"[POSSESSION] Frame {frame_idx}: Clear winner. Player #{closest_id} (dist={dist_1:.2f}) "
                                f"vs Player #{second_closest_id} (dist={dist_2:.2f}), margin check passed."
                            )
                        else:
                            logger.info(
                                f"[POSSESSION] Frame {frame_idx}: Contested due to margin. Player #{closest_id} (dist={dist_1:.2f}) "
                                f"vs Player #{second_closest_id} (dist={dist_2:.2f}), margin check failed."
                            )
                    else:
                        is_possession_assigned = True
                        assigned_player_id = closest_id

            if is_possession_assigned and assigned_player_id is not None:
                self.frame_possession[frame_idx] = assigned_player_id
                # Update player count
                self.player_possession_counts[assigned_player_id] = self.player_possession_counts.get(assigned_player_id, 0) + 1
                # Update team count
                team = team_assignments.get(assigned_player_id, "A")  # Default to "A" if unassigned
                self.team_possession_counts[team] = self.team_possession_counts.get(team, 0) + 1
            else:
                self.frame_possession[frame_idx] = None  # contested/none
                self.team_possession_counts["contested"] += 1
