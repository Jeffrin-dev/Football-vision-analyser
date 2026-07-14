from typing import Dict, List, Any, Optional

class EventDetector:
    def __init__(self, gap_threshold: int = 45, min_consecutive_frames: int = 8):
        """
        Initializes the event detector with a configurable threshold (in frames)
        for classifying transitions as "uncertain", and a min_consecutive_frames
        threshold for smoothing possession noise.
        """
        self.gap_threshold = gap_threshold
        self.min_consecutive_frames = min_consecutive_frames

    def smooth_possession(
        self,
        frame_possession: Dict[int, Optional[int]]
    ) -> Dict[int, Optional[int]]:
        """
        Filters player possession so a player only counts as holding possession
        if they are the nearest-player-within-threshold for at least N consecutive frames.
        Brief flickers below this threshold are treated as noise and folded into
        the closest/surrounding valid possession segment (or None if no valid segments exist).
        Any original None (contested/none) frames are kept as None.
        """
        sorted_frames = sorted(frame_possession.keys())
        if not sorted_frames:
            return {}

        runs = []  # List of tuples: (track_id, [list of consecutive frames])
        current_run_track_id = None
        current_run_frames = []

        for f in sorted_frames:
            track_id = frame_possession[f]
            if track_id is None:
                if current_run_frames:
                    runs.append((current_run_track_id, current_run_frames))
                    current_run_frames = []
                    current_run_track_id = None
            else:
                if current_run_track_id == track_id:
                    # Check if consecutive frame number
                    if current_run_frames and f == current_run_frames[-1] + 1:
                        current_run_frames.append(f)
                    else:
                        if current_run_frames:
                            runs.append((current_run_track_id, current_run_frames))
                        current_run_track_id = track_id
                        current_run_frames = [f]
                else:
                    if current_run_frames:
                        runs.append((current_run_track_id, current_run_frames))
                    current_run_track_id = track_id
                    current_run_frames = [f]

        if current_run_frames:
            runs.append((current_run_track_id, current_run_frames))

        # Separate runs into valid segments (length >= min_consecutive_frames) and invalid frames
        valid_segments = []
        invalid_frames_with_track = []

        for track_id, run_frames in runs:
            if len(run_frames) >= self.min_consecutive_frames:
                valid_segments.append((track_id, run_frames[0], run_frames[-1]))
            else:
                for f in run_frames:
                    invalid_frames_with_track.append((f, track_id))

        # Construct the smoothed possession dictionary
        # Start with a copy of original keys mapped to None
        smoothed = {f: None for f in sorted_frames}

        # Fill in valid segments
        for track_id, start_f, end_f in valid_segments:
            for f in range(start_f, end_f + 1):
                if f in smoothed:
                    smoothed[f] = track_id

        # For invalid frames, fold them into the closest valid segment
        for f, orig_track in invalid_frames_with_track:
            if not valid_segments:
                smoothed[f] = None
            else:
                closest_segment_track = None
                min_dist = float('inf')
                for track_id, start_f, end_f in valid_segments:
                    if f < start_f:
                        dist = start_f - f
                    elif f > end_f:
                        dist = f - end_f
                    else:
                        dist = 0

                    if dist < min_dist:
                        min_dist = dist
                        closest_segment_track = track_id

                smoothed[f] = closest_segment_track

        # Explicitly ensure any frames that were originally None are kept as None
        for f in sorted_frames:
            if frame_possession[f] is None:
                smoothed[f] = None

        return smoothed

    def detect_events(
        self,
        frame_possession: Dict[int, Optional[int]],
        team_assignments: Dict[int, str]
    ) -> List[Dict[str, Any]]:
        """
        Walks through the frame-by-frame possession data in order and detects possession changes.

        A possession change is detected when the player holding the ball changes from one frame
        to a later frame where a different player has possession (skipping over "contested/none" gaps).

        Classifies each possession change as:
        - "pass" if the new holder is on the SAME team as the previous holder.
        - "turnover" if the new holder is on a DIFFERENT team.

        If the gap between the last-known holder's last frame and the new holder's first frame
        is greater than the gap_threshold, the event is marked as "uncertain".
        """
        # Step 1: Smooth possession to filter out proximity noise and flickers
        smoothed_possession = self.smooth_possession(frame_possession)

        # Step 2: Filter smoothed possession data to include only frames where some player has possession
        active_possession = [
            (frame, track_id)
            for frame, track_id in sorted(smoothed_possession.items())
            if track_id is not None
        ]

        events = []
        if not active_possession:
            return events

        # Initialize tracking variables with the first active possession frame
        last_frame, last_track_id = active_possession[0]

        for curr_frame, curr_track_id in active_possession[1:]:
            if curr_track_id != last_track_id:
                # A possession change detected!
                from_team = team_assignments.get(last_track_id, "A")
                to_team = team_assignments.get(curr_track_id, "A")

                # Check gap
                gap = curr_frame - last_frame
                if gap > self.gap_threshold:
                    event_type = "uncertain"
                elif from_team == to_team:
                    event_type = "pass"
                else:
                    event_type = "turnover"

                events.append({
                    "frame": int(curr_frame),
                    "type": event_type,
                    "from_track_id": int(last_track_id),
                    "from_team": from_team,
                    "to_track_id": int(curr_track_id),
                    "to_team": to_team
                })

                # Update the last_track_id and last_frame
                last_track_id = curr_track_id
                last_frame = curr_frame
            else:
                # Same player holding the ball, update their last-seen frame
                last_frame = curr_frame

        return events

    def generate_summary(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Generates summary counts of the detected events.
        """
        summary = {
            "passes": 0,
            "turnovers": 0,
            "uncertain": 0
        }
        for event in events:
            etype = event["type"]
            if etype == "pass":
                summary["passes"] += 1
            elif etype == "turnover":
                summary["turnovers"] += 1
            elif etype == "uncertain":
                summary["uncertain"] += 1
        return summary
