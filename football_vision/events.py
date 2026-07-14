from typing import Dict, List, Any, Optional

class EventDetector:
    def __init__(self, gap_threshold: int = 45):
        """
        Initializes the event detector with a configurable threshold (in frames)
        for classifying transitions as "uncertain".
        """
        self.gap_threshold = gap_threshold

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
        # Filter possession data to include only frames where some player has possession
        active_possession = [
            (frame, track_id)
            for frame, track_id in sorted(frame_possession.items())
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
