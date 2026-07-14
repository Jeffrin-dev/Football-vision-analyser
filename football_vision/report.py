import json
import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class ReportGenerator:
    def __init__(self, clip_name: str, frame_count: int):
        self.clip_name = clip_name
        self.frame_count = frame_count

    def generate_report(
        self,
        team_assignments: Dict[int, str],
        track_coords: Dict[int, List[Tuple[float, float]]],
        output_dir: str,
        ball_stats: Optional[Dict[str, Any]] = None,
        player_possession_counts: Optional[Dict[int, int]] = None,
        team_possession: Optional[Dict[str, int]] = None
    ) -> str:
        """
        Generates and writes a report.json file in the output directory.
        JSON format:
        {
          "clip": "clip.mp4",
          "frame_count": 150,
          "players": [
            {
              "track_id": 1,
              "team": "A",
              "frames_tracked": 45,
              "avg_position": [x, y],
              "possession_frames": 10  # Added in Phase 2
            }
          ],
          "ball": {  # Added in Phase 2
            "frames_detected": 100,
            "frames_interpolated": 10,
            "frames_missing": 40,
            "trajectory": [
              {"frame": 1, "x": 100.0, "y": 200.0, "source": "detected"}
            ]
          },
          "team_possession": {  # Added in Phase 2
            "A": 50,
            "B": 40,
            "contested": 20
          }
        }
        Returns the path to the report.json.
        """
        os.makedirs(output_dir, exist_ok=True)

        if player_possession_counts is None:
            player_possession_counts = {}

        players_report = []
        for track_id, coords in track_coords.items():
            team = team_assignments.get(track_id, "A")
            frames_tracked = len(coords)

            if frames_tracked > 0:
                coords_np = np.array(coords)
                avg_pos = np.mean(coords_np, axis=0).tolist() # [x, y]
            else:
                avg_pos = [0.0, 0.0]

            player_entry = {
                "track_id": int(track_id),
                "team": team,
                "frames_tracked": int(frames_tracked),
                "avg_position": [float(avg_pos[0]), float(avg_pos[1])]
            }

            # Additive: Include possession_frames in each player's stats
            possession_frames = player_possession_counts.get(int(track_id), 0)
            player_entry["possession_frames"] = int(possession_frames)

            players_report.append(player_entry)

        report_data = {
            "clip": self.clip_name,
            "frame_count": int(self.frame_count),
            "players": players_report
        }

        # Additive: Include top-level "ball" section if provided
        if ball_stats is not None:
            report_data["ball"] = ball_stats

        # Additive: Include top-level "team_possession" section if provided
        if team_possession is not None:
            report_data["team_possession"] = team_possession

        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=4)

        return report_path
