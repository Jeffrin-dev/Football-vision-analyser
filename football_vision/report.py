import json
import os
from typing import Dict, List, Tuple
import numpy as np

class ReportGenerator:
    def __init__(self, clip_name: str, frame_count: int):
        self.clip_name = clip_name
        self.frame_count = frame_count

    def generate_report(
        self,
        team_assignments: Dict[int, str],
        track_coords: Dict[int, List[Tuple[float, float]]],
        output_dir: str
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
              "avg_position": [x, y]
            }
          ]
        }
        Returns the path to the report.json.
        """
        os.makedirs(output_dir, exist_ok=True)

        players_report = []
        for track_id, coords in track_coords.items():
            team = team_assignments.get(track_id, "A")
            frames_tracked = len(coords)

            if frames_tracked > 0:
                coords_np = np.array(coords)
                avg_pos = np.mean(coords_np, axis=0).tolist() # [x, y]
            else:
                avg_pos = [0.0, 0.0]

            players_report.append({
                "track_id": int(track_id),
                "team": team,
                "frames_tracked": int(frames_tracked),
                "avg_position": [float(avg_pos[0]), float(avg_pos[1])]
            })

        report_data = {
            "clip": self.clip_name,
            "frame_count": int(self.frame_count),
            "players": players_report
        }

        report_path = os.path.join(output_dir, "report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=4)

        return report_path
