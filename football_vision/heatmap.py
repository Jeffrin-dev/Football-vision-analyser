import os
from typing import Dict, List, Tuple, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class HeatmapGenerator:
    def __init__(self, width: int, height: int):
        """
        Initializes the Heatmap Generator.
        width and height represent the dimensions of the resized video frames.
        """
        self.width = width
        self.height = height
        # Track coordinates of each track_id: Dict[track_id, List[Tuple[float, float]]]
        self.track_coords: Dict[int, List[Tuple[float, float]]] = {}

    def accumulate_position(self, track_id: int, bbox: Tuple[float, float, float, float]):
        """
        Accumulates the center (x, y) of the player's bounding box.
        """
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        if track_id not in self.track_coords:
            self.track_coords[track_id] = []
        self.track_coords[track_id].append((center_x, center_y))

    def generate_and_save_heatmaps(
        self,
        team_assignments: Dict[int, str],
        output_dir: str
    ) -> Tuple[str, str]:
        """
        Generates and saves two heatmaps (one per team) in pixel-space using matplotlib.
        Returns the file paths to the generated PNGs.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Accumulate coordinates for Team A and Team B
        coords_team_a = []
        coords_team_b = []

        for track_id, coords in self.track_coords.items():
            team = team_assignments.get(track_id, "A") # Default to Team A if unassigned
            if team == "A":
                coords_team_a.extend(coords)
            else:
                coords_team_b.extend(coords)

        # Generate paths
        path_a = os.path.join(output_dir, "heatmap_team_a.png")
        path_b = os.path.join(output_dir, "heatmap_team_b.png")

        self._plot_and_save(coords_team_a, "Team A Heatmap", path_a)
        self._plot_and_save(coords_team_b, "Team B Heatmap", path_b)

        return path_a, path_b

    def _plot_and_save(self, coords: List[Tuple[float, float]], title: str, save_path: str):
        """
        Plots a 2D density/heatmap using Matplotlib's hexbin or 2D histogram.
        """
        plt.figure(figsize=(8, 6))

        if len(coords) > 0:
            x, y = zip(*coords)
            # Create a 2D histogram or hexbin
            # OpenCV coordinate system: y starts at top (0) and increases downwards.
            # Matplotlib by default has y-axis increasing upwards, so we invert the y-axis to match the pixel space.
            plt.hexbin(x, y, gridsize=30, cmap='YlOrRd', mincnt=1)
            plt.colorbar(label='Frequency')
        else:
            # Empty plot with placeholder
            plt.text(self.width / 2, self.height / 2, "No Data", ha='center', va='center', fontsize=14)

        plt.xlim(0, self.width)
        plt.ylim(self.height, 0) # Invert y-axis to match video frame (0 at top)
        plt.title(title)
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

    def generate_and_save_ball_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        output_dir: str
    ) -> str:
        """
        Generates and saves the ball_trajectory.png plotting the ball's path over the pitch.
        Detected points: solid/vibrant marker.
        Interpolated points: lighter/dashed or different marker.
        Returns the path to the saved PNG.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "ball_trajectory.png")

        plt.figure(figsize=(8, 6))

        if len(trajectory) > 0:
            # Parse trajectory data
            frames = [pt["frame"] for pt in trajectory]
            xs = [pt["x"] for pt in trajectory]
            ys = [pt["y"] for pt in trajectory]
            sources = [pt["source"] for pt in trajectory]

            # 1. Plot continuous light grey dashed line representing the sequential path
            plt.plot(xs, ys, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Sequential Path")

            # 2. Extract and plot detected coordinates
            det_xs = [xs[i] for i in range(len(xs)) if sources[i] == "detected"]
            det_ys = [ys[i] for i in range(len(ys)) if sources[i] == "detected"]
            if det_xs:
                plt.scatter(
                    det_xs, det_ys,
                    color="blue",
                    marker="o",
                    s=25,
                    alpha=1.0,
                    label="Detected Ball"
                )

            # 3. Extract and plot interpolated coordinates
            interp_xs = [xs[i] for i in range(len(xs)) if sources[i] == "interpolated"]
            interp_ys = [ys[i] for i in range(len(ys)) if sources[i] == "interpolated"]
            if interp_xs:
                plt.scatter(
                    interp_xs, interp_ys,
                    color="orange",
                    marker="^",
                    s=20,
                    alpha=0.6,
                    label="Interpolated Ball"
                )

            plt.legend(loc="upper right")
        else:
            plt.text(self.width / 2, self.height / 2, "No Ball Trajectory Data", ha='center', va='center', fontsize=14)

        plt.xlim(0, self.width)
        plt.ylim(self.height, 0) # Invert y-axis to match video frame (0 at top)
        plt.title("Ball Trajectory Map")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()

        return save_path
