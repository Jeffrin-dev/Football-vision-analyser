import logging
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class TeamClassifier:
    def __init__(self, max_frames: int = 10, referee_threshold: float = 60.0):
        """
        Initializes the Team Classifier.
        For each tracked player, we sample the average color of the torso region
        over their first `max_frames` (default ~10) tracked frames.
        - referee_threshold: distance threshold above which a player is reclassified as a referee.
        """
        self.max_frames = max_frames
        self.referee_threshold = referee_threshold
        # Maps track_id to a list of sampled RGB colors (as numpy arrays or tuples)
        self.player_colors: Dict[int, List[np.ndarray]] = {}
        # Maps track_id to the final assigned team ('A' or 'B')
        self.assignments: Dict[int, str] = {}

    def extract_torso_color(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Extracts the average RGB color of the torso region from the bounding box in the frame.
        bbox is in (x_min, y_min, x_max, y_max) format.
        We crop roughly the middle third vertically, and the middle 60% horizontally to avoid background pixels.
        """
        x_min, y_min, x_max, y_max = bbox

        # Ensure coordinates are within image boundaries
        h, w = frame.shape[:2]
        x_min = max(0, int(round(x_min)))
        y_min = max(0, int(round(y_min)))
        x_max = min(w, int(round(x_max)))
        y_max = min(h, int(round(y_max)))

        box_w = x_max - x_min
        box_h = y_max - y_min

        if box_w <= 0 or box_h <= 0:
            return np.zeros(3)

        # Crop roughly the middle third vertically
        y_start = y_min + int(box_h * (1.0 / 3.0))
        y_end = y_min + int(box_h * (2.0 / 3.0))

        # Crop roughly the middle 60% horizontally (centered)
        x_start = x_min + int(box_w * 0.2)
        x_end = x_min + int(box_w * 0.8)

        # Ensure within image boundaries after cropping adjustments
        y_start = max(0, y_start)
        y_end = min(h, y_end)
        x_start = max(0, x_start)
        x_end = min(w, x_end)

        if (y_end - y_start) <= 0 or (x_end - x_start) <= 0:
            # Fall back to original box if crop is too small/empty
            crop = frame[y_min:y_max, x_min:x_max]
        else:
            crop = frame[y_start:y_end, x_start:x_end]

        if crop.size == 0:
            return np.zeros(3)

        # Frame is likely BGR (OpenCV standard), let's convert to RGB for standard color space representation
        # but average of BGR is fine as long as we are consistent. We will convert BGR to RGB.
        crop_rgb = crop[:, :, ::-1]

        # Compute the average color of the crop (R, G, B)
        avg_color = np.mean(crop_rgb, axis=(0, 1))
        return avg_color

    def add_player_sample(self, track_id: int, frame: np.ndarray, bbox: Tuple[float, float, float, float]):
        """
        If the player has fewer than `max_frames` samples, extract and store their torso color.
        """
        if track_id not in self.player_colors:
            self.player_colors[track_id] = []

        if len(self.player_colors[track_id]) < self.max_frames:
            color = self.extract_torso_color(frame, bbox)
            self.player_colors[track_id].append(color)

    def fit_and_classify(self, track_coords: Dict[int, List[Tuple[float, float]]] = None) -> Dict[int, str]:
        """
        Performs k-means clustering (k=2) across all players' average colors
        to assign team 'A' or 'B'. Logs a warning if the cluster centers are too similar.
        Returns a dictionary mapping track_id to team ('A' or 'B').
        """
        # Calculate the final average color for each player
        player_avg_colors = {}
        for track_id, colors in self.player_colors.items():
            if colors:
                player_avg_colors[track_id] = np.mean(colors, axis=0)
            else:
                player_avg_colors[track_id] = np.zeros(3)

        track_ids = list(player_avg_colors.keys())
        if not track_ids:
            return {}

        features = np.array([player_avg_colors[tid] for tid in track_ids])

        # If there are fewer than 2 players, we can't reliably cluster, so assign them all to Team A
        if len(track_ids) < 2:
            self.assignments = {tid: "A" for tid in track_ids}
            return self.assignments

        # Run KMeans with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features)

        # Check cluster similarity/ambiguity
        centers = kmeans.cluster_centers_
        distance = np.linalg.norm(centers[0] - centers[1])

        # Log warning if Euclidean distance in RGB space is very small (e.g., less than 30.0)
        # 30.0 is a reasonable threshold in RGB [0-255] space for perceptible color difference
        if distance < 30.0:
            logger.warning(
                f"Warning: Team clusters are potentially ambiguous! "
                f"Cluster distance is very small ({distance:.2f}). "
                f"Cluster 1 center (RGB): {centers[0]}, Cluster 2 center (RGB): {centers[1]}"
            )

        # Map labels to 'A' or 'B'
        for tid, label in zip(track_ids, labels):
            self.assignments[tid] = "A" if label == 0 else "B"

        # Calculate movement range for each player
        ranges = {}
        if track_coords is not None:
            for tid, coords in track_coords.items():
                if len(coords) >= 1:
                    xs = [pt[0] for pt in coords]
                    ys = [pt[1] for pt in coords]
                    dx = max(xs) - min(xs)
                    dy = max(ys) - min(ys)
                    rng = float(np.sqrt(dx**2 + dy**2))
                    ranges[tid] = rng
                else:
                    ranges[tid] = 0.0

        for tid in track_ids:
            if tid not in ranges:
                ranges[tid] = 0.0

        # Identify confirmed team A/B player ranges (non-outliers initially)
        team_player_ranges = []
        for tid in track_ids:
            avg_color = player_avg_colors[tid]
            dist_0 = np.linalg.norm(avg_color - centers[0])
            dist_1 = np.linalg.norm(avg_color - centers[1])
            if dist_0 <= self.referee_threshold or dist_1 <= self.referee_threshold:
                team_player_ranges.append(ranges[tid])

        if team_player_ranges:
            median_range = float(np.median(team_player_ranges))
        else:
            median_range = 0.0

        # Classify color outliers as goalkeeper or referee
        for tid in track_ids:
            avg_color = player_avg_colors[tid]
            dist_0 = np.linalg.norm(avg_color - centers[0])
            dist_1 = np.linalg.norm(avg_color - centers[1])

            if dist_0 > self.referee_threshold and dist_1 > self.referee_threshold:
                p_range = ranges[tid]
                if median_range > 0 and p_range < 0.4 * median_range:
                    final_label = "goalkeeper"
                else:
                    final_label = "referee"

                self.assignments[tid] = final_label
                logger.info(
                    f"Classification decision: track_id={tid}, "
                    f"dist_to_centroid_0={dist_0:.2f}, dist_to_centroid_1={dist_1:.2f}, "
                    f"movement_range={p_range:.2f}, median_range={median_range:.2f}, "
                    f"final_label={final_label}"
                )
            else:
                p_range = ranges[tid]
                logger.info(
                    f"Classification decision: track_id={tid}, "
                    f"dist_to_centroid_0={dist_0:.2f}, dist_to_centroid_1={dist_1:.2f}, "
                    f"movement_range={p_range:.2f}, median_range={median_range:.2f}, "
                    f"final_label={self.assignments[tid]}"
                )

        return self.assignments
