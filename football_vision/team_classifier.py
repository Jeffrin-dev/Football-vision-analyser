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
        # Maps track_id to a list of tracked positions (frame_idx, center_x, center_y) across the whole clip
        self.player_positions: Dict[int, List[Tuple[int, float, float]]] = {}

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

    def add_player_sample(self, track_id: int, frame_idx: int, frame: np.ndarray, bbox: Tuple[float, float, float, float]):
        """
        If the player has fewer than `max_frames` samples, extract and store their torso color.
        Also records the player's positions to analyze movement range over the clip.
        """
        if track_id not in self.player_colors:
            self.player_colors[track_id] = []

        if len(self.player_colors[track_id]) < self.max_frames:
            color = self.extract_torso_color(frame, bbox)
            self.player_colors[track_id].append(color)

        if track_id not in self.player_positions:
            self.player_positions[track_id] = []

        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        self.player_positions[track_id].append((frame_idx, center_x, center_y))

    def fit_and_classify(self) -> Dict[int, str]:
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

        # Get tracked frames count for each track_id
        frames_tracked_dict = {}
        for tid in track_ids:
            frames_tracked_dict[tid] = len(self.player_positions.get(tid, []))

        # 3. Compute pan-invariant signal
        # a. Group tracked player positions by frame_idx: frame_idx -> [(track_id, x, y), ...]
        frame_to_players = {}
        for tid in track_ids:
            for frame_idx, x, y in self.player_positions.get(tid, []):
                if frame_idx not in frame_to_players:
                    frame_to_players[frame_idx] = []
                frame_to_players[frame_idx].append((tid, x, y))

        # b. Compute frame centroid for every frame with >= 2 players
        frame_centroids = {}
        for frame_idx, players in frame_to_players.items():
            if len(players) >= 2:
                xs = [p[1] for p in players]
                ys = [p[2] for p in players]
                frame_centroids[frame_idx] = (float(np.mean(xs)), float(np.mean(ys)))

        # c/d. Compute play distance variability for each player (std of distance to frame centroid)
        play_distance_variability = {}
        player_distances = {} # Store distances per player for logging or thresholding if needed
        for tid in track_ids:
            distances = []
            for frame_idx, x, y in self.player_positions.get(tid, []):
                if frame_idx in frame_centroids:
                    cx, cy = frame_centroids[frame_idx]
                    dist = float(np.sqrt((x - cx)**2 + (y - cy)**2))
                    distances.append(dist)

            player_distances[tid] = distances
            if len(distances) >= 10:
                play_distance_variability[tid] = float(np.std(distances))
            else:
                # Treated as insufficient data/None
                play_distance_variability[tid] = None

        # 4. Compute baseline_median_variability
        # Identify confirmed team A/B players (NOT color outliers) with frames_tracked >= 50
        qualifying_variabilities = []
        for tid in track_ids:
            avg_color = player_avg_colors[tid]
            dist_0 = np.linalg.norm(avg_color - centers[0])
            dist_1 = np.linalg.norm(avg_color - centers[1])
            is_outlier = (dist_0 > self.referee_threshold) and (dist_1 > self.referee_threshold)

            if not is_outlier and frames_tracked_dict[tid] >= 50:
                var_val = play_distance_variability[tid]
                if var_val is not None:
                    qualifying_variabilities.append(var_val)

        if qualifying_variabilities:
            baseline_median_variability = float(np.median(qualifying_variabilities))
        else:
            # Fallback to median of all non-outliers with frames_tracked > 0 and valid variability
            non_outlier_variabilities = []
            for tid in track_ids:
                avg_color = player_avg_colors[tid]
                dist_0 = np.linalg.norm(avg_color - centers[0])
                dist_1 = np.linalg.norm(avg_color - centers[1])
                is_outlier = (dist_0 > self.referee_threshold) and (dist_1 > self.referee_threshold)
                if not is_outlier and frames_tracked_dict[tid] > 0:
                    var_val = play_distance_variability[tid]
                    if var_val is not None:
                        non_outlier_variabilities.append(var_val)
            if non_outlier_variabilities:
                baseline_median_variability = float(np.median(non_outlier_variabilities))
            else:
                baseline_median_variability = 100.0

        # Protect against zero/extremely small baseline median variability
        if baseline_median_variability < 1.0:
            baseline_median_variability = 100.0

        # 5. Classification for color-outliers (replaces old absolute movement_range logic)
        GK_VARIABILITY_RATIO = 1.2
        for tid in track_ids:
            avg_color = player_avg_colors[tid]
            dist_0 = np.linalg.norm(avg_color - centers[0])
            dist_1 = np.linalg.norm(avg_color - centers[1])
            is_outlier = (dist_0 > self.referee_threshold) and (dist_1 > self.referee_threshold)

            frames_tracked = frames_tracked_dict[tid]
            var_val = play_distance_variability[tid]
            num_distances = len(player_distances.get(tid, []))

            if is_outlier:
                if frames_tracked < 30 or num_distances < 10:
                    self.assignments[tid] = "uncertain_outlier"
                else:
                    # var_val is guaranteed not None because num_distances >= 10
                    if var_val > baseline_median_variability * GK_VARIABILITY_RATIO:
                        self.assignments[tid] = "goalkeeper"
                    else:
                        self.assignments[tid] = "referee"

            logger.info(
                f"[CLASSIFICATION] Decision: track_id={tid}, frames_tracked={frames_tracked}, "
                f"color_dist_0={dist_0:.2f}, color_dist_1={dist_1:.2f}, "
                f"referee_threshold={self.referee_threshold:.2f}, is_outlier={is_outlier}, "
                f"play_distance_variability={var_val if var_val is not None else 'N/A'}, "
                f"baseline_median_variability={baseline_median_variability:.2f}, "
                f"GK_threshold={baseline_median_variability * GK_VARIABILITY_RATIO:.2f}, "
                f"final_label={self.assignments[tid]}"
            )

        return self.assignments
