import supervision as sv
import weakref
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Easily adjustable constants for tuning tracker behavior
LOST_TRACK_BUFFER = 90             # Number of frames to keep a lost track alive before dropping it (default supervision is 30)

# Tunable constants for Re-Identification via torso color signature
RE_ID_MAX_LOST_FRAMES = 150        # Keep a lost track signature for up to 150 frames (~5 seconds at 30 fps)
RE_ID_SIM_THRESHOLD = 15.0         # Strict Euclidean distance threshold in RGB space. Lower means stricter.
TRACK_ACTIVATION_THRESHOLD = 0.25   # Confidence threshold for track activation
MINIMUM_MATCHING_THRESHOLD = 0.8    # Threshold for matching tracks with detections
FRAME_RATE = 30                    # The frame rate of the video
MIN_CONSECUTIVE_FRAMES = 1         # Number of consecutive frames an object must be tracked to be valid
MIN_TRACK_LENGTH = 10              # Discard track_ids with fewer than this minimum number of tracked frames

# --- TUNING COMMENTS & ANALYSIS ---
# TRADEOFF ANALYSIS (LOST_TRACK_BUFFER):
# Increasing LOST_TRACK_BUFFER (e.g., from 60 to 90 frames / 3 seconds) helps bridge longer gaps where
# a player is temporarily occluded or out of detection range. However, there is a key tradeoff: if the
# buffer is set too high, when two different players cross paths closely, the tracker may incorrectly
# merge their trajectories into a single, persistent track ID when one player reappears. This ID-switching/
# track-merging is a more severe issue than track fragmentation (which only splits a single trajectory)
# because it pollutes individual statistics and scrambles team-classification clusters.
#
# DETECTION THRESHOLD OBSERVATIONS (detector.py):
# In detector.py, the model is called without an explicit confidence threshold (e.g., self.model(frame)),
# meaning it defaults to YOLOv8's built-in threshold of 0.25. If players are partially occluded, far away,
# or blurred, their detection confidence may drop slightly below 0.25. These brief detection dropouts
# prevent the tracker from receiving those detections entirely. Consequently, ByteTrack is forced to
# keep the track "lost" during these frames. If dropouts are frequent or exceed the track buffer, the track
# will fragment into a new ID. Lowering the detector's confidence threshold slightly (e.g., to 0.15 or 0.2)
# and filtering or relying on tracker-internal thresholds might help recover these weak detections and bridge
# the tracker gaps, but we leave detector.py unchanged per instructions.

# Keep a weak reference to the active/most recent tracker instance
_active_tracker_ref = None


class PersonTracker:
    def extract_torso_color(self, frame: np.ndarray, bbox) -> np.ndarray:
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

        # Frame is BGR, convert to RGB
        crop_rgb = crop[:, :, ::-1]

        # Compute the average color of the crop (R, G, B)
        avg_color = np.mean(crop_rgb, axis=(0, 1))
        return avg_color

    def __init__(
        self,
        track_activation_threshold: float = TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = LOST_TRACK_BUFFER,
        minimum_matching_threshold: float = MINIMUM_MATCHING_THRESHOLD,
        frame_rate: int = FRAME_RATE,
        minimum_consecutive_frames: int = MIN_CONSECUTIVE_FRAMES
    ):
        """
        Initializes the ByteTrack tracker using the supervision library wrapper.
        """
        global _active_tracker_ref
        _active_tracker_ref = weakref.ref(self)

        # Encapsulate tracking counts and heatmap references inside the instance
        self.track_counts = {}
        self.heatmap_generators = []

        # Re-Identification state
        self.frame_index = 0
        # Maps raw tracker_id -> canonical_id
        self.id_mapping = {}
        # Stores active canonical track IDs seen in the previous frame
        self.active_canonical_last_frame = set()
        # Maps canonical_id -> list of extracted torso color samples (numpy arrays)
        self.track_colors = {}
        # Maps canonical_id -> { 'last_seen_frame': int, 'avg_color': np.ndarray }
        self.recently_lost_pool = {}

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames
        )

    def update_with_detections(self, detections: sv.Detections, frame: np.ndarray = None) -> sv.Detections:
        """
        Updates the tracker with the given detections and returns the detections with persistent tracker_ids.
        Also updates the tracking counts on the active instance.
        """
        tracked_detections = self.tracker.update_with_detections(detections)

        # Increment frame index
        self.frame_index += 1

        # Purge stale recently lost tracks
        stale_keys = [
            cid for cid, info in self.recently_lost_pool.items()
            if (self.frame_index - info['last_seen_frame']) > RE_ID_MAX_LOST_FRAMES
        ]
        for cid in stale_keys:
            del self.recently_lost_pool[cid]

        current_canonical_ids = set()

        if tracked_detections.tracker_id is not None:
            new_tracker_ids = []
            for bbox, raw_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                # Check if raw_id is already mapped (existing active track)
                if raw_id in self.id_mapping:
                    canonical_id = self.id_mapping[raw_id]
                    # If it was in recently lost pool (e.g. because tracker dropped it but we found it), remove it from pool
                    if canonical_id in self.recently_lost_pool:
                        del self.recently_lost_pool[canonical_id]

                    # Accumulate torso color sample if frame is provided and we have fewer than 10 samples
                    if frame is not None:
                        if canonical_id not in self.track_colors:
                            self.track_colors[canonical_id] = []
                        if len(self.track_colors[canonical_id]) < 10:
                            color = self.extract_torso_color(frame, bbox)
                            self.track_colors[canonical_id].append(color)
                else:
                    # Brand-new track_id. Try matching with recently_lost_pool.
                    matched_canonical_id = None
                    best_dist = float('inf')

                    if frame is not None and self.recently_lost_pool:
                        new_color = self.extract_torso_color(frame, bbox)
                        for lost_cid, info in self.recently_lost_pool.items():
                            dist = np.linalg.norm(new_color - info['avg_color'])
                            if dist < RE_ID_SIM_THRESHOLD and dist < best_dist:
                                best_dist = dist
                                matched_canonical_id = lost_cid

                    if matched_canonical_id is not None:
                        # Log the merge decision
                        logger.info(
                            f"[RE-ID MERGE] Merging new track ID {raw_id} into recently-lost canonical ID {matched_canonical_id} "
                            f"(color Euclidean distance similarity: {best_dist:.2f} < threshold {RE_ID_SIM_THRESHOLD})"
                        )
                        # Establish the mapping
                        self.id_mapping[raw_id] = matched_canonical_id
                        canonical_id = matched_canonical_id
                        # Remove from recently lost pool as it's active again
                        if canonical_id in self.recently_lost_pool:
                            del self.recently_lost_pool[canonical_id]

                        # Accumulate new torso color sample
                        if frame is not None:
                            if canonical_id not in self.track_colors:
                                self.track_colors[canonical_id] = []
                            if len(self.track_colors[canonical_id]) < 10:
                                self.track_colors[canonical_id].append(new_color)
                    else:
                        # No match or frame not provided; map raw_id to itself
                        self.id_mapping[raw_id] = raw_id
                        canonical_id = raw_id

                        # Extract color if frame is provided
                        if frame is not None:
                            if canonical_id not in self.track_colors:
                                self.track_colors[canonical_id] = []
                            new_color = self.extract_torso_color(frame, bbox)
                            self.track_colors[canonical_id].append(new_color)

                new_tracker_ids.append(canonical_id)
                current_canonical_ids.add(canonical_id)

                # Update tracking counts with the canonical ID
                self.track_counts[canonical_id] = self.track_counts.get(canonical_id, 0) + 1

            # Update the returned tracker_id array in-place so downstream sees canonical IDs
            tracked_detections.tracker_id = np.array(new_tracker_ids, dtype=np.int32)

        # Detect tracks that were active in the last frame but not in the current frame (recently lost)
        lost_ids = self.active_canonical_last_frame - current_canonical_ids
        for lost_cid in lost_ids:
            # Only put in recently lost pool if we have at least one color sample for it
            if lost_cid in self.track_colors and self.track_colors[lost_cid]:
                avg_color = np.mean(self.track_colors[lost_cid], axis=0)
                self.recently_lost_pool[lost_cid] = {
                    'last_seen_frame': self.frame_index - 1,
                    'avg_color': avg_color
                }

        # Keep track of active canonical IDs for the next frame
        self.active_canonical_last_frame = current_canonical_ids

        return tracked_detections


# --- Safe Monkeypatching post-processing filters ---

original_heatmap_init = None
original_fit_and_classify = None


def patched_heatmap_init(self, *args, **kwargs):
    if original_heatmap_init is not None:
        original_heatmap_init(self, *args, **kwargs)

    # Store weak reference to this HeatmapGenerator in the active tracker
    tracker_inst = _active_tracker_ref() if _active_tracker_ref is not None else None
    if tracker_inst is not None:
        tracker_inst.heatmap_generators.append(weakref.ref(self))


def patched_fit_and_classify(self, *args, **kwargs):
    if original_fit_and_classify is None:
        return {}

    tracker_inst = _active_tracker_ref() if _active_tracker_ref is not None else None
    if tracker_inst is not None:
        # Identify track_ids that have fewer than MIN_TRACK_LENGTH tracked frames
        to_remove = [tid for tid, count in tracker_inst.track_counts.items() if count < MIN_TRACK_LENGTH]

        # Safe removal from TeamClassifier's internal colors before fitting
        if hasattr(self, 'player_colors') and isinstance(self.player_colors, dict):
            for tid in to_remove:
                if tid in self.player_colors:
                    del self.player_colors[tid]

        # Call original fit_and_classify
        assignments = original_fit_and_classify(self, *args, **kwargs)

        # Safe post-filtering of assignments and classifications
        if isinstance(assignments, dict):
            for tid in to_remove:
                if tid in assignments:
                    del assignments[tid]
        if hasattr(self, 'assignments') and isinstance(self.assignments, dict):
            for tid in to_remove:
                if tid in self.assignments:
                    del self.assignments[tid]
        if hasattr(self, 'player_colors') and isinstance(self.player_colors, dict):
            for tid in to_remove:
                if tid in self.player_colors:
                    del self.player_colors[tid]

        # Safe filtering of all registered HeatmapGenerator instances' coordinates
        for ref in tracker_inst.heatmap_generators:
            hg = ref()
            if hg is not None and hasattr(hg, 'track_coords') and isinstance(hg.track_coords, dict):
                for tid in to_remove:
                    if tid in hg.track_coords:
                        del hg.track_coords[tid]
    else:
        # Fallback to original if no active tracker is bound
        assignments = original_fit_and_classify(self, *args, **kwargs)

    return assignments


def _apply_patches():
    global original_heatmap_init, original_fit_and_classify
    try:
        from football_vision.team_classifier import TeamClassifier
        from football_vision.heatmap import HeatmapGenerator
    except ImportError:
        # Fail-safe in case of import issues (e.g. running outside package context)
        return

    # Patch HeatmapGenerator.__init__
    if getattr(HeatmapGenerator, '__init__', None) is not patched_heatmap_init:
        original_heatmap_init = HeatmapGenerator.__init__
        HeatmapGenerator.__init__ = patched_heatmap_init

    # Patch TeamClassifier.fit_and_classify
    if getattr(TeamClassifier, 'fit_and_classify', None) is not patched_fit_and_classify:
        original_fit_and_classify = TeamClassifier.fit_and_classify
        TeamClassifier.fit_and_classify = patched_fit_and_classify


# Execute the patches at module-load time
_apply_patches()
