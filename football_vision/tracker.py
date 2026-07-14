import supervision as sv
import weakref

# Easily adjustable constants for tuning tracker behavior
LOST_TRACK_BUFFER = 60             # Number of frames to keep a lost track alive before dropping it (default supervision is 30)
TRACK_ACTIVATION_THRESHOLD = 0.25   # Confidence threshold for track activation
MINIMUM_MATCHING_THRESHOLD = 0.8    # Threshold for matching tracks with detections
FRAME_RATE = 30                    # The frame rate of the video
MIN_CONSECUTIVE_FRAMES = 1         # Number of consecutive frames an object must be tracked to be valid
MIN_TRACK_LENGTH = 10              # Discard track_ids with fewer than this minimum number of tracked frames

# Keep a weak reference to the active/most recent tracker instance
_active_tracker_ref = None


class PersonTracker:
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

        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames
        )

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with the given detections and returns the detections with persistent tracker_ids.
        Also updates the tracking counts on the active instance.
        """
        tracked_detections = self.tracker.update_with_detections(detections)
        if tracked_detections.tracker_id is not None:
            for track_id in tracked_detections.tracker_id:
                self.track_counts[track_id] = self.track_counts.get(track_id, 0) + 1
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
