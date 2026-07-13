import supervision as sv

class PersonTracker:
    def __init__(self):
        """
        Initializes the ByteTrack tracker using the supervision library wrapper.
        """
        self.tracker = sv.ByteTrack()

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the tracker with the given detections and returns the detections with persistent tracker_ids.
        """
        tracked_detections = self.tracker.update_with_detections(detections)
        return tracked_detections
