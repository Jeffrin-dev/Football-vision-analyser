import numpy as np
from ultralytics import YOLO
import supervision as sv

class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLOv8n detector.
        """
        self.model = YOLO(model_path)
        self.person_class_id = 0  # 'person' in COCO dataset

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Runs YOLOv8n inference on a frame and returns supervision.Detections
        filtered to 'person' class only.
        """
        # Run inference on the frame
        results = self.model(frame, verbose=False)

        if not results:
            return sv.Detections.empty()

        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results[0])

        # Filter for person class only
        detections = detections[detections.class_id == self.person_class_id]

        return detections
