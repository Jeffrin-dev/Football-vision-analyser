import cv2
import logging
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class VideoAnnotator:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        frame_player_boxes: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
        smoothed_possession: Dict[int, Optional[int]],
        team_assignments: Dict[int, str],
        ball_positions: List[Optional[Tuple[float, float]]],
        w_res: int,
        h_res: int
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.frame_player_boxes = frame_player_boxes
        self.smoothed_possession = smoothed_possession
        self.team_assignments = team_assignments
        self.ball_positions = ball_positions
        self.w_res = w_res
        self.h_res = h_res

    def resize_frame(self, frame: np.ndarray, max_side: int = 640) -> np.ndarray:
        """
        Resizes the frame so that the longest side is at most max_side, maintaining aspect ratio.
        """
        h, w = frame.shape[:2]
        if max(h, w) <= max_side:
            return frame

        if w > h:
            new_w = max_side
            new_h = int(round(h * (max_side / w)))
        else:
            new_h = max_side
            new_w = int(round(w * (max_side / h)))

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def draw_text_with_bg(
        self,
        img: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font=cv2.FONT_HERSHEY_SIMPLEX,
        scale=0.5,
        color=(255, 255, 255),
        thickness=1,
        bg_color=(0, 0, 0)
    ):
        x, y = position
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(img, (x, y - h - baseline), (x + w, y + baseline), bg_color, -1)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def render(self):
        """
        Renders the annotated video by reading the input frame-by-frame,
        overlaying boxes and metadata, and writing to the output path.
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video source for annotation: {self.input_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (self.w_res, self.h_res))

        if not out.isOpened():
            logger.error(f"Failed to open video writer for output: {self.output_path}")
            cap.release()
            return

        logger.info(f"Rendering annotated video to: {self.output_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to the same target resolution
            resized = self.resize_frame(frame, max_side=640)

            # Retrieve smoothed possession holder for this frame
            possession_holder = self.smoothed_possession.get(frame_idx, None)

            # 1. Draw Bounding Boxes around Players
            players = self.frame_player_boxes.get(frame_idx, [])
            for track_id, bbox in players:
                x_min, y_min, x_max, y_max = bbox

                # Determine color based on role
                team = self.team_assignments.get(track_id, "A")
                if track_id == possession_holder:
                    color = (0, 255, 0)  # GREEN for the possession holder
                elif team == "referee":
                    color = (255, 0, 0)  # BLUE for referee (BGR order: B is index 0)
                elif team == "goalkeeper":
                    color = (128, 0, 128)  # PURPLE for goalkeeper (BGR: B=128, G=0, R=128)
                else:
                    color = (0, 0, 255)  # RED for other players

                # Draw bounding box
                cv2.rectangle(
                    resized,
                    (int(round(x_min)), int(round(y_min))),
                    (int(round(x_max)), int(round(y_max))),
                    color,
                    2
                )

                # Draw track ID label above the box
                cv2.putText(
                    resized,
                    f"#{track_id}",
                    (int(round(x_min)), int(round(max(y_min - 5, 15)))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )

            # 2. Draw Ball Marker if position is known (not missing)
            if frame_idx < len(self.ball_positions):
                ball_pos = self.ball_positions[frame_idx]
                if ball_pos is not None:
                    bx, by = ball_pos
                    # Draw a small, precise thin circle outline closer to the ball's actual bounding box
                    cv2.circle(resized, (int(round(bx)), int(round(by))), 4, (0, 255, 255), 1)

            # 3. Draw Persistent Text Overlay for smoothed possession in the corner
            if possession_holder is not None:
                team_name = self.team_assignments.get(possession_holder, "A")
                possession_text = f"Possession: Team {team_name} (Player #{possession_holder})"
            else:
                possession_text = "Possession: Contested"

            self.draw_text_with_bg(
                resized,
                possession_text,
                (15, 30),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                scale=0.6,
                color=(255, 255, 255),
                thickness=1,
                bg_color=(0, 0, 0)
            )

            # Write the annotated frame
            out.write(resized)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"Rendered {frame_idx} annotated frames...")

        cap.release()
        out.release()
        logger.info(f"Rendering complete! Annotated video written to {self.output_path}")
