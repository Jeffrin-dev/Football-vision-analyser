import argparse
import logging
import os
import cv2
import numpy as np
import supervision as sv

from football_vision.detector import PersonDetector
from football_vision.tracker import PersonTracker
from football_vision.team_classifier import TeamClassifier
from football_vision.heatmap import HeatmapGenerator
from football_vision.report import ReportGenerator
from football_vision.ball_detector import BallDetector
from football_vision.possession import PossessionTracker
from football_vision.events import EventDetector
from football_vision.annotator import VideoAnnotator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("football_vision")

def resize_frame(frame: np.ndarray, max_side: int = 640) -> np.ndarray:
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

def main():
    parser = argparse.ArgumentParser(description="Football Vision CLI Application")
    parser.add_argument("--input", required=True, type=str, help="Path to input video clip (e.g. clip.mp4)")
    parser.add_argument("--output", required=True, type=str, help="Directory to save the output report and heatmaps")

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        return

    # Open Video Source
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {input_path}")
        return

    # Read frame dimensions using CAP_PROP to robustly determine resolution before reading
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w_orig <= 0 or h_orig <= 0:
        # Fallback by reading first frame if properties not available
        ret, first_frame = cap.read()
        if not ret:
            logger.error("Empty video or failed to read first frame.")
            cap.release()
            return
        h_orig, w_orig = first_frame.shape[:2]
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Determine resized resolution
    if max(h_orig, w_orig) <= 640:
        w_res, h_res = w_orig, h_orig
    else:
        if w_orig > h_orig:
            w_res = 640
            h_res = int(round(h_orig * (640 / w_orig)))
        else:
            h_res = 640
            w_res = int(round(w_orig * (640 / h_orig)))

    # Generate uniquely named subfolder using input clip's filename without extension + timestamp
    import datetime
    clip_basename = os.path.basename(input_path)
    clip_name_no_ext, _ = os.path.splitext(clip_basename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"{clip_name_no_ext}_{timestamp}")

    # Initialize components
    detector = PersonDetector()
    tracker = PersonTracker()
    team_classifier = TeamClassifier(max_frames=10)
    heatmap_gen = HeatmapGenerator(width=w_res, height=h_res)

    # Initialize Phase 2 components
    ball_detector = BallDetector()
    possession_tracker = PossessionTracker()

    logger.info(f"Initialized modules with target frame resolution: {w_res}x{h_res}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize frame
        resized = resize_frame(frame, max_side=640)

        # 1. Detection: Run YOLOv8n inference once to detect both "person" and "sports ball"
        results = detector.model(resized, verbose=False)

        if not results:
            person_detections = sv.Detections.empty()
            ball_detections = sv.Detections.empty()
        else:
            all_detections = sv.Detections.from_ultralytics(results[0])
            person_detections = all_detections[all_detections.class_id == 0]
            ball_detections = all_detections[all_detections.class_id == 32]

        # 2. Tracking
        tracked_detections = tracker.update_with_detections(person_detections, frame=resized)

        # Record players for proximity possession (Phase 2)
        possession_tracker.record_players(frame_count - 1, tracked_detections)

        # 3. Team Classification Samples & Heatmap Accumulation
        # tracker_id is present in tracked_detections.tracker_id
        if tracked_detections.tracker_id is not None:
            for bbox, track_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
                # Sample color for team classification
                team_classifier.add_player_sample(track_id, resized, bbox)
                # Accumulate positions for heatmap
                heatmap_gen.accumulate_position(track_id, bbox)

        # Process ball detections (Phase 2)
        ball_detector.process_frame_detections(ball_detections, tracked_players=tracked_detections)

        if frame_count % 50 == 0:
            logger.info(f"Processed {frame_count} frames...")

    cap.release()
    logger.info(f"Completed frame processing. Total frames: {frame_count}")

    # Interpolate ball detection gaps (Phase 2)
    ball_detector.interpolate_gaps()
    ball_stats = ball_detector.get_stats()

    # 4. Fit and Classify Teams
    logger.info("Performing team classification via KMeans clustering...")
    team_assignments = team_classifier.fit_and_classify()

    # Compute and aggregate possession (Phase 2)
    possession_tracker.compute_and_aggregate_possession(
        frame_count=frame_count,
        ball_positions=ball_detector.interpolated_positions,
        team_assignments=team_assignments
    )

    # 5. Generate Heatmaps
    logger.info("Generating and saving team heatmaps...")
    heatmap_gen.generate_and_save_heatmaps(team_assignments, run_output_dir)

    # Generate and save ball trajectory map (Phase 2)
    logger.info("Generating and saving ball trajectory...")
    heatmap_gen.generate_and_save_ball_trajectory(ball_stats["trajectory"], run_output_dir)

    # Compute events and summary (Phase 3)
    logger.info("Detecting passes and turnovers...")
    event_detector = EventDetector()
    events = event_detector.detect_events(
        frame_possession=possession_tracker.frame_possession,
        team_assignments=team_assignments
    )
    event_summary = event_detector.generate_summary(events)

    # 6. Generate Report
    logger.info("Writing final report...")
    clip_name = os.path.basename(input_path)
    report_gen = ReportGenerator(clip_name=clip_name, frame_count=frame_count)
    report_path = report_gen.generate_report(
        team_assignments=team_assignments,
        track_coords=heatmap_gen.track_coords,
        output_dir=run_output_dir,
        ball_stats=ball_stats,
        player_possession_counts=possession_tracker.player_possession_counts,
        team_possession=possession_tracker.team_possession_counts,
        events=events,
        event_summary=event_summary
    )

    # 7. Render Annotated Video (Phase 4)
    logger.info("Rendering annotated video output...")
    smoothed_possession = event_detector.smooth_possession(possession_tracker.frame_possession)
    annotated_video_path = os.path.join(run_output_dir, "annotated_output.mp4")
    annotator = VideoAnnotator(
        input_path=input_path,
        output_path=annotated_video_path,
        frame_player_boxes=possession_tracker.frame_player_boxes,
        smoothed_possession=smoothed_possession,
        team_assignments=team_assignments,
        ball_positions=ball_detector.interpolated_positions,
        w_res=w_res,
        h_res=h_res,
        ball_widths=ball_detector.interpolated_widths
    )
    annotator.render()

    logger.info(f"Processing complete! Report saved at {report_path}")
    print(f"\n[OUTPUT_DIR] Output files successfully saved to: {run_output_dir}\n")

if __name__ == "__main__":
    main()
