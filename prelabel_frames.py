#!/usr/bin/env python3
"""
Standalone utility script to pre-label extracted JPEG frames using the YOLOv8 ball detector.
Filters detections to the "sports ball" COCO class (32) only, and outputs YOLO-format
annotations (.txt) in the same directory as the images, with class index "0".
"""

import argparse
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pre-label extracted JPEG frames with YOLO ball detector."
    )
    parser.add_argument(
        "--frames-dir",
        required=True,
        type=str,
        help="Directory containing the extracted JPEG frames."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for sports ball detection (default: 0.15)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Validate frames directory
    if not os.path.exists(args.frames_dir):
        print(f"Error: Frames directory '{args.frames_dir}' does not exist.")
        return

    # Find all JPEG images in the directory
    # Find lower-case and upper-case JPEG extensions to be robust
    image_paths = sorted(
        glob.glob(os.path.join(args.frames_dir, "*.jpg")) +
        glob.glob(os.path.join(args.frames_dir, "*.JPG")) +
        glob.glob(os.path.join(args.frames_dir, "*.jpeg")) +
        glob.glob(os.path.join(args.frames_dir, "*.JPEG"))
    )
    # Remove any potential duplicates and sort again
    image_paths = sorted(list(set(image_paths)))

    total_images = len(image_paths)
    print(f"Found {total_images} image files in '{args.frames_dir}'.")

    if total_images == 0:
        print("No images found to process.")
        return

    # Load YOLOv8n model
    # Note: yolov8n.pt will be downloaded automatically by ultralytics if not cached.
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    skipped_count = 0
    new_label_count = 0
    no_detection_count = 0

    print("Starting pre-labeling process...")

    for idx, img_path in enumerate(image_paths, 1):
        base_path, _ = os.path.splitext(img_path)
        txt_path = base_path + ".txt"

        # Check if a label file already exists (e.g. manually labeled in an earlier session)
        if os.path.exists(txt_path):
            print(f"Skipping (already labeled): {img_path} -> '{txt_path}' exists.")
            skipped_count += 1
        else:
            # Read image to obtain original height and width
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Failed to load image '{img_path}'. Skipping.")
                no_detection_count += 1
                continue

            h, w = frame.shape[:2]

            # Run YOLOv8n inference on the frame
            results = model(frame, verbose=False)

            best_box = None
            best_conf = -1.0

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Extract class and confidence
                        class_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        # Filter to "sports ball" class (COCO 32) above the threshold
                        if class_id == 32 and conf >= args.confidence_threshold:
                            if conf > best_conf:
                                best_conf = conf
                                # Calculate normalized YOLO center coordinates and size
                                xyxy = box.xyxy[0].cpu().numpy()
                                xmin, ymin, xmax, ymax = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                                x_center = ((xmin + xmax) / 2.0) / w
                                y_center = ((ymin + ymax) / 2.0) / h
                                box_width = (xmax - xmin) / w
                                box_height = (ymax - ymin) / h

                                best_box = (x_center, y_center, box_width, box_height)

            if best_box is not None:
                x_c, y_c, box_w, box_h = best_box
                # Write YOLO format label with class index "0"
                with open(txt_path, "w") as f:
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {box_w:.6f} {box_h:.6f}\n")
                new_label_count += 1
            else:
                no_detection_count += 1

        # Print progress every 50 images processed
        if idx % 50 == 0:
            print(f"Processed {idx}/{total_images} images...")

    # Final summary output
    print("\n========================================")
    print("Pre-labeling completed.")
    print(f"Total images processed:                  {total_images}")
    print(f"Images with new pre-labels created:      {new_label_count}")
    print(f"Images with no ball detection:           {no_detection_count}")
    print(f"Images skipped (manual label existed):   {skipped_count}")
    print("========================================")


if __name__ == "__main__":
    main()
