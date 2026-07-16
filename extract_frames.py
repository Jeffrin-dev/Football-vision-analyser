#!/usr/bin/env python3
"""
Standalone utility script to extract candidate frames for ball-detection dataset labeling.
Reads a video frame-by-frame with strict memory discipline (no buffering of all frames),
saves every Nth frame as a JPEG, and limits total extraction to a specified maximum.
"""

import argparse
import os
import sys
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract candidate frames from a video clip for dataset labeling."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to the input video file (e.g., clip.mp4)."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Directory where the extracted JPEG frames will be saved."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Extract every Nth frame (default: 5)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Capped total number of extracted frames (default: 500)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Input validation
    if not os.path.exists(args.input):
        print(f"Error: Input video file not found at '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if args.interval <= 0:
        print("Error: --interval must be a positive integer greater than 0.", file=sys.stderr)
        sys.exit(1)

    if args.max_frames <= 0:
        print("Error: --max-frames must be a positive integer greater than 0.", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Extract clip base name without extension
    clip_basename = os.path.basename(args.input)
    clip_name, _ = os.path.splitext(clip_basename)

    # Open video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Failed to open video file '{args.input}'", file=sys.stderr)
        sys.exit(1)

    frame_count = 0
    saved_count = 0

    print(f"Starting frame extraction from: {args.input}")
    print(f"Saving to: {args.output}")
    print(f"Interval: {args.interval} | Max frames limit: {args.max_frames}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame number (1-based index)
            frame_count += 1

            # Extract every Nth frame
            if frame_count % args.interval == 0:
                # Construct file name using actual frame number
                file_name = f"{clip_name}_frame_{frame_count:06d}.jpg"
                out_path = os.path.join(args.output, file_name)

                # Save frame as JPEG
                success = cv2.imwrite(out_path, frame)
                if success:
                    saved_count += 1
                    # Print progress every 50 saved frames
                    if saved_count % 50 == 0:
                        print(f"Saved {saved_count} frames...")

                    # Check max frames cap
                    if saved_count >= args.max_frames:
                        print(f"Reached max-frames limit of {args.max_frames}. Stopping.")
                        break
                else:
                    print(f"Warning: Failed to save frame {frame_count} to {out_path}", file=sys.stderr)

    finally:
        cap.release()

    print(f"Extraction completed. Total frames processed in video: {frame_count}. Total frames saved: {saved_count}.")


if __name__ == "__main__":
    main()
