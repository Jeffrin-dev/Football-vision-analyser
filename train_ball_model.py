#!/usr/bin/env python3
"""
train_ball_model.py

A standalone script to load pretrained weights (yolov8n.pt),
fine-tune it on the generated custom ball dataset, and save the resulting
fine-tuned weights.

CLI Arguments:
  --data         Path to the dataset data.yaml file (default: ./ball_dataset/data.yaml)
  --epochs       Number of training epochs (default: 50)
  --batch-size   Batch size (default: 4 — small for 4GB RAM CPU setups)
  --output-name  Name of the output run directory (default: "ball_finetune")
"""

import argparse
import os
import sys
from ultralytics import YOLO

# Display startup warnings about training on 4GB CPU-only hardware
# RECOMMENDATION FOR SESSION MANAGEMENT:
# Since CPU training is extremely slow on 4GB RAM CPU-only hardware and can take several hours,
# it is highly recommended to run this script in a way that survives terminal disconnection.
# Options include:
#   1) tmux:
#      $ tmux new -s train_ball
#      $ python3 train_ball_model.py ...
#      (Then press Ctrl+B, then D to detach. To reattach: $ tmux attach -t train_ball)
#
#   2) nohup:
#      $ nohup python3 train_ball_model.py --epochs 50 > train.log 2>&1 &
#      (Then monitor with: $ tail -f train.log)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n model on custom sports ball dataset on CPU."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./ball_dataset/data.yaml",
        help="Path to the dataset config YAML file (default: ./ball_dataset/data.yaml)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4 — recommended small for 4GB RAM)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="ball_finetune",
        help="Output name for training run directory (default: 'ball_finetune')"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()

    print("=====================================================================")
    print("WARNING: Starting training on CPU-only hardware.")
    print("This will be extremely slow on 4GB RAM / CPU-only setups and may")
    print("take multiple hours to complete.")
    print("---------------------------------------------------------------------")
    print("Recommendation: Run this inside a terminal multiplexer (like 'tmux'")
    print("or 'screen') or use 'nohup' so the training survives terminal closure.")
    print("=====================================================================\n")

    # Verify that the data.yaml file exists
    if not os.path.exists(args.data):
        print(f"Error: The dataset config file '{args.data}' does not exist.")
        print("Please run 'build_dataset.py' first to create the dataset and YAML config.")
        sys.exit(1)

    print(f"Dataset YAML:      {os.path.abspath(args.data)}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Output Run Name:   {args.output_name}")
    print("---------------------------------------------------------------------")

    # Load the pretrained yolov8n.pt model as starting point
    # Note: yolov8n.pt will be downloaded automatically if not already cached.
    print("Loading pretrained yolov8n.pt...")
    model = YOLO("yolov8n.pt")

    # Run fine-tuning training
    print("Starting training process with transfer learning on CPU...")

    # We explicitly specify:
    # - data: path to the YAML file
    # - epochs: number of training epochs
    # - batch: batch size
    # - device: 'cpu' (explicit CPU device specification)
    # - project: standard output location project directory 'runs/detect'
    # - name: custom output run name
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        device="cpu",
        project="runs/detect",
        name=args.output_name,
        verbose=True
    )

    print("\n=====================================================================")
    print("Training Complete!")
    # According to Ultralytics convention, best weights are saved at:
    # project / name / weights / best.pt
    expected_weights_path = os.path.join("runs", "detect", args.output_name, "weights", "best.pt")
    print(f"Fine-tuned weights are saved at: {expected_weights_path}")
    print("=====================================================================")

if __name__ == "__main__":
    main()
