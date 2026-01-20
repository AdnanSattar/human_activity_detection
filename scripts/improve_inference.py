"""
Script to improve inference accuracy by adjusting parameters and filtering results.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from src.utils import get_project_root, load_config, setup_logging


def improve_inference(
    video_path: str,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    min_conf_for_activity: float = 0.6,
):
    """
    Run inference with improved settings for better accuracy.

    Args:
        video_path: Path to video file
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        min_conf_for_activity: Minimum confidence to show activity label
    """
    logger = setup_logging()
    logger.info("Running improved inference...")

    project_root = get_project_root()

    # Load model
    model_path = os.path.join(
        project_root, "outputs", "models", "Human_Activity_model.pt"
    )
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train a model first or check the path")
        return

    model = YOLO(model_path)

    # Process video with improved settings
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Confidence threshold: {conf_threshold}")
    logger.info(f"IoU threshold: {iou_threshold}")

    results = model.predict(
        source=video_path,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640,
        device=0,  # GPU
        save=True,
        save_txt=True,
        save_conf=True,
        project=os.path.join(project_root, "outputs", "inference"),
        name="improved_predictions",
        exist_ok=True,
    )

    # Filter results by confidence
    logger.info("\nFiltering low-confidence detections...")
    total_detections = 0
    high_conf_detections = 0

    for result in results:
        boxes = result.boxes
        total_detections += len(boxes)

        # Count high-confidence detections
        if boxes.conf is not None:
            high_conf = (boxes.conf >= min_conf_for_activity).sum().item()
            high_conf_detections += high_conf

    logger.info(f"Total detections: {total_detections}")
    logger.info(
        f"High-confidence detections (>{min_conf_for_activity}): {high_conf_detections}"
    )
    logger.info(f"Filtered out: {total_detections - high_conf_detections}")

    logger.info(f"\nResults saved to: outputs/inference/improved_predictions/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run improved inference with better accuracy"
    )
    parser.add_argument("--source", type=str, required=True, help="Path to video file")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument(
        "--min-activity-conf",
        type=float,
        default=0.6,
        help="Minimum confidence to show activity label",
    )

    args = parser.parse_args()
    improve_inference(args.source, args.conf, args.iou, args.min_activity_conf)
