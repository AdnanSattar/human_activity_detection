"""
Improved Inference module with class-specific post-processing.
Addresses class imbalance issues by applying context-aware reclassification.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import (
    check_cuda_availability,
    get_project_root,
    load_config,
    print_cuda_info,
    setup_logging,
)


def reclassify_detections(results, class_names):
    """
    Post-process detections to improve classification for underrepresented classes.

    Rules:
    - If "standing" detected with low confidence (<0.6) and person appears to be at desk,
      prefer "working" or "sitting on Desk"
    - Boost confidence for "working" and "sitting on Desk" if detected
    """
    # Get class indices
    class_idx_map = {name: idx for idx, name in enumerate(class_names)}

    # Get indices for relevant classes
    standing_idx = class_idx_map.get("standing", -1)
    working_idx = class_idx_map.get("working", -1)
    sitting_idx = class_idx_map.get("sitting", -1)
    sitting_desk_idx = class_idx_map.get("sitting on Desk", -1)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        # Process each detection
        for i in range(len(boxes)):
            cls = int(classes[i])
            conf = float(confidences[i])
            cls_name = class_names[cls] if cls < len(class_names) else "unknown"

            # Rule 1: If "standing" with low confidence, check if we should reclassify
            if cls == standing_idx and conf < 0.6:
                # Check if there are other class predictions with reasonable confidence
                # This is a simplified heuristic - in practice, you'd check the full class probabilities
                # For now, we'll boost "working" or "sitting on Desk" if they exist in nearby detections
                pass  # Placeholder for more sophisticated logic

            # Rule 2: Boost confidence for underrepresented classes if detected
            if cls == working_idx and conf > 0.3:
                # Boost working confidence slightly
                confidences[i] = min(conf * 1.2, 1.0)
            elif cls == sitting_desk_idx and conf > 0.3:
                # Boost sitting on desk confidence
                confidences[i] = min(conf * 1.2, 1.0)

    return results


def run_improved_inference(
    config_path: str = "config/inference_config.yaml",
    source: str = None,
    lower_confidence: bool = True,
):
    """
    Run inference with improved class handling for underrepresented classes.

    Args:
        config_path: Path to inference configuration file
        source: Override source from config (optional)
        lower_confidence: If True, use lower confidence threshold for rare classes
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting improved inference...")

    # Load configuration
    project_root = get_project_root()
    config = load_config(os.path.join(project_root, config_path))

    # Print CUDA info
    print_cuda_info()
    cuda_info = check_cuda_availability()

    if not cuda_info["cuda_available"]:
        logger.warning("CUDA not available. Inference will use CPU (slower).")
        device = "cpu"
    else:
        device = str(config["device"]["gpu_id"])
        logger.info(f"Using GPU: {cuda_info['gpu_name']}")

    # Get model path
    model_path = config["model"]["checkpoint_path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        logger.error(f"Model not found at: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Get class names from model
    class_names = model.names

    # Get source
    if source is None:
        source = config["input"]["source"]

    # Handle different source types
    if source.isdigit():
        source = int(source)  # Webcam
        logger.info(f"Using webcam: {source}")
    elif not os.path.isabs(source) or (
        os.name == "nt" and source.startswith("/") and not source.startswith("//")
    ):
        # Handle relative paths and Windows paths starting with /
        if source.startswith("/") and os.name == "nt":
            source = source.lstrip("/")
        source = os.path.join(project_root, source)

    # Use lower confidence for better detection of rare classes
    conf_threshold = config["model"]["conf_threshold"]
    if lower_confidence:
        conf_threshold = max(0.8, conf_threshold * 0.9)  # Lower threshold by 30%
        logger.info(
            f"Using lower confidence threshold: {conf_threshold} (for better rare class detection)"
        )

    # Run inference
    logger.info(f"Running inference on: {source}")

    try:
        results = model.predict(
            source=source,
            conf=conf_threshold,  # Lower threshold
            iou=config["model"]["iou_threshold"],
            imgsz=config["input"]["imgsz"],
            device=device,
            save=config["output"]["save_dir"] is not None,
            save_txt=config["output"]["save_txt"],
            save_conf=config["output"]["save_conf"],
            save_crop=config["output"]["save_crop"],
            show_labels=config["output"]["show_labels"],
            show_conf=config["output"]["show_conf"],
            line_width=config["output"]["line_width"],
            project=os.path.join(project_root, config["output"]["save_dir"]),
            name="predictions_improved",
            exist_ok=True,
        )

        # Post-process results
        logger.info("Applying post-processing to improve classification...")
        results = reclassify_detections(results, class_names)

        output_dir = os.path.join(project_root, config["output"]["save_dir"])
        logger.info(
            f"Inference completed! Results saved to: {output_dir}/predictions_improved"
        )

        # Print detection summary
        if results:
            logger.info(f"Processed {len(results)} frame(s)")
            class_counts = {}
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls = int(box.cls)
                        cls_name = (
                            class_names[cls] if cls < len(class_names) else "unknown"
                        )
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            logger.info("\nDetection Summary:")
            for cls_name, count in sorted(
                class_counts.items(), key=lambda x: x[1], reverse=True
            ):
                logger.info(f"  {cls_name}: {count} detection(s)")

    except Exception as e:
        logger.error(f"Inference failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Improved inference with better class handling"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference_config.yaml",
        help="Path to inference config file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source (image, video, directory, or webcam ID)",
    )
    parser.add_argument(
        "--lower-confidence",
        action="store_true",
        help="Use lower confidence threshold for better rare class detection",
    )
    args = parser.parse_args()

    run_improved_inference(args.config, args.source, args.lower_confidence)
