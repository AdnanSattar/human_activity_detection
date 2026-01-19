"""
Evaluation module for Human Activity Detection model.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from src.utils import (
    check_cuda_availability,
    get_project_root,
    load_config,
    print_cuda_info,
    setup_logging,
)


def evaluate_model(config_path: str = "config/inference_config.yaml"):
    """
    Evaluate trained model on test set.

    Args:
        config_path: Path to inference configuration file
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting model evaluation...")

    # Load configuration
    project_root = get_project_root()
    config = load_config(os.path.join(project_root, config_path))

    # Print CUDA info
    print_cuda_info()
    cuda_info = check_cuda_availability()

    if not cuda_info["cuda_available"]:
        logger.warning("CUDA not available. Evaluation will use CPU (slower).")
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

    # Get data config path
    data_yaml = os.path.join(project_root, "config", "data.yaml")

    # Run validation
    logger.info("Running validation on test set...")
    try:
        metrics = model.val(
            data=data_yaml,
            conf=config["model"]["conf_threshold"],
            iou=config["model"]["iou_threshold"],
            imgsz=config["input"]["imgsz"],
            device=device,
            save_json=True,
            plots=True,
            project=os.path.join(project_root, "outputs", "runs"),
            name="validation",
            exist_ok=True,
        )

        logger.info("\n" + "=" * 50)
        logger.info("Evaluation Results")
        logger.info("=" * 50)
        logger.info(f"mAP50: {metrics.box.map50:.4f}")
        logger.info(f"mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"Precision: {metrics.box.mp:.4f}")
        logger.info(f"Recall: {metrics.box.mr:.4f}")
        logger.info("=" * 50)

        logger.info(f"\nDetailed results saved to: outputs/runs/validation/")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Human Activity Detection model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference_config.yaml",
        help="Path to inference configuration file",
    )

    args = parser.parse_args()
    evaluate_model(args.config)
