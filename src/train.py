"""
Training module for Human Activity Detection using YOLO.
"""

import argparse
import os
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.utils import (
    check_cuda_availability,
    get_project_root,
    load_config,
    print_cuda_info,
    setup_logging,
    validate_paths,
)


def train_model(config_path: str = "config/training_config.yaml"):
    """
    Train YOLO model for human activity detection.

    Args:
        config_path: Path to training configuration file
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting training process...")

    # Load configuration
    project_root = get_project_root()
    config = load_config(os.path.join(project_root, config_path))

    # Print CUDA info
    print_cuda_info()
    cuda_info = check_cuda_availability()

    if not cuda_info["cuda_available"]:
        logger.warning("CUDA not available. Training will use CPU (much slower).")
        device = "cpu"
    else:
        device = config["training"]["device"]
        logger.info(f"Using GPU: {cuda_info['gpu_name']}")

    # Validate paths
    if not validate_paths(config, project_root):
        logger.warning("Some paths in configuration may not exist. Please check.")

    # Get model path
    model_name = config["model"]["name"]
    if not os.path.isabs(model_name):
        model_path = os.path.join(project_root, model_name)
        if not os.path.exists(model_path):
            # Try without models/ prefix (for backward compatibility)
            model_name_alt = model_name.replace("models/", "")
            model_path_alt = os.path.join(project_root, model_name_alt)
            if os.path.exists(model_path_alt):
                model_path = model_path_alt
            else:
                logger.info(
                    f"Model file not found locally, will download: {model_name}"
                )
                model_path = model_name
    else:
        model_path = model_name

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Get data config path
    data_yaml = config["data"]["yaml_path"]
    if not os.path.isabs(data_yaml):
        data_yaml = os.path.join(project_root, data_yaml)

    # Update data.yaml paths to be absolute before passing to YOLO
    # YOLO resolves paths relative to the config file location, so we need absolute paths
    import yaml

    with open(data_yaml, "r") as f:
        data_config = yaml.safe_load(f)

    # Convert relative paths to absolute
    for key in ["train", "val", "test"]:
        if key in data_config and not os.path.isabs(data_config[key]):
            data_config[key] = os.path.join(project_root, data_config[key]).replace(
                "\\", "/"
            )

    # Write updated config to temp file
    import tempfile

    temp_data_yaml = os.path.join(project_root, "config", "data_temp.yaml")
    with open(temp_data_yaml, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    data_yaml = temp_data_yaml

    # Prepare training arguments
    train_args = {
        "data": data_yaml,
        "epochs": config["training"]["epochs"],
        "imgsz": config["training"]["imgsz"],
        "batch": config["training"]["batch_size"],
        "name": config["output"]["name"],
        "device": device,
        "project": os.path.join(project_root, config["output"]["project"]),
        "exist_ok": config["output"]["exist_ok"],
        "save_period": config["output"]["save_period"],
        "cache": config["training"]["cache"],
        "workers": config["training"]["workers"],
        "rect": config["optimization"]["rect"],
        "resume": config["optimization"]["resume"],
        "amp": config["optimization"]["amp"],
        "save_json": config["validation"]["save_json"],
        "plots": config["validation"]["plots"],
        "augment": config["augmentation"]["enabled"],
        "conf": config["validation"]["conf"],
        "iou": config["validation"]["iou"],
        "multi_scale": config["optimization"]["multi_scale"],
    }

    # Add learning rate if specified
    if "lr0" in config["training"]:
        train_args["lr0"] = config["training"]["lr0"]
    if "lrf" in config["training"]:
        train_args["lrf"] = config["training"]["lrf"]

    # Add augmentation parameters
    if config["augmentation"]["enabled"]:
        train_args.update(
            {
                "mixup": config["augmentation"]["mixup"],
                "flipud": config["augmentation"]["flipud"],
                "fliplr": config["augmentation"]["fliplr"],
            }
        )

    logger.info("Training parameters:")
    for key, value in train_args.items():
        logger.info(f"  {key}: {value}")

    # Start training
    logger.info("Starting training...")
    try:
        results = model.train(**train_args)
        logger.info("Training completed successfully!")

        # Save model to outputs/models
        model_save_path = os.path.join(
            project_root, "outputs", "models", "Human_Activity_model.pt"
        )
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logger.info(f"Model saved to: {model_save_path}")

        # Perform validation (skip if memory issues)
        try:
            logger.info("Running validation...")
            model.val(save_json=True)
        except Exception as e:
            logger.warning(f"Validation skipped due to error: {str(e)}")
            logger.info(
                "You can run validation separately with: python -m src.evaluate"
            )

        # Print results summary
        logger.info("\n" + "=" * 50)
        logger.info("Training Summary")
        logger.info("=" * 50)
        logger.info(
            f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}"
        )
        logger.info(
            f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}"
        )
        logger.info("=" * 50)

        logger.info("\nTo visualize training results with TensorBoard:")
        logger.info(
            f"tensorboard --logdir {os.path.join(project_root, config['output']['project'])}"
        )

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLO model for Human Activity Detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file",
    )

    args = parser.parse_args()
    train_model(args.config)
