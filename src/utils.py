"""
Utility functions for the Human Activity Detection project.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "human_activity_detection.log")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_cuda_availability() -> Dict[str, Any]:
    """
    Check CUDA availability and GPU information.

    Returns:
        Dictionary with CUDA info
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_memory": (
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            if torch.cuda.is_available()
            else None
        ),
    }
    return info


def print_cuda_info():
    """Print CUDA and GPU information."""
    info = check_cuda_availability()
    print("\n" + "=" * 50)
    print("CUDA & GPU Information")
    print("=" * 50)
    print(f"CUDA Available: {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"GPU Count: {info['gpu_count']}")
        print(f"GPU Name: {info['gpu_name']}")
        print(f"GPU Memory: {info['gpu_memory']}")
    print("=" * 50 + "\n")


def ensure_dir(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def validate_paths(config: Dict[str, Any], base_path: Optional[str] = None) -> bool:
    """
    Validate that paths in configuration exist.

    Args:
        config: Configuration dictionary
        base_path: Base path for relative paths

    Returns:
        True if all paths are valid
    """
    if base_path is None:
        base_path = get_project_root()

    required_paths = []

    # Check data paths
    if "data" in config:
        data_config = config["data"]
        if "yaml_path" in data_config:
            required_paths.append(data_config["yaml_path"])

    # Check model paths
    if "model" in config:
        model_config = config["model"]
        if "name" in model_config and not model_config.get("pretrained", True):
            required_paths.append(model_config["name"])

    all_exist = True
    for path in required_paths:
        full_path = os.path.join(base_path, path) if not os.path.isabs(path) else path
        if not os.path.exists(full_path):
            print(f"Warning: Path does not exist: {full_path}")
            all_exist = False

    return all_exist
