"""
Quick start script to verify installation and setup.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    check_cuda_availability,
    load_config,
    print_cuda_info,
    validate_paths,
)


def main():
    """Run quick start checks."""
    print("\n" + "=" * 60)
    print("Human Activity Detection - Quick Start Check")
    print("=" * 60 + "\n")

    # Check Python version
    print(f"Python Version: {sys.version}")
    print(f"Project Root: {project_root}\n")

    # Check CUDA
    print_cuda_info()
    cuda_info = check_cuda_availability()

    if not cuda_info["cuda_available"]:
        print(
            "⚠️  WARNING: CUDA not available. Training/inference will be slow on CPU.\n"
        )
    else:
        print("✅ CUDA is available!\n")

    # Check dependencies
    print("Checking dependencies...")
    required_packages = ["ultralytics", "torch", "cv2", "yaml", "numpy"]
    missing_packages = []

    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "yaml":
                import yaml
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt\n")
    else:
        print("\n✅ All required packages are installed!\n")

    # Check project structure
    print("Checking project structure...")
    required_dirs = [
        "config",
        "src",
        "data/train/images",
        "data/valid/images",
        "data/test/images",
        "outputs/models",
        "outputs/runs",
        "outputs/inference",
        "logs",
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ⚠️  {dir_path}/ - Missing (will be created when needed)")
            missing_dirs.append(dir_path)

    # Check configuration files
    print("\nChecking configuration files...")
    config_files = [
        "config/data.yaml",
        "config/training_config.yaml",
        "config/inference_config.yaml",
    ]

    for config_file in config_files:
        full_path = project_root / config_file
        if full_path.exists():
            print(f"  ✅ {config_file}")
            try:
                config = load_config(str(full_path))
                print(f"     Loaded successfully")
            except Exception as e:
                print(f"     ⚠️  Error loading: {e}")
        else:
            print(f"  ❌ {config_file} - MISSING")

    # Check for model files
    print("\nChecking model files...")
    model_files = ["yolo11n.pt", "outputs/models/Human_Activity_model.pt"]

    for model_file in model_files:
        full_path = project_root / model_file
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {model_file} ({size_mb:.2f} MB)")
        else:
            if model_file == "yolo11n.pt":
                print(f"  ⚠️  {model_file} - Will be downloaded automatically")
            else:
                print(f"  ⚠️  {model_file} - Train a model first")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if missing_packages:
        print("❌ Please install missing packages: pip install -r requirements.txt")
    elif not cuda_info["cuda_available"]:
        print("⚠️  CUDA not available - will use CPU (slower)")
        print("✅ Ready to proceed (CPU mode)")
    else:
        print("✅ Setup looks good! Ready to proceed.")

    print("\nNext steps:")
    print("  1. Prepare your dataset in data/train/, data/valid/, data/test/")
    print("  2. Train: python -m src.train")
    print("  3. Evaluate: python -m src.evaluate")
    print("  4. Inference: python -m src.inference")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
