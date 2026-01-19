"""
Inference module for Human Activity Detection using YOLO.
Supports image, video, webcam, and directory inference.
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.utils import (
    check_cuda_availability,
    get_project_root,
    load_config,
    print_cuda_info,
    setup_logging,
)


def run_inference(
    config_path: str = "config/inference_config.yaml", source: str = None
):
    """
    Run inference on images, videos, or webcam.

    Args:
        config_path: Path to inference configuration file
        source: Override source from config (optional)
    """
    # Setup logging
    logger = setup_logging()
    logger.info("Starting inference...")

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
        logger.info(
            "Please train a model first or update the checkpoint_path in config/inference_config.yaml"
        )
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

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
            # Remove leading / on Windows
            source = source.lstrip("/")
        source = os.path.join(project_root, source)

    # Prepare output directory
    output_dir = config["output"]["save_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run inference
    logger.info(f"Running inference on: {source}")

    try:
        results = model.predict(
            source=source,
            conf=config["model"]["conf_threshold"],
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
            project=output_dir,
            name="predictions",
            exist_ok=True,
        )

        logger.info(f"Inference completed! Results saved to: {output_dir}/predictions")

        # If source is video or webcam, handle video output
        if isinstance(source, int) or (
            isinstance(source, str)
            and source.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
        ):
            logger.info("Processing video/webcam stream...")
            process_video_stream(
                model=model,
                source=source,
                config=config,
                output_dir=output_dir,
                device=device,
                logger=logger,
            )

        # Print summary
        if results:
            logger.info(f"Processed {len(results)} frame(s)")
            for i, result in enumerate(results):
                logger.info(f"Frame {i+1}: {len(result.boxes)} detection(s)")

    except Exception as e:
        logger.error(f"Inference failed with error: {str(e)}", exc_info=True)
        raise


def process_video_stream(model, source, config, output_dir, device, logger):
    """
    Process video stream with OpenCV for real-time display and saving.

    Args:
        model: YOLO model instance
        source: Video source (path or webcam index)
        config: Inference configuration
        output_dir: Output directory
        device: Device to use
        logger: Logger instance
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or config["video"]["fps"]
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width == 0 or frame_height == 0:
        frame_width, frame_height = 640, 480
        logger.warning("Could not determine video dimensions, using default 640x480")

    # Setup video writer
    output_format = config["video"]["format"]
    output_path = os.path.join(output_dir, f"output_video.{output_format}")

    fourcc_map = {
        "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
        "avi": cv2.VideoWriter_fourcc(*"XVID"),
        "mkv": cv2.VideoWriter_fourcc(*"X264"),
    }
    fourcc = fourcc_map.get(output_format, cv2.VideoWriter_fourcc(*"mp4v"))

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    logger.info(
        f"Video output: {output_path} ({frame_width}x{frame_height} @ {fps}fps)"
    )

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = model.predict(
                frame,
                conf=config["model"]["conf_threshold"],
                iou=config["model"]["iou_threshold"],
                imgsz=config["input"]["imgsz"],
                device=device,
                verbose=False,
            )

            # Annotate frame
            annotated_frame = results[0].plot()

            # Display if enabled (skip if GUI not available)
            if config["display"]["show"]:
                try:
                    cv2.imshow("Human Activity Detection", annotated_frame)
                    if cv2.waitKey(config["display"]["wait_key"]) & 0xFF == ord("q"):
                        logger.info("Stopped by user (pressed 'q')")
                        break
                except cv2.error as e:
                    if "not implemented" in str(e).lower():
                        logger.warning(
                            "OpenCV GUI not available, skipping display. Video will still be saved."
                        )
                        config["display"][
                            "show"
                        ] = False  # Disable display for remaining frames
                    else:
                        raise

            # Write frame
            out.write(annotated_frame)
            frame_count += 1

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames...")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Ignore if GUI not available
        logger.info(
            f"Video processing complete. Saved {frame_count} frames to {output_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for Human Activity Detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/inference_config.yaml",
        help="Path to inference configuration file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override source (image path, video path, directory, or webcam index)",
    )

    args = parser.parse_args()
    run_inference(args.config, args.source)
