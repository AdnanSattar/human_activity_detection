# Model Training

Train model `Human_Activity_model.pt` on the Human Activity dataset.

## Step 1: Download the Dataset

1. Visit: <https://universe.roboflow.com/cctv-rfavb/human-activity-kynyq>
2. Click "Download Dataset"
3. Select "YOLOv8" format
4. Extract the ZIP file

## Step 2: Organize Dataset

Place files in this structure:

```text
data/
├── train/
│   ├── images/    # Put all training images here
│   └── labels/    # Put corresponding .txt label files here
├── valid/
│   ├── images/    # Put validation images here
│   └── labels/    # Put validation labels here
└── test/
    ├── images/    # Put test images here
    └── labels/    # Put test labels here
```

**Important**: Each image must have a corresponding `.txt` file with the same name in the `labels/` folder.

### Step 3: Verify Dataset

Check that:

- Images are in `images/` folders
- Labels are in `labels/` folders
- Label files match image names (e.g., `image001.jpg` → `image001.txt`)
- Label format is YOLO (class_id x_center y_center width height, normalized 0-1)

## Step 4: Train the Model

```bash
# Train with default settings
python -m src.train

# This will:
# - Load yolo11n.pt as base model
# - Train on your dataset
# - Save trained model to outputs/models/Human_Activity_model.pt
```

## Step 5: Verify Training

After training, check:

- Training logs in `logs/human_activity_detection.log`
- Training metrics in `outputs/runs/human_activity_detection/`
- Model file size should increase (trained models are larger)

## Step 6: Test Inference

```bash
# Test with trained model
python -m src.inference --source human_activity_video.mp4
```

## Expected Results After Training

- ✅ Accurate activity classification
- ✅ Detects all humans in scene
- ✅ Correct labels (working, sitting, standing, etc.)
- ✅ Higher confidence scores (>0.5)
- ✅ Better detection coverage

## Training Configuration

The training config (`config/training_config.yaml`) is already set up:

- 200 epochs
- Batch size: 8 (adjust if GPU memory issues)
- Image size: 640
- Augmentation enabled
- GPU acceleration

## Training Time Estimate

With your setup (MX450, 2GB VRAM):

- Dataset: ~2291 images
- Epochs: 200
- Estimated time: 2-4 hours (depending on batch size)

## Quick Test Training

To verify everything works, you can do a quick test:

```bash
# Edit config/training_config.yaml
# Set epochs: 5  # Just for testing
# Set batch_size: 4  # Smaller for testing

python -m src.train
```

Then test inference to see if it's better than before.

---
