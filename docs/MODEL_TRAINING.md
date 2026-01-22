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

### Actual Results (100 Epochs, MX450 GPU)

**Training Time**: 42.159 hours (~1.75 days)

**Overall Metrics**:

- mAP50: 23.4%
- mAP50-95: 13.3%
- Precision: 22.5%
- Recall: 23.6%

**Strong Classes** (70-80% mAP50):

- ✅ `sitting`: 79.8% mAP50, 87.2% recall
- ✅ `sitting on Desk`: 79.8% mAP50, 81.0% recall
- ✅ `walking`: 74.8% mAP50, 93.0% recall
- ✅ `standing`: 74.1% mAP50, 78.8% recall

**Moderate Classes** (20-45% mAP50):

- ⚠️ `Cleaning`: 43.1% mAP50, 39.4% recall
- ⚠️ `using_phone`: 25.3% mAP50, 26.8% recall
- ⚠️ `working`: 41.5% mAP50, 22.7% recall

**Poor Classes** (0% detection):

- ❌ `Mopping`, `checking_bag`, `eating`, `holding_walkiee_talkiee`
- ❌ `patient on stretcher`, `patient on wheel chair`, `searching`
- ❌ `sleeping`, `stretcher attendant`, `talking`, `wheelchair attendant`

**Note**: Many classes with very few training instances (1-6 samples) show zero detections. This is expected with class imbalance.

## Training Configuration

The training config (`config/training_config.yaml`) is set up for:

- **100 epochs** (minimum viable for decent results)
- **Batch size: 8** (optimized for MX450 2GB VRAM)
- **Image size: 640**
- **Augmentation enabled**
- **GPU acceleration**
- **Workers: 1** (Windows-compatible)
- **Cache: false** (to avoid paging file errors)

## Training Time Estimate

With your setup (MX450, 2GB VRAM):

- **Dataset**: ~2,291 images (459 validation images)
- **100 epochs**: ~42 hours (actual measured)
- **200 epochs**: ~84 hours estimated (3.5 days)
- **50 epochs**: ~21 hours estimated (minimum viable)

**Tip**: Start with 50 epochs for a sanity check, then train 100+ epochs for production use.

## Training Modes

### 1. Pipeline Sanity Check (1-5 epochs)

Quick test to verify the training pipeline works:

```bash
# Edit config/training_config.yaml
# Set epochs: 5  # Just for testing

python -m src.train
```

**Expected**: Training completes without errors. Model may show 0% detections on rare classes.

### 2. Minimum Viable Training (50 epochs)

Practical starting point for usable results:

```bash
# Edit config/training_config.yaml
# Set epochs: 50

python -m src.train
```

**Expected**: ~21 hours on MX450. Some detections on common classes (sitting, walking, standing).

### 3. Production Training (100+ epochs)

For better accuracy and coverage:

```bash
# Edit config/training_config.yaml
# Set epochs: 100  # or 200 for best results

python -m src.train
```

**Expected**:

- 100 epochs: ~42 hours, mAP50 ~23-25%
- 200 epochs: ~84 hours, mAP50 ~25-30% (estimated)

## Improving Model Performance

### For Rare Classes (Zero Detections)

Many classes have very few training samples (1-6 instances), leading to poor performance:

1. **Collect more data**: Prioritize collecting more samples for rare classes
2. **Data augmentation**: Increase augmentation specifically for underrepresented classes
3. **Class weighting**: Modify loss function to weight rare classes more heavily
4. **Transfer learning**: Use a model pre-trained on similar human activity datasets

### For Overall Accuracy

1. **Train longer**: 200+ epochs can improve mAP50 by 5-10%
2. **Hyperparameter tuning**: Adjust learning rate, batch size, augmentation
3. **Larger model**: Try `yolo11s.pt` or `yolo11m.pt` (if GPU memory allows)
4. **Ensemble**: Combine multiple models for better accuracy

---
