# Inference Issues and Solutions

## Problem: Misclassification of "Working" Activity

### Issue Description

The model frequently misclassifies people who are clearly **sitting at desks and working** as **"standing"** instead of **"working"** or **"sitting on Desk"**.

**Example**: In an office video, three people are sitting at desks operating computers, but the model labels them as "standing" with confidence 0.56.

### Root Cause

**Severe Class Imbalance**:

| Class | Training Instances | Percentage | Issue |
| ----- | ------------------ | ---------- | ----- |
| `standing` | 513 | 37.1% | ✅ Dominant class |
| `working` | 22 | 1.6% | ❌ Severely underrepresented |
| `sitting on Desk` | 21 | 1.5% | ❌ Severely underrepresented |
| `sitting` | 257 | 18.6% | ✅ Well-represented |

**Imbalance Ratio**: 513:22 = **23:1** (standing vs. working)

The model has seen "standing" 23 times more often than "working" during training, causing it to default to "standing" when uncertain.

### Why This Happens

1. **Training Bias**: The model learns that "standing" is the most common class and predicts it more frequently
2. **Low Confidence for Rare Classes**: Even when "working" is detected, its confidence may be lower than "standing"
3. **Similar Visual Features**: Sitting at a desk and standing can have similar upper-body appearances, making classification difficult
4. **Insufficient Training Data**: 22 instances of "working" is insufficient for the model to learn robust patterns

### Validation Metrics vs. Real-World Performance

**Validation Results** (on test set):

- `working`: 41.5% mAP50, 22.7% recall
- `sitting on Desk`: 79.8% mAP50, 81.0% recall
- `standing`: 74.1% mAP50, 78.8% recall

**Real-World Inference**:

- Model defaults to "standing" even when people are clearly working
- "working" detections are rare or have very low confidence
- Model misses some people entirely

**Why the Discrepancy?**

- Validation set may have different distribution than real-world video
- Model may overfit to validation set characteristics
- Class imbalance causes model to favor dominant classes in uncertain cases

## Solutions

### Immediate Fixes (No Retraining Required)

#### 1. Lower Confidence Threshold

**Current**: `conf_threshold: 0.35` (already updated)

**How it helps**: Lower threshold allows more detections, including lower-confidence "working" predictions that might otherwise be filtered out.

**Trade-off**: May increase false positives, but helps detect rare classes.

#### 2. Use Improved Inference Script

```bash
python -m src.inference_improved --source human_activity_video.mp4 --lower-confidence
```

This script:

- Uses lower confidence threshold automatically
- Applies post-processing to boost rare class detections
- Provides better detection summary

#### 3. Check All Class Probabilities

Instead of just the top prediction, check if "working" or "sitting on Desk" appear with reasonable confidence (even if not the top class).

### Training Solutions (Requires Retraining)

#### 1. Collect More Data (Highest Priority)

**Target**: At least 50-100 samples for "working" and "sitting on Desk"

**Priority Classes**:

- `working`: 22 → 100+ instances
- `sitting on Desk`: 21 → 100+ instances
- `operating_comp`: Check current count, target 50+

**Data Collection Strategy**:

- Focus on office/desk work scenarios
- Capture different angles, lighting, desk types
- Include diverse people and computer setups

#### 2. Implement Class Weighting

Modify training to weight rare classes more heavily:

```python
# In training config or code
class_weights = {
    'working': 23.0,  # Inverse of frequency ratio
    'sitting on Desk': 24.0,
    'standing': 0.5,  # Reduce weight for dominant class
}
```

This penalizes misclassification of rare classes more heavily.

#### 3. Data Augmentation for Rare Classes

Apply aggressive augmentation specifically to underrepresented classes:

- Rotation, brightness, contrast adjustments
- Copy-paste augmentation (paste rare class instances into more images)
- Synthetic data generation

#### 4. Longer Training

- Current: 100 epochs (~42 hours)
- Recommended: 200+ epochs (~84 hours)
- May help, but **data imbalance is the root cause**

#### 5. Transfer Learning

Use a model pre-trained on similar human activity datasets, then fine-tune with class-balanced sampling.

### Advanced Solutions

#### 1. Ensemble Methods

Train separate models:

- One for common classes (standing, walking, sitting)
- One for rare classes (working, sitting on Desk)
- Combine predictions

#### 2. Context-Aware Post-Processing

Implement rules based on scene context:

- If person is at desk → prefer "working" or "sitting on Desk"
- If person is in office environment → boost "working" confidence
- Use object detection (desk, computer) to inform activity classification

#### 3. Multi-Stage Detection

1. First detect people
2. Then classify activity with context (location, objects nearby)
3. Apply class-specific confidence thresholds

## Verification Steps

### 1. Check Training Data

```bash
# Count instances per class in training set
python -c "
from pathlib import Path
import yaml

data_yaml = 'config/data.yaml'
with open(data_yaml) as f:
    data = yaml.safe_load(f)
    
# Check label files
train_labels = Path('data/train/labels')
class_counts = {}
for label_file in train_labels.glob('*.txt'):
    with open(label_file) as f:
        for line in f:
            class_id = int(line.split()[0])
            class_name = data['names'][class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

print('Training set class distribution:')
for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'  {cls}: {count}')
"
```

### 2. Review Validation Images

Check validation predictions:

- `outputs/runs/human_activity_detection/val_batch*_pred.jpg`
- Look for "working" or "sitting on Desk" detections
- Note their confidence scores

### 3. Test with Lower Confidence

```bash
# Edit config/inference_config.yaml
# Set conf_threshold: 0.25

python -m src.inference --source human_activity_video.mp4
```

Check if "working" appears with lower confidence.

## Expected Improvements

### After Data Collection (50+ "working" samples)

- **Expected mAP50**: 50-60% (up from 41.5%)
- **Expected Recall**: 40-50% (up from 22.7%)
- **Real-world**: Should see "working" detections in office videos

### After Class Weighting

- **Expected**: Better balance between "standing" and "working"
- **Trade-off**: Slight decrease in "standing" accuracy

### After 200+ Epochs

- **Expected mAP50**: 25-30% overall (up from 23.4%)
- **Expected**: Better convergence, but data imbalance still limits performance

## Conclusion

The misclassification of "working" as "standing" is a **direct result of class imbalance**. While immediate fixes (lower confidence threshold) can help, the **long-term solution requires collecting more training data** for underrepresented classes, particularly "working" and "sitting on Desk".

**Priority**: Collect 50-100 samples of "working" activity before retraining.

---

**Last Updated**: January 2026
