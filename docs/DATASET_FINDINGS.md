# Dataset Findings and Analysis

## Dataset Overview

- **Total Images**: 2,291 images
- **Validation Set**: 459 images with 1,383 annotated instances
- **Classes**: 23 human activity classes
- **Format**: YOLOv8/YOLOv11 format
- **Source**: [Roboflow Universe - Human Activity Dataset](https://universe.roboflow.com/cctv-rfavb/human-activity-kynyq)

## Class Distribution (Validation Set)

Based on validation set analysis (459 images, 1,383 instances):

### High-Frequency Classes (>100 instances)

| Class | Instances | Percentage | Status |
| ----- | --------- | ---------- | ------ |
| `standing` | 513 | 37.1% | ✅ Well-represented |
| `walking` | 413 | 29.9% | ✅ Well-represented |
| `sitting` | 257 | 18.6% | ✅ Well-represented |
| `using_phone` | 82 | 5.9% | ⚠️ Moderate |

**Total**: 1,265 instances (91.5% of dataset)

### Medium-Frequency Classes (10-50 instances)

| Class | Instances | Percentage | Status |
| ----- | --------- | ---------- | ------ |
| `Cleaning` | 33 | 2.4% | ⚠️ Underrepresented |
| `sitting on Desk` | 21 | 1.5% | ⚠️ Underrepresented |
| `working` | 22 | 1.6% | ⚠️ Underrepresented |

**Total**: 76 instances (5.5% of dataset)

### Low-Frequency Classes (<10 instances)

| Class | Instances | Percentage | Status |
| ----- | --------- | ---------- | ------ |
| `patient on stretcher` | 6 | 0.4% | ❌ Severely underrepresented |
| `Mopping` | 10 | 0.7% | ❌ Severely underrepresented |
| `stretcher attendant` | 5 | 0.4% | ❌ Severely underrepresented |
| `keeping_walkie_talkiee_charging` | 3 | 0.2% | ❌ Severely underrepresented |
| `patient on wheel chair` | 3 | 0.2% | ❌ Severely underrepresented |
| `wheelchair attendant` | 3 | 0.2% | ❌ Severely underrepresented |
| `checking_bag` | 1 | 0.1% | ❌ Severely underrepresented |
| `eating` | 4 | 0.3% | ❌ Severely underrepresented |
| `holding_walkiee_talkiee` | 2 | 0.1% | ❌ Severely underrepresented |
| `searching` | 2 | 0.1% | ❌ Severely underrepresented |
| `sleeping` | 1 | 0.1% | ❌ Severely underrepresented |
| `talking` | 2 | 0.1% | ❌ Severely underrepresented |

**Total**: 42 instances (3.0% of dataset)

## Key Findings

### 1. Severe Class Imbalance

- **Top 3 classes** (standing, walking, sitting) account for **85.6%** of all instances
- **11 classes** have **≤6 instances** each, making them extremely difficult to learn
- **Imbalance ratio**: 513:1 (standing vs. checking_bag/sleeping)

### 2. Training Impact

**Strong Performance** (70-80% mAP50):

- Classes with 200+ instances: `sitting` (257), `walking` (413), `standing` (513)
- These classes have sufficient data for the model to learn effectively

**Moderate Performance** (20-45% mAP50):

- Classes with 20-100 instances: `Cleaning` (33), `using_phone` (82), `working` (22)
- Model struggles but can detect some instances

**Poor/Zero Performance** (0% mAP50):

- Classes with <10 instances: All 11 low-frequency classes show 0% detection
- Insufficient data for the model to learn meaningful patterns

### 3. Dataset Quality Issues

1. **Extreme Imbalance**: Some classes have 500+ samples while others have only 1-2
2. **Rare Activities**: Medical/emergency activities (stretcher, wheelchair) are very rare
3. **Specialized Equipment**: Activities involving walkie-talkies are extremely rare
4. **Context-Specific**: Some activities may be context-dependent (e.g., "Battery low" - not visible in validation set)

### 4. Recommendations

#### For Immediate Improvement

1. **Data Collection Priority**:
   - Focus on collecting 50+ samples for each rare class
   - Prioritize: `Mopping`, `checking_bag`, `eating`, `patient on stretcher`
   - Medical scenarios: `stretcher attendant`, `wheelchair attendant`, `patient on wheel chair`

2. **Data Augmentation**:
   - Apply aggressive augmentation to rare classes (rotation, brightness, contrast)
   - Use copy-paste augmentation for rare classes
   - Synthetic data generation for underrepresented classes

3. **Class Weighting**:
   - Implement weighted loss function to penalize misclassification of rare classes more heavily
   - Use inverse frequency weighting: `weight = total_samples / (num_classes * class_samples)`

4. **Transfer Learning**:
   - Use models pre-trained on similar human activity datasets
   - Fine-tune with class-balanced sampling

#### For Long-Term Improvement

1. **Balanced Dataset**: Aim for at least 100 samples per class
2. **Diverse Scenarios**: Collect data in different environments, lighting, angles
3. **Active Learning**: Focus annotation efforts on rare classes
4. **Ensemble Methods**: Train separate models for rare classes and combine

## Expected vs. Actual Performance

| Class Category | Expected mAP50 | Actual mAP50 | Gap |
| -------------- | -------------- | ------------ | --- |
| High-frequency (>100) | 70-85% | 74-80% | ✅ Met expectations |
| Medium-frequency (10-50) | 40-60% | 25-43% | ⚠️ Below expectations |
| Low-frequency (<10) | 10-30% | 0% | ❌ Failed to learn |

## Conclusion

The dataset shows **severe class imbalance** that directly impacts model performance. While the model achieves excellent results (70-80% mAP50) on well-represented classes, it completely fails on rare classes due to insufficient training data. **Collecting more data for rare classes is the highest priority** for improving overall model performance.

---

**Last Updated**: Based on 100-epoch training results (January 2026)
