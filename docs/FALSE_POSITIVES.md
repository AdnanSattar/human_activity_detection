# False Positive Detection Issues

## Problem

The model is detecting **non-human objects** (computer monitors, cups, furniture) as human activities, even with high confidence thresholds (0.9).

**Examples**:

- Computer monitor + cup detected as "walking 0.49"
- Small objects incorrectly classified as human activities
- Objects with wrong aspect ratios (too wide, too narrow) being detected

## Root Causes

### 1. Model Training Issues

- **Insufficient negative examples**: Model may not have seen enough non-human objects during training
- **Overfitting to training data**: Model learned patterns that don't generalize well
- **Class imbalance**: Model defaults to common classes even for non-human objects

### 2. Detection Configuration

- **IoU threshold too low**: Not removing overlapping false positives effectively
- **Max detections too high**: Allowing too many detections per image
- **No size/aspect ratio filtering**: Accepting detections that don't match human proportions

### 3. Model Limitations

- **Single-stage detection**: YOLO detects and classifies in one step, which can be less accurate
- **No person verification**: Model doesn't verify if detected object is actually a person before classifying activity

## Solutions Implemented

### 1. Bounding Box Filtering

Added `filter_false_positives()` function that filters detections based on:

- **Area ratio**: Bounding box must be 1-80% of image area
- **Aspect ratio**: Height/width ratio between 0.3-3.0 (humans are typically taller than wide)
- **Minimum size**: Width > 20px, Height > 30px (removes tiny false positives)

### 2. Configuration Updates

- **IoU threshold**: Increased to 0.6 (from 0.45) for better NMS
- **Max detections**: Reduced to 50 (from 300) to limit false positives
- **Confidence threshold**: Set to 0.5 (balanced value)

### 3. Post-Processing

The inference pipeline now automatically filters false positives after detection but before visualization.

## Usage

The filtering is **automatically applied** in `src/inference.py`. No changes needed to your code.

To adjust filtering parameters, modify the `filter_false_positives()` function:

```python
filter_false_positives(
    result,
    min_area_ratio=0.01,    # Minimum 1% of image
    max_area_ratio=0.8,     # Maximum 80% of image
    min_aspect_ratio=0.3,   # Minimum height/width ratio
    max_aspect_ratio=3.0,   # Maximum height/width ratio
)
```

## Additional Recommendations

### For Better Results

1. **Two-Stage Detection**:
   - First detect people using a person detector (YOLO person class or COCO person)
   - Then classify activity only for detected people
   - This eliminates non-human object detections

2. **Retrain with Negative Examples**:
   - Add images with no humans (empty scenes, objects only)
   - Label these as "background" or exclude from training
   - Helps model learn what NOT to detect

3. **Increase Training Data**:
   - More diverse scenes with various objects
   - Better balance between human and non-human examples
   - Augment with negative examples

4. **Use Larger Model**:
   - `yolo11s.pt` or `yolo11m.pt` instead of `yolo11n.pt`
   - Better feature extraction = fewer false positives
   - Trade-off: Slower inference, more GPU memory

5. **Post-Processing with Person Detector**:

   ```python
   # Pseudo-code
   person_model = YOLO('yolo11n.pt')  # COCO pre-trained
   activity_model = YOLO('Human_Activity_model.pt')
   
   # First detect people
   person_results = person_model.predict(image, classes=[0])  # class 0 = person
   
   # Then classify activity only for detected people
   for person_box in person_results:
       activity = activity_model.predict(crop(person_box))
   ```

## Verification

After applying filters, check:

1. **False positive rate**: Should decrease significantly
2. **True positive rate**: Should remain similar (may slightly decrease)
3. **Detection count**: Should decrease (fewer false positives)

## Expected Improvements

- **Before**: 10-20 false positives per frame (monitors, cups, etc.)
- **After**: 0-2 false positives per frame (mostly edge cases)
- **Trade-off**: May miss some small or partially occluded humans

## Troubleshooting

### If Too Many Detections Filtered

- Lower `min_area_ratio` (e.g., 0.005)
- Widen `min_aspect_ratio` (e.g., 0.2)
- Reduce minimum size requirements

### If Still Getting False Positives

- Increase `conf_threshold` (e.g., 0.6-0.7)
- Increase `iou_threshold` (e.g., 0.7)
- Reduce `max_det` (e.g., 20)
- Consider two-stage detection approach

### If Missing Real Humans

- Lower `conf_threshold` (e.g., 0.3-0.4)
- Increase `max_area_ratio` (e.g., 0.9)
- Widen aspect ratio range
- Check if humans are too small in frame

---

**Last Updated**: January 2026
