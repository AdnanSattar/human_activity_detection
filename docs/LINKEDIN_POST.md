**🚀 From Dataset to Training & Inference: Building Human Activity Detection System with YOLO**

After 100 epochs and 42 hours of training on MX450 GPU, I learned a hard lesson: **"Garbage in, garbage out" is absolutely true** in computer vision.

**📊 The Dataset Reality Check**

I started with a dataset of 2,291 images across 23 activity classes. Sounds good, right?

Wrong. The class distribution was brutal:

- **Standing**: 513 instances (37.1%)
- **Walking**: 413 instances (29.9%)
- **Sitting**: 257 instances (18.6%)
- **Working**: 22 instances (1.6%) ⚠️
- **11 classes**: Only 1-6 instances each ❌

That's a **23:1 imbalance ratio**. The model saw "standing" 23 times more often than "working." So model became biased towards "standing" and "walking" and "sitting" and less towards "working".

The model achieved:

- ✅ **74-80% mAP50** on common classes (sitting, walking, standing)
- ⚠️ **25-43% mAP50** on moderate classes (cleaning, using phone)
- ❌ **0% mAP50** on 11 rare classes (completely failed to learn)

**💡 The Inference Challenges**

When I tested on real video, I found that the model was misclassifying "working" as "standing" and "walking" as "sitting" and "sitting" as "standing".

- People sitting at desks → misclassified as "standing" 😞
- Computer monitors → detected as "walking" (false positives)
- Missing detections for one person in a 3-person scene

The model defaulted to dominant classes when uncertain. Classic class imbalance behavior. So I had to use improved inference script to improve the results.

**🔧 The Solutions**

1. **False Positive Filtering**: Added bounding box size/aspect ratio filters to remove non-human objects
2. **Class Weighting**: Implemented inverse frequency weighting (calculated but YOLO uses focal loss)
3. **Aggressive Augmentation**: Increased copy-paste, rotation, color jittering for rare classes
4. **Lower Confidence Thresholds**: Adjusted from 0.55 to 0.35 to catch rare class detections

**📈 Key Learnings**

1. **Data quality > Model architecture**: No amount of training can fix severe class imbalance
2. **Validation metrics ≠ Real-world performance**: 41.5% mAP50 for "working" but 0% in practice
3. **Post-processing is crucial**: Filtering false positives improved results significantly
4. **Collect more data**: The highest priority fix is getting 50-100 samples per rare class (Mopping, checking_bag, eating, etc.)
5. **Longer training**: Consider 200+ epochs for better convergence
6. **Class weighting**: Use class weights to balance loss for rare classes
7. **Transfer learning**: Consider fine-tuning from a model pre-trained on similar human activity datasets

**🎓 The Takeaway**

This project taught me that building training and inference pipeline for CV systems requires understanding  data distribution FIRST and then building the model accordingly.

The model works, but it's biased.

**The journey continues...** Next step: collect more balanced data and retrain with class weighting enabled or maybe use transfer learning from pre-trained model on similar human activity datasets.

**What would you prioritize: more data collection or better augmentation strategies?**

Drop your thoughts below! 👇

---

# ComputerVision #DeepLearning #YOLO #MachineLearning #ObjectDetection #ClassImbalance #PyTorch #CUDA #GPU #ComputerVisionProjects #MLEngineering #NeuralNetworks #HumanActivityRecognition #MLOps #LearningInPublic
