cat > README.md << 'EOF'
# PRODIGY_ML_04 — Hand Gesture Recognition

## Task
Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image data, enabling gesture-based control systems.

## Dataset
[LeapGestRecog](https://www.kaggle.com/gti-upm/leapgestrecog) — 20,000 grayscale images, 10 gesture classes, 10 subjects.

## Approach
- Images resized to 64x64, grayscale, normalized
- CNN architecture: 3 Conv2D blocks (32→64→128 filters) with BatchNorm, MaxPooling, Dropout
- Data augmentation: rotation, zoom, width/height shift
- Train/test split: 80/20, stratified

## Result
**Test Accuracy: 99.95%**

## Gestures classified
palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down

## Tech stack
Python, TensorFlow/Keras, OpenCV, scikit-learn

## Files
- `gesture_train.py` — data loading, CNN model, training
- `check_accuracy.py` — reload saved model and evaluate
EOF
