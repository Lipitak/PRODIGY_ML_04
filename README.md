cat > README.md << 'EOF'
# PRODIGY_ML_04 — Hand Gesture Recognition ✋

## What I built
A CNN (Convolutional Neural Network) model that recognizes and classifies hand gestures from images — 10 different gesture classes, including palm, fist, thumb, OK sign, and more.

The idea is simple: the model looks at an image (or webcam frame) and predicts which gesture it is — useful for gesture-based control systems, sign language interfaces, and similar HCI applications.

## Dataset
[LeapGestRecog](https://www.kaggle.com/gti-upm/leapgestrecog) — 20,000 grayscale images, 10 gesture classes, captured from 10 different subjects. This dataset was collected using a Leap Motion sensor (infrared camera, controlled lighting conditions).

## Approach
- Resized all images to 64x64, converted to grayscale, normalized pixel values (0-1 range)
- CNN architecture: 3 Conv2D layers (32 → 64 → 128 filters) with BatchNorm, MaxPooling, and Dropout to prevent overfitting
- Data augmentation (rotation, zoom, shift) for better generalization
- Train/test split: 80/20, stratified across all gesture classes

## Result
**Test Accuracy: 99.95%** 🎉

Also generated a confusion matrix and classification report to check for any misclassification patterns across gestures — results were clean with minimal confusion.

## Gestures classified
Palm, L Shape, Fist, Fist (Moved), Thumb, Index, OK Sign, Palm (Moved), C Shape, Down

## Real-time Webcam Demo (Bonus)
`webcam_demo_v2.py` uses MediaPipe Hands to detect the hand region in a live webcam feed and crop it (removing most of the background) before passing it to the CNN for prediction — this reduces background noise compared to a fixed bounding-box approach.

## ⚠️ Known Limitation
While the model achieves 99.95% accuracy on the test set, it performs noticeably worse on live webcam input. This is due to a **domain gap**: the training dataset was captured with a Leap Motion infrared sensor under tightly controlled conditions, while a regular webcam introduces different lighting, backgrounds, and image characteristics. The model learned patterns specific to the training distribution, so it struggles to generalize to unseen, real-world input.

**Two ways to properly address this:**
1. Collect and retrain on webcam-captured data
2. Use MediaPipe's hand landmarks (21 keypoints) as input features instead of raw pixels — this approach is background-independent

This is a common real-world ML deployment challenge — high accuracy on a clean benchmark dataset doesn't always translate to real-world performance.

## Tech Stack
Python, TensorFlow/Keras, OpenCV, MediaPipe, scikit-learn

## Files
- `gesture_train.py` — data loading, CNN model definition, training
- `check_accuracy.py` — reload saved model and evaluate
- `evaluate_detailed.py` — generates confusion matrix and classification report
- `webcam_demo_v2.py` — real-time webcam demo using MediaPipe
- `confusion_matrix.png` — visual confusion matrix result

## Part of
Prodigy InfoTech ML Internship — Task 04
EOF