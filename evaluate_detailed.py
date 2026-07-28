import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = "leapGestRecog/leapgestrecog/leapGestRecog"
IMG_SIZE = 64

images = []
labels = []
label_map = {}
label_counter = 0

for subject_folder in sorted(os.listdir(DATA_DIR)):
    subject_path = os.path.join(DATA_DIR, subject_folder)
    if not os.path.isdir(subject_path):
        continue
    for gesture_folder in sorted(os.listdir(subject_path)):
        gesture_path = os.path.join(subject_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue
        if gesture_folder not in label_map:
            label_map[gesture_folder] = label_counter
            label_counter += 1
        for img_file in os.listdir(gesture_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label_map[gesture_folder])

X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(np.array(labels))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = load_model("gesture_model.h5")

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

gesture_names = list(label_map.keys())

# Classification report (precision, recall, f1 per class)
print("\n--- Classification Report ---")
print(classification_report(y_true_classes, y_pred_classes, target_names=gesture_names))

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=gesture_names, yticklabels=gesture_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Hand Gesture Recognition')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\nConfusion matrix saved as confusion_matrix.png")
plt.show()