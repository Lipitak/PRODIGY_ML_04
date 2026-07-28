import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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
loss, acc = model.evaluate(X_test, y_test)
print(f"\n>>> FINAL TEST ACCURACY: {acc*100:.2f}% <<<\n")