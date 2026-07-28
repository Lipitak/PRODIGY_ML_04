import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

IMG_SIZE = 64
model = load_model("gesture_model.h5")

gesture_names = {
    0: 'Palm', 1: 'L Shape', 2: 'Fist', 3: 'Fist (Moved)', 4: 'Thumb',
    5: 'Index', 6: 'OK Sign', 7: 'Palm (Moved)', 8: 'C Shape', 9: 'Down'
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box around the hand from landmarks
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            padding = 30
            x_min = max(0, int(min(x_coords)) - padding)
            x_max = min(w, int(max(x_coords)) + padding)
            y_min = max(0, int(min(y_coords)) - padding)
            y_max = min(h, int(max(y_coords)) + padding)

            hand_crop = frame[y_min:y_max, x_min:x_max]

            if hand_crop.size > 0:
                gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0
                input_img = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

                pred = model.predict(input_img, verbose=0)
                confidence = np.max(pred)
                gesture = gesture_names[np.argmax(pred)]

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{gesture} ({confidence*100:.1f}%)", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Gesture Recognition v2 - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()