import cv2
import mediapipe as mp

from src.classifier import load_training_data, classify_hand_gesture

# Inicialización de MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

training_data = load_training_data()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Clasificar la letra basada en la posición de la mano
                gesture = classify_hand_gesture(hand_landmarks, training_data)
                if gesture:
                    cv2.putText(image, f"Letra: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Reconocimiento de gestos', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
