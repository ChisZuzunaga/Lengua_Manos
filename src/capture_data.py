import cv2
import mediapipe as mp
import json

def save_hand_position(letter, landmarks):
    data = {
        "letter": letter,
        "landmarks": [
            {"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks.landmark
        ]
    }
    with open("/data/hand_positions.json", "a") as file:
        file.write(json.dumps(data) + "\n")

# Inicialización de MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

                # Guarda la posición de la mano cuando se presiona una tecla
                if cv2.waitKey(1) & 0xFF == ord('a'):
                    save_hand_position("A", hand_landmarks)
                    print("Posición guardada para 'A'.")

                elif cv2.waitKey(1) & 0xFF == ord('e'):
                    save_hand_position("E", hand_landmarks)
                    print("Posición guardada para 'E'.")

                elif cv2.waitKey(1) & 0xFF == ord('i'):
                    save_hand_position("I", hand_landmarks)
                    print("Posición guardada para 'I'.")

                elif cv2.waitKey(1) & 0xFF == ord('o'):
                    save_hand_position("O", hand_landmarks)
                    print("Posición guardada para 'O'.")

                elif cv2.waitKey(1) & 0xFF == ord('u'):
                    save_hand_position("U", hand_landmarks)
                    print("Posición guardada para 'U'.") 

        cv2.imshow('Captura de datos', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
